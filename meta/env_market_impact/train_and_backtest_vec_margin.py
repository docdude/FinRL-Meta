from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import torch as th
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from meta.env_market_impact.backtest_vec_config import VEC_IMPACT_MODEL_CLASSES
from meta.env_market_impact.backtest_vec_config import VEC_MODEL_KWARGS
from meta.env_market_impact.backtest_vec_config import VecMarginBacktestParams
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.envs.utils import get_logger
from meta.env_market_impact.vec.data_prep import build_vec_tiingo_market_data_preparator
from meta.env_market_impact.vec.margin_vec_env import MarginTraderVecEnv
from meta.env_market_impact.vec.runner_utils import SUPPORTED_ELEGANTRL_AGENTS
from meta.env_market_impact.vec.runner_utils import build_tensor_impact_model
from meta.env_market_impact.vec.runner_utils import compute_stats_from_results
from meta.env_market_impact.vec.runner_utils import impact_model_name
from meta.env_market_impact.vec.runner_utils import resolve_default_num_envs
from meta.env_market_impact.vec.runner_utils import resolve_elegantrl_settings
from meta.env_market_impact.vec.runner_utils import run_vec_simulation
from meta.env_market_impact.vec.runner_utils import save_backtest_summary
from meta.env_market_impact.vec.runner_utils import save_vec_backtest_triplet
from meta.env_market_impact.vec.runner_utils import train_with_epoch_evaluation

log = get_logger()


def _resolve_default_gpu_id(gpu_id: int | None) -> int:
    if gpu_id is not None:
        return gpu_id
    return 0 if th.cuda.is_available() else -1


def train_and_backtest(
    data_prep,
    backtest_grid: list[VecMarginBacktestParams],
    num_epochs: int = 20,
    *,
    num_envs: int | None = None,
    gpu_id: int | None = None,
    num_workers: int = 1,
    random_seed: int = 42,
) -> str:
    gpu_id = _resolve_default_gpu_id(gpu_id)
    requested_num_envs = resolve_default_num_envs(num_envs, gpu_id=gpu_id)
    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)
    train_benchmark_df = data_prep.get_benchmark_df(Split.TRAIN)
    trade_benchmark_df = data_prep.get_benchmark_df(Split.TRADE)

    run_id = str(uuid.uuid4())
    results_dir = f"backtest_results/margin_trader_vec_{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    all_backtests_metadata = []
    stock_dim = len(train_config["tic_list"])
    steps_per_epoch = max(1, len(train_config["date_list"]) - 1)

    for idx, params in enumerate(backtest_grid, 1):
        model_name = params.model_name.lower()
        if model_name not in SUPPORTED_ELEGANTRL_AGENTS:
            raise ValueError(
                f"Vec Margin currently supports {SUPPORTED_ELEGANTRL_AGENTS}, "
                f"got '{params.model_name}'."
            )

        impact_model_class = params.impact_model_class
        model_kwargs = params.get_model_kwargs()
        policy_kwargs = params.policy_kwargs

        impact_name = impact_model_name(impact_model_class)
        base_filename = f"mt_vec_{model_name}_{impact_name.replace(' ', '_')}_{idx}"
        run_dir = os.path.join(results_dir, f"{base_filename}_elegantrl")

        log.info(f"[{idx}/{len(configs)}] {model_name.upper()} | {impact_name}")

        probe_train_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "initial_capital": params.initial_capital,
            "num_envs": requested_num_envs,
            "gpu_id": gpu_id,
            "auto_reset": True,
        }
        probe_eval_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "initial_capital": params.initial_capital,
            "num_envs": requested_num_envs,
            "gpu_id": gpu_id,
            "auto_reset": True,
        }
        resolved_settings = resolve_elegantrl_settings(
            model_name,
            model_kwargs,
            policy_kwargs,
            steps_per_epoch=steps_per_epoch,
            env_class=MarginTraderVecEnv,
            train_env_kwargs=probe_train_env_kwargs,
            eval_env_kwargs=probe_eval_env_kwargs,
            requested_num_envs=requested_num_envs,
            gpu_id=gpu_id,
            num_workers=num_workers,
        )
        effective_num_envs = int(resolved_settings["num_envs"])
        log.info(
            "  Vec profile: num_envs=%s horizon_len=%s "
            "batch_size=%s repeat_times=%s",
            effective_num_envs,
            resolved_settings["horizon_len"],
            resolved_settings["batch_size"],
            resolved_settings["repeat_times"],
        )

        train_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "initial_capital": params.initial_capital,
            "num_envs": effective_num_envs,
            "gpu_id": gpu_id,
            "impact_model": build_tensor_impact_model(
                impact_model_class,
                num_envs=effective_num_envs,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            ),
            "auto_reset": True,
        }
        eval_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "initial_capital": params.initial_capital,
            "num_envs": effective_num_envs,
            "gpu_id": gpu_id,
            "impact_model": build_tensor_impact_model(
                impact_model_class,
                num_envs=effective_num_envs,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            ),
            "auto_reset": True,
        }

        def evaluate_epoch(actor):
            continued_impact_model = build_tensor_impact_model(
                impact_model_class,
                num_envs=1,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            )
            train_eval_env = MarginTraderVecEnv(
                config=train_config,
                params=params.env_params,
                initial_capital=params.initial_capital,
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=continued_impact_model,
                auto_reset=False,
            )
            train_results_df, train_trades_df = run_vec_simulation(
                train_eval_env,
                actor,
                train_config["date_list"],
                train_benchmark_df,
                reset_impact_model=True,
            )

            blank_env = MarginTraderVecEnv(
                config=trade_config,
                params=params.env_params,
                initial_capital=params.initial_capital,
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=build_tensor_impact_model(
                    impact_model_class,
                    num_envs=1,
                    stock_dim=stock_dim,
                    gpu_id=gpu_id,
                ),
                auto_reset=False,
            )
            blank_results_df, blank_trades_df = run_vec_simulation(
                blank_env,
                actor,
                trade_config["date_list"],
                trade_benchmark_df,
                reset_impact_model=True,
            )
            return (
                compute_stats_from_results(train_results_df, train_trades_df),
                compute_stats_from_results(blank_results_df, blank_trades_df),
            )

        trained_actor, epoch_stats_train, epoch_stats_test_blank, _ = (
            train_with_epoch_evaluation(
                env_class=MarginTraderVecEnv,
                train_env_kwargs=train_env_kwargs,
                eval_env_kwargs=eval_env_kwargs,
                agent_name=model_name,
                model_kwargs=model_kwargs,
                policy_kwargs=policy_kwargs,
                num_epochs=num_epochs,
                steps_per_epoch=steps_per_epoch,
                run_dir=run_dir,
                evaluate_epoch=evaluate_epoch,
                gpu_id=gpu_id,
                num_workers=num_workers,
                random_seed=random_seed,
                if_single_process=True,
                resolved_settings=resolved_settings,
            )
        )

        log.info(
            f"  Train Sharpe={epoch_stats_train[-1]['annualized_sharpe']:.3f} | "
            f"Test Sharpe={epoch_stats_test_blank[-1]['annualized_sharpe']:.3f}"
        )

        continued_impact_model = build_tensor_impact_model(
            impact_model_class,
            num_envs=1,
            stock_dim=stock_dim,
            gpu_id=gpu_id,
        )
        train_eval_env = MarginTraderVecEnv(
            config=train_config,
            params=params.env_params,
            initial_capital=params.initial_capital,
            num_envs=1,
            gpu_id=gpu_id,
            impact_model=continued_impact_model,
            auto_reset=False,
        )
        artifacts = save_vec_backtest_triplet(
            actor=trained_actor,
            train_env=train_eval_env,
            build_trade_env=lambda evaluated_train_env: MarginTraderVecEnv(
                config=trade_config,
                params=params.env_params,
                initial_capital=float(evaluated_train_env.total_asset[0].item()),
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=evaluated_train_env.impact_model,
                initial_margin_state=evaluated_train_env.get_margin_state(),
                auto_reset=False,
            ),
            build_blank_env=lambda: MarginTraderVecEnv(
                config=trade_config,
                params=params.env_params,
                initial_capital=params.initial_capital,
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=build_tensor_impact_model(
                    impact_model_class,
                    num_envs=1,
                    stock_dim=stock_dim,
                    gpu_id=gpu_id,
                ),
                auto_reset=False,
            ),
            train_dates=train_config["date_list"],
            trade_dates=trade_config["date_list"],
            train_benchmark_df=train_benchmark_df,
            trade_benchmark_df=trade_benchmark_df,
            results_dir=results_dir,
            base_filename=base_filename,
        )

        all_backtests_metadata.append(
            {
                "drl_agent": model_name,
                "impact_model": impact_name,
                "initial_capital": params.initial_capital,
                "num_envs": resolved_settings.get("num_envs"),
                "requested_num_envs": resolved_settings.get(
                    "requested_num_envs"
                ),
                "results_csv_train": artifacts["results_csv_train"],
                "results_csv_test": artifacts["results_csv_test"],
                "results_csv_test_blank": artifacts["results_csv_test_blank"],
                "trades_csv_train": artifacts["trades_csv_train"],
                "trades_csv_test": artifacts["trades_csv_test"],
                "trades_csv_test_blank": artifacts["trades_csv_test_blank"],
                "learning_rate": resolved_settings.get("learning_rate"),
                "gamma": resolved_settings.get("gamma"),
                "batch_size": resolved_settings.get("batch_size"),
                "horizon_len": resolved_settings.get("horizon_len"),
                "repeat_times": resolved_settings.get("repeat_times"),
                "lambda_gae_adv": resolved_settings.get("lambda_gae_adv"),
                "lambda_entropy": resolved_settings.get("lambda_entropy"),
                "clip_grad_norm": resolved_settings.get("clip_grad_norm"),
                "ratio_clip": resolved_settings.get("ratio_clip"),
                "if_use_v_trace": resolved_settings.get("if_use_v_trace"),
                "net_arch": resolved_settings.get("net_dims"),
                "env_type": "margin_trader_vec",
                "training_engine": "elegantrl_vec",
                "model_kwargs": model_kwargs,
                "policy_kwargs": policy_kwargs,
                "epoch_stats_train": epoch_stats_train,
                "epoch_stats_test_blank": epoch_stats_test_blank,
            }
        )

    return save_backtest_summary(
        results_dir=results_dir,
        benchmark_ticker=data_prep.benchmark_ticker,
        all_backtests_metadata=all_backtests_metadata,
    )


def run_example() -> None:
    np.random.seed(42)

    start_date = "2010-01-01"
    end_date = "2026-01-01"
    num_epochs = 20

    data_prep = build_vec_tiingo_market_data_preparator(
        tickers=NAS_100_TICKER,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=INDICATORS,
        train_ratio=0.9,
        benchmark_ticker="QQEW",
    )

    configs = []
    for algo in SUPPORTED_ELEGANTRL_AGENTS:
        for impact_model_class in VEC_IMPACT_MODEL_CLASSES:
            config = {
                "model_name": algo,
                "impact_model_class": impact_model_class,
                "initial_capital": 1e9,
                "model_kwargs": dict(VEC_MODEL_KWARGS[algo]),
            }
            configs.append(config)

    backtest_grid = VecMarginBacktestParams.from_explicit(
        configs,
        num_stocks=data_prep.universe_size,
    )

    summary_path = train_and_backtest(
        data_prep,
        backtest_grid,
        num_epochs=num_epochs,
    )
    BacktestReportGenerator(summary_path).generate_report()
    log.info("Vec margin-trader backtests complete.")


if __name__ == "__main__":
    run_example()