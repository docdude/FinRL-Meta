from __future__ import annotations

import os
import sys
import uuid

import numpy as np
import torch as th
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from meta.env_market_impact.backtest_config import BacktestParams
from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from meta.env_market_impact.envs.impact_models import ACImpactModel
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.envs.utils import get_logger
from meta.env_market_impact.vec.data_prep import build_vec_tiingo_market_data_preparator
from meta.env_market_impact.vec.mace_vec_env import MACEVecEnv
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


def _load_mace_normalizer(env: MACEVecEnv, normalizer_state_path: str) -> None:
    if env.params.use_obs_normalizer and os.path.isfile(normalizer_state_path):
        env.load_normalizer_state(normalizer_state_path, freeze=True)


def train_and_backtest(
    data_prep,
    backtest_grid: list[BacktestParams],
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
    results_dir = f"backtest_results/mace_vec_{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    all_backtests_metadata = []
    stock_dim = len(train_config["tic_list"])
    steps_per_epoch = max(1, len(train_config["date_list"]) - 1)

    for idx, params in enumerate(backtest_grid, 1):
        if params.model_name.lower() not in SUPPORTED_ELEGANTRL_AGENTS:
            raise ValueError(
                f"Vec MACE currently supports {SUPPORTED_ELEGANTRL_AGENTS}, "
                f"got '{params.model_name}'."
            )

        base_filename = params.base_filename
        run_dir = os.path.join(results_dir, f"{base_filename}_elegantrl")
        normalizer_state_path = os.path.join(run_dir, "mace_obs_normalizer.pt")

        log.info(
            f"[{idx}/{len(backtest_grid)}] {params.model_name.upper()} | "
            f"{params.impact_model_name} | capital=${params.initial_capital:,.0f}"
        )

        probe_train_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "num_envs": requested_num_envs,
            "gpu_id": gpu_id,
            "initial_capital": params.initial_capital,
            "auto_reset": True,
        }
        probe_eval_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "num_envs": requested_num_envs,
            "gpu_id": gpu_id,
            "initial_capital": params.initial_capital,
            "auto_reset": True,
        }
        resolved_settings = resolve_elegantrl_settings(
            params.model_name,
            params.get_model_kwargs(),
            params.policy_kwargs,
            steps_per_epoch=steps_per_epoch,
            env_class=MACEVecEnv,
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
            "num_envs": effective_num_envs,
            "gpu_id": gpu_id,
            "impact_model": build_tensor_impact_model(
                params.impact_model_class,
                num_envs=effective_num_envs,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            ),
            "initial_capital": params.initial_capital,
            "normalizer_state_path": normalizer_state_path,
            "auto_reset": True,
        }
        eval_env_kwargs = {
            "config": train_config,
            "params": params.env_params,
            "num_envs": effective_num_envs,
            "gpu_id": gpu_id,
            "impact_model": build_tensor_impact_model(
                params.impact_model_class,
                num_envs=effective_num_envs,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            ),
            "initial_capital": params.initial_capital,
            "auto_reset": True,
        }

        def evaluate_epoch(actor):
            continued_impact_model = build_tensor_impact_model(
                params.impact_model_class,
                num_envs=1,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            )
            eval_train_env = MACEVecEnv(
                config=train_config,
                params=params.env_params,
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=continued_impact_model,
                initial_capital=params.initial_capital,
                auto_reset=False,
            )
            _load_mace_normalizer(eval_train_env, normalizer_state_path)
            train_results_df, train_trades_df = run_vec_simulation(
                eval_train_env,
                actor,
                train_config["date_list"],
                train_benchmark_df,
                reset_impact_model=True,
            )

            blank_env = MACEVecEnv(
                config=trade_config,
                params=params.env_params,
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=build_tensor_impact_model(
                    params.impact_model_class,
                    num_envs=1,
                    stock_dim=stock_dim,
                    gpu_id=gpu_id,
                ),
                initial_capital=params.initial_capital,
                auto_reset=False,
            )
            _load_mace_normalizer(blank_env, normalizer_state_path)
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
                env_class=MACEVecEnv,
                train_env_kwargs=train_env_kwargs,
                eval_env_kwargs=eval_env_kwargs,
                agent_name=params.model_name,
                model_kwargs=params.get_model_kwargs(),
                policy_kwargs=params.policy_kwargs,
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
            params.impact_model_class,
            num_envs=1,
            stock_dim=stock_dim,
            gpu_id=gpu_id,
        )
        train_eval_env = MACEVecEnv(
            config=train_config,
            params=params.env_params,
            num_envs=1,
            gpu_id=gpu_id,
            impact_model=continued_impact_model,
            initial_capital=params.initial_capital,
            auto_reset=False,
        )
        _load_mace_normalizer(train_eval_env, normalizer_state_path)
        artifacts = save_vec_backtest_triplet(
            actor=trained_actor,
            train_env=train_eval_env,
            build_trade_env=lambda evaluated_train_env: _build_mace_trade_env(
                trade_config=trade_config,
                params=params,
                gpu_id=gpu_id,
                train_eval_env=evaluated_train_env,
                normalizer_state_path=normalizer_state_path,
            ),
            build_blank_env=lambda: _build_mace_blank_env(
                trade_config=trade_config,
                params=params,
                gpu_id=gpu_id,
                stock_dim=stock_dim,
                normalizer_state_path=normalizer_state_path,
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
                "drl_agent": params.model_name,
                "impact_model": impact_model_name(params.impact_model_class),
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
                "with_perm": params.env_params.include_permanent_impact_in_state,
                "with_cooldown": params.env_params.include_cooldown_in_state,
                "with_tbill": params.env_params.include_tbill_in_state,
                "eta_dd": params.env_params.eta_dd,
                "use_obs_normalizer": params.env_params.use_obs_normalizer,
                "reward_scaling": params.env_params.reward_scaling,
                "horizon": params.env_params.horizon,
                "obs_clip": params.env_params.obs_clip,
                "learning_rate": resolved_settings.get("learning_rate"),
                "gamma": resolved_settings.get("gamma"),
                "batch_size": resolved_settings.get("batch_size"),
                "horizon_len": resolved_settings.get("horizon_len"),
                "repeat_times": resolved_settings.get("repeat_times"),
                "net_arch": resolved_settings.get("net_dims"),
                "training_engine": "elegantrl_vec",
                "epoch_stats_train": epoch_stats_train,
                "epoch_stats_test_blank": epoch_stats_test_blank,
            }
        )

    return save_backtest_summary(
        results_dir=results_dir,
        benchmark_ticker=data_prep.benchmark_ticker,
        all_backtests_metadata=all_backtests_metadata,
    )


def _build_mace_trade_env(
    *,
    trade_config,
    params: BacktestParams,
    gpu_id: int,
    train_eval_env: MACEVecEnv,
    normalizer_state_path: str,
) -> MACEVecEnv:
    trade_env = MACEVecEnv(
        config=trade_config,
        params=params.env_params,
        num_envs=1,
        gpu_id=gpu_id,
        impact_model=train_eval_env.impact_model,
        initial_capital=float(train_eval_env.cash[0].item()),
        initial_stocks=train_eval_env.stocks[0].clone(),
        auto_reset=False,
    )
    _load_mace_normalizer(trade_env, normalizer_state_path)
    return trade_env


def _build_mace_blank_env(
    *,
    trade_config,
    params: BacktestParams,
    gpu_id: int,
    stock_dim: int,
    normalizer_state_path: str,
) -> MACEVecEnv:
    blank_env = MACEVecEnv(
        config=trade_config,
        params=params.env_params,
        num_envs=1,
        gpu_id=gpu_id,
        impact_model=build_tensor_impact_model(
            params.impact_model_class,
            num_envs=1,
            stock_dim=stock_dim,
            gpu_id=gpu_id,
        ),
        initial_capital=params.initial_capital,
        auto_reset=False,
    )
    _load_mace_normalizer(blank_env, normalizer_state_path)
    return blank_env


def run_example() -> None:
    np.random.seed(42)
    log.info("Preparing Tiingo-backed vec MACE data...")

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

    backtest_grid = BacktestParams.from_explicit(
        [
            {
                "model_name": "a2c",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
                "policy_kwargs": {"net_arch": [256, 128]},
                "eta_dd": 0.5,
                "horizon": 10,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "use_obs_normalizer": True,
            },
            {
                "model_name": "ppo",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
                "policy_kwargs": {"net_arch": [256, 128]},
                "eta_dd": 0.5,
                "horizon": 10,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "use_obs_normalizer": True,
            },
        ],
        num_stocks=data_prep.universe_size,
    )

    summary_path = train_and_backtest(data_prep, backtest_grid, num_epochs=num_epochs)
    BacktestReportGenerator(summary_path).generate_report()
    log.info("Vec MACE backtests complete.")


if __name__ == "__main__":
    run_example()