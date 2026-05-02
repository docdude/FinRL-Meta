from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime

import numpy as np
import optuna
import torch as th
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER

from meta.env_market_impact.backtest_vec_config import VEC_IMPACT_MODEL_CLASSES
from meta.env_market_impact.backtest_vec_config import VEC_NET_DIMS
from meta.env_market_impact.backtest_vec_config import VecMarginEnvParams
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.envs.utils import get_logger
from meta.env_market_impact.vec.data_prep import build_vec_tiingo_market_data_preparator
from meta.env_market_impact.vec.margin_vec_env import MarginTraderVecEnv
from meta.env_market_impact.vec.runner_utils import build_tensor_impact_model
from meta.env_market_impact.vec.runner_utils import compute_stats_from_results
from meta.env_market_impact.vec.runner_utils import impact_model_name
from meta.env_market_impact.vec.runner_utils import resolve_default_num_envs
from meta.env_market_impact.vec.runner_utils import resolve_elegantrl_settings
from meta.env_market_impact.vec.runner_utils import run_vec_simulation
from meta.env_market_impact.vec.runner_utils import train_with_epoch_evaluation

log = get_logger()

IMPACT_MODEL_NAME_MAP = {
    str(impact_model_class()).lower(): impact_model_class
    for impact_model_class in VEC_IMPACT_MODEL_CLASSES
}


def _resolve_default_gpu_id(gpu_id: int | None) -> int:
    if gpu_id is not None:
        return gpu_id
    return 0 if th.cuda.is_available() else -1


def sample_env_params(trial: optuna.Trial) -> VecMarginEnvParams:
    maintenance_margin = trial.suggest_float("maintenance_margin", 0.2, 0.5)
    maintenance_warning = trial.suggest_float(
        "maintenance_warning",
        max(maintenance_margin + 0.05, 0.3),
        min(maintenance_margin + 0.25, 0.8),
    )
    return VecMarginEnvParams(
        max_stock_pct=trial.suggest_float("max_stock_pct", 0.01, 0.05),
        margin_rate=trial.suggest_float("margin_rate", 1.5, 3.0),
        long_short_ratio=trial.suggest_float("long_short_ratio", 0.5, 1.5),
        maintenance_margin=maintenance_margin,
        maintenance_warning=maintenance_warning,
        max_trade_volume_pct=trial.suggest_float("max_trade_volume_pct", 0.05, 0.2),
        lambda_1=trial.suggest_float("lambda_1", 1e-6, 1e-4, log=True),
        lambda_2=trial.suggest_float("lambda_2", 1e-4, 5e-2, log=True),
        sharpe_window=trial.suggest_categorical("sharpe_window", [5, 10, 20]),
        margin_adjust_period=trial.suggest_categorical("margin_adjust_period", [10, 20, 30, 60]),
    )


def sample_model_kwargs(trial: optuna.Trial, model_name: str) -> dict:
    net_dims_key = trial.suggest_categorical("net_dims_key", list(VEC_NET_DIMS))
    common = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "net_dims": VEC_NET_DIMS[net_dims_key],
        "eval_times": 1,
    }

    if model_name == "a2c":
        return {
            **common,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "repeat_times": trial.suggest_categorical("repeat_times", [1, 2, 4]),
            "lambda_gae_adv": trial.suggest_float("lambda_gae_adv", 0.8, 1.0),
            "lambda_entropy": trial.suggest_float("lambda_entropy", 1e-4, 0.05, log=True),
            "clip_grad_norm": 3.0,
            "if_use_v_trace": trial.suggest_categorical(
                "if_use_v_trace",
                [True, False],
            ),
        }
    if model_name == "ppo":
        return {
            **common,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "repeat_times": trial.suggest_categorical("repeat_times", [4, 8, 16]),
            "lambda_gae_adv": trial.suggest_float("lambda_gae_adv", 0.8, 1.0),
            "lambda_entropy": trial.suggest_float("lambda_entropy", 1e-4, 0.05, log=True),
            "clip_grad_norm": 3.0,
            "if_use_v_trace": trial.suggest_categorical(
                "if_use_v_trace",
                [True, False],
            ),
            "ratio_clip": 0.25,
        }
    if model_name == "ddpg":
        return {
            **common,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 500_000, 1_000_000]),
            "repeat_times": trial.suggest_categorical("repeat_times", [1, 2, 4]),
            "soft_update_tau": trial.suggest_float("soft_update_tau", 1e-3, 5e-2, log=True),
            "horizon_len": trial.suggest_categorical("horizon_len", [256, 512, 1024]),
            "reward_scale": 1.0,
            "clip_grad_norm": 3.0,
            "state_value_tau": 0.0,
            "if_use_per": False,
            "lambda_fit_cum_r": 0.0,
            "explore_noise": trial.suggest_categorical("explore_noise", [0.02, 0.05, 0.1]),
        }
    if model_name == "sac":
        return {
            **common,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 500_000, 1_000_000]),
            "repeat_times": trial.suggest_categorical("repeat_times", [1, 2, 4]),
            "soft_update_tau": trial.suggest_float("soft_update_tau", 1e-3, 5e-2, log=True),
            "horizon_len": trial.suggest_categorical("horizon_len", [256, 512, 1024]),
            "reward_scale": 1.0,
            "clip_grad_norm": 3.0,
            "state_value_tau": 0.0,
            "if_use_per": False,
            "lambda_fit_cum_r": 0.0,
            "num_ensembles": trial.suggest_categorical("num_ensembles", [2, 4, 8]),
        }
    if model_name == "td3":
        return {
            **common,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "buffer_size": trial.suggest_categorical("buffer_size", [100_000, 500_000, 1_000_000]),
            "repeat_times": trial.suggest_categorical("repeat_times", [1, 2, 4]),
            "soft_update_tau": trial.suggest_float("soft_update_tau", 1e-3, 5e-2, log=True),
            "horizon_len": trial.suggest_categorical("horizon_len", [256, 512, 1024]),
            "reward_scale": 1.0,
            "clip_grad_norm": 3.0,
            "state_value_tau": 0.0,
            "if_use_per": False,
            "lambda_fit_cum_r": 0.0,
            "update_freq": trial.suggest_categorical("update_freq", [2, 4]),
            "num_ensembles": trial.suggest_categorical("num_ensembles", [4, 8]),
            "policy_noise_std": trial.suggest_categorical("policy_noise_std", [0.05, 0.1, 0.2]),
            "explore_noise_std": trial.suggest_categorical("explore_noise_std", [0.02, 0.05, 0.1]),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def objective(
    trial: optuna.Trial,
    *,
    data_prep,
    model_name: str,
    impact_model_class: type,
    initial_capital: float,
    num_epochs: int,
    num_envs: int | None,
    gpu_id: int,
    num_workers: int,
    seed: int,
    results_dir: str,
) -> float:
    env_params = sample_env_params(trial)
    model_kwargs = sample_model_kwargs(trial, model_name)

    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)
    train_benchmark_df = data_prep.get_benchmark_df(Split.TRAIN)
    trade_benchmark_df = data_prep.get_benchmark_df(Split.TRADE)
    requested_num_envs = resolve_default_num_envs(num_envs, gpu_id=gpu_id)
    stock_dim = len(train_config["tic_list"])
    steps_per_epoch = max(1, len(train_config["date_list"]) - 1)
    run_dir = os.path.join(
        results_dir,
        f"trial_{trial.number:04d}_{uuid.uuid4().hex[:8]}",
    )

    probe_train_env_kwargs = {
        "config": train_config,
        "params": env_params,
        "initial_capital": initial_capital,
        "num_envs": requested_num_envs,
        "gpu_id": gpu_id,
        "auto_reset": True,
    }
    resolved_settings = resolve_elegantrl_settings(
        model_name,
        model_kwargs,
        None,
        steps_per_epoch=steps_per_epoch,
        env_class=MarginTraderVecEnv,
        train_env_kwargs=probe_train_env_kwargs,
        eval_env_kwargs=dict(probe_train_env_kwargs),
        requested_num_envs=requested_num_envs,
        gpu_id=gpu_id,
        num_workers=num_workers,
    )
    effective_num_envs = int(resolved_settings["num_envs"])

    train_env_kwargs = {
        "config": train_config,
        "params": env_params,
        "initial_capital": initial_capital,
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
        "params": env_params,
        "initial_capital": initial_capital,
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
        train_env = MarginTraderVecEnv(
            config=train_config,
            params=env_params,
            initial_capital=initial_capital,
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
        train_results_df, train_trades_df = run_vec_simulation(
            train_env,
            actor,
            train_config["date_list"],
            train_benchmark_df,
            reset_impact_model=True,
        )

        blank_env = MarginTraderVecEnv(
            config=trade_config,
            params=env_params,
            initial_capital=initial_capital,
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

    try:
        _, epoch_stats_train, epoch_stats_test_blank, _ = train_with_epoch_evaluation(
            env_class=MarginTraderVecEnv,
            train_env_kwargs=train_env_kwargs,
            eval_env_kwargs=eval_env_kwargs,
            agent_name=model_name,
            model_kwargs=model_kwargs,
            policy_kwargs=None,
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            run_dir=run_dir,
            evaluate_epoch=evaluate_epoch,
            gpu_id=gpu_id,
            num_workers=num_workers,
            random_seed=seed,
            if_single_process=True,
            resolved_settings=resolved_settings,
        )
    except Exception as exc:
        log.warning("Trial %s failed for vec margin %s/%s: %s", trial.number, model_name, impact_model_name(impact_model_class), exc)
        raise optuna.TrialPruned() from exc

    best_index, best_blank = max(
        enumerate(epoch_stats_test_blank),
        key=lambda item: item[1]["annualized_sharpe"],
    )
    trial.set_user_attr("best_epoch", int(best_blank["epoch"]))
    trial.set_user_attr(
        "train_sharpe_at_best_epoch",
        float(epoch_stats_train[best_index]["annualized_sharpe"]),
    )
    trial.set_user_attr("model_kwargs", model_kwargs)
    trial.set_user_attr("resolved_num_envs", effective_num_envs)
    return float(best_blank["annualized_sharpe"])


def _save_best_params(study: optuna.Study, output_path: str) -> None:
    payload = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "trial_number": study.best_trial.number,
        "params": study.best_params,
        "user_attrs": study.best_trial.user_attrs,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_study(
    *,
    model_name: str,
    impact_model_name_input: str,
    n_trials: int,
    num_epochs: int,
    num_envs: int | None,
    gpu_id: int | None,
    num_workers: int,
    seed: int,
    results_dir: str | None,
    storage_path: str | None,
) -> optuna.Study:
    gpu_id = _resolve_default_gpu_id(gpu_id)
    impact_model_class = IMPACT_MODEL_NAME_MAP[impact_model_name_input.strip().lower()]

    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        impact_slug = impact_model_name(impact_model_class).replace(" ", "_").lower()
        results_dir = f"hpo_results/vec_margin_{model_name}_{impact_slug}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    data_prep = build_vec_tiingo_market_data_preparator(
        tickers=NAS_100_TICKER,
        start_date="2010-01-01",
        end_date="2026-01-01",
        tech_indicators=INDICATORS,
        train_ratio=0.9,
        benchmark_ticker="QQEW",
    )

    storage = None
    if storage_path is not None:
        os.makedirs(os.path.dirname(storage_path) or ".", exist_ok=True)
        storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="maximize",
        study_name=f"vec_margin_{model_name}_{impact_model_name(impact_model_class).replace(' ', '_').lower()}",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            data_prep=data_prep,
            model_name=model_name,
            impact_model_class=impact_model_class,
            initial_capital=1e9,
            num_epochs=num_epochs,
            num_envs=num_envs,
            gpu_id=gpu_id,
            num_workers=num_workers,
            seed=seed,
            results_dir=results_dir,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    _save_best_params(study, os.path.join(results_dir, "best_params.json"))
    return study


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna HPO for vec Margin + ElegantRL.")
    parser.add_argument("--agent", default="a2c")
    parser.add_argument("--impact-model", default="baseline impact model")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--storage-path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_study(
        model_name=args.agent.strip().lower(),
        impact_model_name_input=args.impact_model,
        n_trials=args.n_trials,
        num_epochs=args.num_epochs,
        num_envs=args.num_envs,
        gpu_id=args.gpu_id,
        num_workers=args.num_workers,
        seed=args.seed,
        results_dir=args.results_dir,
        storage_path=args.storage_path,
    )