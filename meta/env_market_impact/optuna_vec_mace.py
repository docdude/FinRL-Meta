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
from meta.env_market_impact.backtest_vec_config import VecMACEEnvParams
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.envs.utils import get_logger
from meta.env_market_impact.vec.data_prep import build_vec_tiingo_market_data_preparator
from meta.env_market_impact.vec.mace_vec_env import MACEVecEnv
from meta.env_market_impact.vec.runner_utils import build_tensor_impact_model
from meta.env_market_impact.vec.runner_utils import impact_model_name
from meta.env_market_impact.vec.runner_utils import resolve_default_num_envs
from meta.env_market_impact.vec.runner_utils import resolve_elegantrl_settings
from meta.env_market_impact.vec.runner_utils import run_vec_simulation_stats
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


def _default_max_stock_pct(num_stocks: int) -> float:
    return float(np.clip((1.0 / num_stocks) * 2.0, 0.01, 1.0))


def _load_mace_normalizer(env: MACEVecEnv, normalizer_state_path: str) -> None:
    if env.params.use_obs_normalizer and os.path.isfile(normalizer_state_path):
        env.load_normalizer_state(normalizer_state_path, freeze=True)


def sample_env_params(trial: optuna.Trial, num_stocks: int) -> VecMACEEnvParams:
    return VecMACEEnvParams(
        max_stock_pct=_default_max_stock_pct(num_stocks),
        max_trade_volume_pct=trial.suggest_float("max_trade_volume_pct", 0.05, 0.2),
        reward_scaling=trial.suggest_float("reward_scaling", 2**-14, 2**-8, log=True),
        include_permanent_impact_in_state=trial.suggest_categorical(
            "include_permanent_impact_in_state", [True, False]
        ),
        include_cooldown_in_state=trial.suggest_categorical(
            "include_cooldown_in_state", [True, False]
        ),
        include_tbill_in_state=trial.suggest_categorical(
            "include_tbill_in_state", [True, False]
        ),
        sharpe_window=trial.suggest_categorical("sharpe_window", [10, 20, 40]),
        horizon=trial.suggest_categorical("horizon", [10, 20, 40, 80]),
        eta_dd=trial.suggest_float("eta_dd", 0.0, 3.0),
        use_obs_normalizer=trial.suggest_categorical("use_obs_normalizer", [True, False]),
        obs_clip=trial.suggest_categorical("obs_clip", [5.0, 10.0, 20.0]),
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
    env_params = sample_env_params(trial, data_prep.universe_size)
    model_kwargs = sample_model_kwargs(trial, model_name)

    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)
    requested_num_envs = resolve_default_num_envs(num_envs, gpu_id=gpu_id)
    user_forced_num_envs = num_envs is not None
    stock_dim = len(train_config["tic_list"])
    steps_per_epoch = max(1, len(train_config["date_list"]) - 1)
    run_dir = os.path.join(
        results_dir,
        f"trial_{trial.number:04d}_{uuid.uuid4().hex[:8]}",
    )
    normalizer_state_path = os.path.join(run_dir, "vec_normalize.pt")

    probe_train_env_kwargs = {
        "config": train_config,
        "params": env_params,
        "num_envs": requested_num_envs,
        "gpu_id": gpu_id,
        "initial_capital": initial_capital,
        "auto_reset": True,
    }
    resolved_settings = resolve_elegantrl_settings(
        model_name,
        model_kwargs,
        None,
        steps_per_epoch=steps_per_epoch,
        env_class=MACEVecEnv,
        train_env_kwargs=probe_train_env_kwargs,
        eval_env_kwargs=dict(probe_train_env_kwargs),
        requested_num_envs=requested_num_envs,
        gpu_id=gpu_id,
        num_workers=num_workers,
        disable_num_envs_scaling=user_forced_num_envs,
    )
    effective_num_envs = int(resolved_settings["num_envs"])

    train_env_kwargs = {
        "config": train_config,
        "params": env_params,
        "num_envs": effective_num_envs,
        "gpu_id": gpu_id,
        "impact_model": build_tensor_impact_model(
            impact_model_class,
            num_envs=effective_num_envs,
            stock_dim=stock_dim,
            gpu_id=gpu_id,
        ),
        "initial_capital": initial_capital,
        "normalizer_state_path": normalizer_state_path,
        "auto_reset": True,
    }
    eval_env_kwargs = {
        "config": train_config,
        "params": env_params,
        "num_envs": effective_num_envs,
        "gpu_id": gpu_id,
        "impact_model": build_tensor_impact_model(
            impact_model_class,
            num_envs=effective_num_envs,
            stock_dim=stock_dim,
            gpu_id=gpu_id,
        ),
        "initial_capital": initial_capital,
        "normalizer_state_path": normalizer_state_path,
        "auto_reset": True,
    }

    def evaluate_epoch(actor):
        train_env = MACEVecEnv(
            config=train_config,
            params=env_params,
            num_envs=1,
            gpu_id=gpu_id,
            impact_model=build_tensor_impact_model(
                impact_model_class,
                num_envs=1,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            ),
            initial_capital=initial_capital,
            auto_reset=False,
        )
        _load_mace_normalizer(train_env, normalizer_state_path)
        train_stats = run_vec_simulation_stats(train_env, actor, reset_impact_model=True)

        blank_env = MACEVecEnv(
            config=trade_config,
            params=env_params,
            num_envs=1,
            gpu_id=gpu_id,
            impact_model=build_tensor_impact_model(
                impact_model_class,
                num_envs=1,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            ),
            initial_capital=initial_capital,
            auto_reset=False,
        )
        _load_mace_normalizer(blank_env, normalizer_state_path)
        blank_stats = run_vec_simulation_stats(blank_env, actor, reset_impact_model=True)
        return train_stats, blank_stats

    try:
        _, epoch_stats_train, epoch_stats_test_blank, _ = train_with_epoch_evaluation(
            env_class=MACEVecEnv,
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
        log.warning("Trial %s failed for vec MACE %s/%s: %s", trial.number, model_name, impact_model_name(impact_model_class), exc)
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
        results_dir = f"hpo_results/vec_mace_{model_name}_{impact_slug}_{timestamp}"
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
        study_name=f"vec_mace_{model_name}_{impact_model_name(impact_model_class).replace(' ', '_').lower()}",
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
    parser = argparse.ArgumentParser(description="Optuna HPO for vec MACE + ElegantRL.")
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