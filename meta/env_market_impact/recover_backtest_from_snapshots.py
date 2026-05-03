#!/usr/bin/env python3
"""
Recover backtests from epoch snapshots after a crashed training run.

Usage:
    python -m meta.env_market_impact.recover_backtest_from_snapshots \
        --run-dir backtest_results/mace_vec_XXXX/backtest_a2c_..._elegantrl \
        --agent a2c \
        --impact-model "Almgren-Chriss Impact Model" \
        [--use-intensity-wrapper]  \
        [--gpu-id 0]

This script:
  1. Finds all actor__*.pt + normalizer__*.pt in epoch_snapshots/
  2. Rebuilds the env config (same data pipeline as training)
  3. Runs evaluate_epoch for each snapshot → epoch_stats
  4. Runs save_vec_backtest_triplet for the best epoch actor
  5. Writes backtest_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch as th
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from meta.env_market_impact.backtest_vec_config import VecMACEBacktestParams
from meta.env_market_impact.backtest_vec_config import VecMACEEnvParams
from meta.env_market_impact.backtest_vec_config import VEC_IMPACT_MODEL_CLASSES
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.envs.utils import get_logger
from meta.env_market_impact.vec.data_prep import (
    build_vec_tiingo_market_data_preparator,
)
from meta.env_market_impact.vec.mace_vec_env import MACEVecEnv
from meta.env_market_impact.vec.runner_utils import (
    SUPPORTED_ELEGANTRL_AGENTS,
    build_tensor_impact_model,
    build_training_args,
    load_trained_actor,
    resolve_elegantrl_settings,
    run_vec_simulation_stats,
    save_backtest_summary,
    save_vec_backtest_triplet,
    _load_epoch_steps,
    _epoch_actor_snapshot_path,
    _epoch_normalizer_snapshot_path,
    _restore_epoch_normalizer_snapshot,
)

log = get_logger()

IMPACT_MODEL_MAP = {
    str(cls()).lower(): cls for cls in VEC_IMPACT_MODEL_CLASSES
}


def _resolve_impact_class(name: str):
    key = name.strip().lower()
    if key in IMPACT_MODEL_MAP:
        return IMPACT_MODEL_MAP[key]
    # Fuzzy match
    for k, v in IMPACT_MODEL_MAP.items():
        if name.lower().replace("-", "").replace(" ", "") in k.replace("-", "").replace(" ", ""):
            return v
    raise ValueError(f"Unknown impact model '{name}'. Available: {list(IMPACT_MODEL_MAP)}")


def _load_mace_normalizer(env, path):
    if env.params.use_obs_normalizer and os.path.isfile(path):
        env.load_normalizer_state(path, freeze=True)


def recover(
    run_dir: str,
    agent_name: str,
    impact_model_name: str,
    use_intensity_wrapper: bool = False,
    gpu_id: int = 0,
    num_epochs: int = 20,
):
    np.random.seed(42)
    impact_model_class = _resolve_impact_class(impact_model_name)

    # Check snapshots exist
    snapshots_dir = os.path.join(run_dir, "epoch_snapshots")
    if not os.path.isdir(snapshots_dir):
        raise FileNotFoundError(f"No epoch_snapshots in {run_dir}")

    actor_files = sorted(f for f in os.listdir(snapshots_dir) if f.startswith("actor__"))
    log.info(f"Found {len(actor_files)} epoch snapshots in {snapshots_dir}")

    # Rebuild data pipeline
    log.info("Preparing market data (same pipeline as training)...")
    data_prep = build_vec_tiingo_market_data_preparator(
        tickers=NAS_100_TICKER,
        start_date="2010-01-01",
        end_date="2026-01-01",
        tech_indicators=INDICATORS,
        train_ratio=0.9,
        benchmark_ticker="QQEW",
    )
    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)
    train_benchmark_df = data_prep.get_benchmark_df(Split.TRAIN)
    trade_benchmark_df = data_prep.get_benchmark_df(Split.TRADE)
    stock_dim = len(train_config["tic_list"])
    steps_per_epoch = max(1, len(train_config["date_list"]) - 1)

    # Build env params matching training
    from meta.env_market_impact.backtest_vec_config import VEC_MODEL_KWARGS
    model_kwargs = dict(VEC_MODEL_KWARGS.get(agent_name, {}))

    # Use optuna preset if it was an a2c run
    from meta.env_market_impact.backtest_vec_config import apply_vec_mace_a2c_preset
    config_dict = {
        "model_name": agent_name,
        "impact_model_class": impact_model_class,
        "initial_capital": 1e9,
        "model_kwargs": model_kwargs,
    }
    if agent_name == "a2c":
        config_dict = apply_vec_mace_a2c_preset(config_dict, "optuna-20260420")

    params = VecMACEBacktestParams.from_explicit([config_dict], num_stocks=stock_dim)[0]
    normalizer_state_path = os.path.join(run_dir, "vec_normalize.pt")

    # Determine env class
    if use_intensity_wrapper:
        from meta.env_market_impact.train_and_backtest_vec_mace_intensity import (
            IntensityMACEVecEnv,
            INTENSITY_DEFAULTS,
            _wrap_with_intensity,
        )
        env_class = IntensityMACEVecEnv
        extra_env_kwargs = {"intensity_kwargs": dict(INTENSITY_DEFAULTS)}
    else:
        env_class = MACEVecEnv
        extra_env_kwargs = {}

    # Build args for actor loading
    train_env_kwargs = {
        "config": train_config,
        "params": params.env_params,
        "num_envs": 1,
        "gpu_id": gpu_id,
        "impact_model": build_tensor_impact_model(
            impact_model_class, num_envs=1, stock_dim=stock_dim, gpu_id=gpu_id,
        ),
        "initial_capital": params.initial_capital,
        "normalizer_state_path": normalizer_state_path,
        "auto_reset": True,
        **extra_env_kwargs,
    }

    resolved_settings = resolve_elegantrl_settings(
        agent_name,
        model_kwargs,
        params.policy_kwargs,
        steps_per_epoch=steps_per_epoch,
        env_class=env_class,
        train_env_kwargs=train_env_kwargs,
        eval_env_kwargs=train_env_kwargs,
        requested_num_envs=1,
        gpu_id=gpu_id,
        disable_num_envs_scaling=True,
    )

    args = build_training_args(
        env_class=env_class,
        train_env_kwargs=train_env_kwargs,
        eval_env_kwargs=train_env_kwargs,
        agent_name=agent_name,
        model_kwargs=model_kwargs,
        policy_kwargs=params.policy_kwargs,
        steps_per_epoch=steps_per_epoch,
        epoch_index=0,
        run_dir=run_dir,
        gpu_id=gpu_id,
        num_workers=1,
        random_seed=42,
        resolved_settings=resolved_settings,
    )

    # Define evaluate_epoch
    def evaluate_epoch(actor):
        base_train = MACEVecEnv(
            config=train_config, params=params.env_params, num_envs=1,
            gpu_id=gpu_id,
            impact_model=build_tensor_impact_model(
                impact_model_class, num_envs=1, stock_dim=stock_dim, gpu_id=gpu_id,
            ),
            initial_capital=params.initial_capital, auto_reset=False,
        )
        _load_mace_normalizer(base_train, normalizer_state_path)
        eval_train_env = _wrap_with_intensity(base_train) if use_intensity_wrapper else base_train
        train_stats = run_vec_simulation_stats(eval_train_env, actor, reset_impact_model=True)

        base_blank = MACEVecEnv(
            config=trade_config, params=params.env_params, num_envs=1,
            gpu_id=gpu_id,
            impact_model=build_tensor_impact_model(
                impact_model_class, num_envs=1, stock_dim=stock_dim, gpu_id=gpu_id,
            ),
            initial_capital=params.initial_capital, auto_reset=False,
        )
        _load_mace_normalizer(base_blank, normalizer_state_path)
        blank_env = _wrap_with_intensity(base_blank) if use_intensity_wrapper else base_blank
        blank_stats = run_vec_simulation_stats(blank_env, actor, reset_impact_model=True)
        return train_stats, blank_stats

    # Evaluate all epochs
    epoch_steps = _load_epoch_steps(run_dir, num_epochs)
    epoch_stats_train = []
    epoch_stats_test_blank = []
    best_actor = None
    best_sr = -999

    for epoch, total_step in enumerate(epoch_steps, start=1):
        actor_path = _epoch_actor_snapshot_path(run_dir, total_step)
        if not os.path.isfile(actor_path):
            log.warning(f"  Epoch {epoch}: missing {actor_path}, skipping")
            continue

        _restore_epoch_normalizer_snapshot(
            normalizer_state_path,
            _epoch_normalizer_snapshot_path(run_dir, total_step),
        )
        actor = load_trained_actor(args, actor_path=actor_path)
        train_stats, test_blank_stats = evaluate_epoch(actor)
        train_stats["epoch"] = epoch
        test_blank_stats["epoch"] = epoch
        epoch_stats_train.append(train_stats)
        epoch_stats_test_blank.append(test_blank_stats)

        sr = test_blank_stats.get("annualized_sharpe", -999)
        log.info(
            f"  Epoch {epoch}/{len(epoch_steps)}: "
            f"Train SR={train_stats.get('annualized_sharpe', 0):.3f} | "
            f"Test SR={sr:.3f}"
        )
        if sr > best_sr:
            best_sr = sr
            best_actor = actor
            best_epoch = epoch
            best_total_step = total_step

    if best_actor is None:
        raise RuntimeError("No valid epoch snapshots found")

    log.info(f"\nBest epoch: {best_epoch} (SR={best_sr:.3f})")

    # Run full backtest triplet for best actor
    _restore_epoch_normalizer_snapshot(
        normalizer_state_path,
        _epoch_normalizer_snapshot_path(run_dir, best_total_step),
    )

    base_train_eval = MACEVecEnv(
        config=train_config, params=params.env_params, num_envs=1,
        gpu_id=gpu_id,
        impact_model=build_tensor_impact_model(
            impact_model_class, num_envs=1, stock_dim=stock_dim, gpu_id=gpu_id,
        ),
        initial_capital=params.initial_capital, auto_reset=False,
    )
    _load_mace_normalizer(base_train_eval, normalizer_state_path)
    train_eval_env = _wrap_with_intensity(base_train_eval) if use_intensity_wrapper else base_train_eval

    results_dir = os.path.dirname(run_dir)
    base_filename = os.path.basename(run_dir).replace("_elegantrl", "")

    def build_trade_env(evaluated_train_env):
        base_src = evaluated_train_env.base_env if use_intensity_wrapper else evaluated_train_env
        base_trade = MACEVecEnv(
            config=trade_config, params=params.env_params, num_envs=1,
            gpu_id=gpu_id,
            impact_model=base_src.impact_model,
            initial_capital=float(base_src.cash[0].item()),
            initial_stocks=base_src.stocks[0].clone(),
            auto_reset=False,
        )
        _load_mace_normalizer(base_trade, normalizer_state_path)
        return _wrap_with_intensity(base_trade) if use_intensity_wrapper else base_trade

    def build_blank_env():
        base_blank = MACEVecEnv(
            config=trade_config, params=params.env_params, num_envs=1,
            gpu_id=gpu_id,
            impact_model=build_tensor_impact_model(
                impact_model_class, num_envs=1, stock_dim=stock_dim, gpu_id=gpu_id,
            ),
            initial_capital=params.initial_capital, auto_reset=False,
        )
        _load_mace_normalizer(base_blank, normalizer_state_path)
        return _wrap_with_intensity(base_blank) if use_intensity_wrapper else base_blank

    artifacts = save_vec_backtest_triplet(
        actor=best_actor,
        train_env=train_eval_env,
        build_trade_env=build_trade_env,
        build_blank_env=build_blank_env,
        train_dates=train_config["date_list"],
        trade_dates=trade_config["date_list"],
        train_benchmark_df=train_benchmark_df,
        trade_benchmark_df=trade_benchmark_df,
        results_dir=results_dir,
        base_filename=base_filename + f"_recovered_ep{best_epoch}",
    )

    from meta.env_market_impact.vec.runner_utils import impact_model_name
    metadata = {
        "drl_agent": agent_name,
        "impact_model": impact_model_name(impact_model_class),
        "initial_capital": params.initial_capital,
        "intensity_wrapper": use_intensity_wrapper,
        "recovered_from": run_dir,
        "best_epoch": best_epoch,
        "best_test_sharpe": best_sr,
        "training_engine": "elegantrl_vec",
        "epoch_stats_train": epoch_stats_train,
        "epoch_stats_test_blank": epoch_stats_test_blank,
        **{k: v for k, v in artifacts.items() if k.endswith("_csv") or k.startswith("results_csv") or k.startswith("trades_csv")},
    }

    summary_path = save_backtest_summary(
        results_dir=results_dir,
        benchmark_ticker="QQEW",
        all_backtests_metadata=[metadata],
    )
    log.info(f"Saved recovery summary → {summary_path}")

    try:
        BacktestReportGenerator(summary_path).generate_report()
    except Exception as e:
        log.warning(f"Report generation failed: {e}")

    log.info("Recovery complete.")
    return summary_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Recover backtests from epoch snapshots")
    parser.add_argument("--run-dir", required=True, help="Path to the _elegantrl run directory")
    parser.add_argument("--agent", required=True, choices=sorted(SUPPORTED_ELEGANTRL_AGENTS))
    parser.add_argument("--impact-model", required=True, help="Impact model name")
    parser.add_argument("--use-intensity-wrapper", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    cli = _parse_args()
    recover(
        run_dir=cli.run_dir,
        agent_name=cli.agent,
        impact_model_name=cli.impact_model,
        use_intensity_wrapper=cli.use_intensity_wrapper,
        gpu_id=cli.gpu_id,
        num_epochs=cli.num_epochs,
    )
