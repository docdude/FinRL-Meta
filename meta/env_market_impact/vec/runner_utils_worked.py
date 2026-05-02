from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import pandas as pd
import torch as th

from meta.env_market_impact.envs.impact_models import ACImpactModel
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.env_market_impact.envs.impact_models import OWImpactModel
from meta.env_market_impact.envs.impact_models import SqrtImpactModel
from meta.env_market_impact.envs.utils import compute_performance_stats
from meta.env_market_impact.envs.utils import get_logger

from .common import resolve_device
from .elegantrl_bridge import build_elegantrl_config
from .tensor_impact import TensorACImpactConfig
from .tensor_impact import TensorACImpactModel
from .tensor_impact import TensorBaselineImpactConfig
from .tensor_impact import TensorBaselineImpactModel
from .tensor_impact import TensorImpactBase
from .tensor_impact import TensorImpactConfig
from .tensor_impact import TensorOWImpactConfig
from .tensor_impact import TensorOWImpactModel
from .tensor_impact import TensorSqrtImpactModel

log = get_logger()

ELEGANTRL_REPO = "/mnt/ssd_backup/ElegantRL"
SUPPORTED_ELEGANTRL_AGENTS = ("a2c", "ppo")
EMPTY_TRADES_COLUMNS = ["date", "step", "notional", "pov", "turnover_percentile"]
FULL_TRADES_COLUMNS = [
    "date",
    "step",
    "stock_idx",
    "side",
    "shares",
    "notional",
    "pov",
    "turnover_percentile",
]
DEFAULT_GPU_NUM_ENVS = 2**11
DEFAULT_CPU_NUM_ENVS = 128
GPU_NUM_ENVS_ALIGNMENT = 256
GPU_MEMORY_RESERVE_BYTES = 512 * 1024**2
GPU_WORKER_CONTEXT_BYTES = 445 * 1024**2
GPU_LEARNER_CONTEXT_BYTES = 205 * 1024**2
GPU_EVALUATOR_CONTEXT_BYTES = 164 * 1024**2

ELEGANTRL_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "a2c": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "repeat_times": 1,
        "gamma": 0.99,
        "lambda_gae_adv": 0.95,
        "lambda_entropy": 0.01,
        "net_dims": [256, 128],
        "eval_times": 1,
    },
    "ppo": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "repeat_times": 8,
        "gamma": 0.99,
        "lambda_gae_adv": 0.95,
        "lambda_entropy": 0.01,
        "net_dims": [256, 128],
        "eval_times": 1,
    },
}

MODEL_KWARG_ALIASES = {
    "n_steps": "horizon_len",
    "ent_coef": "lambda_entropy",
    "gae_lambda": "lambda_gae_adv",
}
ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME = "epoch_snapshots"


def ensure_elegantrl_on_path(repo_path: str = ELEGANTRL_REPO) -> str:
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return repo_path


def empty_trades_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=FULL_TRADES_COLUMNS)


def _epoch_snapshot_dir(run_dir: str) -> str:
    return os.path.join(run_dir, ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME)


def _epoch_actor_snapshot_path(run_dir: str, total_step: int) -> str:
    return os.path.join(
        _epoch_snapshot_dir(run_dir),
        f"actor__{int(total_step):012}.pt",
    )


def _epoch_normalizer_snapshot_path(run_dir: str, total_step: int) -> str:
    return os.path.join(
        _epoch_snapshot_dir(run_dir),
        f"normalizer__{int(total_step):012}.pt",
    )


def _load_epoch_steps(run_dir: str, num_epochs: int) -> list[int]:
    recorder_path = os.path.join(run_dir, "recorder.npy")
    if not os.path.isfile(recorder_path):
        raise RuntimeError(f"Expected ElegantRL recorder at {recorder_path}")

    recorder = np.load(recorder_path)
    if recorder.ndim != 2 or recorder.shape[1] < 1:
        raise RuntimeError(
            f"Unexpected ElegantRL recorder shape {recorder.shape!r}"
        )
    if recorder.shape[0] < num_epochs:
        raise RuntimeError(
            "ElegantRL recorded fewer evaluation points than requested "
            f"epochs ({recorder.shape[0]} < {num_epochs})"
        )
    return [int(step) for step in recorder[:num_epochs, 0].tolist()]


def _restore_epoch_normalizer_snapshot(
    normalizer_state_path: Optional[str],
    snapshot_path: str,
) -> None:
    if normalizer_state_path is None or not os.path.isfile(snapshot_path):
        return

    dir_name = os.path.dirname(normalizer_state_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    shutil.copyfile(snapshot_path, normalizer_state_path)


@contextlib.contextmanager
def _patch_elegantrl_evaluator_for_epoch_snapshots(run_dir: str):
    ensure_elegantrl_on_path()
    import elegantrl.train.evaluator as evaluator_module
    import elegantrl.train.run as run_module

    os.makedirs(_epoch_snapshot_dir(run_dir), exist_ok=True)
    original_run_evaluator = run_module.Evaluator
    original_module_evaluator = evaluator_module.Evaluator

    class SnapshotEvaluator(original_run_evaluator):
        def evaluate_and_save(
            self,
            actor: th.nn.Module,
            steps: int,
            exp_r: float,
            logging_tuple: tuple,
        ) -> None:
            recorder_len = len(self.recorder)
            super().evaluate_and_save(
                actor=actor,
                steps=steps,
                exp_r=exp_r,
                logging_tuple=logging_tuple,
            )
            if len(self.recorder) == recorder_len:
                return

            total_step = int(self.recorder[-1][0])
            os.makedirs(_epoch_snapshot_dir(run_dir), exist_ok=True)
            th.save(actor, _epoch_actor_snapshot_path(run_dir, total_step))
            if hasattr(self.env, "save"):
                self.env.save(
                    _epoch_normalizer_snapshot_path(run_dir, total_step)
                )

    run_module.Evaluator = SnapshotEvaluator
    evaluator_module.Evaluator = SnapshotEvaluator
    try:
        yield
    finally:
        run_module.Evaluator = original_run_evaluator
        evaluator_module.Evaluator = original_module_evaluator


def scalarize(value: Any) -> Any:
    if isinstance(value, th.Tensor):
        if value.ndim == 0:
            return value.item()
        if value.numel() == 1:
            return value.reshape(-1)[0].item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def select_env_value(value: Any, env_index: int = 0) -> Any:
    if isinstance(value, th.Tensor):
        if value.ndim == 0 or value.numel() == 1:
            return value
        return value[env_index]
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return value
        return value[env_index]
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], dict):
        return value[env_index]
    return value


def resolve_net_dims(policy_kwargs: Optional[dict[str, Any]]) -> Optional[list[int]]:
    if not policy_kwargs:
        return None

    net_arch = policy_kwargs.get("net_arch")
    if isinstance(net_arch, list):
        if net_arch and all(isinstance(item, (int, float)) for item in net_arch):
            return [int(item) for item in net_arch]
        return None

    if isinstance(net_arch, dict):
        for key in ("pi", "vf"):
            value = net_arch.get(key)
            if isinstance(value, list) and value:
                return [int(item) for item in value]
    return None


def resolve_default_num_envs(
    requested_num_envs: Optional[int],
    *,
    gpu_id: int,
) -> int:
    if requested_num_envs is not None:
        num_envs = int(requested_num_envs)
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        return num_envs

    if gpu_id >= 0 and th.cuda.is_available():
        return DEFAULT_GPU_NUM_ENVS
    return DEFAULT_CPU_NUM_ENVS


def _estimate_max_step(
    env_kwargs: Optional[dict[str, Any]],
    fallback_steps: int,
) -> int:
    if env_kwargs is None:
        return max(1, int(fallback_steps))

    config = env_kwargs.get("config")
    if isinstance(config, dict) and "date_list" in config:
        return max(1, len(config["date_list"]) - 1)
    return max(1, int(fallback_steps))


def _estimate_state_action_dims(
    env_class: type,
    env_kwargs: dict[str, Any],
) -> tuple[int, int]:
    config = env_kwargs["config"]
    stock_dim = int(np.asarray(config["price_array"]).shape[1])
    tech_dim = int(np.asarray(config["tech_array"]).shape[1])

    if env_class.__name__ == "MACEVecEnv":
        params = env_kwargs["params"]
        state_dim = 1 + (3 * stock_dim) + tech_dim
        if params.include_permanent_impact_in_state:
            state_dim += stock_dim
        if params.include_cooldown_in_state:
            state_dim += stock_dim
        if params.include_tbill_in_state:
            state_dim += 1
        return state_dim, stock_dim

    if env_class.__name__ == "MarginTraderVecEnv":
        return 6 + (4 * stock_dim) + tech_dim, stock_dim

    raise ValueError(
        f"Unsupported vec env class for scaling: {env_class.__name__}"
    )


def _scale_num_envs_for_available_gpu_memory(
    requested_num_envs: int,
    *,
    state_dim: int,
    action_dim: int,
    train_max_step: int,
    eval_max_step: int,
    gpu_id: int,
    num_workers: int,
) -> tuple[int, Optional[dict[str, float]]]:
    if requested_num_envs < 1:
        raise ValueError("requested_num_envs must be at least 1")

    if gpu_id < 0 or not th.cuda.is_available():
        return requested_num_envs, None

    try:
        gpu_free, gpu_total = th.cuda.mem_get_info(gpu_id)
    except Exception:
        return requested_num_envs, None

    worker_count = max(1, int(num_workers))
    usable = max(
        gpu_free - GPU_MEMORY_RESERVE_BYTES,
        gpu_total - 2 * 1024**3,
    )
    fixed_overhead = (
        worker_count * GPU_WORKER_CONTEXT_BYTES
        + GPU_LEARNER_CONTEXT_BYTES
        + GPU_EVALUATOR_CONTEXT_BYTES
    )
    buffer_budget = usable - fixed_overhead
    bytes_per_step = max(1, (state_dim + action_dim + 3) * 4)
    total_steps_per_env = max(
        1,
        (2 * worker_count * max(1, train_max_step)) + max(1, eval_max_step),
    )
    max_envs = max(1, int(buffer_budget // (total_steps_per_env * bytes_per_step)))
    if max_envs >= GPU_NUM_ENVS_ALIGNMENT:
        max_envs = (
            max_envs // GPU_NUM_ENVS_ALIGNMENT
        ) * GPU_NUM_ENVS_ALIGNMENT

    effective_num_envs = max(1, min(requested_num_envs, max_envs))
    return effective_num_envs, {
        "gpu_free_gib": gpu_free / 1024**3,
        "gpu_total_gib": gpu_total / 1024**3,
        "gpu_used_mib": (gpu_total - gpu_free) / 1024**2,
        "projected_gib": (
            fixed_overhead
            + (effective_num_envs * total_steps_per_env * bytes_per_step)
        )
        / 1024**3,
        "worker_count": float(worker_count),
        "train_max_step": float(train_max_step),
        "eval_max_step": float(eval_max_step),
    }


def resolve_elegantrl_settings(
    agent_name: str,
    model_kwargs: Optional[dict[str, Any]] = None,
    policy_kwargs: Optional[dict[str, Any]] = None,
    *,
    steps_per_epoch: int,
    env_class: Optional[type] = None,
    train_env_kwargs: Optional[dict[str, Any]] = None,
    eval_env_kwargs: Optional[dict[str, Any]] = None,
    requested_num_envs: Optional[int] = None,
    gpu_id: int = -1,
    num_workers: int = 1,
) -> dict[str, Any]:
    agent_key = agent_name.lower()
    if agent_key not in ELEGANTRL_MODEL_KWARGS:
        raise ValueError(
            f"Unsupported vec ElegantRL agent '{agent_name}'. "
            f"Available: {sorted(ELEGANTRL_MODEL_KWARGS)}"
        )

    settings = dict(ELEGANTRL_MODEL_KWARGS[agent_key])
    for key, value in (model_kwargs or {}).items():
        settings[MODEL_KWARG_ALIASES.get(key, key)] = value

    net_dims = resolve_net_dims(policy_kwargs)
    if net_dims is not None:
        settings["net_dims"] = net_dims

    resolved_num_envs = resolve_default_num_envs(
        requested_num_envs,
        gpu_id=gpu_id,
    )
    scaling_info = None
    if env_class is not None and train_env_kwargs is not None:
        state_dim, action_dim = _estimate_state_action_dims(
            env_class,
            train_env_kwargs,
        )
        train_max_step = _estimate_max_step(train_env_kwargs, steps_per_epoch)
        eval_max_step = _estimate_max_step(eval_env_kwargs, steps_per_epoch)
        resolved_num_envs, scaling_info = _scale_num_envs_for_available_gpu_memory(
            resolved_num_envs,
            state_dim=state_dim,
            action_dim=action_dim,
            train_max_step=train_max_step,
            eval_max_step=eval_max_step,
            gpu_id=gpu_id,
            num_workers=num_workers,
        )
        if scaling_info is not None and resolved_num_envs < int(
            resolve_default_num_envs(requested_num_envs, gpu_id=gpu_id)
        ):
            log.info(
                "Scaling vec num_envs %s -> %s "
                "(GPU %.1f/%.1f GiB free, used %.0f MiB, "
                "train_horizon=%s, eval_horizon=%s, workers=%s, proj=%.1f GiB)",
                resolve_default_num_envs(requested_num_envs, gpu_id=gpu_id),
                resolved_num_envs,
                scaling_info["gpu_free_gib"],
                scaling_info["gpu_total_gib"],
                scaling_info["gpu_used_mib"],
                int(scaling_info["train_max_step"]),
                int(scaling_info["eval_max_step"]),
                int(scaling_info["worker_count"]),
                scaling_info["projected_gib"],
            )

    settings["num_envs"] = int(resolved_num_envs)
    settings["requested_num_envs"] = int(
        resolve_default_num_envs(requested_num_envs, gpu_id=gpu_id)
    )

    settings["horizon_len"] = max(
        1,
        min(int(settings.get("horizon_len", steps_per_epoch)), int(steps_per_epoch)),
    )
    settings["repeat_times"] = max(1, int(settings.get("repeat_times", 1)))
    rollout_batch_size = max(
        1,
        settings["horizon_len"]
        * max(1, int(num_workers))
        * settings["num_envs"],
    )
    # ElegantRL's PPO/A2C on-policy update count is based on horizon_len,
    # not horizon_len * num_envs, so keep batch_size within that limit.
    max_update_batch_size = max(
        1,
        settings["horizon_len"] * settings["repeat_times"],
    )
    settings["batch_size"] = min(
        int(settings.get("batch_size", 128)),
        rollout_batch_size,
        max_update_batch_size,
    )
    settings["rollout_batch_size"] = rollout_batch_size
    settings["eval_times"] = int(settings.get("eval_times", 1))
    return settings


def build_tensor_impact_model(
    impact_model: object,
    *,
    num_envs: int,
    stock_dim: int,
    gpu_id: int = -1,
    device: Optional[str] = None,
) -> TensorImpactBase:
    resolved_device = resolve_device(gpu_id=gpu_id, device=device)
    impact_model_class = (
        impact_model if isinstance(impact_model, type) else type(impact_model)
    )
    source = impact_model if not isinstance(impact_model, type) else impact_model()

    if issubclass(impact_model_class, ACImpactModel):
        return TensorACImpactModel(
            num_envs=num_envs,
            stock_dim=stock_dim,
            device=resolved_device,
            config=TensorACImpactConfig(
                alpha=float(source.alpha),
                beta=float(source.beta),
                epsilon=float(source.epsilon),
                perm_half_life_days=float(source.perm_half_life_days),
            ),
        )

    if issubclass(impact_model_class, BaselineImpactModel):
        return TensorBaselineImpactModel(
            num_envs=num_envs,
            stock_dim=stock_dim,
            device=resolved_device,
            config=TensorBaselineImpactConfig(
                basis_points=float(source.basis_points),
                perm_half_life_days=float(source.perm_half_life_days),
            ),
        )

    if issubclass(impact_model_class, OWImpactModel):
        return TensorOWImpactModel(
            num_envs=num_envs,
            stock_dim=stock_dim,
            device=resolved_device,
            config=TensorOWImpactConfig(
                Y=float(source.Y),
                perm_fraction=float(source.perm_fraction),
                half_life_days=float(source.half_life_days),
                perm_half_life_days=float(source.perm_half_life_days),
            ),
        )

    if issubclass(impact_model_class, SqrtImpactModel):
        return TensorSqrtImpactModel(
            num_envs=num_envs,
            stock_dim=stock_dim,
            device=resolved_device,
            config=TensorImpactConfig(
                Y=float(source.Y),
                perm_fraction=float(source.perm_fraction),
                perm_half_life_days=float(source.perm_half_life_days),
            ),
        )

    raise ValueError(f"Unsupported impact model class: {impact_model_class!r}")


def impact_model_name(impact_model: object) -> str:
    if isinstance(impact_model, type):
        return str(impact_model())
    return str(impact_model)


def run_vec_simulation(
    env: Any,
    actor: th.nn.Module,
    dates: list[str],
    benchmark_df: pd.DataFrame,
    *,
    reset_impact_model: bool = True,
    initial_benchmark_value: Optional[float] = None,
    env_index: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_envs = int(getattr(env, "num_envs", 1))
    if env_index < 0 or env_index >= num_envs:
        raise ValueError(
            f"env_index must be in [0, {num_envs - 1}] for num_envs={num_envs}"
        )

    actor.eval()
    first_param = next(actor.parameters(), None)
    device = first_param.device if first_param is not None else th.device("cpu")
    state, _ = env.reset(options={"reset_impact_model": reset_impact_model})
    results_log: list[dict[str, Any]] = []
    trades_log: list[dict[str, Any]] = []
    last_asset_value = float(scalarize(select_env_value(env.total_asset, env_index)))

    start_value = (
        float(initial_benchmark_value)
        if initial_benchmark_value is not None
        else float(getattr(env, "initial_capital", last_asset_value))
    )
    benchmark_cumulative_return = benchmark_df["close"] / benchmark_df["close"].iloc[0]
    benchmark_value_series = start_value * benchmark_cumulative_return

    results_log.append(
        {
            "date": dates[env.time],
            "step": -1,
            "portfolio_value": float(
                scalarize(select_env_value(env.total_asset, env_index))
            ),
            "pnl": 0.0,
            "reward": 0.0,
            "turnover": 0.0,
            "cost": 0.0,
            "total_buy_value": 0.0,
            "total_sell_value": 0.0,
            "benchmark_value": float(benchmark_value_series.iloc[env.time]),
            "cash": float(scalarize(select_env_value(env.cash, env_index))),
        }
    )

    for step in range(env.max_step):
        tensor_state = state.to(device) if isinstance(state, th.Tensor) else th.as_tensor(
            state,
            dtype=th.float32,
            device=device,
        )
        if tensor_state.ndim == 1:
            tensor_state = tensor_state.unsqueeze(0)

        with th.no_grad():
            action = actor(tensor_state)

        state, reward, done, truncated, info = env.step(action)
        current_asset_value = float(
            scalarize(select_env_value(env.total_asset, env_index))
        )
        pnl = current_asset_value - last_asset_value
        last_asset_value = current_asset_value

        results_log.append(
            {
                "date": dates[env.time],
                "step": step,
                "portfolio_value": current_asset_value,
                "pnl": pnl,
                "reward": float(
                    scalarize(select_env_value(reward, env_index))
                ),
                "turnover": float(
                    scalarize(select_env_value(info["turnover"], env_index))
                ),
                "cost": float(
                    scalarize(select_env_value(info["cost"], env_index))
                ),
                "total_buy_value": float(
                    scalarize(select_env_value(info["total_buy_value"], env_index))
                ),
                "total_sell_value": float(
                    scalarize(select_env_value(info["total_sell_value"], env_index))
                ),
                "benchmark_value": float(benchmark_value_series.iloc[env.time]),
                "cash": float(
                    scalarize(select_env_value(info["cash"], env_index))
                ),
            }
        )

        for trade in info.get("trades", []):
            trades_log.append(
                {
                    "date": dates[env.time],
                    "step": step,
                    **trade,
                }
            )

        if bool(scalarize(select_env_value(done, env_index))) or bool(
            scalarize(select_env_value(truncated, env_index))
        ):
            break

    if trades_log:
        return pd.DataFrame(results_log), pd.DataFrame(trades_log)
    return pd.DataFrame(results_log), empty_trades_dataframe()


def save_vec_backtest_triplet(
    *,
    actor: th.nn.Module,
    train_env: Any,
    build_trade_env: Callable[[Any], Any],
    build_blank_env: Callable[[], Any],
    train_dates: list[str],
    trade_dates: list[str],
    train_benchmark_df: pd.DataFrame,
    trade_benchmark_df: pd.DataFrame,
    results_dir: str,
    base_filename: str,
    env_index: int = 0,
) -> dict[str, Any]:
    train_results_df, train_trades_df = run_vec_simulation(
        train_env,
        actor,
        train_dates,
        train_benchmark_df,
        reset_impact_model=True,
        env_index=env_index,
    )
    train_csv_filename = os.path.join(results_dir, f"{base_filename}_train.csv")
    train_trades_csv_filename = os.path.join(
        results_dir,
        f"{base_filename}_train_trades.csv",
    )
    train_results_df.to_csv(train_csv_filename, index=False)
    train_trades_df.to_csv(train_trades_csv_filename, index=False)

    trade_env = build_trade_env(train_env)
    last_train_benchmark_value = float(train_results_df["benchmark_value"].iloc[-1])
    trade_results_df, trade_trades_df = run_vec_simulation(
        trade_env,
        actor,
        trade_dates,
        trade_benchmark_df,
        reset_impact_model=False,
        initial_benchmark_value=last_train_benchmark_value,
        env_index=env_index,
    )
    test_csv_filename = os.path.join(results_dir, f"{base_filename}_test.csv")
    test_trades_csv_filename = os.path.join(
        results_dir,
        f"{base_filename}_test_trades.csv",
    )
    trade_results_df.to_csv(test_csv_filename, index=False)
    trade_trades_df.to_csv(test_trades_csv_filename, index=False)

    blank_env = build_blank_env()
    blank_results_df, blank_trades_df = run_vec_simulation(
        blank_env,
        actor,
        trade_dates,
        trade_benchmark_df,
        reset_impact_model=True,
        env_index=env_index,
    )
    test_blank_csv_filename = os.path.join(
        results_dir,
        f"{base_filename}_test_blank.csv",
    )
    test_blank_trades_csv_filename = os.path.join(
        results_dir,
        f"{base_filename}_test_blank_trades.csv",
    )
    blank_results_df.to_csv(test_blank_csv_filename, index=False)
    blank_trades_df.to_csv(test_blank_trades_csv_filename, index=False)

    return {
        "train_results_df": train_results_df,
        "train_trades_df": train_trades_df,
        "trade_results_df": trade_results_df,
        "trade_trades_df": trade_trades_df,
        "blank_results_df": blank_results_df,
        "blank_trades_df": blank_trades_df,
        "results_csv_train": train_csv_filename,
        "results_csv_test": test_csv_filename,
        "results_csv_test_blank": test_blank_csv_filename,
        "trades_csv_train": train_trades_csv_filename,
        "trades_csv_test": test_trades_csv_filename,
        "trades_csv_test_blank": test_blank_trades_csv_filename,
    }


def save_backtest_summary(
    *,
    results_dir: str,
    benchmark_ticker: str,
    all_backtests_metadata: list[dict[str, Any]],
) -> str:
    summary_path = os.path.join(results_dir, "backtest_summary.json")
    with open(summary_path, "w") as file:
        json.dump(
            {
                "benchmark_ticker": benchmark_ticker,
                "backtests": all_backtests_metadata,
            },
            file,
            indent=4,
            default=str,
        )
    log.info("Saved backtest summary to %s", summary_path)
    return summary_path


def compute_stats_from_results(
    results_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    return compute_performance_stats(
        results_df["portfolio_value"],
        results_df["turnover"],
        results_df["cost"],
        trades_df=trades_df,
        rewards=results_df["reward"],
    )


def build_training_args(
    *,
    env_class: type,
    train_env_kwargs: dict[str, Any],
    eval_env_kwargs: Optional[dict[str, Any]],
    agent_name: str,
    model_kwargs: Optional[dict[str, Any]],
    policy_kwargs: Optional[dict[str, Any]],
    steps_per_epoch: int,
    epoch_index: int,
    run_dir: str,
    gpu_id: int,
    num_workers: int,
    random_seed: int,
    resolved_settings: Optional[dict[str, Any]] = None,
) -> Any:
    settings = dict(resolved_settings) if resolved_settings is not None else resolve_elegantrl_settings(
        agent_name,
        model_kwargs,
        policy_kwargs,
        steps_per_epoch=steps_per_epoch,
        env_class=env_class,
        train_env_kwargs=train_env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        requested_num_envs=train_env_kwargs.get("num_envs"),
        gpu_id=gpu_id,
        num_workers=num_workers,
    )

    probe_train_env_kwargs = dict(train_env_kwargs)
    normalizer_state_path = probe_train_env_kwargs.pop("normalizer_state_path", None)
    probe_train_env_kwargs.setdefault("freeze_loaded_normalizer", False)

    probe_eval_env_kwargs = dict(eval_env_kwargs) if eval_env_kwargs is not None else dict(probe_train_env_kwargs)
    probe_eval_env_kwargs.pop("normalizer_state_path", None)
    probe_eval_env_kwargs.setdefault("freeze_loaded_normalizer", True)

    args = build_elegantrl_config(
        env_class,
        probe_train_env_kwargs,
        agent_name=agent_name,
        gpu_id=gpu_id,
        eval_env_kwargs=probe_eval_env_kwargs,
        break_step=steps_per_epoch * (epoch_index + 1),
        horizon_len=settings["horizon_len"],
        net_dims=settings["net_dims"],
        learning_rate=float(settings["learning_rate"]),
        repeat_times=int(settings["repeat_times"]),
        eval_per_step=int(settings.get("eval_per_step", steps_per_epoch)),
        eval_times=int(settings.get("eval_times", 1)),
        num_workers=num_workers,
        random_seed=random_seed,
    )
    args.cwd = run_dir
    args.if_remove = epoch_index == 0
    args.continue_train = epoch_index > 0
    args.if_keep_save = True
    args.if_over_write = True
    args.batch_size = int(settings.get("batch_size", args.batch_size))
    args.gamma = float(settings.get("gamma", args.gamma))
    args.lambda_gae_adv = float(settings.get("lambda_gae_adv", getattr(args, "lambda_gae_adv", 0.95)))
    args.lambda_entropy = float(settings.get("lambda_entropy", getattr(args, "lambda_entropy", 0.01)))
    args.horizon_len = int(settings["horizon_len"])
    args.eval_times = int(settings.get("eval_times", args.eval_times))
    args.eval_per_step = int(settings.get("eval_per_step", args.eval_per_step))

    if normalizer_state_path is not None:
        args.env_args["normalizer_state_path"] = normalizer_state_path

    return args


def load_trained_actor(
    args: Any,
    actor_path: Optional[str] = None,
) -> th.nn.Module:
    ensure_elegantrl_on_path()
    agent = args.agent_class(
        args.net_dims,
        args.state_dim,
        args.action_dim,
        gpu_id=args.gpu_id,
        args=args,
    )
    if actor_path is None:
        agent.save_or_load_agent(args.cwd, if_save=False)
    else:
        loaded_actor = th.load(
            actor_path,
            map_location=agent.device,
            weights_only=False,
        )
        if hasattr(loaded_actor, "state_dict"):
            agent.act.load_state_dict(loaded_actor.state_dict())
        else:
            agent.act.load_state_dict(loaded_actor)
    agent.act.eval()
    return agent.act


def train_with_epoch_evaluation(
    *,
    env_class: type,
    train_env_kwargs: dict[str, Any],
    eval_env_kwargs: Optional[dict[str, Any]],
    agent_name: str,
    model_kwargs: Optional[dict[str, Any]],
    policy_kwargs: Optional[dict[str, Any]],
    num_epochs: int,
    steps_per_epoch: int,
    run_dir: str,
    evaluate_epoch: Callable[[th.nn.Module], tuple[dict[str, float], dict[str, float]]],
    gpu_id: int = 0,
    num_workers: int = 1,
    random_seed: int = 42,
    if_single_process: bool = True,
    resolved_settings: Optional[dict[str, Any]] = None,
) -> tuple[th.nn.Module, list[dict[str, float]], list[dict[str, float]], Any]:
    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")

    params = train_env_kwargs.get("params")
    if (
        num_workers > 1
        and params is not None
        and getattr(params, "use_obs_normalizer", False)
    ):
        raise ValueError(
            "Observation normalizer stats are not shared across workers; "
            "use num_workers=1 when observation normalization is enabled."
        )

    ensure_elegantrl_on_path()
    from elegantrl.train.run import train_agent

    args = build_training_args(
        env_class=env_class,
        train_env_kwargs=train_env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        agent_name=agent_name,
        model_kwargs=model_kwargs,
        policy_kwargs=policy_kwargs,
        steps_per_epoch=steps_per_epoch,
        epoch_index=0,
        run_dir=run_dir,
        gpu_id=gpu_id,
        num_workers=num_workers,
        random_seed=random_seed,
        resolved_settings=resolved_settings,
    )
    # ElegantRL stops when total_step > break_step, so subtract one step to
    # land exactly on the requested final epoch boundary.
    total_training_steps = max(1, int(steps_per_epoch) * int(num_epochs))
    args.break_step = total_training_steps - 1
    args.eval_per_step = max(1, int(steps_per_epoch))
    args.if_remove = True
    args.continue_train = False

    epoch_stats_train: list[dict[str, float]] = []
    epoch_stats_test_blank: list[dict[str, float]] = []
    last_actor = None

    with _patch_elegantrl_evaluator_for_epoch_snapshots(run_dir):
        train_agent(args, if_single_process=if_single_process)

    normalizer_state_path = train_env_kwargs.get("normalizer_state_path")
    epoch_steps = _load_epoch_steps(run_dir, num_epochs)
    for epoch, total_step in enumerate(epoch_steps, start=1):
        actor_path = _epoch_actor_snapshot_path(run_dir, total_step)
        if not os.path.isfile(actor_path):
            raise RuntimeError(
                f"Expected ElegantRL epoch checkpoint at {actor_path}"
            )

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
        last_actor = actor

    if last_actor is None:
        raise RuntimeError("Training did not produce a final actor")

    return last_actor, epoch_stats_train, epoch_stats_test_blank, args
