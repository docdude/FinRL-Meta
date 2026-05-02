from __future__ import annotations

import contextlib
import inspect
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

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
ELEGANTRL_REPO = REPO_ROOT
if ELEGANTRL_REPO in sys.path:
    sys.path.remove(ELEGANTRL_REPO)
sys.path.insert(0, ELEGANTRL_REPO)

from elegantrl.agents import AgentA2C
from elegantrl.agents import AgentDDPG
from elegantrl.agents import AgentPPO
from elegantrl.agents import AgentSAC
from elegantrl.agents import AgentTD3
from elegantrl.train import Config

from meta.env_market_impact.backtest_vec_config import SUPPORTED_VEC_AGENTS
from meta.env_market_impact.backtest_vec_config import VEC_MODEL_KWARGS
from meta.env_market_impact.backtest_summary_utils import (
    prepare_summary_payload,
)
from meta.env_market_impact.envs.impact_models import ACImpactModel
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.env_market_impact.envs.impact_models import OWImpactModel
from meta.env_market_impact.envs.impact_models import SqrtImpactModel
from meta.env_market_impact.envs.utils import compute_performance_stats
from meta.env_market_impact.envs.utils import (
    compute_performance_stats_from_aggregates,
)
from meta.env_market_impact.envs.utils import get_logger

from .common import resolve_device
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

SUPPORTED_ELEGANTRL_AGENTS = SUPPORTED_VEC_AGENTS
ON_POLICY_ELEGANTRL_AGENTS = {"a2c", "ppo"}
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
ELEGANTRL_MODEL_KWARGS = VEC_MODEL_KWARGS
ELEGANTRL_AGENT_CLASSES = {
    "a2c": AgentA2C,
    "ppo": AgentPPO,
    "ddpg": AgentDDPG,
    "sac": AgentSAC,
    "td3": AgentTD3,
}

MODEL_KWARG_ALIASES = {
    "n_steps": "horizon_len",
    "ent_coef": "lambda_entropy",
    "gae_lambda": "lambda_gae_adv",
    "tau": "soft_update_tau",
    "if_use_vtrace": "if_use_v_trace",
}
SCALAR_TO_VEC_MODEL_KWARG_ALLOWLIST: dict[str, set[str]] = {
    "a2c": {"learning_rate", "ent_coef", "gamma", "gae_lambda"},
    "ppo": {
        "learning_rate",
        "batch_size",
        "ent_coef",
        "gamma",
        "gae_lambda",
    },
    "ddpg": {
        "learning_rate",
        "batch_size",
        "buffer_size",
        "buffer_init_size",
        "gamma",
        "tau",
        "reward_scale",
        "clip_grad_norm",
        "state_value_tau",
        "if_use_per",
        "lambda_fit_cum_r",
        "explore_noise",
    },
    "sac": {
        "learning_rate",
        "batch_size",
        "buffer_size",
        "buffer_init_size",
        "gamma",
        "tau",
        "ent_coef",
        "reward_scale",
        "clip_grad_norm",
        "state_value_tau",
        "if_use_per",
        "lambda_fit_cum_r",
        "num_ensembles",
    },
    "td3": {
        "learning_rate",
        "batch_size",
        "buffer_size",
        "buffer_init_size",
        "gamma",
        "tau",
        "reward_scale",
        "clip_grad_norm",
        "state_value_tau",
        "if_use_per",
        "lambda_fit_cum_r",
        "update_freq",
        "num_ensembles",
        "policy_noise_std",
        "explore_noise_std",
    },
}
ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME = "epoch_snapshots"
DIRECT_ELEGANTRL_ARG_FIELDS = (
    "reward_scale",
    "clip_grad_norm",
    "state_value_tau",
    "if_use_per",
    "if_use_v_trace",
    "lambda_fit_cum_r",
    "ratio_clip",
    "update_freq",
    "num_ensembles",
    "policy_noise_std",
    "explore_noise_std",
    "explore_noise",
)


def ensure_elegantrl_on_path(repo_path: str = ELEGANTRL_REPO) -> str:
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)
    return repo_path


def empty_trades_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=FULL_TRADES_COLUMNS)


def _build_direct_env_args(env: Any, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_name": env.env_name,
        "num_envs": env.num_envs,
        "max_step": env.max_step,
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "if_discrete": env.if_discrete,
        **env_kwargs,
    }


def _epoch_snapshot_dir(run_dir: str) -> str:
    return os.path.join(run_dir, ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME)


def _epoch_steps_path(run_dir: str) -> str:
    return os.path.join(_epoch_snapshot_dir(run_dir), "epoch_steps.npy")


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
    epoch_steps_path = _epoch_steps_path(run_dir)
    if os.path.isfile(epoch_steps_path):
        epoch_steps = np.load(epoch_steps_path)
        if epoch_steps.ndim != 1:
            raise RuntimeError(
                f"Unexpected saved epoch-step shape {epoch_steps.shape!r}"
            )
        if epoch_steps.shape[0] < num_epochs:
            raise RuntimeError(
                "Captured fewer epoch snapshots than requested "
                f"epochs ({epoch_steps.shape[0]} < {num_epochs})"
            )
        return [int(step) for step in epoch_steps[:num_epochs].tolist()]

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
def _patch_elegantrl_evaluator_for_epoch_snapshots(
    run_dir: str,
    epoch_step_targets: Optional[list[int]] = None,
):
    ensure_elegantrl_on_path()
    import elegantrl.train.evaluator as evaluator_module
    import elegantrl.train.run as run_module

    os.makedirs(_epoch_snapshot_dir(run_dir), exist_ok=True)
    original_run_evaluator = run_module.Evaluator
    original_module_evaluator = evaluator_module.Evaluator
    epoch_targets = [int(step) for step in (epoch_step_targets or [])]
    captured_epoch_steps: list[int] = []

    def _save_epoch_snapshot(actor: th.nn.Module, env: Any, total_step: int) -> None:
        os.makedirs(_epoch_snapshot_dir(run_dir), exist_ok=True)
        th.save(actor, _epoch_actor_snapshot_path(run_dir, total_step))
        if hasattr(env, "save"):
            env.save(_epoch_normalizer_snapshot_path(run_dir, total_step))

    def _persist_epoch_steps() -> None:
        np.save(
            _epoch_steps_path(run_dir),
            np.asarray(captured_epoch_steps, dtype=np.int64),
        )

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

            if epoch_targets:
                total_step = int(self.total_step)
                while (
                    len(captured_epoch_steps) < len(epoch_targets)
                    and total_step >= epoch_targets[len(captured_epoch_steps)]
                ):
                    _save_epoch_snapshot(actor, self.env, total_step)
                    captured_epoch_steps.append(total_step)
                    _persist_epoch_steps()
                return

            if len(self.recorder) == recorder_len:
                return

            total_step = int(self.recorder[-1][0])
            _save_epoch_snapshot(actor, self.env, total_step)

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


def filter_scalar_model_kwargs_for_vec(
    agent_name: str,
    model_kwargs: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if model_kwargs is None:
        return None

    allowlist = SCALAR_TO_VEC_MODEL_KWARG_ALLOWLIST.get(agent_name.lower())
    if allowlist is None:
        return dict(model_kwargs)
    return {
        key: value
        for key, value in model_kwargs.items()
        if key in allowlist
    }


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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _summarize_debug_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _summarize_debug_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 12:
            return {"type": type(value).__name__, "len": len(value)}
        return [_summarize_debug_value(item) for item in value]
    if isinstance(value, th.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return value


def _summarize_env_config_for_debug(config: Any) -> Any:
    if not isinstance(config, dict):
        return _summarize_debug_value(config)

    summary: dict[str, Any] = {}
    if "date_list" in config:
        summary["date_count"] = len(config["date_list"])
    if "tbill_rates" in config:
        summary["tbill_count"] = len(config["tbill_rates"])
    if "tic_list" in config:
        summary["tic_count"] = len(config["tic_list"])
    for key in (
        "price_array",
        "tech_array",
        "volatility_array",
        "volume_array",
        "adv20_array",
    ):
        if key in config:
            summary[f"{key}_shape"] = list(np.asarray(config[key]).shape)
    return summary


def _summarize_env_kwargs_for_debug(env_kwargs: Optional[dict[str, Any]]) -> Any:
    if env_kwargs is None:
        return None

    summary: dict[str, Any] = {}
    for key, value in env_kwargs.items():
        if key == "config":
            summary[key] = _summarize_env_config_for_debug(value)
        elif key == "params":
            summary[key] = repr(value)
        elif key == "impact_model":
            summary[key] = None if value is None else type(value).__name__
        else:
            summary[key] = _summarize_debug_value(value)
    return summary


def _scale_num_envs_for_available_gpu_memory(
    requested_num_envs: int,
    *,
    state_dim: int,
    action_dim: int,
    train_max_step: int,
    train_rollout_step: int,
    eval_max_step: int,
    gpu_id: int,
    num_workers: int,
    if_off_policy: bool = False,
    replay_buffer_size: Optional[int] = None,
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
    rollout_bytes_per_step = max(1, (state_dim + action_dim + 3) * 4)
    replay_bytes_per_step = max(1, (state_dim + action_dim + 4) * 4)
    total_steps_per_env = max(
        1,
        (2 * worker_count * max(1, train_rollout_step))
        + max(1, eval_max_step),
    )
    replay_buffer_bytes_per_env = 0
    if if_off_policy and replay_buffer_size is not None:
        replay_buffer_bytes_per_env = (
            max(1, int(replay_buffer_size)) * replay_bytes_per_step
        )

    bytes_per_env = (
        total_steps_per_env * rollout_bytes_per_step
    ) + replay_buffer_bytes_per_env
    max_envs = max(1, int(buffer_budget // max(1, bytes_per_env)))
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
            + (effective_num_envs * bytes_per_env)
        )
        / 1024**3,
        "worker_count": float(worker_count),
        "train_max_step": float(train_max_step),
        "train_rollout_step": float(train_rollout_step),
        "eval_max_step": float(eval_max_step),
        "replay_buffer_gib": (
            (effective_num_envs * replay_buffer_bytes_per_env) / 1024**3
        ),
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
    disable_num_envs_scaling: bool = False,
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

    if agent_key in ON_POLICY_ELEGANTRL_AGENTS:
        settings["horizon_len"] = max(1, int(steps_per_epoch))
    else:
        explicit_horizon_len = any(
            MODEL_KWARG_ALIASES.get(key, key) == "horizon_len"
            for key in (model_kwargs or {})
        )
        default_horizon_len = max(1, int(steps_per_epoch) // 4)
        if explicit_horizon_len:
            configured_horizon_len = int(
                settings.get("horizon_len", default_horizon_len)
            )
        else:
            configured_horizon_len = default_horizon_len
        settings["horizon_len"] = max(
            1,
            min(configured_horizon_len, int(steps_per_epoch)),
        )

    resolved_num_envs = resolve_default_num_envs(
        requested_num_envs,
        gpu_id=gpu_id,
    )
    scaling_info = None
    if (
        not disable_num_envs_scaling
        and env_class is not None
        and train_env_kwargs is not None
    ):
        state_dim, action_dim = _estimate_state_action_dims(
            env_class,
            train_env_kwargs,
        )
        train_max_step = _estimate_max_step(train_env_kwargs, steps_per_epoch)
        eval_max_step = _estimate_max_step(eval_env_kwargs, steps_per_epoch)
        resolved_num_envs, scaling_info = (
            _scale_num_envs_for_available_gpu_memory(
                resolved_num_envs,
                state_dim=state_dim,
                action_dim=action_dim,
                train_max_step=train_max_step,
                train_rollout_step=int(settings["horizon_len"]),
                eval_max_step=eval_max_step,
                gpu_id=gpu_id,
                num_workers=num_workers,
                if_off_policy=agent_key not in ON_POLICY_ELEGANTRL_AGENTS,
                replay_buffer_size=settings.get("buffer_size"),
            )
        )
        if scaling_info is not None and resolved_num_envs < int(
            resolve_default_num_envs(requested_num_envs, gpu_id=gpu_id)
        ):
            log.info(
                "Scaling vec num_envs %s -> %s "
                "(GPU %.1f/%.1f GiB free, used %.0f MiB, "
                "train_horizon=%s, eval_horizon=%s, "
                "workers=%s, proj=%.1f GiB)",
                resolve_default_num_envs(requested_num_envs, gpu_id=gpu_id),
                resolved_num_envs,
                scaling_info["gpu_free_gib"],
                scaling_info["gpu_total_gib"],
                scaling_info["gpu_used_mib"],
                int(scaling_info["train_rollout_step"]),
                int(scaling_info["eval_max_step"]),
                int(scaling_info["worker_count"]),
                scaling_info["projected_gib"],
            )

    settings["num_envs"] = int(resolved_num_envs)
    settings["requested_num_envs"] = int(
        resolve_default_num_envs(requested_num_envs, gpu_id=gpu_id)
    )
    settings["repeat_times"] = max(1, int(settings.get("repeat_times", 1)))
    rollout_batch_size = max(
        1,
        settings["horizon_len"]
        * max(1, int(num_workers))
        * settings["num_envs"],
    )
    configured_batch_size = max(1, int(settings.get("batch_size", 128)))
    if agent_key in ON_POLICY_ELEGANTRL_AGENTS:
        # ElegantRL's PPO/A2C on-policy update count is based on horizon_len,
        # not horizon_len * num_envs, so keep batch_size within that limit.
        max_update_batch_size = max(
            1,
            settings["horizon_len"] * settings["repeat_times"],
        )
        settings["batch_size"] = min(
            configured_batch_size,
            rollout_batch_size,
            max_update_batch_size,
        )
    else:
        settings["batch_size"] = configured_batch_size
        if settings.get("buffer_init_size") is None:
            # ReplayBuffer.cur_size grows along the time axis, so the first
            # off-policy update only sees horizon_len samples regardless of
            # how many parallel env sequences were collected.
            settings["buffer_init_size"] = min(
                settings["batch_size"] * 8,
                settings["horizon_len"],
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


def run_vec_simulation_stats(
    env: Any,
    actor: th.nn.Module,
    *,
    reset_impact_model: bool = True,
    env_index: int = 0,
) -> dict[str, float]:
    num_envs = int(getattr(env, "num_envs", 1))
    if env_index < 0 or env_index >= num_envs:
        raise ValueError(
            f"env_index must be in [0, {num_envs - 1}] for num_envs={num_envs}"
        )

    actor.eval()
    first_param = next(actor.parameters(), None)
    device = first_param.device if first_param is not None else th.device("cpu")
    state, _ = env.reset(options={"reset_impact_model": reset_impact_model})

    max_rows = int(env.max_step) + 1
    portfolio_values = np.empty(max_rows, dtype=np.float64)
    turnovers = np.zeros(max_rows, dtype=np.float64)
    costs = np.zeros(max_rows, dtype=np.float64)
    rewards = np.zeros(max_rows, dtype=np.float64)

    row = 0
    portfolio_values[row] = float(
        scalarize(select_env_value(env.total_asset, env_index))
    )

    trade_count = 0
    pov_sum = 0.0
    total_notional = 0.0
    turnover_percentile_notional_sum = 0.0

    for _step in range(env.max_step):
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

        row += 1
        portfolio_values[row] = current_asset_value
        turnovers[row] = float(
            scalarize(select_env_value(info["turnover"], env_index))
        )
        costs[row] = float(
            scalarize(select_env_value(info["cost"], env_index))
        )
        rewards[row] = float(
            scalarize(select_env_value(reward, env_index))
        )

        for trade in info.get("trades", []):
            trade_count += 1
            pov_sum += float(trade["pov"])
            notional = float(trade["notional"])
            total_notional += notional
            turnover_percentile_notional_sum += (
                float(trade["turnover_percentile"]) * notional
            )

        if bool(scalarize(select_env_value(done, env_index))) or bool(
            scalarize(select_env_value(truncated, env_index))
        ):
            break

    used_rows = row + 1
    return compute_performance_stats_from_aggregates(
        portfolio_values[:used_rows],
        turnovers[:used_rows],
        costs[:used_rows],
        trade_count=trade_count,
        pov_sum=pov_sum,
        total_notional=total_notional,
        turnover_percentile_notional_sum=turnover_percentile_notional_sum,
        rewards=rewards[:used_rows],
    )


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
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            prepare_summary_payload(
                benchmark_ticker=benchmark_ticker,
                backtests=all_backtests_metadata,
            ),
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
    supports_frozen_normalizer = (
        "freeze_loaded_normalizer"
        in inspect.signature(env_class.__init__).parameters
    )
    if supports_frozen_normalizer:
        probe_train_env_kwargs.setdefault("freeze_loaded_normalizer", False)

    probe_eval_env_kwargs = (
        dict(eval_env_kwargs)
        if eval_env_kwargs is not None
        else dict(probe_train_env_kwargs)
    )
    probe_eval_env_kwargs.pop("normalizer_state_path", None)
    if supports_frozen_normalizer:
        probe_eval_env_kwargs.setdefault("freeze_loaded_normalizer", True)

    agent_key = agent_name.lower()
    agent_class = ELEGANTRL_AGENT_CLASSES.get(agent_key)
    if agent_class is None:
        raise ValueError(
            f"Unsupported ElegantRL agent '{agent_name}'. "
            f"Available: {sorted(ELEGANTRL_AGENT_CLASSES)}"
        )

    probe_env = env_class(**probe_train_env_kwargs)
    env_args = _build_direct_env_args(probe_env, probe_train_env_kwargs)
    args = Config(agent_class, env_class, env_args)
    args.gpu_id = gpu_id
    args.break_step = int(steps_per_epoch * (epoch_index + 1))
    args.net_dims = list(settings["net_dims"])
    args.learning_rate = float(settings["learning_rate"])
    args.repeat_times = int(settings["repeat_times"])
    args.horizon_len = int(settings["horizon_len"])
    args.eval_per_step = int(settings.get("eval_per_step", steps_per_epoch))
    args.eval_times = int(settings.get("eval_times", 1))
    args.num_workers = int(num_workers)
    args.random_seed = int(random_seed)

    eval_probe = env_class(**probe_eval_env_kwargs)
    args.eval_env_class = env_class
    args.eval_env_args = _build_direct_env_args(eval_probe, probe_eval_env_kwargs)

    with contextlib.suppress(Exception):
        probe_env.close()
    with contextlib.suppress(Exception):
        eval_probe.close()

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

    if "buffer_size" in settings:
        args.buffer_size = int(settings["buffer_size"])
    if settings.get("buffer_init_size") is not None:
        args.buffer_init_size = int(settings["buffer_init_size"])
    if "soft_update_tau" in settings:
        args.soft_update_tau = float(settings["soft_update_tau"])
    for field_name in DIRECT_ELEGANTRL_ARG_FIELDS:
        if field_name in settings:
            setattr(args, field_name, settings[field_name])

    if normalizer_state_path is not None:
        args.env_args["normalizer_state_path"] = normalizer_state_path

    if _env_flag("ERL_VEC_DEBUG_CONFIG"):
        debug_payload = {
            "agent_name": agent_name,
            "epoch_index": epoch_index,
            "run_dir": run_dir,
            "steps_per_epoch": steps_per_epoch,
            "model_kwargs": model_kwargs,
            "policy_kwargs": policy_kwargs,
            "resolved_settings": settings,
            "final_args": {
                "cwd": args.cwd,
                "if_remove": args.if_remove,
                "continue_train": args.continue_train,
                "break_step": args.break_step,
                "horizon_len": args.horizon_len,
                "batch_size": args.batch_size,
                "repeat_times": getattr(args, "repeat_times", None),
                "learning_rate": getattr(args, "learning_rate", None),
                "gamma": getattr(args, "gamma", None),
                "lambda_gae_adv": getattr(args, "lambda_gae_adv", None),
                "lambda_entropy": getattr(args, "lambda_entropy", None),
                "clip_grad_norm": getattr(args, "clip_grad_norm", None),
                "ratio_clip": getattr(args, "ratio_clip", None),
                "if_use_v_trace": getattr(args, "if_use_v_trace", None),
                "eval_per_step": args.eval_per_step,
                "eval_times": args.eval_times,
                "num_workers": getattr(args, "num_workers", None),
                "gpu_id": getattr(args, "gpu_id", None),
                "random_seed": getattr(args, "random_seed", None),
            },
            "train_env_kwargs": _summarize_env_kwargs_for_debug(train_env_kwargs),
            "eval_env_kwargs": _summarize_env_kwargs_for_debug(eval_env_kwargs),
            "probe_train_env_kwargs": _summarize_env_kwargs_for_debug(probe_train_env_kwargs),
            "probe_eval_env_kwargs": _summarize_env_kwargs_for_debug(probe_eval_env_kwargs),
            "args_env_args": _summarize_env_kwargs_for_debug(getattr(args, "env_args", None)),
            "args_eval_env_args": _summarize_env_kwargs_for_debug(getattr(args, "eval_env_args", None)),
            "normalizer_state_path": normalizer_state_path,
        }
        print(
            "[ERL_VEC_CONFIG] "
            + json.dumps(debug_payload, indent=2, default=str),
            flush=True,
        )

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
    epoch_step_targets = [
        int(steps_per_epoch) * epoch for epoch in range(1, int(num_epochs) + 1)
    ]

    epoch_stats_train: list[dict[str, float]] = []
    epoch_stats_test_blank: list[dict[str, float]] = []
    last_actor = None

    with _patch_elegantrl_evaluator_for_epoch_snapshots(
        run_dir,
        epoch_step_targets=epoch_step_targets,
    ):
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
