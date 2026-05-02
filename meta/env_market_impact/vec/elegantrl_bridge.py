from __future__ import annotations

from importlib import import_module
from typing import Any
from typing import Optional


def _safe_get(module_name: str, attr: str) -> Any:
    try:
        return getattr(import_module(module_name), attr, None)
    except ModuleNotFoundError:
        return None


def _load_elegantrl_symbols() -> tuple[Any, Any, dict[str, Any]]:
    config_module = import_module("elegantrl.train.config")
    run_module = import_module("elegantrl.train.run")
    # AgentA2C / AgentDiscretePPO live inside AgentPPO.py in current ElegantRL;
    # older versions ship them as separate modules. Try both.
    ppo_module = "elegantrl.agents.AgentPPO"
    td3_module = "elegantrl.agents.AgentTD3"
    agents = {
        "ppo": _safe_get(ppo_module, "AgentPPO"),
        "a2c": _safe_get(ppo_module, "AgentA2C") or _safe_get("elegantrl.agents.AgentA2C", "AgentA2C"),
        "ddpg": _safe_get(td3_module, "AgentDDPG") or _safe_get("elegantrl.agents", "AgentDDPG"),
        "sac": _safe_get("elegantrl.agents.AgentSAC", "AgentSAC"),
        "td3": _safe_get(td3_module, "AgentTD3"),
        "discrete_ppo": _safe_get(ppo_module, "AgentDiscretePPO"),
        "discrete_a2c": _safe_get(ppo_module, "AgentDiscreteA2C"),
    }
    agents = {key: value for key, value in agents.items() if value is not None}
    return getattr(config_module, "Config"), getattr(run_module, "train_agent"), agents


def build_env_args(env: Any, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_name": env.env_name,
        "num_envs": env.num_envs,
        "max_step": env.max_step,
        "state_dim": env.state_dim,
        "action_dim": env.action_dim,
        "if_discrete": env.if_discrete,
        **env_kwargs,
    }


def build_elegantrl_config(
    env_class: type,
    env_kwargs: dict[str, Any],
    *,
    agent_name: str = "ppo",
    gpu_id: int = 0,
    eval_env_kwargs: Optional[dict[str, Any]] = None,
    break_step: int = 200_000,
    horizon_len: Optional[int] = None,
    net_dims: Optional[list[int]] = None,
    learning_rate: float = 2e-4,
    repeat_times: int = 8,
    eval_per_step: int = 20_000,
    eval_times: int = 32,
    num_workers: int = 1,
    random_seed: int = 0,
) -> Any:
    Config, _, agents = _load_elegantrl_symbols()
    agent_key = agent_name.lower()
    if agent_key not in agents:
        raise ValueError(f"Unsupported ElegantRL agent '{agent_name}'. Available: {sorted(agents)}")

    probe_env = env_class(**env_kwargs)
    env_args = build_env_args(probe_env, env_kwargs)
    args = Config(agents[agent_key], env_class, env_args)
    args.gpu_id = gpu_id
    args.break_step = int(break_step)
    args.net_dims = net_dims or [256, 128]
    args.learning_rate = learning_rate
    args.repeat_times = repeat_times
    args.horizon_len = int(horizon_len or probe_env.max_step)
    args.eval_per_step = int(eval_per_step)
    args.eval_times = int(eval_times)
    args.num_workers = int(num_workers)
    args.random_seed = int(random_seed)

    if eval_env_kwargs is not None:
        eval_probe = env_class(**eval_env_kwargs)
        args.eval_env_class = env_class
        args.eval_env_args = build_env_args(eval_probe, eval_env_kwargs)

    return args


def train_elegantrl_agent(*args: Any, if_single_process: bool = False, **kwargs: Any) -> Any:
    config = build_elegantrl_config(*args, **kwargs)
    _, train_agent, _ = _load_elegantrl_symbols()
    return train_agent(args=config, if_single_process=if_single_process)