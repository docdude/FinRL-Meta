from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from dataclasses import field

from meta.env_market_impact.envs.impact_models import ACImpactModel
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.env_market_impact.envs.impact_models import OWImpactModel
from meta.env_market_impact.envs.impact_models import SqrtImpactModel

SUPPORTED_VEC_AGENTS = ("a2c", "ppo", "ddpg", "sac", "td3")
VEC_IMPACT_MODEL_CLASSES: tuple[type, ...] = (
    BaselineImpactModel,
    ACImpactModel,
    SqrtImpactModel,
    OWImpactModel,
)
VEC_NET_DIMS: dict[str, list[int]] = {
    "small": [128, 64],
    "medium": [256, 128],
    "large": [512, 256],
    "wide": [512, 256, 128],
}
VEC_MODEL_KWARGS: dict[str, dict] = {
    "a2c": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "repeat_times": 1,
        "gamma": 0.99,
        "lambda_gae_adv": 0.95,
        "lambda_entropy": 0.01,
        "clip_grad_norm": 3.0,
        "if_use_v_trace": True,
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
        "clip_grad_norm": 3.0,
        "if_use_v_trace": True,
        "ratio_clip": 0.25,
        "net_dims": [256, 128],
        "eval_times": 1,
    },
    "ddpg": {
        "learning_rate": 9.6e-5,
        "batch_size": 128,
        "buffer_size": int(5e4),
        "repeat_times": 2,
        "soft_update_tau": 5e-3,
        "gamma": 0.92,
        "net_dims": [256, 128, 64],
        "eval_times": 1,
        "reward_scale": 1.0,
        "clip_grad_norm": 3.0,
        "state_value_tau": 0.0,
        "if_use_per": False,
        "lambda_fit_cum_r": 0.0,
        "explore_noise_std": 0.05,
    },
    "sac": {
        "learning_rate": 8.8e-5,
        "batch_size": 128,
        "buffer_size": int(5e4),
        "repeat_times": 2,
        "soft_update_tau": 5e-3,
        "gamma": 0.92,
        "net_dims": [256, 128, 64],
        "eval_times": 1,
        "reward_scale": 1.0,
        "clip_grad_norm": 3.0,
        "state_value_tau": 0.0,
        "if_use_per": False,
        "lambda_fit_cum_r": 0.0,
        "num_ensembles": 4,
    },
    "td3": {
        "learning_rate": 1.04e-5,
        "batch_size": 128,
        "buffer_size": int(5e4),
        "repeat_times": 2,
        "soft_update_tau": 5e-3,
        "gamma": 0.95,
        "net_dims": [256, 128, 64],
        "eval_times": 1,
        "reward_scale": 1.0,
        "clip_grad_norm": 3.0,
        "state_value_tau": 0.0,
        "if_use_per": False,
        "lambda_fit_cum_r": 0.0,
        "update_freq": 2,
        "num_ensembles": 8,
        "policy_noise_std": 0.10,
        "explore_noise_std": 0.05,
    },
}
VEC_MODEL_KWARG_KEYS: dict[str, set[str]] = {
    "a2c": {
        "learning_rate",
        "batch_size",
        "repeat_times",
        "gamma",
        "lambda_gae_adv",
        "lambda_entropy",
        "clip_grad_norm",
        "if_use_v_trace",
        "net_dims_key",
    },
    "ppo": {
        "learning_rate",
        "batch_size",
        "repeat_times",
        "gamma",
        "lambda_gae_adv",
        "lambda_entropy",
        "clip_grad_norm",
        "if_use_v_trace",
        "ratio_clip",
        "net_dims_key",
    },
    "ddpg": {
        "learning_rate",
        "batch_size",
        "buffer_size",
        "repeat_times",
        "soft_update_tau",
        "gamma",
        "horizon_len",
        "net_dims_key",
        "explore_noise",
    },
    "sac": {
        "learning_rate",
        "batch_size",
        "buffer_size",
        "repeat_times",
        "soft_update_tau",
        "gamma",
        "horizon_len",
        "net_dims_key",
        "num_ensembles",
    },
    "td3": {
        "learning_rate",
        "batch_size",
        "buffer_size",
        "repeat_times",
        "soft_update_tau",
        "gamma",
        "horizon_len",
        "net_dims_key",
        "update_freq",
        "num_ensembles",
        "policy_noise_std",
        "explore_noise_std",
    },
}
VEC_A2C_PRESETS: dict[str, dict[str, dict]] = {
    "canonical-hpo": {
        "model_kwargs": {
            "learning_rate": 9.345473014207437e-05,
            "gamma": 0.9088236029155305,
            "lambda_gae_adv": 0.8085192755608721,
            "lambda_entropy": 0.0007879004401207801,
        },
        "policy_kwargs": {"net_arch": [256, 128, 64]},
        "env_kwargs": {
            "eta_dd": 0.5920255491056137,
            "horizon": 40,
            "include_cooldown_in_state": False,
            "include_permanent_impact_in_state": False,
            "include_tbill_in_state": True,
            "obs_clip": 7.43605870267289,
            "reward_scaling": 0.000230613280155283,
            "use_obs_normalizer": True,
        },
    },
    "optuna-20260420": {
        "model_kwargs": {
            "learning_rate": 7.50605554641839e-05,
            "gamma": 0.927638999007113,
            "lambda_gae_adv": 0.9958518878203044,
            "lambda_entropy": 0.006538517515790021,
        },
        "policy_kwargs": {"net_arch": [256, 128]},
        "env_kwargs": {
            "eta_dd": 2.9294904746679213,
            "horizon": 80,
            "include_cooldown_in_state": False,
            "include_permanent_impact_in_state": True,
            "include_tbill_in_state": False,
            "reward_scaling": 0.000536836975727163,
            "use_obs_normalizer": True,
        },
    },
}
SUPPORTED_VEC_A2C_PRESETS = tuple(VEC_A2C_PRESETS)


def _default_max_stock_pct(
    num_stocks: int,
    max_stock_weight_multiplier: float = 2.0,
    max_stock_pct_clip: tuple[float, float] = (0.01, 1.0),
) -> float:
    return float(
        np.clip(
            (1.0 / num_stocks) * max_stock_weight_multiplier,
            max_stock_pct_clip[0],
            max_stock_pct_clip[1],
        )
    )


def _clone_model_kwargs(model_name: str, model_kwargs: dict | None) -> dict:
    base = dict(VEC_MODEL_KWARGS[model_name.lower()])
    if model_kwargs:
        base.update(model_kwargs)
    return base


def apply_vec_mace_a2c_preset(config: dict, preset_name: str | None) -> dict:
    if preset_name is None:
        return dict(config)

    preset = VEC_A2C_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unsupported vec A2C preset '{preset_name}'. "
            f"Available: {sorted(VEC_A2C_PRESETS)}"
        )

    updated = dict(config)
    model_kwargs = dict(updated.get("model_kwargs") or {})
    model_kwargs.update(preset.get("model_kwargs", {}))
    if model_kwargs:
        updated["model_kwargs"] = model_kwargs

    policy_kwargs = dict(updated.get("policy_kwargs") or {})
    policy_kwargs.update(preset.get("policy_kwargs", {}))
    if policy_kwargs:
        updated["policy_kwargs"] = policy_kwargs

    updated.update(preset.get("env_kwargs", {}))
    return updated


def reconstruct_vec_model_kwargs(
    flat_params: dict,
    model_name: str,
) -> tuple[dict, None]:
    agent_name = model_name.lower()
    agent_keys = VEC_MODEL_KWARG_KEYS.get(agent_name, set())
    model_kwargs = {
        key: value
        for key, value in flat_params.items()
        if key in agent_keys and key != "net_dims_key"
    }
    net_dims_key = flat_params.get("net_dims_key")
    if net_dims_key is not None:
        model_kwargs["net_dims"] = VEC_NET_DIMS[net_dims_key]
    return _clone_model_kwargs(agent_name, model_kwargs), None


@dataclass
class VecMACEEnvParams:
    max_stock_pct: float = 0.02
    max_trade_volume_pct: float = 0.1
    reward_scaling: float = 2**-11
    include_permanent_impact_in_state: bool = True
    include_cooldown_in_state: bool = True
    include_tbill_in_state: bool = True
    sharpe_window: int = 20
    horizon: int = 20
    eta_dd: float = 0.5
    use_obs_normalizer: bool = True
    obs_clip: float = 10.0
    obs_norm_update: bool = True


@dataclass
class VecMarginEnvParams:
    max_stock_pct: float = 0.02
    margin_rate: float = 2.0
    long_short_ratio: float = 1.0
    maintenance_margin: float = 0.3
    maintenance_warning: float = 0.4
    max_trade_volume_pct: float = 0.1
    lambda_1: float = 1e-5
    lambda_2: float = 0.01
    sharpe_window: int = 5
    margin_adjust_period: int = 30


@dataclass
class VecMACEBacktestParams:
    model_name: str
    impact_model_class: type
    initial_capital: float = 1e9
    env_params: VecMACEEnvParams = field(default_factory=VecMACEEnvParams)
    model_kwargs: dict | None = None
    policy_kwargs: dict | None = None

    impact_model_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_name = self.model_name.lower()
        self.impact_model_name = str(self.impact_model_class())

    def get_model_kwargs(self) -> dict:
        return _clone_model_kwargs(self.model_name, self.model_kwargs)

    _BACKTEST_KEYS = {
        "model_name",
        "impact_model_class",
        "initial_capital",
        "model_kwargs",
        "policy_kwargs",
    }

    @staticmethod
    def from_explicit(
        configs: list[dict],
        num_stocks: int,
        max_stock_weight_multiplier: float = 2.0,
        max_stock_pct_clip: tuple[float, float] = (0.01, 1.0),
    ) -> list["VecMACEBacktestParams"]:
        default_max_stock_pct = _default_max_stock_pct(
            num_stocks,
            max_stock_weight_multiplier=max_stock_weight_multiplier,
            max_stock_pct_clip=max_stock_pct_clip,
        )

        result: list[VecMACEBacktestParams] = []
        for cfg in configs:
            bt_kwargs = {}
            env_kwargs: dict = {"max_stock_pct": default_max_stock_pct}
            for key, value in cfg.items():
                if key in VecMACEBacktestParams._BACKTEST_KEYS:
                    bt_kwargs[key] = value
                else:
                    env_kwargs[key] = value
            bt_kwargs["env_params"] = VecMACEEnvParams(**env_kwargs)
            result.append(VecMACEBacktestParams(**bt_kwargs))
        return result


@dataclass
class VecMarginBacktestParams:
    model_name: str
    impact_model_class: type
    initial_capital: float = 1e9
    env_params: VecMarginEnvParams = field(default_factory=VecMarginEnvParams)
    model_kwargs: dict | None = None
    policy_kwargs: dict | None = None

    impact_model_name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_name = self.model_name.lower()
        self.impact_model_name = str(self.impact_model_class())

    def get_model_kwargs(self) -> dict:
        return _clone_model_kwargs(self.model_name, self.model_kwargs)

    _BACKTEST_KEYS = {
        "model_name",
        "impact_model_class",
        "initial_capital",
        "model_kwargs",
        "policy_kwargs",
    }

    @staticmethod
    def from_explicit(
        configs: list[dict],
        num_stocks: int,
        max_stock_weight_multiplier: float = 2.0,
        max_stock_pct_clip: tuple[float, float] = (0.01, 1.0),
    ) -> list["VecMarginBacktestParams"]:
        default_max_stock_pct = _default_max_stock_pct(
            num_stocks,
            max_stock_weight_multiplier=max_stock_weight_multiplier,
            max_stock_pct_clip=max_stock_pct_clip,
        )

        result: list[VecMarginBacktestParams] = []
        for cfg in configs:
            bt_kwargs = {}
            env_kwargs: dict = {"max_stock_pct": default_max_stock_pct}
            for key, value in cfg.items():
                if key in VecMarginBacktestParams._BACKTEST_KEYS:
                    bt_kwargs[key] = value
                else:
                    env_kwargs[key] = value
            bt_kwargs["env_params"] = VecMarginEnvParams(**env_kwargs)
            result.append(VecMarginBacktestParams(**bt_kwargs))
        return result
