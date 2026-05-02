from __future__ import annotations

import argparse
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
from meta.env_market_impact.backtest_vec_config import (
    SUPPORTED_VEC_A2C_PRESETS,
)
from meta.env_market_impact.backtest_vec_config import VecMACEBacktestParams
from meta.env_market_impact.backtest_vec_config import (
    apply_vec_mace_a2c_preset,
)
from meta.env_market_impact.envs.market_data import Split
from meta.env_market_impact.envs.utils import get_logger
from meta.env_market_impact.vec.data_prep import (
    build_vec_tiingo_market_data_preparator,
)
from meta.env_market_impact.vec.mace_vec_env import MACEVecEnv
from meta.env_market_impact.vec.intensity_timing_wrapper import IntensityTimingWrapper
from meta.env_market_impact.vec.runner_utils import SUPPORTED_ELEGANTRL_AGENTS
from meta.env_market_impact.vec.runner_utils import build_tensor_impact_model
from meta.env_market_impact.vec.runner_utils import impact_model_name
from meta.env_market_impact.vec.runner_utils import resolve_default_num_envs
from meta.env_market_impact.vec.runner_utils import resolve_elegantrl_settings
from meta.env_market_impact.vec.runner_utils import run_vec_simulation_stats
from meta.env_market_impact.vec.runner_utils import save_backtest_summary
from meta.env_market_impact.vec.runner_utils import save_vec_backtest_triplet
from meta.env_market_impact.vec.runner_utils import train_with_epoch_evaluation

log = get_logger()

IMPACT_MODEL_NAME_MAP = {
    str(impact_model_class()).lower(): impact_model_class
    for impact_model_class in VEC_IMPACT_MODEL_CLASSES
}

# Intensity timing wrapper defaults (from V5 sweep: ρ=0.002 optimal)
INTENSITY_DEFAULTS = dict(
    M=5.0,
    eta=0.005,
    rho=0.002,
    dt=0.25,
    Psi=0.20,
    varpi=0.5,
    k_loss=2.0,
    gamma=1.0,
    iota=1.0,
    R=0.0,
    entropy_reward_scale=0.0,
    augment_state=True,
)


def _wrap_with_intensity(env: MACEVecEnv, **overrides) -> IntensityTimingWrapper:
    """Wrap a MACEVecEnv with the intensity timing gate."""
    kwargs = {**INTENSITY_DEFAULTS, **overrides}
    return IntensityTimingWrapper(env, **kwargs)


class IntensityMACEVecEnv(IntensityTimingWrapper):
    """MACEVecEnv + IntensityTimingWrapper in one class.

    ElegantRL's ``build_env`` calls ``kwargs_filter(cls.__init__, env_args)``
    to select valid constructor arguments. By accepting the same signature as
    ``MACEVecEnv.__init__`` plus ``intensity_kwargs``, the filter keeps all
    the parameters that the base env needs.
    """

    def __init__(
        self,
        config,
        params=None,
        num_envs=128,
        gpu_id=-1,
        device=None,
        impact_config=None,
        impact_model=None,
        initial_capital=1e6,
        initial_stocks=None,
        normalizer_state_path=None,
        freeze_loaded_normalizer=False,
        if_random_reset=False,
        auto_reset=True,
        intensity_kwargs=None,
    ):
        base_env = MACEVecEnv(
            config=config,
            params=params,
            num_envs=num_envs,
            gpu_id=gpu_id,
            device=device,
            impact_config=impact_config,
            impact_model=impact_model,
            initial_capital=initial_capital,
            initial_stocks=initial_stocks,
            normalizer_state_path=normalizer_state_path,
            freeze_loaded_normalizer=freeze_loaded_normalizer,
            if_random_reset=if_random_reset,
            auto_reset=auto_reset,
        )
        ikw = intensity_kwargs or {}
        super().__init__(base_env, **{**INTENSITY_DEFAULTS, **ikw})


def _resolve_default_gpu_id(gpu_id: int | None) -> int:
    if gpu_id is not None:
        return gpu_id
    return 0 if th.cuda.is_available() else -1


def _load_mace_normalizer(env: MACEVecEnv, normalizer_state_path: str) -> None:
    if env.params.use_obs_normalizer and os.path.isfile(normalizer_state_path):
        env.load_normalizer_state(normalizer_state_path, freeze=True)


def _build_vec_base_filename(
    params: VecMACEBacktestParams,
    resolved_settings: dict[str, object],
) -> str:
    ep = params.env_params
    base = (
        f"backtest_{params.model_name}_"
        f"{params.impact_model_name.replace(' ', '_')}_"
        f"{str(int(params.initial_capital))}"
    )
    resolved_fingerprint = {
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
        "eval_times": resolved_settings.get("eval_times"),
        "num_envs": resolved_settings.get("num_envs"),
        "requested_num_envs": resolved_settings.get("requested_num_envs"),
        "net_dims": tuple(resolved_settings.get("net_dims") or ()),
    }
    param_string = (
        f"model={params.model_name};"
        f"impact={params.impact_model_name};"
        f"capital={params.initial_capital};"
        f"perm={ep.include_permanent_impact_in_state};"
        f"cooldown={ep.include_cooldown_in_state};"
        f"tbill={ep.include_tbill_in_state};"
        f"eta_dd={ep.eta_dd};"
        f"norm={ep.use_obs_normalizer};"
        f"reward_scaling={ep.reward_scaling};"
        f"horizon={ep.horizon};"
        f"obs_clip={ep.obs_clip};"
        f"resolved_settings={sorted(resolved_fingerprint.items())}"
    )
    uid = uuid.uuid5(uuid.NAMESPACE_URL, param_string).hex[:8]
    return f"{base}_{uid}"


def _normalize_selected_agents(
    agents: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if not agents:
        return tuple(SUPPORTED_ELEGANTRL_AGENTS)

    resolved_agents = tuple(agent.strip().lower() for agent in agents if agent.strip())
    unsupported = sorted(set(resolved_agents) - set(SUPPORTED_ELEGANTRL_AGENTS))
    if unsupported:
        raise ValueError(
            f"Unsupported agent(s) {unsupported}. "
            f"Available: {list(SUPPORTED_ELEGANTRL_AGENTS)}"
        )
    return resolved_agents


def _normalize_selected_impact_models(
    impact_models: list[str] | tuple[str, ...] | None,
) -> tuple[type, ...]:
    if not impact_models:
        return tuple(VEC_IMPACT_MODEL_CLASSES)

    resolved_classes = []
    unknown_models = []
    for name in impact_models:
        normalized_name = name.strip().lower()
        if not normalized_name:
            continue
        impact_model_class = IMPACT_MODEL_NAME_MAP.get(normalized_name)
        if impact_model_class is None:
            unknown_models.append(name)
            continue
        resolved_classes.append(impact_model_class)

    if unknown_models:
        raise ValueError(
            f"Unsupported impact model(s) {unknown_models}. "
            f"Available: {list(IMPACT_MODEL_NAME_MAP)}"
        )
    return tuple(resolved_classes)


def _parse_csv_option(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _build_vec_mace_configs(
    *,
    num_stocks: int,
    agents: list[str] | tuple[str, ...] | None = None,
    impact_models: list[str] | tuple[str, ...] | None = None,
    a2c_preset: str | None = None,
) -> list[VecMACEBacktestParams]:
    selected_agents = _normalize_selected_agents(agents)
    selected_impact_models = _normalize_selected_impact_models(impact_models)
    configs = []
    for agent_name in selected_agents:
        for impact_model_class in selected_impact_models:
            config = {
                "model_name": agent_name,
                "impact_model_class": impact_model_class,
                "initial_capital": 1e9,
                "model_kwargs": dict(VEC_MODEL_KWARGS[agent_name]),
            }
            if agent_name == "a2c":
                config = apply_vec_mace_a2c_preset(config, a2c_preset)
            configs.append(config)
    return VecMACEBacktestParams.from_explicit(configs, num_stocks=num_stocks)


def train_and_backtest(
    data_prep,
    backtest_grid: list[VecMACEBacktestParams],
    num_epochs: int = 20,
    *,
    num_envs: int | None = None,
    gpu_id: int | None = None,
    num_workers: int = 1,
    random_seed: int = 42,
) -> str:
    gpu_id = _resolve_default_gpu_id(gpu_id)
    user_forced_num_envs = num_envs is not None
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

        model_kwargs = params.get_model_kwargs()

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
        # Inject intensity_kwargs into env construction for the factory class
        probe_train_env_kwargs["intensity_kwargs"] = dict(INTENSITY_DEFAULTS)
        probe_eval_env_kwargs["intensity_kwargs"] = dict(INTENSITY_DEFAULTS)
        resolved_settings = resolve_elegantrl_settings(
            params.model_name,
            model_kwargs,
            params.policy_kwargs,
            steps_per_epoch=steps_per_epoch,
            env_class=IntensityMACEVecEnv,
            train_env_kwargs=probe_train_env_kwargs,
            eval_env_kwargs=probe_eval_env_kwargs,
            requested_num_envs=requested_num_envs,
            gpu_id=gpu_id,
            num_workers=num_workers,
            disable_num_envs_scaling=user_forced_num_envs,
        )
        base_filename = _build_vec_base_filename(params, resolved_settings)
        run_dir = os.path.join(results_dir, f"{base_filename}_elegantrl")
        normalizer_state_path = os.path.join(run_dir, "vec_normalize.pt")
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
            "intensity_kwargs": dict(INTENSITY_DEFAULTS),
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
            "normalizer_state_path": normalizer_state_path,
            "auto_reset": True,
            "intensity_kwargs": dict(INTENSITY_DEFAULTS),
        }

        def evaluate_epoch(actor):
            continued_impact_model = build_tensor_impact_model(
                params.impact_model_class,
                num_envs=1,
                stock_dim=stock_dim,
                gpu_id=gpu_id,
            )
            base_train_env = MACEVecEnv(
                config=train_config,
                params=params.env_params,
                num_envs=1,
                gpu_id=gpu_id,
                impact_model=continued_impact_model,
                initial_capital=params.initial_capital,
                auto_reset=False,
            )
            _load_mace_normalizer(base_train_env, normalizer_state_path)
            eval_train_env = _wrap_with_intensity(base_train_env)
            train_stats = run_vec_simulation_stats(
                eval_train_env,
                actor,
                reset_impact_model=True,
            )

            base_blank_env = MACEVecEnv(
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
            _load_mace_normalizer(base_blank_env, normalizer_state_path)
            blank_env = _wrap_with_intensity(base_blank_env)
            blank_stats = run_vec_simulation_stats(
                blank_env,
                actor,
                reset_impact_model=True,
            )
            return train_stats, blank_stats

        trained_actor, epoch_stats_train, epoch_stats_test_blank, _ = (
            train_with_epoch_evaluation(
                env_class=IntensityMACEVecEnv,
                train_env_kwargs=train_env_kwargs,
                eval_env_kwargs=eval_env_kwargs,
                agent_name=params.model_name,
                model_kwargs=model_kwargs,
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
        base_train_eval_env = MACEVecEnv(
            config=train_config,
            params=params.env_params,
            num_envs=1,
            gpu_id=gpu_id,
            impact_model=continued_impact_model,
            initial_capital=params.initial_capital,
            auto_reset=False,
        )
        _load_mace_normalizer(base_train_eval_env, normalizer_state_path)
        train_eval_env = _wrap_with_intensity(base_train_eval_env)
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
                "lambda_gae_adv": resolved_settings.get("lambda_gae_adv"),
                "lambda_entropy": resolved_settings.get("lambda_entropy"),
                "clip_grad_norm": resolved_settings.get("clip_grad_norm"),
                "ratio_clip": resolved_settings.get("ratio_clip"),
                "if_use_v_trace": resolved_settings.get("if_use_v_trace"),
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
    params: VecMACEBacktestParams,
    gpu_id: int,
    train_eval_env,
    normalizer_state_path: str,
):
    # Unwrap to get base env's impact model and state
    base_train = train_eval_env.base_env if isinstance(train_eval_env, IntensityTimingWrapper) else train_eval_env
    base_trade = MACEVecEnv(
        config=trade_config,
        params=params.env_params,
        num_envs=1,
        gpu_id=gpu_id,
        impact_model=base_train.impact_model,
        initial_capital=float(base_train.cash[0].item()),
        initial_stocks=base_train.stocks[0].clone(),
        auto_reset=False,
    )
    _load_mace_normalizer(base_trade, normalizer_state_path)
    return _wrap_with_intensity(base_trade)


def _build_mace_blank_env(
    *,
    trade_config,
    params: VecMACEBacktestParams,
    gpu_id: int,
    stock_dim: int,
    normalizer_state_path: str,
):
    base_blank = MACEVecEnv(
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
    _load_mace_normalizer(base_blank, normalizer_state_path)
    return _wrap_with_intensity(base_blank)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vec MACE backtests.")
    parser.add_argument(
        "--agents",
        help=(
            "Comma-separated ElegantRL agent names to run "
            "(for example: ddpg,sac,td3). Defaults to all agents."
        ),
    )
    parser.add_argument(
        "--impact-models",
        help=(
            "Comma-separated impact model display names to run. "
            "Defaults to all impact models."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=(
            "Force the vec env count and bypass GPU memory auto-scaling. "
            "Useful for reproducing prior runs."
        ),
    )
    parser.add_argument(
        "--a2c-preset",
        choices=sorted(SUPPORTED_VEC_A2C_PRESETS),
        default=None,
        help=(
            "Apply a named A2C preset when building the default vec MACE "
            "grid. "
            "Leave unset to keep the current vec defaults."
        ),
    )
    return parser.parse_args()


def run_example(
    *,
    agents: list[str] | tuple[str, ...] | None = None,
    impact_models: list[str] | tuple[str, ...] | None = None,
    num_envs: int | None = None,
    a2c_preset: str | None = None,
) -> None:
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

    backtest_grid = _build_vec_mace_configs(
        num_stocks=data_prep.universe_size,
        agents=agents,
        impact_models=impact_models,
        a2c_preset=a2c_preset,
    )
    log.info(
        "Vec MACE grid: %s configs across agents=%s impact_models=%s "
        "a2c_preset=%s",
        len(backtest_grid),
        sorted({params.model_name for params in backtest_grid}),
        sorted({params.impact_model_name for params in backtest_grid}),
        a2c_preset,
    )

    summary_path = train_and_backtest(
        data_prep,
        backtest_grid,
        num_epochs=num_epochs,
        num_envs=num_envs,
    )
    BacktestReportGenerator(summary_path).generate_report()
    log.info("Vec MACE backtests complete.")


if __name__ == "__main__":
    cli_args = _parse_args()
    run_example(
        agents=_parse_csv_option(cli_args.agents),
        impact_models=_parse_csv_option(cli_args.impact_models),
        num_envs=cli_args.num_envs,
        a2c_preset=cli_args.a2c_preset,
    )
