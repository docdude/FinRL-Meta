from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch as th
from pandas.testing import assert_frame_equal

from meta.env_market_impact.backtest_config import BacktestParams
from meta.env_market_impact.envs.env_mace_stock_trading import EnvParams
from meta.env_market_impact.envs.env_mace_stock_trading import MACEStockTradingEnv
from meta.env_market_impact.envs.env_margin_trader_impact import MarginEnvParams
from meta.env_market_impact.envs.env_margin_trader_impact import MarginTraderImpactEnv
from meta.env_market_impact.envs.impact_models import ACImpactModel
from meta.env_market_impact.envs.impact_models import BaselineImpactModel
from meta.env_market_impact.envs.impact_models import OWImpactModel
from meta.env_market_impact.envs.impact_models import SqrtImpactModel
from meta.env_market_impact.vec import runner_utils
from meta.env_market_impact.vec.runner_utils import build_tensor_impact_model
from meta.env_market_impact.vec.runner_utils import resolve_elegantrl_settings
from meta.env_market_impact.vec.runner_utils import run_vec_simulation
from meta.env_market_impact.vec.runner_utils import run_vec_simulation_stats
from meta.env_market_impact.vec.runner_utils import scalarize
from meta.env_market_impact.vec.runner_utils import train_with_epoch_evaluation
from meta.env_market_impact.train_and_backtest_vec_mace import (
    _build_vec_mace_configs,
    _build_vec_base_filename,
)
from meta.env_market_impact.vec.mace_vec_env import MACEVecEnv
from meta.env_market_impact.vec.margin_vec_env import MarginTraderVecEnv


def _build_market_config() -> dict:
    return {
        "date_list": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "price_array": np.array(
            [[10.0, 20.0], [11.0, 19.0], [12.0, 18.0]],
            dtype=np.float32,
        ),
        "tech_array": np.array(
            [[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]],
            dtype=np.float32,
        ),
        "volatility_array": np.array(
            [[0.02, 0.03], [0.02, 0.03], [0.02, 0.03]],
            dtype=np.float32,
        ),
        "volume_array": np.array(
            [[1000.0, 1500.0], [1000.0, 1500.0], [1000.0, 1500.0]],
            dtype=np.float32,
        ),
        "adv20_array": np.array(
            [[1000.0, 1500.0], [1000.0, 1500.0], [1000.0, 1500.0]],
            dtype=np.float32,
        ),
        "tbill_rates": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "tic_list": ["AAA", "BBB"],
    }


def _build_long_market_config() -> dict:
    price_array = np.array(
        [
            [10.0, 20.0],
            [10.5, 19.5],
            [11.0, 19.0],
            [10.8, 19.2],
            [11.3, 18.7],
            [11.6, 18.4],
        ],
        dtype=np.float32,
    )
    tech_array = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.4, 0.3],
            [0.5, 0.4],
        ],
        dtype=np.float32,
    )
    vol = np.full_like(price_array, 0.02, dtype=np.float32)
    volume = np.array(
        [[1000.0, 1500.0]] * len(price_array),
        dtype=np.float32,
    )
    return {
        "date_list": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ],
        "price_array": price_array,
        "tech_array": tech_array,
        "volatility_array": vol,
        "volume_array": volume,
        "adv20_array": volume.copy(),
        "tbill_rates": np.zeros(len(price_array), dtype=np.float32),
        "tic_list": ["AAA", "BBB"],
    }


def _build_benchmark_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [100.0, 101.0, 102.0],
        }
    )


class ConstantActor(th.nn.Module):
    def __init__(self, action: list[float]) -> None:
        super().__init__()
        self.action = th.nn.Parameter(th.tensor(action, dtype=th.float32))

    def forward(self, state: th.Tensor) -> th.Tensor:
        return self.action.unsqueeze(0).expand(state.shape[0], -1)


def _scalarize_info_value(value) -> float:
    return float(scalarize(value))


def _assert_margin_float32_close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, rel=1e-6, abs=64.0)


def test_mace_apply_trades_batched_zero_trade_returns_zero_tensors() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
    )

    perm_before = env.impact_model.get_perm_state_array().clone()
    cost, price_shift = env.impact_model.apply_trades_batched(
        trade_size=th.zeros((1, 2), dtype=th.float32),
        price=th.tensor([[10.0, 20.0]], dtype=th.float32),
        volatility=th.tensor([[0.02, 0.03]], dtype=th.float32),
        volume=th.tensor([[1000.0, 1500.0]], dtype=th.float32),
    )

    assert cost.shape == (1, 2)
    assert price_shift.shape == (1, 2)
    assert cost.dtype == th.float32
    assert price_shift.dtype == th.float32
    assert th.equal(cost, th.zeros_like(cost))
    assert th.equal(price_shift, th.zeros_like(price_shift))
    assert th.equal(env.impact_model.get_perm_state_array(), perm_before)


def test_margin_invalid_sell_helpers_return_zero_tuple() -> None:
    env = MarginTraderVecEnv(
        config=_build_market_config(),
        num_envs=1,
        gpu_id=-1,
    )
    env.reset()

    sell_long = env._sell_long(
        stock_idx=0,
        shares=th.tensor([1], dtype=th.int32),
        price=th.tensor([0.0, 20.0]),
        volatility=th.tensor([0.02, 0.03]),
        volume=th.tensor([1000.0, 1500.0]),
    )
    sell_short = env._sell_short(
        stock_idx=0,
        shares=th.tensor([1], dtype=th.int32),
        price=th.tensor([0.0, 20.0]),
        volatility=th.tensor([0.02, 0.03]),
        volume=th.tensor([1000.0, 1500.0]),
    )

    for value, cost, executed in (sell_long, sell_short):
        assert value.tolist() == [0.0]
        assert cost.tolist() == [0.0]
        assert executed.tolist() == [0]


def test_build_tensor_impact_model_copies_instance_parameters() -> None:
    ac_model = ACImpactModel(
        alpha=2.0,
        beta=0.5,
        epsilon=0.001,
        perm_half_life_days=7.0,
    )
    tensor_model = build_tensor_impact_model(
        ac_model,
        num_envs=2,
        stock_dim=3,
        gpu_id=-1,
    )

    assert tensor_model.alpha == 2.0
    assert tensor_model.beta == 0.5
    assert tensor_model.epsilon == 0.001
    assert tensor_model.perm_half_life_days == 7.0


def test_build_tensor_impact_model_supports_ow_instances() -> None:
    ow_model = OWImpactModel(
        Y=0.9,
        perm_fraction=0.4,
        half_life_days=0.2,
        perm_half_life_days=9.0,
    )
    tensor_model = build_tensor_impact_model(
        ow_model,
        num_envs=2,
        stock_dim=3,
        gpu_id=-1,
    )

    assert tensor_model.Y == 0.9
    assert tensor_model.perm_fraction == 0.4
    assert tensor_model.half_life_days == 0.2
    assert tensor_model.perm_half_life_days == 9.0


def test_save_normalizer_state_accepts_basename(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
    )

    env.save_normalizer_state("normalizer.pt")

    assert (tmp_path / "normalizer.pt").is_file()


def test_legacy_vec_normalize_redirects_to_explicit_normalizer_path(
    tmp_path: Path,
) -> None:
    normalizer_path = tmp_path / "mace_obs_normalizer.pt"

    train_env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        normalizer_state_path=str(normalizer_path),
    )
    train_env.reset()
    train_env.step(th.tensor([[0.5, 0.0]], dtype=th.float32))

    legacy_path = tmp_path / "vec_normalize.pt"
    train_env.save(str(legacy_path))

    assert normalizer_path.is_file()
    assert not legacy_path.exists()

    eval_env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        normalizer_state_path=str(normalizer_path),
        freeze_loaded_normalizer=True,
    )
    eval_env.load(str(legacy_path))

    train_state = train_env.get_normalizer_state()
    eval_state = eval_env.get_normalizer_state()
    assert train_state is not None
    assert eval_state is not None
    assert np.array_equal(train_state["mean"], eval_state["mean"])
    assert np.array_equal(train_state["var"], eval_state["var"])
    assert eval_state["count"] == pytest.approx(train_state["count"])


def test_single_env_impact_history_is_recorded() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
    )
    env.reset()

    env.step(th.tensor([[0.5, 0.0]], dtype=th.float32))
    history = env.get_impact_history()

    assert list(history.columns) == ["date", "symbol", "permanent_impact"]
    assert history["date"].tolist() == ["2024-01-02", "2024-01-02"]
    assert history["symbol"].tolist() == ["AAA", "BBB"]


def test_build_vec_mace_configs_uses_canonical_env_defaults() -> None:
    configs = _build_vec_mace_configs(
        num_stocks=99,
        agents=["a2c"],
        impact_models=["Baseline Impact Model"],
    )

    assert len(configs) == 1
    env_params = configs[0].env_params
    assert env_params.include_permanent_impact_in_state is True
    assert env_params.include_cooldown_in_state is True
    assert env_params.include_tbill_in_state is True
    assert env_params.eta_dd == pytest.approx(0.5)
    assert env_params.use_obs_normalizer is True
    assert env_params.reward_scaling == pytest.approx(2**-11)
    assert env_params.horizon == 20
    assert env_params.obs_clip == pytest.approx(10.0)


def test_margin_single_env_impact_history_is_recorded() -> None:
    env = MarginTraderVecEnv(
        config=_build_market_config(),
        num_envs=1,
        gpu_id=-1,
    )
    env.reset()

    env.step(th.tensor([[0.5, 0.0]], dtype=th.float32))
    history = env.get_impact_history()

    assert list(history.columns) == ["date", "symbol", "permanent_impact"]
    assert history["date"].tolist() == ["2024-01-02", "2024-01-02"]


def test_margin_state_single_env_returns_tensors() -> None:
    env = MarginTraderVecEnv(
        config=_build_market_config(),
        num_envs=1,
        gpu_id=-1,
    )
    env.reset()

    state = env.get_margin_state()

    assert all(isinstance(value, th.Tensor) for value in state.values())


def test_mace_multi_env_step_emits_aggregate_trades() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=2,
        gpu_id=-1,
        auto_reset=False,
    )
    env.reset()

    _, _, _, _, info = env.step(
        th.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=th.float32)
    )

    assert info["trades"]
    assert info["trades"][0]["side"] == "buy"
    assert info["trades"][0]["shares"] == 200


def test_margin_multi_env_step_emits_aggregate_trades() -> None:
    env = MarginTraderVecEnv(
        config=_build_market_config(),
        num_envs=2,
        gpu_id=-1,
        auto_reset=False,
    )
    env.reset()

    _, _, _, _, info = env.step(
        th.tensor([[-1.0, 0.0], [-1.0, 0.0]], dtype=th.float32)
    )

    assert info["trades"]
    assert info["trades"][0]["shares"] > 0


def test_run_vec_simulation_supports_selected_env_from_multienv() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=2,
        gpu_id=-1,
        auto_reset=False,
    )
    actor = ConstantActor([0.0, 0.0])

    results_df, trades_df = run_vec_simulation(
        env,
        actor,
        env.date_list,
        _build_benchmark_df(),
        env_index=0,
    )

    assert not results_df.empty
    assert list(trades_df.columns)


def test_mace_normalizer_transfer_reproduces_eval_observation() -> None:
    source_env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    source_env.reset()
    source_env.step(th.tensor([[0.25, -0.25]], dtype=th.float32))
    saved_state = source_env.get_normalizer_state()

    eval_env_a = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    eval_env_b = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    eval_env_a.set_normalizer_state(saved_state, freeze=True)
    eval_env_b.set_normalizer_state(saved_state, freeze=True)

    obs_a, _ = eval_env_a.reset()
    obs_b, _ = eval_env_b.reset()

    assert th.allclose(obs_a, obs_b)


def test_mace_get_state_returns_clone_when_normalizer_disabled() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    obs, _ = env.reset()
    obs_copy = obs.clone()

    env.step(th.tensor([[0.25, 0.0]], dtype=th.float32))

    assert th.equal(obs, obs_copy)


def test_run_vec_simulation_is_deterministic_for_fixed_actor() -> None:
    actor = ConstantActor([0.0, 0.0])
    benchmark_df = _build_benchmark_df()

    env_a = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    env_b = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )

    results_a, trades_a = run_vec_simulation(
        env_a,
        actor,
        env_a.date_list,
        benchmark_df,
    )
    results_b, trades_b = run_vec_simulation(
        env_b,
        actor,
        env_b.date_list,
        benchmark_df,
    )

    assert_frame_equal(results_a, results_b)
    assert_frame_equal(trades_a, trades_b)


@pytest.mark.parametrize(
    "action",
    ([0.0, 0.0], [1.0, 0.0]),
)
def test_run_vec_simulation_stats_matches_full_simulation(action) -> None:
    actor = ConstantActor(action)
    benchmark_df = _build_benchmark_df()

    full_env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    stats_env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )

    results_df, trades_df = run_vec_simulation(
        full_env,
        actor,
        full_env.date_list,
        benchmark_df,
    )
    full_stats = runner_utils.compute_stats_from_results(results_df, trades_df)
    fast_stats = run_vec_simulation_stats(stats_env, actor)

    assert full_stats.keys() == fast_stats.keys()
    for key, expected_value in full_stats.items():
        assert fast_stats[key] == pytest.approx(
            expected_value,
            rel=1e-6,
            abs=1e-9,
        )


def test_mace_multi_env_action_divergence_tracks_per_env_state() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=False),
        num_envs=2,
        gpu_id=-1,
        auto_reset=False,
    )
    env.reset()

    _, _, _, _, info = env.step(
        th.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=th.float32)
    )

    assert env.stocks[0].tolist() == [100.0, 0.0]
    assert env.stocks[1].tolist() == [0.0, 150.0]
    trades_by_stock = {trade["stock_idx"]: trade["shares"] for trade in info["trades"]}
    assert trades_by_stock == {0: 100, 1: 150}


def test_mace_multienv_normalizer_keeps_identical_env_rows_in_sync() -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=4,
        gpu_id=-1,
        auto_reset=False,
    )
    obs, _ = env.reset()
    assert th.allclose(obs[0], obs[1])
    assert th.allclose(obs[0], obs[2])

    next_obs, _, _, _, _ = env.step(
        th.tensor([[0.5, -0.25]] * 4, dtype=th.float32)
    )
    assert th.allclose(next_obs[0], next_obs[1])
    assert th.allclose(next_obs[0], next_obs[3])


def test_margin_step_at_short_maintenance_warning_does_not_crash() -> None:
    env = MarginTraderVecEnv(
        config=_build_market_config(),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )
    env.reset()
    env.stocks[0, 0] = -10.0
    env.short_equity[0] = 40.0
    env.short_limit[0] = 100.0
    env.short_credit[0] = 140.0

    _, _, done, truncated, info = env.step(
        th.tensor([[-1.0, 0.0]], dtype=th.float32)
    )

    assert bool(done[0].item()) is False
    assert bool(truncated[0].item()) is False
    assert "cost" in info


def test_train_with_epoch_evaluation_rejects_mult_worker_normalizer() -> None:
    with pytest.raises(ValueError, match="use num_workers=1"):
        train_with_epoch_evaluation(
            env_class=MACEVecEnv,
            train_env_kwargs={
                "config": _build_market_config(),
                "params": EnvParams(use_obs_normalizer=True),
                "num_envs": 1,
                "gpu_id": -1,
                "auto_reset": True,
            },
            eval_env_kwargs=None,
            agent_name="ppo",
            model_kwargs=None,
            policy_kwargs=None,
            num_epochs=1,
            steps_per_epoch=2,
            run_dir="/tmp/vec-test",
            evaluate_epoch=lambda actor: ({}, {}),
            gpu_id=-1,
            num_workers=2,
            random_seed=42,
            if_single_process=True,
        )


def test_resolve_elegantrl_settings_caps_on_policy_batch_size() -> None:
    settings = resolve_elegantrl_settings(
        "a2c",
        model_kwargs={"n_steps": 5, "batch_size": 128},
        policy_kwargs=None,
        steps_per_epoch=100,
        env_class=MACEVecEnv,
        train_env_kwargs={
            "config": _build_market_config(),
            "params": EnvParams(use_obs_normalizer=False),
            "num_envs": 120,
            "gpu_id": -1,
            "auto_reset": True,
        },
        eval_env_kwargs={
            "config": _build_market_config(),
            "params": EnvParams(use_obs_normalizer=False),
            "num_envs": 120,
            "gpu_id": -1,
            "auto_reset": True,
        },
        requested_num_envs=120,
        gpu_id=-1,
        num_workers=1,
    )

    assert settings["horizon_len"] == 100
    assert settings["rollout_batch_size"] == 12000
    assert settings["repeat_times"] >= 1
    assert settings["batch_size"] == (
        settings["horizon_len"] * settings["repeat_times"]
    )
    assert settings["batch_size"] <= (
        settings["horizon_len"] * settings["repeat_times"]
    )


def test_resolve_elegantrl_settings_caps_default_off_policy_buffer_warmup() -> None:
    settings = resolve_elegantrl_settings(
        "ddpg",
        model_kwargs={"batch_size": 128, "buffer_size": 50000},
        policy_kwargs=None,
        steps_per_epoch=100,
        env_class=MACEVecEnv,
        train_env_kwargs={
            "config": _build_market_config(),
            "params": EnvParams(use_obs_normalizer=False),
            "num_envs": 120,
            "gpu_id": -1,
            "auto_reset": True,
        },
        eval_env_kwargs={
            "config": _build_market_config(),
            "params": EnvParams(use_obs_normalizer=False),
            "num_envs": 120,
            "gpu_id": -1,
            "auto_reset": True,
        },
        requested_num_envs=120,
        gpu_id=-1,
        num_workers=1,
    )

    assert settings["horizon_len"] == 25
    assert settings["buffer_init_size"] == 25
    assert settings["buffer_init_size"] <= settings["horizon_len"]


def test_mace_vec_env_save_and_load_normalizer_state(tmp_path: Path) -> None:
    env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        auto_reset=True,
    )
    env.reset()
    env.step(th.zeros((1, env.action_dim), dtype=th.float32))

    snapshot_path = tmp_path / "mace_normalizer.pt"
    env.save(str(snapshot_path))

    loaded_env = MACEVecEnv(
        config=_build_market_config(),
        params=EnvParams(use_obs_normalizer=True),
        num_envs=1,
        gpu_id=-1,
        freeze_loaded_normalizer=True,
        auto_reset=True,
    )
    loaded_env.load(str(snapshot_path))

    state = env.get_normalizer_state()
    loaded_state = loaded_env.get_normalizer_state()

    assert snapshot_path.exists()
    assert state is not None
    assert loaded_state is not None
    assert np.allclose(state["mean"], loaded_state["mean"])
    assert np.allclose(state["var"], loaded_state["var"])
    assert state["count"] == pytest.approx(loaded_state["count"])


def test_train_with_epoch_evaluation_uses_single_training_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_utils.ensure_elegantrl_on_path()
    import elegantrl.train.run as elegantrl_run

    call_count = {"train_agent": 0}
    run_dir = tmp_path / "vec-run"
    normalizer_state_path = run_dir / "vec_normalize.pt"
    loaded_actor_paths: list[str] = []
    seen_normalizer_markers: list[int] = []

    def fake_build_training_args(**_: object) -> SimpleNamespace:
        return SimpleNamespace(
            cwd=str(run_dir),
            agent_class=object,
            net_dims=[1],
            state_dim=1,
            action_dim=1,
            gpu_id=-1,
            break_step=0,
            eval_per_step=0,
            if_remove=False,
            continue_train=True,
        )

    def fake_train_agent(args: object, if_single_process: bool) -> None:
        call_count["train_agent"] += 1
        assert if_single_process is True
        assert args.break_step == 7
        assert args.eval_per_step == 4
        assert args.if_remove is True
        assert args.continue_train is False

        snapshot_dir = run_dir / runner_utils.ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            run_dir / "recorder.npy",
            np.array(
                [
                    [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        for step, marker in ((4, 11), (8, 22)):
            th.save(
                th.nn.Linear(1, 1),
                snapshot_dir / f"actor__{step:012}.pt",
            )
            th.save(
                {"marker": marker},
                snapshot_dir / f"normalizer__{step:012}.pt",
            )

    class DummyActor:
        def __init__(self, source: str) -> None:
            self.source = source

    def fake_load_trained_actor(
        args: object,
        actor_path: str | None = None,
    ) -> DummyActor:
        assert args.cwd == str(run_dir)
        assert actor_path is not None
        loaded_actor_paths.append(actor_path)
        return DummyActor(actor_path)

    def fake_evaluate_epoch(
        _actor: DummyActor,
    ) -> tuple[dict[str, float], dict[str, float]]:
        state = th.load(normalizer_state_path, map_location="cpu", weights_only=False)
        seen_normalizer_markers.append(int(state["marker"]))
        score = float(len(loaded_actor_paths))
        return ({"score": score}, {"score": score})

    monkeypatch.setattr(
        runner_utils,
        "build_training_args",
        fake_build_training_args,
    )
    monkeypatch.setattr(elegantrl_run, "train_agent", fake_train_agent)
    monkeypatch.setattr(
        runner_utils,
        "load_trained_actor",
        fake_load_trained_actor,
    )

    actor, epoch_stats_train, epoch_stats_test_blank, args = (
        train_with_epoch_evaluation(
            env_class=MACEVecEnv,
            train_env_kwargs={
                "config": _build_market_config(),
                "params": EnvParams(use_obs_normalizer=True),
                "num_envs": 1,
                "gpu_id": -1,
                "normalizer_state_path": str(normalizer_state_path),
                "auto_reset": True,
            },
            eval_env_kwargs=None,
            agent_name="ppo",
            model_kwargs=None,
            policy_kwargs=None,
            num_epochs=2,
            steps_per_epoch=4,
            run_dir=str(run_dir),
            evaluate_epoch=fake_evaluate_epoch,
            gpu_id=-1,
            num_workers=1,
            random_seed=42,
            if_single_process=True,
        )
    )

    assert call_count["train_agent"] == 1
    assert args.cwd == str(run_dir)
    assert seen_normalizer_markers == [11, 22]
    assert len(loaded_actor_paths) == 2
    assert loaded_actor_paths[0].endswith("actor__000000000004.pt")
    assert loaded_actor_paths[1].endswith("actor__000000000008.pt")
    assert actor.source.endswith("actor__000000000008.pt")
    assert [stats["epoch"] for stats in epoch_stats_train] == [1, 2]
    assert [stats["epoch"] for stats in epoch_stats_test_blank] == [1, 2]


def test_epoch_snapshot_evaluator_recreates_deleted_snapshot_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_utils.ensure_elegantrl_on_path()
    import elegantrl.train.evaluator as elegantrl_evaluator
    import elegantrl.train.run as elegantrl_run

    class FakeEvaluator:
        def __init__(
            self,
            cwd: str,
            env: object,
            args: object,
            if_tensorboard: bool = False,
        ) -> None:
            del cwd, args, if_tensorboard
            self.env = env
            self.recorder: list[tuple[float, float]] = []

        def evaluate_and_save(
            self,
            actor: th.nn.Module,
            steps: int,
            exp_r: float,
            logging_tuple: tuple,
        ) -> None:
            del actor, exp_r, logging_tuple
            self.recorder.append((float(steps), 0.0))

    monkeypatch.setattr(elegantrl_run, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(elegantrl_evaluator, "Evaluator", FakeEvaluator)

    run_dir = tmp_path / "vec-run"
    snapshot_dir = run_dir / runner_utils.ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME

    def save_normalizer(path: str) -> None:
        th.save({"marker": 1}, path)

    with runner_utils._patch_elegantrl_evaluator_for_epoch_snapshots(str(run_dir)):
        snapshot_dir.rmdir()
        evaluator = elegantrl_run.Evaluator(
            cwd=str(run_dir),
            env=SimpleNamespace(save=save_normalizer),
            args=SimpleNamespace(),
            if_tensorboard=False,
        )
        evaluator.evaluate_and_save(
            actor=th.nn.Linear(1, 1),
            steps=4,
            exp_r=0.0,
            logging_tuple=(0.0, 0.0, 0.0, ""),
        )

    assert snapshot_dir.is_dir()
    assert (snapshot_dir / "actor__000000000004.pt").is_file()
    assert (snapshot_dir / "normalizer__000000000004.pt").is_file()


def test_epoch_snapshot_evaluator_tracks_epoch_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_utils.ensure_elegantrl_on_path()
    import elegantrl.train.evaluator as elegantrl_evaluator
    import elegantrl.train.run as elegantrl_run

    class FakeEvaluator:
        def __init__(
            self,
            cwd: str,
            env: object,
            args: object,
            if_tensorboard: bool = False,
        ) -> None:
            del cwd, args, if_tensorboard
            self.env = env
            self.total_step = 0
            self.recorder: list[tuple[float, float]] = []

        def evaluate_and_save(
            self,
            actor: th.nn.Module,
            steps: int,
            exp_r: float,
            logging_tuple: tuple,
        ) -> None:
            del actor, exp_r, logging_tuple
            self.total_step += int(steps)

    monkeypatch.setattr(elegantrl_run, "Evaluator", FakeEvaluator)
    monkeypatch.setattr(elegantrl_evaluator, "Evaluator", FakeEvaluator)

    run_dir = tmp_path / "vec-run"
    snapshot_dir = run_dir / runner_utils.ELEGANTRL_EPOCH_SNAPSHOT_DIRNAME

    def save_normalizer(path: str) -> None:
        th.save({"marker": 1}, path)

    with runner_utils._patch_elegantrl_evaluator_for_epoch_snapshots(
        str(run_dir),
        epoch_step_targets=[4, 8, 12],
    ):
        evaluator = elegantrl_run.Evaluator(
            cwd=str(run_dir),
            env=SimpleNamespace(save=save_normalizer),
            args=SimpleNamespace(),
            if_tensorboard=False,
        )
        for _ in range(4):
            evaluator.evaluate_and_save(
                actor=th.nn.Linear(1, 1),
                steps=3,
                exp_r=0.0,
                logging_tuple=(0.0, 0.0, 0.0, ""),
            )

    assert snapshot_dir.is_dir()
    assert runner_utils._load_epoch_steps(str(run_dir), 3) == [6, 9, 12]
    assert (snapshot_dir / "actor__000000000006.pt").is_file()
    assert (snapshot_dir / "actor__000000000009.pt").is_file()
    assert (snapshot_dir / "actor__000000000012.pt").is_file()
    assert (snapshot_dir / "normalizer__000000000006.pt").is_file()
    assert (snapshot_dir / "normalizer__000000000009.pt").is_file()
    assert (snapshot_dir / "normalizer__000000000012.pt").is_file()


def test_vec_mace_base_filename_uses_resolved_settings() -> None:
    params = BacktestParams(
        model_name="a2c",
        impact_model_class=BaselineImpactModel,
        env_params=EnvParams(use_obs_normalizer=False),
    )
    resolved_settings = resolve_elegantrl_settings(
        "a2c",
        params.model_kwargs,
        params.policy_kwargs,
        steps_per_epoch=100,
        env_class=MACEVecEnv,
        train_env_kwargs={
            "config": _build_market_config(),
            "params": params.env_params,
            "num_envs": 120,
            "gpu_id": -1,
            "auto_reset": True,
        },
        eval_env_kwargs={
            "config": _build_market_config(),
            "params": params.env_params,
            "num_envs": 120,
            "gpu_id": -1,
            "auto_reset": True,
        },
        requested_num_envs=120,
        gpu_id=-1,
        num_workers=1,
    )

    vec_base_filename = _build_vec_base_filename(params, resolved_settings)

    assert vec_base_filename.startswith(
        "backtest_a2c_Baseline_Impact_Model_1000000_"
    )
    assert vec_base_filename != params.base_filename

    changed_settings = dict(resolved_settings)
    changed_settings["horizon_len"] = 50
    assert (
        _build_vec_base_filename(params, changed_settings) != vec_base_filename
    )


@pytest.mark.parametrize(
    "impact_model_cls",
    [BaselineImpactModel, ACImpactModel, SqrtImpactModel],
)
def test_mace_scalar_vec_single_env_parity(impact_model_cls) -> None:
    config = _build_market_config()
    params = EnvParams(use_obs_normalizer=False)
    scalar_env = MACEStockTradingEnv(
        config=config,
        params=params,
        impact_model=impact_model_cls(),
    )
    vec_env = MACEVecEnv(
        config=config,
        params=params,
        num_envs=1,
        gpu_id=-1,
        impact_model=build_tensor_impact_model(
            impact_model_cls(),
            num_envs=1,
            stock_dim=2,
            gpu_id=-1,
        ),
        auto_reset=False,
    )
    scalar_env.reset()
    vec_env.reset()

    for action in (
        np.array([0.5, -0.25], dtype=np.float32),
        np.array([-1.0, 0.5], dtype=np.float32),
    ):
        _, scalar_reward, scalar_done, scalar_truncated, scalar_info = scalar_env.step(
            action
        )
        _, vec_reward, vec_done, vec_truncated, vec_info = vec_env.step(
            th.from_numpy(action).unsqueeze(0)
        )

        assert _scalarize_info_value(vec_reward[0]) == pytest.approx(
            float(scalar_reward),
            rel=1e-5,
            abs=1e-5,
        )
        for key in (
            "turnover",
            "cost",
            "total_buy_value",
            "total_sell_value",
            "cash",
        ):
            assert _scalarize_info_value(vec_info[key][0]) == pytest.approx(
                float(scalar_info[key]),
                rel=1e-5,
                abs=1e-5,
            )
        assert float(vec_env.total_asset[0].item()) == pytest.approx(
            float(scalar_env.total_asset),
            rel=1e-5,
            abs=1e-5,
        )
        assert bool(vec_done[0].item()) is scalar_done
        assert bool(vec_truncated[0].item()) is scalar_truncated


def test_mace_multistep_permanent_impact_and_path_parity() -> None:
    config = _build_long_market_config()
    params = EnvParams(use_obs_normalizer=False)
    scalar_env = MACEStockTradingEnv(
        config=config,
        params=params,
        impact_model=SqrtImpactModel(),
    )
    vec_env = MACEVecEnv(
        config=config,
        params=params,
        num_envs=1,
        gpu_id=-1,
        impact_model=build_tensor_impact_model(
            SqrtImpactModel(),
            num_envs=1,
            stock_dim=2,
            gpu_id=-1,
        ),
        auto_reset=False,
    )
    scalar_env.reset()
    vec_env.reset()

    actions = (
        np.array([0.5, -0.25], dtype=np.float32),
        np.array([-0.5, 0.25], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.75], dtype=np.float32),
        np.array([-1.0, -0.5], dtype=np.float32),
    )
    cumulative_scalar = {
        "cost": 0.0,
        "turnover": 0.0,
        "total_buy_value": 0.0,
        "total_sell_value": 0.0,
    }
    cumulative_vec = {key: 0.0 for key in cumulative_scalar}

    for action in actions:
        _, _, _, _, scalar_info = scalar_env.step(action)
        _, _, _, _, vec_info = vec_env.step(th.from_numpy(action).unsqueeze(0))
        for key in cumulative_scalar:
            cumulative_scalar[key] += float(scalar_info[key])
            cumulative_vec[key] += _scalarize_info_value(vec_info[key][0])
        scalar_perm = scalar_env.impact_model.get_perm_state_array(scalar_env.stock_symbols)
        vec_perm = vec_env.impact_model.get_perm_state_array()[0].cpu().numpy()
        assert np.allclose(vec_perm, scalar_perm, rtol=1e-5, atol=1e-5)
        assert float(vec_env.total_asset[0].item()) == pytest.approx(
            float(scalar_env.total_asset),
            rel=1e-5,
            abs=1e-5,
        )

    for key in cumulative_scalar:
        assert cumulative_vec[key] == pytest.approx(
            cumulative_scalar[key],
            rel=1e-5,
            abs=1e-5,
        )


def test_mace_scalar_vec_single_env_parity_ow() -> None:
    config = _build_long_market_config()
    params = EnvParams(use_obs_normalizer=False)
    scalar_model = OWImpactModel()
    scalar_env = MACEStockTradingEnv(
        config=config,
        params=params,
        impact_model=scalar_model,
    )
    vec_env = MACEVecEnv(
        config=config,
        params=params,
        num_envs=1,
        gpu_id=-1,
        impact_model=build_tensor_impact_model(
            OWImpactModel(),
            num_envs=1,
            stock_dim=2,
            gpu_id=-1,
        ),
        auto_reset=False,
    )
    scalar_env.reset()
    vec_env.reset()

    for action in (
        np.array([0.5, 0.0], dtype=np.float32),
        np.array([-0.25, 0.75], dtype=np.float32),
        np.array([0.0, -0.5], dtype=np.float32),
    ):
        scalar_env.step(action)
        vec_env.step(th.from_numpy(action).unsqueeze(0))

    scalar_perm = scalar_env.impact_model.get_perm_state_array(scalar_env.stock_symbols)
    vec_perm = vec_env.impact_model.get_perm_state_array()[0].cpu().numpy()
    assert np.allclose(vec_perm, scalar_perm, rtol=1e-5, atol=1e-5)


def test_margin_adjustment_cascade_parity() -> None:
    config = _build_market_config()
    params = MarginEnvParams(margin_adjust_period=1)
    initial_state = {
        "long_cash": 5.0,
        "loan": 25.0,
        "long_equity": 10.0,
        "short_limit": 5.0,
        "short_credit": 55.0,
        "short_equity": 10.0,
        "stocks": np.array([3.0, -2.0], dtype=np.float32),
    }
    scalar_env = MarginTraderImpactEnv(
        config=config,
        params=params,
        impact_model=BaselineImpactModel(),
        initial_margin_state=initial_state,
    )
    vec_env = MarginTraderVecEnv(
        config=config,
        params=params,
        num_envs=1,
        gpu_id=-1,
        impact_model=build_tensor_impact_model(
            BaselineImpactModel(),
            num_envs=1,
            stock_dim=2,
            gpu_id=-1,
        ),
        initial_margin_state=initial_state,
        auto_reset=False,
    )
    scalar_env.reset()
    vec_env.reset()

    scalar_env.step(np.array([0.0, 0.0], dtype=np.float32))
    vec_env.step(th.tensor([[0.0, 0.0]], dtype=th.float32))

    vec_state = vec_env.get_margin_state()
    assert _scalarize_info_value(vec_state["long_cash"]) == pytest.approx(
        float(scalar_env.long_cash),
        rel=1e-5,
        abs=1e-5,
    )
    assert _scalarize_info_value(vec_state["loan"]) == pytest.approx(
        float(scalar_env.loan),
        rel=1e-5,
        abs=1e-5,
    )
    assert _scalarize_info_value(vec_state["short_limit"]) == pytest.approx(
        float(scalar_env.short_limit),
        rel=1e-5,
        abs=1e-5,
    )
    assert _scalarize_info_value(vec_state["short_credit"]) == pytest.approx(
        float(scalar_env.short_credit),
        rel=1e-5,
        abs=1e-5,
    )
    assert _scalarize_info_value(vec_state["short_equity"]) == pytest.approx(
        float(scalar_env.short_equity),
        rel=1e-5,
        abs=1e-5,
    )
    assert np.allclose(
        vec_state["stocks"].cpu().numpy(),
        scalar_env.stocks,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("impact_model_cls", [BaselineImpactModel, ACImpactModel])
def test_margin_scalar_vec_single_env_parity(impact_model_cls) -> None:
    config = _build_market_config()
    params = MarginEnvParams()
    scalar_env = MarginTraderImpactEnv(
        config=config,
        params=params,
        impact_model=impact_model_cls(),
        initial_capital=1e6,
    )
    vec_env = MarginTraderVecEnv(
        config=config,
        params=params,
        num_envs=1,
        gpu_id=-1,
        impact_model=build_tensor_impact_model(
            impact_model_cls(),
            num_envs=1,
            stock_dim=2,
            gpu_id=-1,
        ),
        initial_capital=1e6,
        auto_reset=False,
    )
    scalar_env.reset()
    vec_env.reset()

    for action in (
        np.array([0.5, -0.25], dtype=np.float32),
        np.array([-1.0, 0.5], dtype=np.float32),
    ):
        _, scalar_reward, scalar_done, scalar_truncated, scalar_info = scalar_env.step(
            action
        )
        _, vec_reward, vec_done, vec_truncated, vec_info = vec_env.step(
            th.from_numpy(action).unsqueeze(0)
        )

        assert _scalarize_info_value(vec_reward[0]) == pytest.approx(
            float(scalar_reward),
            rel=1e-4,
            abs=1e-3,
        )
        assert _scalarize_info_value(vec_info["turnover"][0]) == pytest.approx(
            float(scalar_info["turnover"]),
            rel=1e-5,
            abs=1e-8,
        )
        for key in ("cost", "total_buy_value", "total_sell_value", "cash"):
            _assert_margin_float32_close(
                _scalarize_info_value(vec_info[key][0]),
                float(scalar_info[key]),
            )
        _assert_margin_float32_close(
            float(vec_env.total_asset[0].item()),
            float(scalar_env.total_asset),
        )
        assert bool(vec_done[0].item()) is scalar_done
        assert bool(vec_truncated[0].item()) is scalar_truncated
