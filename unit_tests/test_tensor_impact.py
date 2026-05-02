"""Tests for the tensor impact models.

Covers:
- ``TensorSqrtImpactModel`` parity against :class:`SqrtImpactModel`.
- ``TensorACImpactModel`` parity against :class:`ACImpactModel`.
- The batched (vmap-equivalent) ``apply_trades_batched`` path agrees with
  looping over stocks via ``apply_trade``.
"""
from __future__ import annotations

import numpy as np
import pytest

th = pytest.importorskip("torch")

from meta.env_market_impact.envs.impact_models import ACImpactModel  # noqa: E402
from meta.env_market_impact.envs.impact_models import SqrtImpactModel  # noqa: E402
from meta.env_market_impact.vec import (  # noqa: E402
    TensorACImpactConfig,
    TensorACImpactModel,
    TensorImpactConfig,
    TensorSqrtImpactModel,
)


DEVICE = th.device("cpu")


def _check_model_parity(ref_model, tensor_model, symbols, trades, prices, vols, volumes):
    for step_idx in range(trades.shape[0]):
        for s_idx, sym in enumerate(symbols):
            ts = float(trades[step_idx, s_idx])
            if ts == 0:
                continue
            ref = ref_model.apply_trade(
                ts, float(prices[s_idx]), float(vols[s_idx]),
                float(volumes[s_idx]), sym,
            )
            t_trade = th.tensor([ts], dtype=th.float32)
            t_price = th.tensor([prices[s_idx]], dtype=th.float32)
            t_vol = th.tensor([vols[s_idx]], dtype=th.float32)
            t_volm = th.tensor([volumes[s_idx]], dtype=th.float32)
            env_idx = th.tensor([0], dtype=th.int64)
            cost_t, shift_t = tensor_model.apply_trade(
                trade_size=t_trade, price=t_price, volatility=t_vol,
                volume=t_volm, stock_index=s_idx, env_indices=env_idx,
            )
            assert np.isclose(float(cost_t[0]), ref.cost, rtol=1e-5, atol=1e-6), (
                f"cost mismatch step={step_idx} stock={sym}: "
                f"ref={ref.cost:.6f} tensor={float(cost_t[0]):.6f}"
            )
            assert np.isclose(float(shift_t[0]), ref.price_shift, rtol=1e-5, atol=1e-8), (
                f"shift mismatch step={step_idx} stock={sym}"
            )


def test_tensor_sqrt_impact_parity():
    stock_dim = 4
    symbols = [f"S{i}" for i in range(stock_dim)]
    tensor_model = TensorSqrtImpactModel(
        num_envs=1, stock_dim=stock_dim, device=DEVICE,
        config=TensorImpactConfig(),
    )
    ref_model = SqrtImpactModel()

    rng = np.random.default_rng(5)
    trades = rng.integers(-5000, 5000, size=(6, stock_dim)).astype(np.float32)
    prices = rng.uniform(50.0, 200.0, size=stock_dim).astype(np.float32)
    vols = np.full(stock_dim, 0.02, dtype=np.float32)
    volumes = np.full(stock_dim, 1e6, dtype=np.float32)

    _check_model_parity(ref_model, tensor_model, symbols, trades, prices, vols, volumes)


def test_tensor_ac_impact_parity():
    stock_dim = 3
    symbols = [f"T{i}" for i in range(stock_dim)]
    cfg = TensorACImpactConfig(alpha=1.0, beta=1.0, epsilon=5e-4)
    tensor_model = TensorACImpactModel(
        num_envs=1, stock_dim=stock_dim, device=DEVICE, config=cfg,
    )
    ref_model = ACImpactModel(
        alpha=cfg.alpha, beta=cfg.beta, epsilon=cfg.epsilon,
    )

    rng = np.random.default_rng(11)
    trades = rng.integers(-2000, 2000, size=(5, stock_dim)).astype(np.float32)
    prices = rng.uniform(30.0, 300.0, size=stock_dim).astype(np.float32)
    vols = np.full(stock_dim, 0.015, dtype=np.float32)
    volumes = np.full(stock_dim, 5e5, dtype=np.float32)

    _check_model_parity(ref_model, tensor_model, symbols, trades, prices, vols, volumes)


def test_batched_path_matches_per_stock_loop():
    """apply_trades_batched should match the per-stock apply_trade loop."""
    num_envs, stock_dim = 8, 5
    cfg = TensorImpactConfig()

    a = TensorSqrtImpactModel(num_envs=num_envs, stock_dim=stock_dim, device=DEVICE, config=cfg)
    b = TensorSqrtImpactModel(num_envs=num_envs, stock_dim=stock_dim, device=DEVICE, config=cfg)

    rng = np.random.default_rng(23)
    trades = th.from_numpy(
        rng.integers(-1000, 1000, size=(num_envs, stock_dim)).astype(np.float32)
    )
    prices = th.from_numpy(rng.uniform(50.0, 200.0, size=stock_dim).astype(np.float32))
    vols = th.full((stock_dim,), 0.02, dtype=th.float32)
    volumes = th.full((stock_dim,), 1e6, dtype=th.float32)

    # Loop path (model a)
    cost_loop = th.zeros(num_envs, stock_dim)
    shift_loop = th.zeros(num_envs, stock_dim)
    for s_idx in range(stock_dim):
        c, s = a.apply_trade(
            trade_size=trades[:, s_idx],
            price=prices[s_idx].expand(num_envs),
            volatility=vols[s_idx].expand(num_envs),
            volume=volumes[s_idx].expand(num_envs),
            stock_index=s_idx,
            env_indices=None,
        )
        cost_loop[:, s_idx] = c
        shift_loop[:, s_idx] = s

    # Batched path (model b) — broadcasts price/vol/volume to (num_envs, stock_dim)
    cost_batched, shift_batched = b.apply_trades_batched(
        trade_size=trades,
        price=prices,
        volatility=vols,
        volume=volumes,
    )

    assert th.allclose(cost_loop, cost_batched, rtol=1e-5, atol=1e-7)
    assert th.allclose(shift_loop, shift_batched, rtol=1e-5, atol=1e-7)
    assert th.allclose(a.perm_state, b.perm_state, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    test_tensor_sqrt_impact_parity()
    test_tensor_ac_impact_parity()
    test_batched_path_matches_per_stock_loop()
    print("ok")
