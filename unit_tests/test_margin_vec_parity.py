"""Parity check: MarginTraderVecEnv(num_envs=1) vs MarginTraderImpactEnv.

The two code paths differ in trade-ordering (single-env uses argsort over
``trade_shares``; vec iterates stocks then buys/sells), and in FP32 vs
FP64 accumulation.  Equity is checked with a looser tolerance than the
MACE parity test.
"""
from __future__ import annotations

import numpy as np
import pytest

th = pytest.importorskip("torch")

from meta.env_market_impact.envs.env_margin_trader_impact import MarginTraderImpactEnv  # noqa: E402
from meta.env_market_impact.vec import MarginTraderVecEnv  # noqa: E402


def _make_synthetic_config(n_steps: int = 10, stock_dim: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(50.0, 150.0, size=stock_dim).astype(np.float32)
    returns = rng.normal(0.0, 0.005, size=(n_steps, stock_dim)).astype(np.float32)
    price = base * np.cumprod(1.0 + returns, axis=0)
    tech = rng.normal(0.0, 1.0, size=(n_steps, stock_dim * 2)).astype(np.float32)
    volatility = np.full((n_steps, stock_dim), 0.02, dtype=np.float32)
    volume = np.full((n_steps, stock_dim), 1.0e7, dtype=np.float32)
    return {
        "date_list": [f"2020-01-{i + 1:02d}" for i in range(n_steps)],
        "price_array": price,
        "tech_array": tech,
        "volatility_array": volatility,
        "volume_array": volume,
        "tic_list": [f"S{i}" for i in range(stock_dim)],
    }


def test_margin_vec_parity_total_asset():
    n_steps = 10
    stock_dim = 3
    config = _make_synthetic_config(n_steps=n_steps, stock_dim=stock_dim, seed=7)

    common = dict(
        initial_capital=1e8,
        max_stock_pct=0.02,
        margin_rate=2.0,
        long_short_ratio=1.0,
        maintenance_margin=0.3,
        maintenance_warning=0.4,
        max_trade_volume_pct=0.1,
        lambda_1=1e-5,
        lambda_2=0.01,
        sharpe_window=5,
        margin_adjust_period=30,
    )

    single = MarginTraderImpactEnv(config=config, **common)
    vec = MarginTraderVecEnv(
        config=config, num_envs=1, gpu_id=-1, auto_reset=False, **common
    )

    single.reset()
    vec.reset()

    # Small action magnitude so cash never binds and sell/buy ordering
    # between the two envs is irrelevant.
    rng = np.random.default_rng(31)
    actions = rng.uniform(-0.25, 0.25, size=(n_steps - 1, stock_dim)).astype(np.float32)

    for t in range(n_steps - 1):
        a = actions[t]
        _, _, done_s, trunc_s, _ = single.step(a)
        _, _, done_v, _, _ = vec.step(th.from_numpy(a).unsqueeze(0))
        if bool(done_s) or bool(trunc_s):
            break
        assert not bool(done_v[0].item())

    single_asset = float(single.total_asset)
    vec_asset = float(vec.total_asset[0].item())
    rel_err = abs(vec_asset - single_asset) / max(abs(single_asset), 1.0)
    # 10 bps tolerance — trade ordering diverges across the two envs.
    assert rel_err < 1e-3, (
        f"Margin vec vs single-env total_asset diverged: "
        f"single={single_asset:.4f} vec={vec_asset:.4f} rel_err={rel_err:.2e}"
    )


if __name__ == "__main__":
    test_margin_vec_parity_total_asset()
    print("ok")
