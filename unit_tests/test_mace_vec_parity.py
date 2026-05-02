"""Parity test: MACEVecEnv(num_envs=1) vs the single-env MACEStockTradingEnv.

Runs a short synthetic episode with a fixed action sequence in both envs and
checks that the final ``total_asset`` agrees within a small tolerance.

Float32 vs float64 drift plus order-of-ops differences in the torch vs numpy
paths make exact equality impossible; a relative tolerance of 5 bps per
episode is our correctness bar.
"""
from __future__ import annotations

import numpy as np
import pytest

th = pytest.importorskip("torch")

from meta.env_market_impact.envs.env_mace_stock_trading import (  # noqa: E402
    EnvParams,
    MACEStockTradingEnv,
)
from meta.env_market_impact.vec import MACEVecEnv  # noqa: E402


def _make_synthetic_config(n_steps: int = 20, stock_dim: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    base_price = rng.uniform(50.0, 200.0, size=stock_dim).astype(np.float32)
    returns = rng.normal(0.0, 0.01, size=(n_steps, stock_dim)).astype(np.float32)
    price = base_price * np.cumprod(1.0 + returns, axis=0)
    tech = rng.normal(0.0, 1.0, size=(n_steps, stock_dim * 2)).astype(np.float32)
    volatility = np.full((n_steps, stock_dim), 0.02, dtype=np.float32)
    volume = np.full((n_steps, stock_dim), 1.0e6, dtype=np.float32)
    adv20 = volume.copy()
    tbill = np.full(n_steps, 2.0, dtype=np.float32)
    return {
        "date_list": [f"2020-01-{i + 1:02d}" for i in range(n_steps)],
        "price_array": price,
        "tech_array": tech,
        "volatility_array": volatility,
        "volume_array": volume,
        "adv20_array": adv20,
        "tbill_rates": tbill,
        "tic_list": [f"S{i}" for i in range(stock_dim)],
    }


def _params_no_norm() -> EnvParams:
    # Disable the obs normalizer so the two envs see identical state scaling,
    # and keep other knobs at their defaults.
    p = EnvParams()
    p.use_obs_normalizer = False
    return p


def test_mace_vec_parity_final_asset():
    n_steps = 20
    stock_dim = 3
    config = _make_synthetic_config(n_steps=n_steps, stock_dim=stock_dim, seed=42)

    single = MACEStockTradingEnv(config=config, params=_params_no_norm())
    vec = MACEVecEnv(
        config=config,
        params=_params_no_norm(),
        num_envs=1,
        gpu_id=-1,
        auto_reset=False,
    )

    single.reset()
    vec.reset()

    rng = np.random.default_rng(123)
    actions = rng.uniform(-1.0, 1.0, size=(n_steps - 1, stock_dim)).astype(np.float32)

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
    assert rel_err < 5e-4, (
        f"MACE vec vs single-env total_asset diverged: "
        f"single={single_asset:.4f} vec={vec_asset:.4f} rel_err={rel_err:.2e}"
    )


if __name__ == "__main__":
    test_mace_vec_parity_final_asset()
    print("ok")
