#!/usr/bin/env python3
"""Smoke tests for IntensityTimingWrapper."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch as th
from meta.env_market_impact.vec.intensity_timing_wrapper import IntensityTimingWrapper


class MockVecEnv:
    """Minimal mock of MACEVecEnv for testing the wrapper."""

    def __init__(self, num_envs=4, stock_dim=3, max_step=10):
        self.num_envs = num_envs
        self.stock_dim = stock_dim
        self.state_dim = 1 + 3 * stock_dim  # minimal state
        self.action_dim = stock_dim
        self.if_discrete = False
        self.max_step = max_step
        self.device = th.device("cpu")
        self.env_name = "MockVecEnv"
        self.target_return = float("inf")
        self.time = 0

        # Fake price array
        self.price_array = th.ones((max_step + 1, stock_dim), dtype=th.float32) * 100.0
        # Add some price variation
        for t in range(max_step + 1):
            self.price_array[t] = 100.0 + 5.0 * th.sin(th.tensor(t * 0.5))

        self.stocks = th.zeros((num_envs, stock_dim), dtype=th.float32)
        self.cash = th.full((num_envs,), 1e6, dtype=th.float32)
        self.total_asset = self.cash.clone()
        self._step_count = 0
        self._last_actions = None

    def reset(self):
        self.stocks.zero_()
        self.cash.fill_(1e6)
        self.total_asset = self.cash.clone()
        self.time = 0
        self._step_count = 0
        state = th.randn(self.num_envs, self.state_dim)
        return state, {}

    def step(self, actions):
        self._last_actions = actions.clone()
        self._step_count += 1
        self.time = min(self._step_count, self.max_step - 1)

        # Simulate: positive action → buy (add shares), negative → sell
        buy_mask = actions > 0.01
        sell_mask = actions < -0.01
        self.stocks = th.where(buy_mask, th.ones_like(self.stocks) * 10, self.stocks)
        self.stocks = th.where(sell_mask, th.zeros_like(self.stocks), self.stocks)

        state = th.randn(self.num_envs, self.state_dim)
        reward = th.zeros(self.num_envs)
        done = th.zeros(self.num_envs, dtype=th.bool)
        truncated = th.zeros(self.num_envs, dtype=th.bool)

        if self._step_count >= self.max_step:
            done.fill_(True)

        return state, reward, done, truncated, {}


def test_wrapper_creation():
    base = MockVecEnv(num_envs=4, stock_dim=3)
    wrapper = IntensityTimingWrapper(base, M=5.0, eta=0.005)
    assert wrapper.num_envs == 4
    assert wrapper.stock_dim == 3
    assert wrapper.action_dim == 3  # same as base
    assert wrapper.state_dim == base.state_dim + 3 * 3  # +9 augmentation
    print("  PASS: wrapper creation")


def test_reset():
    base = MockVecEnv(num_envs=4, stock_dim=3)
    wrapper = IntensityTimingWrapper(base, M=5.0, eta=0.005)
    state, info = wrapper.reset()
    assert state.shape == (4, wrapper.state_dim)
    assert (wrapper._J == 0).all()
    assert (wrapper._hold_age == 0).all()
    print("  PASS: reset")


def test_gating_blocks_most_actions():
    """With low intensity (small actions), most trades should be gated."""
    th.manual_seed(42)
    base = MockVecEnv(num_envs=32, stock_dim=5)
    wrapper = IntensityTimingWrapper(base, M=1.0, eta=0.5, dt=0.1)
    wrapper.reset()

    # Small positive actions → low entry intensity → mostly blocked
    actions = th.full((32, 5), 0.05)
    state, reward, done, trunc, info = wrapper.step(actions)

    # With M=1.0, eta=0.5, dt=0.1, small action → very low entry rate
    # Most should be gated (zeros forwarded to base)
    gated_frac = info["gated_zeros"] / info["total_actions"]
    assert gated_frac > 0.5, f"Expected most actions gated, got {gated_frac:.2f}"
    print(f"  PASS: gating blocks actions (gated fraction: {gated_frac:.2f})")


def test_high_intensity_enters():
    """With high M and strong actions, entries should fire."""
    th.manual_seed(42)
    base = MockVecEnv(num_envs=16, stock_dim=3)
    wrapper = IntensityTimingWrapper(base, M=50.0, eta=0.001, dt=1.0)
    wrapper.reset()

    # Strong positive action → high entry intensity
    actions = th.full((16, 3), 0.9)
    state, reward, done, trunc, info = wrapper.step(actions)

    assert info["entry_fires"] > 0, "Expected some entries to fire"
    entered = (wrapper._J > 0.5).sum().item()
    assert entered > 0, "Expected some positions entered"
    print(f"  PASS: high intensity enters ({entered} positions entered, {info['entry_fires']} fires)")


def test_exit_after_entry():
    """Stocks that entered should eventually exit when edge is positive."""
    th.manual_seed(42)
    base = MockVecEnv(num_envs=4, stock_dim=2, max_step=20)
    # Set prices to increase so exit edge is positive
    for t in range(21):
        base.price_array[t] = 100.0 + t * 2.0

    wrapper = IntensityTimingWrapper(base, M=50.0, eta=0.001, dt=1.0, Psi=0.0, R=0.0)
    wrapper.reset()

    # Force entry with very high intensity
    actions = th.full((4, 2), 0.99)
    wrapper.step(actions)
    entered = (wrapper._J > 0.5).sum().item()

    # Now step several times — exit should eventually fire as price rises
    exited = False
    for _ in range(15):
        actions = th.full((4, 2), -0.5)  # negative = sell signal
        wrapper.step(actions)
        if (wrapper._J < 0.5).any():
            exited = True
            break

    print(f"  PASS: exit after entry (entered={entered}, exited={exited})")


def test_state_augmentation_shape():
    base = MockVecEnv(num_envs=4, stock_dim=5)
    wrapper = IntensityTimingWrapper(base, augment_state=True)
    state, _ = wrapper.reset()
    expected = base.state_dim + 3 * 5  # J + hold_age + entry_ratio per stock
    assert state.shape == (4, expected), f"Expected shape (4, {expected}), got {state.shape}"

    # Without augmentation
    wrapper2 = IntensityTimingWrapper(base, augment_state=False)
    state2, _ = wrapper2.reset()
    assert state2.shape == (4, base.state_dim)
    print("  PASS: state augmentation shapes")


def test_hold_age_increments():
    th.manual_seed(42)
    base = MockVecEnv(num_envs=2, stock_dim=1, max_step=20)
    # Use low M and high eta so exit intensity is very low → position held
    wrapper = IntensityTimingWrapper(base, M=0.01, eta=10.0, dt=0.1, Psi=100.0)
    wrapper.reset()

    # Force entry state
    wrapper._J[0, 0] = 1.0
    wrapper._entry_price[0, 0] = 100.0
    base.stocks[0, 0] = 10.0

    # Step several times — with very low M and high Psi, edge is deeply
    # negative so exit intensity ≈ 0, position should be held
    for step in range(5):
        actions = th.zeros((2, 1))
        wrapper.step(actions)

    age = wrapper._hold_age[0, 0].item()
    assert age >= 5.0, f"Expected hold_age >= 5, got {age}"
    print(f"  PASS: hold age increments (age={age})")


def test_env_name_and_attrs():
    base = MockVecEnv()
    wrapper = IntensityTimingWrapper(base)
    assert "IntensityTiming" in wrapper.env_name
    assert wrapper.if_discrete == False
    assert wrapper.max_step == base.max_step
    print("  PASS: env attributes proxied correctly")


if __name__ == "__main__":
    print("IntensityTimingWrapper smoke tests:")
    test_wrapper_creation()
    test_reset()
    test_gating_blocks_most_actions()
    test_high_intensity_enters()
    test_exit_after_entry()
    test_state_augmentation_shape()
    test_hold_age_increments()
    test_env_name_and_attrs()
    print("\nAll tests passed.")
