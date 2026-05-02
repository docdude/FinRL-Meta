"""
Single-instrument Wyckoff Range-Bar Trading Environment.

Designed for NQ futures (or any single instrument) with Wyckoff features.
Unlike StockTradingEnv (multi-stock portfolio), this env trades ONE asset
with discrete position states: short (-1), flat (0), long (+1).

Data format (NPZ):
    close_ary : (n_bars, 1) — range-bar close prices
    tech_ary  : (n_bars, n_features) — Wyckoff feature matrix

Action space: continuous [-1, 1]
    Discretized to: -1 (short), 0 (flat), +1 (long)
    Thresholds at ±0.33

State: [position, unrealized_pnl_norm, cash_norm, *tech_features]

Reward: configurable via reward_mode parameter
    "pnl"       — realized + unrealized PnL change (default)
    "log_ret"   — log return of portfolio value
    "sharpe"    — differential Sharpe (Moody & Saffell 1998)
    "sortino"   — differential Sortino
"""

import os
import numpy as np
from typing import Tuple

ARY = np.ndarray

# Transaction cost for NQ futures: ~$4.50 round-trip per contract
# At NQ ~20000, 1 point = $20, so cost ≈ 0.01125 points per side
# We use a conservative 0.5 point per side as default (covers slippage)
DEFAULT_COST_PER_TRADE = 0.5  # points per side


class WyckoffTradingEnv:
    """
    Single-instrument trading env for Wyckoff range bars.

    Parameters
    ----------
    initial_amount : float
        Starting capital in points (e.g. 1000 NQ points).
    cost_per_trade : float
        Transaction cost per side in price points.
    gamma : float
        Discount factor for terminal reward shaping.
    reward_mode : str
        One of: "pnl", "log_ret", "sharpe", "sortino"
    reward_scale : float
        Multiplier applied to raw reward.
    beg_idx : int
        Start index into the data arrays.
    end_idx : int
        End index into the data arrays.
    npz_path : str
        Path to NPZ file with close_ary and tech_ary.
    """

    def __init__(
        self,
        initial_amount: float = 1000.0,
        cost_per_trade: float = DEFAULT_COST_PER_TRADE,
        gamma: float = 0.99,
        reward_mode: str = "pnl",
        reward_scale: float = 1.0,
        beg_idx: int = 0,
        end_idx: int = 0,
        npz_path: str = None,
        **kwargs,  # absorb extra kwargs from build_env
    ):
        self.npz_path = npz_path
        self.close_ary, self.tech_ary = self._load_data(npz_path)

        # Slice to requested range
        if end_idx <= 0:
            end_idx = len(self.close_ary)
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]

        self.initial_amount = initial_amount
        self.cost_per_trade = cost_per_trade
        self.gamma = gamma
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale

        # Position tracking
        self.position = 0        # -1, 0, +1
        self.entry_price = 0.0
        self.cash = 0.0          # accumulated PnL in points
        self.day = 0
        self.rewards = []
        self.total_asset = 0.0
        self.cumulative_returns = 0.0
        self.if_random_reset = False

        # Differential Sharpe state
        self._A = 0.0  # EMA of returns
        self._B = 0.0  # EMA of squared returns
        self._eta = 0.005  # EMA decay for diff Sharpe

        # Env metadata (ElegantRL interface)
        n_features = self.tech_ary.shape[1]
        self.env_name = 'WyckoffTradingEnv-v1'
        # state = [position(1), unrealized_pnl(1), cash_norm(1), tech_features(n)]
        self.state_dim = 3 + n_features
        self.action_dim = 1
        self.if_discrete = False
        self.max_step = self.close_ary.shape[0] - 1
        self.target_return = +np.inf
        self.num_envs = 1

    def _load_data(self, npz_path: str) -> Tuple[ARY, ARY]:
        if npz_path is None or not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        close_ary = data['close_ary'].astype(np.float32)
        tech_ary = data['tech_ary'].astype(np.float32)
        # Ensure close is 1D for single instrument
        if close_ary.ndim == 2:
            close_ary = close_ary[:, 0]
        return close_ary, tech_ary

    def reset(self) -> Tuple[ARY, dict]:
        self.day = 0
        self.position = 0
        self.entry_price = 0.0
        self.cash = 0.0
        self.rewards = []
        self.total_asset = self.initial_amount
        self.cumulative_returns = 0.0
        self._A = 0.0
        self._B = 0.0
        return self.get_state(), {}

    def get_state(self) -> ARY:
        price = self.close_ary[self.day]

        # Unrealized PnL normalized by initial capital
        if self.position != 0:
            unrealized = self.position * (price - self.entry_price)
        else:
            unrealized = 0.0

        state = np.concatenate([
            np.array([
                float(self.position),
                unrealized / self.initial_amount,
                self.cash / self.initial_amount,
            ], dtype=np.float32),
            self.tech_ary[self.day],
        ])
        return state

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        # Decode action: continuous [-1, 1] → discrete {-1, 0, +1}
        if isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action[0]
        if action > 0.33:
            target_pos = 1
        elif action < -0.33:
            target_pos = -1
        else:
            target_pos = 0

        prev_price = self.close_ary[self.day]
        self.day += 1
        curr_price = self.close_ary[self.day]

        # Execute position change
        old_position = self.position
        if target_pos != old_position:
            # Close existing position
            if old_position != 0:
                pnl = old_position * (curr_price - self.entry_price)
                self.cash += pnl - self.cost_per_trade
            # Open new position
            if target_pos != 0:
                self.entry_price = curr_price
                self.cash -= self.cost_per_trade
            else:
                self.entry_price = 0.0
            self.position = target_pos

        # Current portfolio value
        unrealized = 0.0
        if self.position != 0:
            unrealized = self.position * (curr_price - self.entry_price)
        new_total = self.initial_amount + self.cash + unrealized
        prev_total = self.total_asset

        # Compute reward
        reward = self._compute_reward(new_total, prev_total, curr_price, prev_price)
        self.rewards.append(reward)
        self.total_asset = new_total

        # Terminal
        terminal = self.day == self.max_step
        if terminal:
            # Force close any open position
            if self.position != 0:
                pnl = self.position * (curr_price - self.entry_price)
                self.cash += pnl - self.cost_per_trade
                self.position = 0
                self.entry_price = 0.0
                self.total_asset = self.initial_amount + self.cash

            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = self.total_asset / self.initial_amount * 100

        return self.get_state(), float(reward * self.reward_scale), terminal, False, {}

    def _compute_reward(
        self, new_total: float, prev_total: float,
        curr_price: float, prev_price: float,
    ) -> float:
        if self.reward_mode == "pnl":
            # Normalized PnL change
            return (new_total - prev_total) / self.initial_amount

        elif self.reward_mode == "log_ret":
            if prev_total > 0 and new_total > 0:
                return np.log(new_total / prev_total)
            return 0.0

        elif self.reward_mode == "sharpe":
            # Differential Sharpe ratio (Moody & Saffell 1998)
            r = (new_total - prev_total) / max(prev_total, 1e-8)
            dA = r - self._A
            dB = r * r - self._B
            if self._B - self._A ** 2 > 1e-12:
                denom = (self._B - self._A ** 2) ** 1.5
                dsr = (self._B * dA - 0.5 * self._A * dB) / denom
            else:
                dsr = 0.0
            self._A += self._eta * dA
            self._B += self._eta * dB
            return dsr

        elif self.reward_mode == "sortino":
            r = (new_total - prev_total) / max(prev_total, 1e-8)
            dA = r - self._A
            # Only penalize downside
            down_r2 = r * r if r < 0 else 0.0
            dB = down_r2 - self._B
            if self._B > 1e-12:
                denom = self._B ** 1.5
                dsr = (self._B * dA - 0.5 * self._A * dB) / denom
            else:
                dsr = 0.0
            self._A += self._eta * dA
            self._B += self._eta * dB
            return dsr

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
