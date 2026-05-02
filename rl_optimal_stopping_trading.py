#!/usr/bin/env python3
"""
RL speculative trading agent with entry/exit modeled as exploratory optimal stopping.

Implements a practical discrete-time approximation of the Cox-process/intensity idea:

    flat state:      policy outputs entry intensity lambda_in(t, X_t)
    in-position:    policy outputs exit intensity lambda_out(t, X_t)

Daily event probability is:

    p(stop at t | X_t) = 1 - exp(-lambda(t, X_t) * dt)

The agent is trained with an entropy-regularized actor-critic objective using
utility-based rewards from completed trades.

Default experiment:
    - NASDAQ 100 index via Yahoo Finance: ^NDX
    - Data: 2018-2025
    - Train: 2018-2021
    - Test: 2022-2025
    - Baseline: simple threshold momentum entry/exit rules

Install:
    pip install numpy pandas yfinance torch matplotlib

Run:
    python rl_optimal_stopping_trading.py

Optional:
    python rl_optimal_stopping_trading.py --episodes 3000 --symbol ^NDX
    python rl_optimal_stopping_trading.py --symbol QQQ
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class Config:
    symbol: str = "^NDX"
    start: str = "2018-01-01"
    end: str = "2026-01-01"

    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"

    seed: int = 42
    device: str = "auto"

    episodes: int = 2000
    batch_episodes: int = 16
    episode_length: int = 252

    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 1.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 1.0

    lambda_max: float = 4.0
    dt: float = 1.0

    cost_bps: float = 2.0
    risk_aversion: float = 5.0
    utility: str = "cara"
    reward_scale: float = 100.0

    mc_runs: int = 25
    deterministic_threshold: float = 0.5

    output_dir: str = "outputs_rl_stopping"
    log_every: int = 10


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def utility_value(
    trade_return: float,
    kind: str = "cara",
    risk_aversion: float = 5.0,
) -> float:
    """
    Utility of a completed trade return.

    trade_return is a fractional return, e.g. 0.05 for +5%.
    """
    r = float(trade_return)

    if kind == "linear":
        return r

    if kind == "log":
        return math.log(max(1.0 + r, 1e-8))

    if kind == "cara":
        if abs(risk_aversion) < 1e-12:
            return r
        z = np.clip(-risk_aversion * r, -60.0, 60.0)
        return float((1.0 - np.exp(z)) / risk_aversion)

    raise ValueError(f"Unknown utility type: {kind}")


def vector_utility(
    returns: np.ndarray,
    kind: str = "cara",
    risk_aversion: float = 5.0,
) -> np.ndarray:
    return np.array(
        [utility_value(float(x), kind=kind, risk_aversion=risk_aversion) for x in returns],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def download_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Please install yfinance: pip install yfinance") from exc

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"No data downloaded for {symbol}. "
            f"Try another ticker, e.g. QQQ, or check internet/Yahoo access."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close_col = "Close" if "Close" in raw.columns else "Adj Close"
    close = raw[close_col]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    df = pd.DataFrame({"close": close.astype(float)})
    df = df.dropna()
    return df


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def make_features(price_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates price, momentum, volatility, drawdown, and regime-style features.
    All features use information available up to the current close.
    """
    df = price_df.copy()
    close = df["close"]
    log_close = np.log(close)

    df["log_ret_1"] = log_close.diff()

    for h in [2, 5, 10, 20, 60]:
        df[f"ret_{h}"] = close.pct_change(h)

    for h in [5, 10, 20, 60]:
        df[f"vol_{h}"] = df["log_ret_1"].rolling(h).std() * np.sqrt(252.0)

    for h in [20, 50, 100, 200]:
        ma = close.rolling(h).mean()
        df[f"ma_{h}_ratio"] = close / ma - 1.0

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["zscore_20"] = (close - ma20) / std20

    roll_max_60 = close.rolling(60).max()
    df["drawdown_60"] = close / roll_max_60 - 1.0

    df["vol_ratio_20_60"] = df["vol_20"] / df["vol_60"] - 1.0

    df["rsi_14"] = rsi(close, 14) / 100.0 - 0.5

    df["dow"] = df.index.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["dow"] / 5.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dow"] / 5.0)

    df = df.drop(columns=["dow"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c for c in df.columns if c != "close"]
    return df, feature_cols


@dataclass
class FeatureScaler:
    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame, feature_cols: List[str]) -> "FeatureScaler":
        mean = df[feature_cols].mean()
        std = df[feature_cols].std().replace(0.0, 1.0)
        return cls(mean=mean, std=std)

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        out.loc[:, feature_cols] = ((out[feature_cols] - self.mean) / self.std).clip(
            -10.0, 10.0
        )
        return out


def arrays_from_df(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    prices = df["close"].to_numpy(dtype=np.float64)
    features = df[feature_cols].to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(df.index)
    return prices, features, dates


# ---------------------------------------------------------------------
# Trading environment
# ---------------------------------------------------------------------


class TradingStoppingEnv:
    """
    Sparse-reward long-only speculative trading environment.

    The policy does not choose buy/sell/hold each bar directly.
    It chooses a stopping intensity. The realized stopping event is sampled
    from the Bernoulli probability induced by the Cox-process discretization.

    Observation:
        [market features, position flag, unrealized return, normalized holding time]

    Reward:
        Utility(completed trade return) * reward_scale
    """

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        episode_length: int = 252,
        cost_bps: float = 2.0,
        utility_kind: str = "cara",
        risk_aversion: float = 5.0,
        reward_scale: float = 100.0,
        seed: int = 42,
    ):
        if len(prices) != len(features):
            raise ValueError("prices and features must have equal length")
        if len(prices) < episode_length:
            raise ValueError("Not enough data for requested episode_length")

        self.prices = prices
        self.features = features
        self.dates = dates
        self.episode_length = int(episode_length)
        self.cost = float(cost_bps) / 10000.0
        self.utility_kind = utility_kind
        self.risk_aversion = float(risk_aversion)
        self.reward_scale = float(reward_scale)
        self.rng = np.random.default_rng(seed)

        self.obs_dim = features.shape[1] + 3

        self.start_idx = 0
        self.end_idx = 0
        self.t = 0

        self.position = 0
        self.entry_price = np.nan
        self.entry_t = -1
        self.trades: List[Dict] = []

    def reset(self, random_start: bool = True) -> np.ndarray:
        max_start = len(self.prices) - self.episode_length

        if random_start and max_start > 0:
            self.start_idx = int(self.rng.integers(0, max_start + 1))
        else:
            self.start_idx = 0

        self.end_idx = self.start_idx + self.episode_length - 1
        self.t = self.start_idx

        self.position = 0
        self.entry_price = np.nan
        self.entry_t = -1
        self.trades = []

        return self._obs()

    def _obs(self) -> np.ndarray:
        if self.position == 1:
            unrealized = self.prices[self.t] / self.entry_price - 1.0
            holding = (self.t - self.entry_t) / max(1.0, float(self.episode_length))
        else:
            unrealized = 0.0
            holding = 0.0

        state = np.array([float(self.position), unrealized, holding], dtype=np.float32)
        return np.concatenate([self.features[self.t], state]).astype(np.float32)

    def _close_trade(self, forced: bool = False) -> Tuple[float, Dict]:
        exit_price = float(self.prices[self.t])

        net_return = (exit_price / self.entry_price) * ((1.0 - self.cost) ** 2) - 1.0
        util = utility_value(
            net_return,
            kind=self.utility_kind,
            risk_aversion=self.risk_aversion,
        )
        reward = self.reward_scale * util

        trade = {
            "entry_index": self.entry_t,
            "exit_index": self.t,
            "entry_date": None if self.dates is None else self.dates[self.entry_t],
            "exit_date": None if self.dates is None else self.dates[self.t],
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "return": net_return,
            "holding_days": self.t - self.entry_t,
            "forced": forced,
        }
        self.trades.append(trade)

        self.position = 0
        self.entry_price = np.nan
        self.entry_t = -1

        return reward, trade

    def step(self, event: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        event = 1 means:
            if flat: enter
            if invested: exit
        event = 0 means wait.
        """
        event = int(event)
        reward = 0.0
        info: Dict = {
            "entered": False,
            "exited": False,
            "forced_exit": False,
            "trade_return": np.nan,
        }

        price = float(self.prices[self.t])

        if self.position == 0:
            if event == 1:
                self.position = 1
                self.entry_price = price
                self.entry_t = self.t
                info["entered"] = True

        else:
            if event == 1:
                r, trade = self._close_trade(forced=False)
                reward += r
                info["exited"] = True
                info["trade_return"] = trade["return"]

        done = self.t >= self.end_idx

        if done:
            if self.position == 1:
                r, trade = self._close_trade(forced=True)
                reward += r
                info["exited"] = True
                info["forced_exit"] = True
                info["trade_return"] = trade["return"]

            next_obs = np.zeros(self.obs_dim, dtype=np.float32)
            return next_obs, float(reward), True, info

        self.t += 1
        return self._obs(), float(reward), False, info


# ---------------------------------------------------------------------
# Entropy-regularized intensity actor-critic
# ---------------------------------------------------------------------


class IntensityActorCritic(nn.Module):
    """
    Neural policy.

    Outputs bounded entry and exit intensities:

        lambda_in  = lambda_max * sigmoid(raw_in)
        lambda_out = lambda_max * sigmoid(raw_out)

    Discrete-time Cox event probabilities:

        p_in  = 1 - exp(-lambda_in * dt)
        p_out = 1 - exp(-lambda_out * dt)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        lambda_max: float = 4.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.dt = float(dt)

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        self.intensity_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def intensities_and_probs(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        z = self.backbone(obs)
        raw = self.intensity_head(z)

        entry_lambda = self.lambda_max * torch.sigmoid(raw[:, 0])
        exit_lambda = self.lambda_max * torch.sigmoid(raw[:, 1])

        entry_prob = 1.0 - torch.exp(-entry_lambda * self.dt)
        exit_prob = 1.0 - torch.exp(-exit_lambda * self.dt)

        entry_prob = entry_prob.clamp(1e-6, 1.0 - 1e-6)
        exit_prob = exit_prob.clamp(1e-6, 1.0 - 1e-6)

        value = self.value_head(z).squeeze(-1)

        return entry_lambda, exit_lambda, entry_prob, exit_prob, value

    def event_distribution(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.distributions.Bernoulli, torch.Tensor, Dict[str, torch.Tensor]]:
        entry_lambda, exit_lambda, entry_prob, exit_prob, value = self.intensities_and_probs(obs)

        # Observation layout:
        # [features..., position, unrealized_return, holding_time]
        position_flag = obs[:, -3]

        chosen_prob = torch.where(position_flag > 0.5, exit_prob, entry_prob)
        chosen_lambda = torch.where(position_flag > 0.5, exit_lambda, entry_lambda)

        dist = torch.distributions.Bernoulli(probs=chosen_prob)

        stats = {
            "chosen_prob": chosen_prob,
            "chosen_lambda": chosen_lambda,
            "entry_prob": entry_prob,
            "exit_prob": exit_prob,
            "entry_lambda": entry_lambda,
            "exit_lambda": exit_lambda,
        }

        return dist, value, stats


def discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return out


def train_agent(
    train_prices: np.ndarray,
    train_features: np.ndarray,
    train_dates: pd.DatetimeIndex,
    cfg: Config,
    device: torch.device,
) -> Tuple[IntensityActorCritic, pd.DataFrame]:
    env = TradingStoppingEnv(
        prices=train_prices,
        features=train_features,
        dates=train_dates,
        episode_length=cfg.episode_length,
        cost_bps=cfg.cost_bps,
        utility_kind=cfg.utility,
        risk_aversion=cfg.risk_aversion,
        reward_scale=cfg.reward_scale,
        seed=cfg.seed,
    )

    model = IntensityActorCritic(
        obs_dim=env.obs_dim,
        hidden_dim=cfg.hidden_dim,
        lambda_max=cfg.lambda_max,
        dt=cfg.dt,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    updates = max(1, cfg.episodes // cfg.batch_episodes)
    history = []

    for update in range(1, updates + 1):
        log_probs_batch: List[torch.Tensor] = []
        entropies_batch: List[torch.Tensor] = []
        values_batch: List[torch.Tensor] = []
        returns_batch: List[float] = []

        ep_total_rewards = []
        ep_trade_counts = []

        model.train()

        for _ in range(cfg.batch_episodes):
            obs = env.reset(random_start=True)
            done = False

            step_rewards: List[float] = []
            ep_reward = 0.0
            ep_trades = 0

            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                dist, value, _ = model.event_distribution(obs_t)
                action = dist.sample()

                log_prob = dist.log_prob(action).squeeze()
                entropy = dist.entropy().squeeze()

                next_obs, reward, done, info = env.step(int(action.item()))

                log_probs_batch.append(log_prob)
                entropies_batch.append(entropy)
                values_batch.append(value.squeeze())

                step_rewards.append(float(reward))
                ep_reward += float(reward)

                if info.get("exited", False):
                    ep_trades += 1

                obs = next_obs

            returns_batch.extend(discounted_returns(step_rewards, cfg.gamma))
            ep_total_rewards.append(ep_reward)
            ep_trade_counts.append(ep_trades)

        log_probs = torch.stack(log_probs_batch)
        entropies = torch.stack(entropies_batch)
        values = torch.stack(values_batch)

        returns_t = torch.as_tensor(returns_batch, dtype=torch.float32, device=device)

        advantages = returns_t - values.detach()
        if len(advantages) > 1 and advantages.std().item() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_t)
        entropy = entropies.mean()

        loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        row = {
            "update": update,
            "episodes_seen": update * cfg.batch_episodes,
            "loss": float(loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "mean_episode_reward": float(np.mean(ep_total_rewards)),
            "mean_episode_trades": float(np.mean(ep_trade_counts)),
        }
        history.append(row)

        if update == 1 or update % cfg.log_every == 0 or update == updates:
            print(
                f"[train] update={update:04d}/{updates:04d} "
                f"reward={row['mean_episode_reward']:.4f} "
                f"trades={row['mean_episode_trades']:.2f} "
                f"entropy={row['entropy']:.4f} "
                f"loss={row['loss']:.4f}"
            )

    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------


def build_obs_for_backtest(
    features: np.ndarray,
    prices: np.ndarray,
    t: int,
    position: int,
    entry_price: float,
    entry_t: int,
) -> np.ndarray:
    if position == 1:
        unrealized = prices[t] / entry_price - 1.0
        holding = (t - entry_t) / max(1.0, float(len(prices)))
    else:
        unrealized = 0.0
        holding = 0.0

    state = np.array([float(position), unrealized, holding], dtype=np.float32)
    return np.concatenate([features[t], state]).astype(np.float32)


def backtest_policy(
    model: IntensityActorCritic,
    prices: np.ndarray,
    features: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: Config,
    device: torch.device,
    stochastic: bool = True,
    seed: int = 123,
    deterministic_threshold: Optional[float] = None,
) -> Dict:
    model.eval()

    if deterministic_threshold is None:
        deterministic_threshold = cfg.deterministic_threshold

    rng = np.random.default_rng(seed)
    cost = cfg.cost_bps / 10000.0

    n = len(prices)
    equity = np.ones(n, dtype=float)
    current_equity = 1.0

    position = 0
    entry_price = np.nan
    entry_t = -1

    trades = []
    policy_records = []

    for t in range(n - 1):
        obs = build_obs_for_backtest(
            features=features,
            prices=prices,
            t=t,
            position=position,
            entry_price=entry_price,
            entry_t=entry_t,
        )

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            entry_lam, exit_lam, entry_p, exit_p, _ = model.intensities_and_probs(obs_t)

        entry_lam_f = float(entry_lam.item())
        exit_lam_f = float(exit_lam.item())
        entry_p_f = float(entry_p.item())
        exit_p_f = float(exit_p.item())

        active_p = exit_p_f if position == 1 else entry_p_f

        if stochastic:
            event = bool(rng.random() < active_p)
        else:
            event = bool(active_p >= deterministic_threshold)

        policy_records.append(
            {
                "date": dates[t],
                "position_before": position,
                "entry_lambda": entry_lam_f,
                "exit_lambda": exit_lam_f,
                "entry_prob": entry_p_f,
                "exit_prob": exit_p_f,
                "active_prob": active_p,
                "event": event,
            }
        )

        if position == 0:
            if event:
                position = 1
                entry_price = float(prices[t])
                entry_t = t
                current_equity *= 1.0 - cost

        else:
            if event:
                exit_price = float(prices[t])
                current_equity *= 1.0 - cost

                net_ret = (exit_price / entry_price) * ((1.0 - cost) ** 2) - 1.0

                trades.append(
                    {
                        "entry_date": dates[entry_t],
                        "exit_date": dates[t],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": net_ret,
                        "holding_days": t - entry_t,
                        "forced": False,
                    }
                )

                position = 0
                entry_price = np.nan
                entry_t = -1

        equity[t] = current_equity

        if position == 1:
            current_equity *= float(prices[t + 1] / prices[t])

        equity[t + 1] = current_equity

    if position == 1:
        t = n - 1
        exit_price = float(prices[t])
        current_equity = equity[t]
        current_equity *= 1.0 - cost
        equity[t] = current_equity

        net_ret = (exit_price / entry_price) * ((1.0 - cost) ** 2) - 1.0

        trades.append(
            {
                "entry_date": dates[entry_t],
                "exit_date": dates[t],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": net_ret,
                "holding_days": t - entry_t,
                "forced": True,
            }
        )

    return {
        "dates": dates,
        "equity": equity,
        "trades": pd.DataFrame(trades),
        "policy": pd.DataFrame(policy_records),
    }


def threshold_backtest(
    df: pd.DataFrame,
    params: Dict,
    cost_bps: float = 2.0,
) -> Dict:
    """
    Simple threshold baseline.

    Entry:
        ret_20 > entry_ret
        ma_50_ratio > ma_thresh

    Exit:
        ret_10 < exit_ret
        or stop-loss
        or take-profit
        or max holding reached
    """
    cost = cost_bps / 10000.0

    prices = df["close"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(df.index)

    entry_ret = params["entry_ret"]
    exit_ret = params["exit_ret"]
    ma_thresh = params["ma_thresh"]
    stop_loss = params["stop_loss"]
    take_profit = params["take_profit"]
    max_holding = params["max_holding"]

    n = len(df)
    equity = np.ones(n, dtype=float)
    current_equity = 1.0

    position = 0
    entry_price = np.nan
    entry_t = -1
    trades = []

    for t in range(n - 1):
        if position == 1:
            pnl = prices[t] / entry_price - 1.0
            holding = t - entry_t

            exit_signal = (
                df["ret_10"].iloc[t] < exit_ret
                or pnl <= -stop_loss
                or pnl >= take_profit
                or holding >= max_holding
            )

            if exit_signal:
                current_equity *= 1.0 - cost
                exit_price = float(prices[t])
                net_ret = (exit_price / entry_price) * ((1.0 - cost) ** 2) - 1.0

                trades.append(
                    {
                        "entry_date": dates[entry_t],
                        "exit_date": dates[t],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": net_ret,
                        "holding_days": holding,
                        "forced": False,
                    }
                )

                position = 0
                entry_price = np.nan
                entry_t = -1

        else:
            enter_signal = (
                df["ret_20"].iloc[t] > entry_ret
                and df["ma_50_ratio"].iloc[t] > ma_thresh
            )

            if enter_signal:
                position = 1
                entry_price = float(prices[t])
                entry_t = t
                current_equity *= 1.0 - cost

        equity[t] = current_equity

        if position == 1:
            current_equity *= float(prices[t + 1] / prices[t])

        equity[t + 1] = current_equity

    if position == 1:
        t = n - 1
        exit_price = float(prices[t])
        current_equity = equity[t]
        current_equity *= 1.0 - cost
        equity[t] = current_equity

        net_ret = (exit_price / entry_price) * ((1.0 - cost) ** 2) - 1.0

        trades.append(
            {
                "entry_date": dates[entry_t],
                "exit_date": dates[t],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": net_ret,
                "holding_days": t - entry_t,
                "forced": True,
            }
        )

    return {
        "dates": dates,
        "equity": equity,
        "trades": pd.DataFrame(trades),
    }


def buy_and_hold_backtest(df: pd.DataFrame) -> Dict:
    prices = df["close"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(df.index)
    equity = prices / prices[0]
    return {"dates": dates, "equity": equity, "trades": pd.DataFrame()}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(np.min(dd))


def performance_metrics(
    equity: np.ndarray,
    trades: pd.DataFrame,
    dates: Optional[pd.DatetimeIndex] = None,
    utility_kind: str = "cara",
    risk_aversion: float = 5.0,
) -> Dict:
    equity = np.asarray(equity, dtype=float)
    equity = pd.Series(equity).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0).to_numpy()

    # Prepend initial capital so first-day transaction cost is reflected.
    equity_aug = np.r_[1.0, equity]

    returns = pd.Series(equity_aug).pct_change().dropna().to_numpy(dtype=float)

    if dates is not None and len(dates) > 1:
        years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1.0 / 252.0)
    else:
        years = max((len(equity) - 1) / 252.0, 1.0 / 252.0)

    ending = float(equity[-1])
    total_return = ending - 1.0
    cagr = ending ** (1.0 / years) - 1.0 if ending > 0 else -1.0

    ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252.0)) if len(returns) > 1 else np.nan
    sharpe = (
        float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252.0))
        if len(returns) > 1 and np.std(returns, ddof=1) > 1e-12
        else np.nan
    )

    mdd = max_drawdown(equity_aug)
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-12 else np.nan

    out = {
        "ending_equity": ending,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
    }

    if trades is not None and not trades.empty:
        tr = trades["return"].astype(float).to_numpy()
        util = vector_utility(tr, kind=utility_kind, risk_aversion=risk_aversion)

        out.update(
            {
                "num_trades": int(len(trades)),
                "win_rate": float(np.mean(tr > 0.0)),
                "avg_trade_return": float(np.mean(tr)),
                "median_trade_return": float(np.median(tr)),
                "std_trade_return": float(np.std(tr, ddof=1)) if len(tr) > 1 else np.nan,
                "avg_holding_days": float(trades["holding_days"].mean()),
                "mean_trade_utility": float(np.mean(util)),
            }
        )
    else:
        out.update(
            {
                "num_trades": 0,
                "win_rate": np.nan,
                "avg_trade_return": np.nan,
                "median_trade_return": np.nan,
                "std_trade_return": np.nan,
                "avg_holding_days": np.nan,
                "mean_trade_utility": np.nan,
            }
        )

    return out


def optimize_threshold_baseline(
    train_df: pd.DataFrame,
    cost_bps: float,
    utility_kind: str,
    risk_aversion: float,
) -> Tuple[Dict, Dict]:
    """
    Selects simple baseline thresholds on training data.

    This is intentionally simple and transparent, not meant to be a complex
    benchmark.
    """
    grid = {
        "entry_ret": [0.00, 0.02, 0.04, 0.06, 0.08],
        "exit_ret": [-0.04, -0.02, 0.00, 0.02],
        "ma_thresh": [-0.02, 0.00, 0.02],
        "stop_loss": [0.05, 0.08, 0.12],
        "take_profit": [0.08, 0.12, 0.20, 0.30],
        "max_holding": [20, 40, 60, 120],
    }

    best_params = None
    best_metrics = None
    best_score = -np.inf

    keys = list(grid.keys())

    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))

        result = threshold_backtest(train_df, params=params, cost_bps=cost_bps)
        metrics = performance_metrics(
            result["equity"],
            result["trades"],
            result["dates"],
            utility_kind=utility_kind,
            risk_aversion=risk_aversion,
        )

        if metrics["num_trades"] < 2:
            continue

        sharpe = metrics["sharpe"]
        if not np.isfinite(sharpe):
            continue

        # Sharpe as main selection criterion; small utility tie-breaker.
        utility_bonus = metrics["mean_trade_utility"]
        if not np.isfinite(utility_bonus):
            utility_bonus = 0.0

        score = sharpe + 0.10 * utility_bonus

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if best_params is None:
        best_params = {
            "entry_ret": 0.02,
            "exit_ret": 0.00,
            "ma_thresh": 0.00,
            "stop_loss": 0.08,
            "take_profit": 0.12,
            "max_holding": 60,
        }
        result = threshold_backtest(train_df, params=best_params, cost_bps=cost_bps)
        best_metrics = performance_metrics(
            result["equity"],
            result["trades"],
            result["dates"],
            utility_kind=utility_kind,
            risk_aversion=risk_aversion,
        )

    return best_params, best_metrics


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------


def evaluate_rl_monte_carlo(
    model: IntensityActorCritic,
    test_prices: np.ndarray,
    test_features: np.ndarray,
    test_dates: pd.DatetimeIndex,
    cfg: Config,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    first_result = None

    for i in range(cfg.mc_runs):
        result = backtest_policy(
            model=model,
            prices=test_prices,
            features=test_features,
            dates=test_dates,
            cfg=cfg,
            device=device,
            stochastic=True,
            seed=cfg.seed + 1000 + i,
        )

        metrics = performance_metrics(
            result["equity"],
            result["trades"],
            result["dates"],
            utility_kind=cfg.utility,
            risk_aversion=cfg.risk_aversion,
        )

        metrics["run"] = i
        rows.append(metrics)

        if first_result is None:
            first_result = result

    return pd.DataFrame(rows), first_result


def plot_equity_curves(curves: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping plot.")
        return

    plt.figure(figsize=(12, 6))

    for name, (dates, equity) in curves.items():
        plt.plot(dates, equity, label=name, linewidth=1.8)

    plt.title("Out-of-sample equity curves")
    plt.xlabel("Date")
    plt.ylabel("Equity, initial capital = 1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def smoke_tests(prices: np.ndarray, features: np.ndarray, dates: pd.DatetimeIndex, cfg: Config) -> None:
    assert len(prices) == len(features) == len(dates)
    assert np.isfinite(prices).all()
    assert np.isfinite(features).all()

    env_len = min(max(30, cfg.episode_length), len(prices))
    env = TradingStoppingEnv(
        prices=prices,
        features=features,
        dates=dates,
        episode_length=env_len,
        cost_bps=cfg.cost_bps,
        utility_kind=cfg.utility,
        risk_aversion=cfg.risk_aversion,
        reward_scale=cfg.reward_scale,
        seed=cfg.seed,
    )
    obs = env.reset()
    assert obs.shape[0] == features.shape[1] + 3

    for _ in range(5):
        obs, reward, done, info = env.step(0)
        assert np.isfinite(reward)
        if done:
            break


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> Config:
    d = Config()

    p = argparse.ArgumentParser()

    p.add_argument("--symbol", type=str, default=d.symbol)
    p.add_argument("--start", type=str, default=d.start)
    p.add_argument("--end", type=str, default=d.end)
    p.add_argument("--train-end", type=str, default=d.train_end)
    p.add_argument("--test-start", type=str, default=d.test_start)

    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", type=str, default=d.device)

    p.add_argument("--episodes", type=int, default=d.episodes)
    p.add_argument("--batch-episodes", type=int, default=d.batch_episodes)
    p.add_argument("--episode-length", type=int, default=d.episode_length)

    p.add_argument("--hidden-dim", type=int, default=d.hidden_dim)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--value-coef", type=float, default=d.value_coef)
    p.add_argument("--entropy-coef", type=float, default=d.entropy_coef)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)

    p.add_argument("--lambda-max", type=float, default=d.lambda_max)
    p.add_argument("--dt", type=float, default=d.dt)

    p.add_argument("--cost-bps", type=float, default=d.cost_bps)
    p.add_argument("--risk-aversion", type=float, default=d.risk_aversion)
    p.add_argument("--utility", type=str, default=d.utility, choices=["cara", "log", "linear"])
    p.add_argument("--reward-scale", type=float, default=d.reward_scale)

    p.add_argument("--mc-runs", type=int, default=d.mc_runs)
    p.add_argument("--deterministic-threshold", type=float, default=d.deterministic_threshold)

    p.add_argument("--output-dir", type=str, default=d.output_dir)
    p.add_argument("--log-every", type=int, default=d.log_every)

    args = p.parse_args()

    return Config(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        train_end=args.train_end,
        test_start=args.test_start,
        seed=args.seed,
        device=args.device,
        episodes=args.episodes,
        batch_episodes=args.batch_episodes,
        episode_length=args.episode_length,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        lambda_max=args.lambda_max,
        dt=args.dt,
        cost_bps=args.cost_bps,
        risk_aversion=args.risk_aversion,
        utility=args.utility,
        reward_scale=args.reward_scale,
        mc_runs=args.mc_runs,
        deterministic_threshold=args.deterministic_threshold,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()

    os.makedirs(cfg.output_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    print(f"Using device: {device}")
    print(f"Downloading {cfg.symbol} from {cfg.start} to {cfg.end}...")

    raw_prices = download_prices(cfg.symbol, cfg.start, cfg.end)
    data, feature_cols = make_features(raw_prices)

    data = data.loc[(data.index >= pd.Timestamp(cfg.start)) & (data.index < pd.Timestamp(cfg.end))]

    train_raw = data.loc[data.index <= pd.Timestamp(cfg.train_end)].copy()
    test_raw = data.loc[data.index >= pd.Timestamp(cfg.test_start)].copy()

    if train_raw.empty:
        raise RuntimeError("Training set is empty after feature construction.")
    if test_raw.empty:
        raise RuntimeError("Test set is empty after feature construction.")

    scaler = FeatureScaler.fit(train_raw, feature_cols)
    scaled_all = scaler.transform(data, feature_cols)

    train_scaled = scaled_all.loc[train_raw.index].copy()
    test_scaled = scaled_all.loc[test_raw.index].copy()

    train_prices, train_features, train_dates = arrays_from_df(train_scaled, feature_cols)
    test_prices, test_features, test_dates = arrays_from_df(test_scaled, feature_cols)

    if len(train_prices) < 30:
        raise RuntimeError("Too little training data after feature construction.")

    if cfg.episode_length > len(train_prices):
        warnings.warn(
            f"episode_length={cfg.episode_length} exceeds train length={len(train_prices)}. "
            f"Reducing episode_length to {len(train_prices)}."
        )
        cfg.episode_length = len(train_prices)

    smoke_tests(train_prices, train_features, train_dates, cfg)

    print(f"Train observations: {len(train_raw)} | {train_raw.index[0].date()} to {train_raw.index[-1].date()}")
    print(f"Test observations:  {len(test_raw)} | {test_raw.index[0].date()} to {test_raw.index[-1].date()}")
    print(f"Features: {len(feature_cols)}")

    pd.DataFrame({"feature": feature_cols}).to_csv(
        os.path.join(cfg.output_dir, "feature_columns.csv"),
        index=False,
    )

    # -----------------------------------------------------------------
    # Train RL intensity policy
    # -----------------------------------------------------------------
    model, train_history = train_agent(
        train_prices=train_prices,
        train_features=train_features,
        train_dates=train_dates,
        cfg=cfg,
        device=device,
    )

    train_history.to_csv(os.path.join(cfg.output_dir, "training_history.csv"), index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "feature_cols": feature_cols,
            "scaler_mean": scaler.mean.to_dict(),
            "scaler_std": scaler.std.to_dict(),
        },
        os.path.join(cfg.output_dir, "rl_intensity_actor_critic.pt"),
    )

    # -----------------------------------------------------------------
    # Baseline optimization
    # -----------------------------------------------------------------
    print("\nOptimizing simple threshold baseline on training period...")
    best_baseline_params, baseline_train_metrics = optimize_threshold_baseline(
        train_raw,
        cost_bps=cfg.cost_bps,
        utility_kind=cfg.utility,
        risk_aversion=cfg.risk_aversion,
    )

    print("Best baseline params:")
    print(best_baseline_params)

    pd.DataFrame([best_baseline_params]).to_csv(
        os.path.join(cfg.output_dir, "baseline_best_params.csv"),
        index=False,
    )
    pd.DataFrame([baseline_train_metrics]).to_csv(
        os.path.join(cfg.output_dir, "baseline_train_metrics.csv"),
        index=False,
    )

    # -----------------------------------------------------------------
    # Out-of-sample evaluation
    # -----------------------------------------------------------------
    print("\nEvaluating RL stochastic policy...")
    rl_runs_metrics, rl_first = evaluate_rl_monte_carlo(
        model=model,
        test_prices=test_prices,
        test_features=test_features,
        test_dates=test_dates,
        cfg=cfg,
        device=device,
    )

    rl_runs_metrics.to_csv(
        os.path.join(cfg.output_dir, "rl_stochastic_mc_metrics.csv"),
        index=False,
    )

    rl_first["trades"].to_csv(
        os.path.join(cfg.output_dir, "rl_stochastic_sample_trades.csv"),
        index=False,
    )
    rl_first["policy"].to_csv(
        os.path.join(cfg.output_dir, "rl_stochastic_sample_policy_probs.csv"),
        index=False,
    )

    print("Evaluating RL deterministic probability-threshold policy...")
    rl_det = backtest_policy(
        model=model,
        prices=test_prices,
        features=test_features,
        dates=test_dates,
        cfg=cfg,
        device=device,
        stochastic=False,
        deterministic_threshold=cfg.deterministic_threshold,
    )

    rl_det_metrics = performance_metrics(
        rl_det["equity"],
        rl_det["trades"],
        rl_det["dates"],
        utility_kind=cfg.utility,
        risk_aversion=cfg.risk_aversion,
    )

    rl_det["trades"].to_csv(
        os.path.join(cfg.output_dir, "rl_deterministic_trades.csv"),
        index=False,
    )
    rl_det["policy"].to_csv(
        os.path.join(cfg.output_dir, "rl_deterministic_policy_probs.csv"),
        index=False,
    )

    print("Evaluating threshold baseline...")
    baseline_test = threshold_backtest(
        test_raw,
        params=best_baseline_params,
        cost_bps=cfg.cost_bps,
    )

    baseline_test_metrics = performance_metrics(
        baseline_test["equity"],
        baseline_test["trades"],
        baseline_test["dates"],
        utility_kind=cfg.utility,
        risk_aversion=cfg.risk_aversion,
    )

    baseline_test["trades"].to_csv(
        os.path.join(cfg.output_dir, "baseline_test_trades.csv"),
        index=False,
    )

    print("Evaluating buy-and-hold...")
    buy_hold = buy_and_hold_backtest(test_raw)
    buy_hold_metrics = performance_metrics(
        buy_hold["equity"],
        buy_hold["trades"],
        buy_hold["dates"],
        utility_kind=cfg.utility,
        risk_aversion=cfg.risk_aversion,
    )

    # -----------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------
    rl_stoch_mean = rl_runs_metrics.mean(numeric_only=True).to_dict()
    rl_stoch_std = rl_runs_metrics.std(numeric_only=True).to_dict()

    rows = []

    r = dict(rl_stoch_mean)
    r["strategy"] = "RL_Cox_entropy_stochastic_mean"
    rows.append(r)

    r = dict(rl_stoch_std)
    r["strategy"] = "RL_Cox_entropy_stochastic_std"
    rows.append(r)

    r = dict(rl_det_metrics)
    r["strategy"] = "RL_Cox_entropy_deterministic"
    rows.append(r)

    r = dict(baseline_test_metrics)
    r["strategy"] = "Threshold_momentum_baseline"
    rows.append(r)

    r = dict(buy_hold_metrics)
    r["strategy"] = "Buy_and_hold"
    rows.append(r)

    metrics_table = pd.DataFrame(rows).set_index("strategy")
    metrics_table.to_csv(os.path.join(cfg.output_dir, "test_metrics_summary.csv"))

    display_cols = [
        "ending_equity",
        "total_return",
        "cagr",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "num_trades",
        "win_rate",
        "avg_trade_return",
        "avg_holding_days",
        "mean_trade_utility",
    ]

    print("\nOut-of-sample test metrics:")
    print(metrics_table[[c for c in display_cols if c in metrics_table.columns]].round(4))

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    plot_equity_curves(
        {
            "RL stochastic sample": (rl_first["dates"], rl_first["equity"]),
            "RL deterministic": (rl_det["dates"], rl_det["equity"]),
            "Threshold baseline": (baseline_test["dates"], baseline_test["equity"]),
            "Buy and hold": (buy_hold["dates"], buy_hold["equity"]),
        },
        os.path.join(cfg.output_dir, "equity_curves.png"),
    )

    print(f"\nSaved outputs to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
