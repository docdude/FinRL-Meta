#!/usr/bin/env python3
"""
RL Speculative Trading via Exploratory Optimal Stopping.

Based on: "Reinforcement Learning for Speculative Trading under Exploratory Framework"
- Entry/exit modeled as sequential optimal stopping with Cox process intensities
- Entropy-regularized policy (exploratory RL)
- CRRA utility-based rewards
- Compared against threshold baselines on NASDAQ 100 (QQQ) 2018-2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# Configuration
# ================================================================

@dataclass
class Config:
    ticker: str = "QQQ"
    start_date: str = "2018-01-01"
    end_date: str = "2025-06-01"
    train_end: str = "2022-01-01"

    return_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 21])
    vol_windows: List[int] = field(default_factory=lambda: [10, 21, 63])
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63])

    max_wait_steps: int = 63
    max_hold_steps: int = 42
    lambda_min: float = 0.01
    lambda_max: float = 5.0
    transaction_cost: float = 0.001

    gamma_risk: float = 2.0          # CRRA risk aversion
    alpha_entropy: float = 0.05      # entropy regularization weight
    hidden_dim: int = 128
    lr: float = 3e-4
    episodes_per_epoch: int = 256
    n_epochs: int = 150
    grad_clip: float = 1.0
    discount: float = 0.99

    baseline_ma_short: int = 10
    baseline_ma_long: int = 50
    baseline_stop_loss: float = -0.05
    baseline_take_profit: float = 0.10


# ================================================================
# Data Pipeline
# ================================================================

class DataPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def download(self) -> pd.DataFrame:
        df = yf.download(self.cfg.ticker, start=self.cfg.start_date,
                         end=self.cfg.end_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close"]].copy()
        df.columns = ["close"]
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        log_ret = np.log(close / close.shift(1))
        for h in self.cfg.return_horizons:
            df[f"ret_{h}d"] = close.pct_change(h)
        for w in self.cfg.vol_windows:
            df[f"rvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
        for w in self.cfg.momentum_windows:
            ma = close.rolling(w).mean()
            df[f"mom_{w}d"] = (close - ma) / ma
        sv, lv = f"rvol_{self.cfg.vol_windows[0]}d", f"rvol_{self.cfg.vol_windows[-1]}d"
        df["vol_regime"] = df[sv] / df[lv].replace(0, np.nan)
        df.dropna(inplace=True)
        return df

    def feature_cols(self) -> List[str]:
        cols = [f"ret_{h}d" for h in self.cfg.return_horizons]
        cols += [f"rvol_{w}d" for w in self.cfg.vol_windows]
        cols += [f"mom_{w}d" for w in self.cfg.momentum_windows]
        cols.append("vol_regime")
        return cols

    def split(self, df):
        return df[df.index < self.cfg.train_end].copy(), df[df.index >= self.cfg.train_end].copy()

    def normalize(self, train, test, cols):
        stats = {}
        for c in cols:
            mu, sig = train[c].mean(), train[c].std() + 1e-8
            stats[c] = (mu, sig)
            train[c] = (train[c] - mu) / sig
            test[c] = (test[c] - mu) / sig
        return train, test, stats


# ================================================================
# Trading Environment — Sequential Optimal Stopping
# ================================================================

class TradingEnv:
    """
    Phase 0: flat — agent controls entry intensity λ_in(t, X_t)
    Phase 1: in position — agent controls exit intensity λ_out(t, X_t)
    Phase 2: done
    """

    def __init__(self, prices: np.ndarray, features: np.ndarray, cfg: Config):
        self.prices = prices
        self.features = features
        self.cfg = cfg
        self.n = len(prices)

    def reset(self, start: Optional[int] = None):
        if start is None:
            hi = self.n - self.cfg.max_wait_steps - self.cfg.max_hold_steps - 2
            start = np.random.randint(0, max(1, hi))
        self.idx = self.start = start
        self.phase = 0
        self.entry_idx = self.entry_price = None
        self.steps_in_phase = 0
        return self._state()

    def _state(self):
        feat = self.features[min(self.idx, self.n - 1)]
        pos_flag = float(self.phase == 1)
        cap = self.cfg.max_hold_steps if self.phase == 1 else self.cfg.max_wait_steps
        t_norm = self.steps_in_phase / cap
        return np.concatenate([feat, [pos_flag, t_norm]]).astype(np.float32)

    def step(self, intensity: float):
        """Returns (next_state, reward, done, info)."""
        p_stop = 1.0 - np.exp(-intensity)
        p_stop = np.clip(p_stop, 0.0, 1.0)
        stop = np.random.random() < p_stop

        self.steps_in_phase += 1
        self.idx += 1
        done = False
        reward = 0.0
        info = {"phase": self.phase, "pnl": 0.0, "traded": False}

        if self.idx >= self.n - 1:
            done = True
            if self.phase == 1:
                pnl = self.prices[self.idx] / self.entry_price - 1.0 - self.cfg.transaction_cost
                reward = self._utility(pnl)
                info["pnl"] = pnl
                info["traded"] = True
            self.phase = 2
            return self._state(), reward, done, info

        if self.phase == 0:
            if stop:
                self.entry_idx = self.idx
                self.entry_price = self.prices[self.idx]
                self.phase = 1
                self.steps_in_phase = 0
            elif self.steps_in_phase >= self.cfg.max_wait_steps:
                done = True
                self.phase = 2
        elif self.phase == 1:
            if stop or self.steps_in_phase >= self.cfg.max_hold_steps:
                pnl = self.prices[self.idx] / self.entry_price - 1.0 - self.cfg.transaction_cost
                reward = self._utility(pnl)
                info["pnl"] = pnl
                info["traded"] = True
                done = True
                self.phase = 2

        info["phase"] = self.phase
        return self._state(), reward, done, info

    def _utility(self, pnl: float) -> float:
        g = self.cfg.gamma_risk
        w = 1.0 + pnl
        if w <= 0:
            return -100.0
        if abs(g - 1.0) < 1e-6:
            return np.log(w)
        return (w ** (1 - g) - 1) / (1 - g)


# ================================================================
# Policy Network — Beta-distributed Cox Intensities
# ================================================================

class IntensityPolicy(nn.Module):
    def __init__(self, state_dim: int, cfg: Config):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
        )
        self.entry_head = nn.Sequential(
            nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 2), nn.Softplus()
        )
        self.exit_head = nn.Sequential(
            nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 2), nn.Softplus()
        )
        self.value_head = nn.Sequential(
            nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1)
        )

    def forward(self, state: torch.Tensor, phase: int):
        h = self.backbone(state)
        params = (self.entry_head(h) if phase == 0 else self.exit_head(h)) + 1.0
        dist = torch.distributions.Beta(params[..., 0], params[..., 1])
        val = self.value_head(h).squeeze(-1)
        return dist, val

    def act(self, state_np: np.ndarray, phase: int, deterministic: bool = False):
        s = torch.FloatTensor(state_np).unsqueeze(0)
        dist, val = self.forward(s, phase)
        if deterministic:
            sample = dist.mean
        else:
            sample = dist.sample()
        lp = dist.log_prob(sample)
        ent = dist.entropy()
        lam = self.cfg.lambda_min + sample.item() * (self.cfg.lambda_max - self.cfg.lambda_min)
        return lam, lp, ent, val


# ================================================================
# Trainer — REINFORCE + Entropy Regularization
# ================================================================

class Trainer:
    def __init__(self, policy: IntensityPolicy, cfg: Config):
        self.policy = policy
        self.cfg = cfg
        self.optim = optim.Adam(policy.parameters(), lr=cfg.lr)
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=cfg.n_epochs, eta_min=1e-5)

    def _rollout(self, env: TradingEnv):
        state = env.reset()
        log_probs, entropies, values, rewards = [], [], [], []
        done = False
        while not done and env.phase < 2:
            lam, lp, ent, val = self.policy.act(state, env.phase)
            ns, r, done, info = env.step(lam)
            log_probs.append(lp)
            entropies.append(ent)
            values.append(val)
            rewards.append(r)
            state = ns
        return log_probs, entropies, values, rewards, info

    def train_epoch(self, env: TradingEnv) -> dict:
        all_pnl = []
        n_traded = 0
        total_loss = torch.tensor(0.0)
        n_ep = 0

        for _ in range(self.cfg.episodes_per_epoch):
            lps, ents, vals, rews, info = self._rollout(env)
            if not lps:
                continue
            T = len(rews)
            returns = torch.zeros(T)
            R = 0.0
            for t in reversed(range(T)):
                R = rews[t] + self.cfg.discount * R
                returns[t] = R

            lps_t = torch.stack(lps).squeeze()
            ents_t = torch.stack(ents).squeeze()
            vals_t = torch.stack(vals).squeeze()
            if lps_t.dim() == 0:
                lps_t, ents_t, vals_t = lps_t.unsqueeze(0), ents_t.unsqueeze(0), vals_t.unsqueeze(0)

            adv = returns - vals_t.detach()
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            pol_loss = -(lps_t * adv).mean()
            val_loss = ((vals_t - returns) ** 2).mean()
            ent_bonus = -ents_t.mean()
            total_loss = total_loss + pol_loss + 0.5 * val_loss + self.cfg.alpha_entropy * ent_bonus
            n_ep += 1

            if info.get("traded"):
                all_pnl.append(info["pnl"])
                n_traded += 1

        if n_ep > 0:
            loss = total_loss / n_ep
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
            self.optim.step()
        self.sched.step()

        return {
            "mean_pnl": float(np.mean(all_pnl)) if all_pnl else 0.0,
            "median_pnl": float(np.median(all_pnl)) if all_pnl else 0.0,
            "trade_rate": n_traded / max(self.cfg.episodes_per_epoch, 1),
            "n_trades": n_traded,
            "entropy": float(
                np.mean([e.item() for e in (ents if 'ents' in dir() else [torch.tensor(0.0)])])
            ) if n_ep > 0 else 0.0,
        }


# ================================================================
# Evaluation helpers
# ================================================================

def evaluate_rl(policy: IntensityPolicy, env: TradingEnv, cfg: Config,
                n_episodes: int = 1000, deterministic: bool = True) -> List[dict]:
    policy.eval()
    trades = []
    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            while not done and env.phase < 2:
                lam, _, _, _ = policy.act(state, env.phase, deterministic=deterministic)
                state, _, done, info = env.step(lam)
            if info.get("traded"):
                trades.append({
                    "entry_idx": env.entry_idx,
                    "exit_idx": env.idx,
                    "pnl": info["pnl"],
                    "holding_period": env.idx - env.entry_idx,
                })
    policy.train()
    return trades


def metrics(trades: List[dict], label: str) -> dict:
    if not trades:
        return {"label": label, "n_trades": 0}
    pnls = np.array([t["pnl"] for t in trades])
    holds = np.array([t["holding_period"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    return {
        "label": label, "n_trades": len(pnls),
        "mean_pnl": pnls.mean(), "median_pnl": np.median(pnls), "std_pnl": pnls.std(),
        "sharpe": pnls.mean() / (pnls.std() + 1e-8),
        "win_rate": (pnls > 0).mean(),
        "profit_factor": abs(wins.sum() / (losses.sum() + 1e-8)) if len(losses) else float("inf"),
        "mean_hold": holds.mean(), "total_return": pnls.sum(),
        "max_pnl": pnls.max(), "min_pnl": pnls.min(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
    }


# ================================================================
# Baseline Strategies
# ================================================================

class MABaseline:
    """MA crossover entry + stop-loss / take-profit exit."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, prices: np.ndarray) -> List[dict]:
        c = self.cfg
        ma_s = pd.Series(prices).rolling(c.baseline_ma_short).mean().values
        ma_l = pd.Series(prices).rolling(c.baseline_ma_long).mean().values
        trades, i = [], c.baseline_ma_long
        while i < len(prices):
            if ma_s[i] > ma_l[i] and ma_s[i - 1] <= ma_l[i - 1]:
                ep = prices[i]
                for j in range(i + 1, min(i + c.max_hold_steps + 1, len(prices))):
                    pnl = prices[j] / ep - 1.0
                    if pnl <= c.baseline_stop_loss or pnl >= c.baseline_take_profit or j == min(i + c.max_hold_steps, len(prices) - 1):
                        trades.append({"entry_idx": i, "exit_idx": j, "pnl": pnl - c.transaction_cost,
                                       "holding_period": j - i})
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1
        return trades


class VolBaseline:
    """Enter on low-vol + positive momentum; exit on vol spike / reversal."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, prices: np.ndarray, features: pd.DataFrame) -> List[dict]:
        c = self.cfg
        vc = f"rvol_{c.vol_windows[0]}d"
        mc = f"mom_{c.momentum_windows[-1]}d"
        if vc not in features.columns or mc not in features.columns:
            return []
        vol = features[vc].values
        mom = features[mc].values
        vol_med = np.nanmedian(vol)
        trades, i = [], 0
        while i < len(prices):
            if not (np.isnan(vol[i]) or np.isnan(mom[i])) and vol[i] < vol_med and mom[i] > 0:
                ep = prices[i]
                for j in range(i + 1, min(i + c.max_hold_steps + 1, len(prices))):
                    pnl = prices[j] / ep - 1.0
                    if (pnl <= c.baseline_stop_loss or pnl >= c.baseline_take_profit
                            or (not np.isnan(vol[j]) and vol[j] > 1.5 * vol_med)
                            or (not np.isnan(mom[j]) and mom[j] < -0.02)
                            or j == min(i + c.max_hold_steps, len(prices) - 1)):
                        trades.append({"entry_idx": i, "exit_idx": j, "pnl": pnl - c.transaction_cost,
                                       "holding_period": j - i})
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1
        return trades


# ================================================================
# Visualization
# ================================================================

def plot_results(history, results, test_prices, test_dates):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("RL Speculative Trading — Exploratory Optimal Stopping", fontsize=14, fontweight="bold")

    # -- Training PnL --
    ax = axes[0, 0]
    mpnl = [h["mean_pnl"] for h in history]
    ax.plot(mpnl, alpha=0.4, label="mean pnl")
    ax.plot(pd.Series(mpnl).rolling(10).mean(), lw=2, label="smoothed")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean PnL"); ax.set_title("Training PnL"); ax.legend(); ax.grid(alpha=0.3)

    # -- Entropy & trade rate --
    ax = axes[0, 1]
    ax.plot([h["entropy"] for h in history], color="green", alpha=0.7, label="entropy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy"); ax.set_title("Entropy & Trade Rate"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot([h["trade_rate"] for h in history], color="orange", alpha=0.7, label="trade rate")
    ax2.set_ylabel("Trade Rate", color="orange")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # -- PnL distributions --
    ax = axes[1, 0]
    for lab, res in results.items():
        if res["trades"]:
            ax.hist([t["pnl"] for t in res["trades"]], bins=40, alpha=0.4,
                    label=f'{lab} (n={len(res["trades"])})')
    ax.set_xlabel("PnL"); ax.set_ylabel("Count"); ax.set_title("OOS PnL Distributions"); ax.legend(); ax.grid(alpha=0.3)

    # -- Holding period --
    ax = axes[1, 1]
    for lab, res in results.items():
        if res["trades"]:
            ax.hist([t["holding_period"] for t in res["trades"]], bins=25, alpha=0.4, label=lab)
    ax.set_xlabel("Days"); ax.set_ylabel("Count"); ax.set_title("Holding Period Distributions"); ax.legend(); ax.grid(alpha=0.3)

    # -- Cumulative PnL --
    ax = axes[2, 0]
    for lab, res in results.items():
        if res["trades"]:
            st = sorted(res["trades"], key=lambda x: x["exit_idx"])
            ax.plot(np.cumsum([t["pnl"] for t in st]), lw=2, label=lab)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL"); ax.set_title("OOS Cumulative PnL"); ax.legend(); ax.grid(alpha=0.3)

    # -- Metrics table --
    ax = axes[2, 1]; ax.axis("off")
    rows = []
    for lab, res in results.items():
        m = res["metrics"]
        rows.append([lab, f'{m.get("n_trades",0)}', f'{m.get("mean_pnl",0):.4f}',
                      f'{m.get("sharpe",0):.3f}', f'{m.get("win_rate",0):.1%}',
                      f'{m.get("profit_factor",0):.2f}', f'{m.get("mean_hold",0):.1f}d'])
    tbl = ax.table(cellText=rows, colLabels=["Strategy","Trades","Mean PnL","Sharpe","Win%","PF","Hold"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.8)
    ax.set_title("OOS Metrics", pad=20)

    plt.tight_layout()
    plt.savefig("rl_speculative_trading_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved rl_speculative_trading_results.png")


# ================================================================
# Entry Point
# ================================================================

def main():
    cfg = Config()
    torch.manual_seed(42); np.random.seed(42)

    # ---- data ----
    print("Downloading NASDAQ-100 (QQQ) 2018-2025 …")
    pipe = DataPipeline(cfg)
    df = pipe.download()
    df = pipe.build_features(df)
    fcols = pipe.feature_cols()
    train_df, test_df = pipe.split(df)
    train_df, test_df, _ = pipe.normalize(train_df, test_df, fcols)

    print(f"Train: {train_df.index[0].date()} → {train_df.index[-1].date()}  ({len(train_df)} bars)")
    print(f"Test:  {test_df.index[0].date()} → {test_df.index[-1].date()}  ({len(test_df)} bars)")

    state_dim = len(fcols) + 2  # features + position_flag + time_normalized

    train_env = TradingEnv(train_df["close"].values, train_df[fcols].values, cfg)
    test_env  = TradingEnv(test_df["close"].values,  test_df[fcols].values, cfg)

    # ---- train RL agent ----
    print(f"\nTraining RL agent  (state_dim={state_dim}, epochs={cfg.n_epochs}) …")
    policy = IntensityPolicy(state_dim, cfg)
    trainer = Trainer(policy, cfg)
    history = []
    for ep in range(cfg.n_epochs):
        m = trainer.train_epoch(train_env)
        history.append(m)
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d} │ PnL {m['mean_pnl']:+.4f} │ "
                  f"Trade Rate {m['trade_rate']:.1%} │ Trades {m['n_trades']}")

    # ---- OOS evaluation ----
    print("\nOut-of-sample evaluation …")
    rl_trades  = evaluate_rl(policy, test_env, cfg, n_episodes=1000)
    ma_trades  = MABaseline(cfg).run(test_df["close"].values)
    vol_trades = VolBaseline(cfg).run(test_df["close"].values, test_df)

    results = {
        "RL Optimal Stopping": {"trades": rl_trades,  "metrics": metrics(rl_trades,  "RL Optimal Stopping")},
        "MA Crossover":        {"trades": ma_trades,  "metrics": metrics(ma_trades,  "MA Crossover")},
        "Vol Regime":          {"trades": vol_trades, "metrics": metrics(vol_trades, "Vol Regime")},
    }

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE  RESULTS")
    print("=" * 80)
    for lab, res in results.items():
        m = res["metrics"]
        print(f"\n{'─'*40}")
        print(f"  {lab}")
        print(f"{'─'*40}")
        if m["n_trades"] == 0:
            print("  No trades."); continue
        print(f"  Trades:         {m['n_trades']}")
        print(f"  Mean PnL:       {m['mean_pnl']:+.4f}")
        print(f"  Median PnL:     {m['median_pnl']:+.4f}")
        print(f"  Std PnL:        {m['std_pnl']:.4f}")
        print(f"  Sharpe:         {m['sharpe']:.3f}")
        print(f"  Win Rate:       {m['win_rate']:.1%}")
        print(f"  Profit Factor:  {m['profit_factor']:.2f}")
        print(f"  Avg Holding:    {m['mean_hold']:.1f} days")
        print(f"  Total Return:   {m['total_return']:+.4f}")
        print(f"  Best / Worst:   {m['max_pnl']:+.4f} / {m['min_pnl']:+.4f}")

    plot_results(history, results, test_df["close"].values, test_df.index)
    return policy, results, history


if __name__ == "__main__":
    policy, results, history = main()
