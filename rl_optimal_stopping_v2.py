"""
RL-based Speculative Trading Agent
Optimal Stopping (Entry/Exit) under Exploratory Framework with Cox-process intensities.

Reference idea:
    - State X_t observed continuously.
    - When flat:    sample entry  jump from a Cox process with intensity lambda_in(t, X_t).
    - When in pos:  sample exit   jump from a Cox process with intensity lambda_out(t, X_t).
    - Maximize  E[ U(PnL) ] + alpha * H(policy)
    - U is CRRA / exponential; H is differential entropy of the (Bernoulli) jump distribution.

We discretize: in a step of size dt the jump probability is p = 1 - exp(-lambda*dt).
This is the standard discrete approximation to a Cox process.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

torch.manual_seed(0); np.random.seed(0)

# ------------------------------------------------------------------ #
# 1. Data
# ------------------------------------------------------------------ #
def get_data(ticker="^NDX", start="2018-01-01", end="2025-01-01"):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df[["Close"]].rename(columns={"Close": "px"}).dropna()
    df["ret"] = np.log(df["px"]).diff()
    df["ret_5"]   = df["ret"].rolling(5).sum()
    df["ret_20"]  = df["ret"].rolling(20).sum()
    df["vol_20"]  = df["ret"].rolling(20).std()
    df["mom_60"]  = df["ret"].rolling(60).sum()
    df["zscore"]  = (df["px"] - df["px"].rolling(60).mean()) / df["px"].rolling(60).std()
    return df.dropna()

# ------------------------------------------------------------------ #
# 2. Environment
# ------------------------------------------------------------------ #
FEATURES = ["ret", "ret_5", "ret_20", "vol_20", "mom_60", "zscore"]

class TradingEnv:
    """
    Episode = one walk through the price series.  At each bar we are either
    FLAT (decide whether to enter long) or IN POSITION (decide whether to exit).
    A trade pair (tau, sigma) yields PnL = log(P_sigma) - log(P_tau)  (continuous return).
    """
    def __init__(self, df, episode_len=252, max_hold=60, cost_bps=1.0):
        self.df = df.reset_index(drop=True)
        self.feat = df[FEATURES].values.astype(np.float32)
        # standardize features
        self.feat = (self.feat - self.feat.mean(0)) / (self.feat.std(0) + 1e-8)
        self.px = df["px"].values.astype(np.float32)
        self.episode_len = episode_len
        self.max_hold = max_hold
        self.cost = cost_bps * 1e-4

    def reset(self, start=None):
        if start is None:
            start = np.random.randint(0, len(self.df) - self.episode_len - 1)
        self.t0 = start
        self.t  = start
        self.end = start + self.episode_len
        self.in_pos = False
        self.entry_t = None
        self.entry_px = None
        self.hold = 0
        return self._obs()

    def _obs(self):
        s = self.feat[self.t]
        flag = np.array([1.0 if self.in_pos else 0.0,
                         self.hold / self.max_hold], dtype=np.float32)
        return np.concatenate([s, flag])

    @property
    def obs_dim(self):
        return len(FEATURES) + 2

    def step(self, action):
        """
        action = 0 (do nothing) or 1 (jump: enter if flat, exit if in pos).
        Returns: next_obs, reward (only on exit), done
        """
        reward = 0.0
        info = {}
        if action == 1 and not self.in_pos:
            self.in_pos   = True
            self.entry_t  = self.t
            self.entry_px = self.px[self.t]
            self.hold     = 0
        elif action == 1 and self.in_pos:
            exit_px = self.px[self.t]
            reward = float(np.log(exit_px / self.entry_px) - 2 * self.cost)
            info = dict(entry=self.entry_t, exit=self.t,
                        pnl=reward, hold=self.hold)
            self.in_pos = False
            self.entry_t = None; self.hold = 0

        self.t += 1
        if self.in_pos:
            self.hold += 1
            # forced exit on max_hold
            if self.hold >= self.max_hold:
                exit_px = self.px[self.t]
                reward = float(np.log(exit_px / self.entry_px) - 2 * self.cost)
                info = dict(entry=self.entry_t, exit=self.t,
                            pnl=reward, hold=self.hold, forced=True)
                self.in_pos = False
                self.entry_t = None; self.hold = 0

        done = self.t >= self.end - 1
        if done and self.in_pos:           # close at episode end
            exit_px = self.px[self.t]
            reward = float(np.log(exit_px / self.entry_px) - 2 * self.cost)
            info = dict(entry=self.entry_t, exit=self.t,
                        pnl=reward, hold=self.hold, forced=True)
            self.in_pos = False
        return self._obs(), reward, done, info

# ------------------------------------------------------------------ #
# 3. Policy (Cox-intensity network)
# ------------------------------------------------------------------ #
class IntensityPolicy(nn.Module):
    """
    Outputs two log-intensities (entry, exit).  The relevant one is selected
    by the position flag.  The discrete jump probability per dt=1 step is
        p = 1 - exp(-lambda).
    """
    def __init__(self, obs_dim, hidden=64, lam_max=1.5):
        super().__init__()
        self.lam_max = lam_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 2))                 # (log lam_in, log lam_out)

    def intensities(self, obs):
        raw = self.net(obs)
        # bound intensities in (0, lam_max) via sigmoid * lam_max
        lam = torch.sigmoid(raw) * self.lam_max
        return lam                                # (..., 2)

    def jump_prob(self, obs, in_pos):
        lam = self.intensities(obs)               # (B,2)
        idx = in_pos.long().unsqueeze(-1)         # 0=entry, 1=exit
        lam_use = lam.gather(-1, idx).squeeze(-1)
        p = 1.0 - torch.exp(-lam_use)
        return p.clamp(1e-5, 1 - 1e-5), lam_use

# ------------------------------------------------------------------ #
# 4. Utility & exploratory objective
# ------------------------------------------------------------------ #
def crra_utility(pnl, gamma=2.0):
    # use exponential utility (numerically friendlier for log-returns):  U(x) = (1 - exp(-gamma x))/gamma
    return (1.0 - torch.exp(-gamma * pnl)) / gamma

# ------------------------------------------------------------------ #
# 5. Training loop (REINFORCE + entropy regularization)
# ------------------------------------------------------------------ #
def train(env, policy, episodes=400, alpha=0.01, gamma_u=2.0, lr=3e-4):
    opt = optim.Adam(policy.parameters(), lr=lr)
    history = []
    for ep in range(episodes):
        obs = env.reset()
        log_probs, entropies, rewards_assigned = [], [], []
        ep_actions = []
        ep_pnls = []
        traj_rewards = []          # per step utility reward (mostly 0)
        done = False
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            in_pos = torch.tensor([1.0 if env.in_pos else 0.0])
            p, lam = policy.jump_prob(obs_t, in_pos)
            dist   = torch.distributions.Bernoulli(probs=p)
            a      = dist.sample()
            logp   = dist.log_prob(a).squeeze()
            ent    = dist.entropy().squeeze()
            action = int(a.item())

            obs, r, done, info = env.step(action)
            # reward is non-zero only on exit; convert PnL -> utility
            if r != 0.0:
                u = crra_utility(torch.tensor(r), gamma_u).item()
                ep_pnls.append(r)
            else:
                u = 0.0
            log_probs.append(logp)
            entropies.append(ent)
            traj_rewards.append(u)

        # Monte-Carlo return-to-go for each step
        R = 0.0
        returns = []
        for u in reversed(traj_rewards):
            R = u + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        loss = -(log_probs * returns).mean() - alpha * entropies.mean()

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        total_pnl = float(np.sum(ep_pnls)) if ep_pnls else 0.0
        history.append(dict(ep=ep, pnl=total_pnl, n_trades=len(ep_pnls),
                            ent=float(entropies.mean().item()),
                            loss=float(loss.item())))
        if (ep + 1) % 25 == 0:
            last = history[-25:]
            print(f"[ep {ep+1:4d}] avg_pnl={np.mean([h['pnl'] for h in last]):+.4f} "
                  f"avg_trades={np.mean([h['n_trades'] for h in last]):.1f} "
                  f"avg_ent={np.mean([h['ent'] for h in last]):.3f}")
    return history

# ------------------------------------------------------------------ #
# 6. Evaluation (deterministic walk-forward over a date range)
# ------------------------------------------------------------------ #
def evaluate_agent(env, policy, start_idx, end_idx, stochastic=False):
    """Run policy over [start_idx, end_idx) sequentially; return trades + equity."""
    policy.eval()
    env.t0 = start_idx
    env.t  = start_idx
    env.end = end_idx
    env.in_pos = False; env.entry_t = None; env.hold = 0
    obs = env._obs()
    trades = []
    equity = [0.0]
    pnl_running = 0.0
    while env.t < end_idx - 1:
        obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        in_pos = torch.tensor([1.0 if env.in_pos else 0.0])
        with torch.no_grad():
            p, _ = policy.jump_prob(obs_t, in_pos)
        if stochastic:
            action = int(torch.bernoulli(p).item())
        else:
            action = int((p > 0.5).item())
        obs, r, done, info = env.step(action)
        if r != 0.0:
            trades.append(info)
            pnl_running += r
        equity.append(pnl_running)
        if done: break
    policy.train()
    return trades, np.array(equity)

# ------------------------------------------------------------------ #
# 7. Threshold baselines
# ------------------------------------------------------------------ #
def threshold_baseline(df, start_idx, end_idx, mode="momentum",
                       enter_thr=0.5, exit_thr=-0.5, cost_bps=1.0, max_hold=60):
    cost = cost_bps * 1e-4
    px = df["px"].values
    sig = df["zscore"].values if mode == "meanrev" else df["mom_60"].values
    in_pos = False; entry_px = None; hold = 0
    trades = []; equity = [0.0]; pnl = 0.0
    # for momentum: enter when sig > enter_thr, exit when sig < exit_thr
    # for meanrev:  enter when sig < -enter_thr, exit when sig > exit_thr
    for t in range(start_idx, end_idx - 1):
        s = sig[t]
        if not in_pos:
            cond = (s > enter_thr) if mode == "momentum" else (s < -enter_thr)
            if cond:
                in_pos = True; entry_px = px[t]; entry_t = t; hold = 0
        else:
            hold += 1
            cond = (s < exit_thr) if mode == "momentum" else (s > exit_thr)
            if cond or hold >= max_hold:
                r = float(np.log(px[t] / entry_px) - 2 * cost)
                pnl += r
                trades.append(dict(entry=entry_t, exit=t, pnl=r, hold=hold))
                in_pos = False
        equity.append(pnl)
    return trades, np.array(equity)

# ------------------------------------------------------------------ #
# 8. Metrics
# ------------------------------------------------------------------ #
def metrics(trades, equity, name=""):
    if len(trades) == 0:
        return dict(name=name, trades=0)
    pnls = np.array([t["pnl"] for t in trades])
    holds = np.array([t["hold"] for t in trades])
    total = equity[-1]
    daily = np.diff(equity)
    sharpe = (daily.mean() / (daily.std() + 1e-9)) * np.sqrt(252) if daily.std() > 0 else 0.0
    win = (pnls > 0).mean()
    dd = (np.maximum.accumulate(equity) - equity).max()
    return dict(name=name, trades=len(trades), total_logret=total,
                avg_pnl=pnls.mean(), win_rate=win, avg_hold=holds.mean(),
                sharpe=sharpe, max_drawdown=dd)

# ------------------------------------------------------------------ #
# 9. Main
# ------------------------------------------------------------------ #
def main():
    print("Downloading NASDAQ 100 ...")
    df = get_data("^NDX", "2018-01-01", "2025-01-01")
    print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # train/test split (walk-forward style)
    split_date = "2023-01-01"
    split_idx  = df.index.get_indexer([pd.Timestamp(split_date)], method="nearest")[0]
    print(f"Train: 0..{split_idx}    Test: {split_idx}..{len(df)}")

    env = TradingEnv(df.iloc[:split_idx].copy(), episode_len=252, max_hold=60)
    policy = IntensityPolicy(env.obs_dim, hidden=64, lam_max=1.5)

    print("\n=== Training exploratory RL agent ===")
    history = train(env, policy, episodes=400, alpha=0.02, gamma_u=2.0, lr=3e-4)

    # ---- Evaluate on the test set -------------------------------------
    test_env = TradingEnv(df.copy(), episode_len=len(df), max_hold=60)
    print("\n=== Out-of-sample evaluation ===")
    rl_trades, rl_eq = evaluate_agent(test_env, policy, split_idx, len(df),
                                      stochastic=False)

    mom_trades, mom_eq = threshold_baseline(df, split_idx, len(df),
                                            mode="momentum",
                                            enter_thr=0.02, exit_thr=-0.02)
    mr_trades,  mr_eq  = threshold_baseline(df, split_idx, len(df),
                                            mode="meanrev",
                                            enter_thr=1.0,  exit_thr=0.0)

    results = [
        metrics(rl_trades,  rl_eq,  "RL (exploratory, Cox)"),
        metrics(mom_trades, mom_eq, "Momentum threshold"),
        metrics(mr_trades,  mr_eq,  "Mean-reversion threshold"),
    ]
    res_df = pd.DataFrame(results).set_index("name")
    print("\n", res_df.round(4))

    # ---- Plots --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot([h["pnl"] for h in history], alpha=0.4, label="ep PnL")
    ax.plot(pd.Series([h["pnl"] for h in history]).rolling(25).mean(),
            color="black", label="25-ep MA")
    ax.set_title("Training: episode PnL"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot([h["ent"] for h in history], color="purple")
    ax.set_title("Training: policy entropy (exploration)"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    test_dates = df.index[split_idx:split_idx + len(rl_eq)]
    ax.plot(test_dates, rl_eq,  label=f"RL  ({len(rl_trades)} trades)")
    ax.plot(df.index[split_idx:split_idx + len(mom_eq)], mom_eq,
            label=f"Momentum ({len(mom_trades)})", alpha=0.8)
    ax.plot(df.index[split_idx:split_idx + len(mr_eq)], mr_eq,
            label=f"Mean-rev ({len(mr_trades)})", alpha=0.8)
    ax.set_title("Out-of-sample equity (cumulative log-PnL)")
    ax.legend(); ax.grid(alpha=0.3)

    # Visualize entry/exit intensities across the test set
    ax = axes[1, 1]
    feats = test_env.feat[split_idx:len(df)]
    flags = np.zeros((len(feats), 2), dtype=np.float32)  # show entry intensity (flat=0)
    obs_arr = np.concatenate([feats, flags], axis=1)
    with torch.no_grad():
        lam = policy.intensities(torch.tensor(obs_arr)).numpy()
    ax.plot(df.index[split_idx:split_idx + len(lam)], lam[:, 0],
            label=r"$\lambda_{in}$ (entry)", color="tab:green")
    ax.plot(df.index[split_idx:split_idx + len(lam)], lam[:, 1],
            label=r"$\lambda_{out}$ (exit)", color="tab:red", alpha=0.7)
    ax.set_title("Learned Cox intensities (test period)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_speculative_trading.png", dpi=120)
    print("\nSaved figure -> rl_speculative_trading.png")

    res_df.to_csv("rl_speculative_results.csv")
    print("Saved metrics -> rl_speculative_results.csv")

if __name__ == "__main__":
    main()
