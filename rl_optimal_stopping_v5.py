#!/usr/bin/env python3
"""
Exploratory RL for Speculative Trading on NASDAQ 100  (V9)
=========================================================
Based on: Zhao, Tse & Zheng (2026), arXiv:2604.02035v1

This version keeps the practical Nasdaq data and backtest layer, but moves the
decision and training core closer to Section 4:
  1. live entry and exit use pure Gibbs mean intensities
  2. there is no mandatory minimum hold in simulation or backtest
  3. policy iteration uses one-step offline TD on simulated (P, J, B) paths
  4. the default training loop does not use target-network smoothing,
     re-anchoring, or anchor penalties
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, time

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AVGO",
    "COST","NFLX","AMD","ADBE","PEP","CSCO","INTC","CMCSA",
    "TMUS","TXN","AMGN","QCOM","INTU","ISRG","AMAT","LRCX",
    "SBUX","GILD","ADP","MDLZ","PYPL","MELI",
]


class Cfg:
    gamma=1.0; iota=1.0; R=0.0; rho=0.002
    M=5.0; varpi=0.5; k_loss=2.0; dt=0.25
    eta_start=0.05; eta_end=0.005
    Psi=0.20
    trade_process="rolling_residual"  # "rolling_residual" or "raw_price"
    entry_admissibility="crossing_gate"  # "crossing_gate" or "free"
    diag_price_ref=100.0
    diag_scale_ref=1.0
    diag_hold_age_ref=20.0
    signal_gate=-2.0          # pretrain / diagnostic entry anchor only
    exit_target_low=0.0       # sampled pretrain exit anchor lower bound
    exit_target_high=0.75     # sampled pretrain exit anchor upper bound
    exit_hard=0.75            # pretrain / diagnostic exit anchor only
    train_entry_softness=0.25
    train_entry_floor=0.005
    max_hold_pt=90            # max pretrain hold (long enough to find 1.5)
    zscore_window=60; window_len=200
    min_hold=0                # no mandatory hold in the live decision rule
    cooldown=5
    single_round_trip_eval=True
    n_sims=6; batch_size=32
    n_pretrain=0              # legacy supervised warm-start disabled by default
    n_iter=1000
    tau=0.0
    td_nstep=1; reanchor_every=0; reanchor_steps=0
    v1_anchor_weight=0.0
    diag_anchor_weight=0.0
    diag_anchor_batch=64
    lr=5e-4; hidden=64; taylor_thr=0.1
    train_end="2022-12-31"; test_start="2023-01-01"; n_eval=5

    @property
    def eta(self): return self._eta if hasattr(self,"_eta") else self.eta_start
    @eta.setter
    def eta(self, v): self._eta = v

cfg = Cfg()


# ── Utility / HJB ─────────────────────────────────────────────────────────────
def _U_t(x):
    a = torch.abs(x) + 1e-8
    return torch.where(x >= 0, a.pow(cfg.varpi), -cfg.k_loss * a.pow(cfg.varpi))

def _G_t(p, b):
    return _U_t(cfg.gamma * p - cfg.iota * b - cfg.Psi - cfg.R)

def _u_s(x):
    a = abs(x) + 1e-8
    return a**cfg.varpi if x >= 0 else -cfg.k_loss * a**cfg.varpi

def _trade_scale_t(scale):
    return torch.clamp(scale.abs(), min=1e-6)

def _trade_scale_s(scale):
    return max(abs(scale), 1e-6)

def _state_coord_t(price, scale):
    return torch.asinh(price / _trade_scale_t(scale))

def _hold_age_t(hold_age):
    denom = max(float(cfg.max_hold_pt), 1.0)
    return hold_age / denom

def _trade_edge_t(price, entry_price, entry_scale):
    denom = _trade_scale_t(entry_scale)
    return (cfg.gamma * price - cfg.iota * entry_price - cfg.Psi - cfg.R) / denom

def _trade_edge_s(price, entry_price, entry_scale):
    denom = _trade_scale_s(entry_scale)
    return (cfg.gamma * price - cfg.iota * entry_price - cfg.Psi - cfg.R) / denom

def _G_trade_t(price, entry_price, entry_scale):
    return _U_t(_trade_edge_t(price, entry_price, entry_scale))

def _v0_features_t(signal, price, scale, armed):
    return torch.stack([signal, _state_coord_t(price, scale), armed], -1)

def _v1_features_t(signal, entry_signal, price, scale, entry_price, entry_scale, hold_age=None):
    edge = _trade_edge_t(price, entry_price, entry_scale)
    if hold_age is None:
        hold_age = torch.zeros_like(signal)
    return torch.stack([signal, entry_signal, _state_coord_t(price, scale), edge, _hold_age_t(hold_age)], -1)

def _entry_active_mask_t(signal, armed=None, level=None):
    if cfg.entry_admissibility == "free":
        return torch.ones_like(signal, dtype=torch.float32)
    gate = cfg.signal_gate if level is None else level
    mask = (signal <= gate).to(torch.float32)
    if armed is not None:
        mask = mask * (armed > 0.5).to(torch.float32)
    return mask

def _entry_admissibility_step_np(current, prev, armed, level=None):
    current = np.asarray(current)
    if cfg.entry_admissibility == "free":
        ones = np.ones_like(current, dtype=bool)
        return ones, ones, ones
    gate = cfg.signal_gate if level is None else level
    prev_arr = np.full_like(current, np.inf, dtype=np.float32) if prev is None else np.asarray(prev)
    crossed = (current <= gate) & (prev_arr > gate)
    armed_now = np.asarray(armed, dtype=bool) | crossed
    admissible = armed_now & (current <= gate)
    armed_next = np.where(current > gate, False, armed_now)
    return admissible, armed_now, armed_next

def _entry_admissibility_step_s(current, prev, armed, level=None):
    if cfg.entry_admissibility == "free":
        return True, True, True
    gate = cfg.signal_gate if level is None else level
    prev_v = np.inf if prev is None else float(prev)
    crossed = current <= gate and prev_v > gate
    armed_now = bool(armed) or crossed
    admissible = armed_now and current <= gate
    armed_next = False if current > gate else armed_now
    return admissible, armed_now, armed_next

def _find_entry_crossings(path, level=None):
    path = np.asarray(path, dtype=np.float32)
    if len(path) == 0:
        return np.array([], dtype=np.int64)
    if cfg.entry_admissibility == "free":
        return np.arange(len(path), dtype=np.int64)
    gate = cfg.signal_gate if level is None else level
    prev = np.concatenate(([np.inf], path[:-1]))
    return np.flatnonzero((path <= gate) & (prev > gate)).astype(np.int64)

def _sample_regime0_start(path, limit):
    valid = max(int(limit), 1)
    crossings = _find_entry_crossings(path[:valid])
    if len(crossings):
        return int(np.random.choice(crossings))
    return int(np.random.randint(valid))

def _sample_regime1_anchor(path, limit):
    valid = max(int(limit), 1)
    crossings = _find_entry_crossings(path[:valid])
    if len(crossings) == 0:
        return None, None
    entry_idx = int(np.random.choice(crossings))
    start_hi = min(valid - 1, entry_idx + max(int(cfg.max_hold_pt), 1))
    start_hi = max(entry_idx, start_hi)
    start_idx = int(np.random.randint(entry_idx, start_hi + 1))
    return entry_idx, start_idx

def _hjb_src(delta, eta=None):
    eta = eta or cfg.eta
    z = cfg.M * delta / eta; az = torch.abs(z); sa = az + 1e-6
    exact = eta * (torch.relu(z) + torch.log1p(-torch.exp(-sa)) - torch.log(sa))
    return torch.where(az < cfg.taylor_thr, cfg.M * delta / 2., exact)

def _mean_lam(delta, eta=None):
    eta = eta or cfg.eta
    z = cfg.M * delta / eta; az = torch.abs(z)
    safe = torch.where(az > cfg.taylor_thr, z, torch.ones_like(z))
    exact = torch.clamp(cfg.M / (1 - torch.exp(-safe) + 1e-30) - cfg.M / safe, 0, cfg.M)
    taylor = torch.clamp(cfg.M / 2 + cfg.M * z / 12, 0, cfg.M)
    return torch.where(az > cfg.taylor_thr, exact, taylor)

def _ent_cost(delta, eta=None):
    eta = eta or cfg.eta
    return torch.clamp(_mean_lam(delta, eta) * delta - _hjb_src(delta, eta), min=0)

def _entry_time_weights(path, limit):
    idx = np.arange(limit, dtype=np.int64)
    if limit <= 0:
        return idx, np.array([], dtype=np.float64)
    width = max(cfg.train_entry_softness, 1e-6)
    gap = np.maximum(path[idx] - cfg.signal_gate, 0.0)
    weights = cfg.train_entry_floor + np.exp(-gap / width)
    return idx, weights / weights.sum()

def _sample_entry_time(path, limit):
    idx, probs = _entry_time_weights(path, limit)
    if len(idx) == 0:
        return None
    return int(np.random.choice(idx, p=probs))

def _sample_exit_target():
    lo = min(cfg.exit_target_low, cfg.exit_target_high)
    hi = max(cfg.exit_target_low, cfg.exit_target_high)
    if hi <= lo:
        return hi
    return float(np.random.uniform(lo, hi))

def _sample_exit_time(path, entry_time, limit):
    target = _sample_exit_target()
    exit_time = min(entry_time + cfg.max_hold_pt, limit)
    for candidate in range(entry_time + max(1, cfg.min_hold),
                           min(entry_time + cfg.max_hold_pt + 1, limit + 1)):
        if candidate > limit:
            break
        if path[candidate] > target:
            exit_time = candidate
            break
    return min(exit_time, limit)

def _entry_lam(delta, eta=None, _signal_p=None):
    eta = eta or cfg.eta
    lam = torch.clamp(_mean_lam(delta, eta), 0, cfg.M)
    if _signal_p is None:
        return lam
    return lam * _entry_active_mask_t(_signal_p)

def _exit_lam(delta, eta=None, _signal_p=None):
    eta = eta or cfg.eta
    _ = _signal_p
    return torch.clamp(_mean_lam(delta, eta), 0, cfg.M)


# ── Networks ──────────────────────────────────────────────────────────────────
class V0Net(nn.Module):
    def __init__(self):
        super().__init__(); h = cfg.hidden
        self.f = nn.Sequential(
            nn.Linear(3,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
    def forward(self, p): return self.f(p).squeeze(-1)

class V1Net(nn.Module):
    def __init__(self):
        super().__init__(); h = cfg.hidden
        self.f = nn.Sequential(
            nn.Linear(5,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, pb): return self.f(pb).squeeze(-1)


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v0 = V0Net().to(self.dev)
        self.v1 = V1Net().to(self.dev)
        self.opt = torch.optim.Adam(
            list(self.v0.parameters()) + list(self.v1.parameters()),
            lr=cfg.lr,
        )
        self.opt_v1 = torch.optim.Adam(self.v1.parameters(), lr=cfg.lr)

    def _update_target(self):
        return

    def _d1(self, p, px, scale, armed):
        return self.v1(_v1_features_t(p, p, px, scale, px, scale)) - self.v0(_v0_features_t(p, px, scale, armed))

    def _d1_t(self, p, px, scale, armed):
        return self._d1(p, px, scale, armed)

    def _d2(self, p, b, px, scale, entry_px, entry_scale, hold_age=None):
        return _G_trade_t(px, entry_px, entry_scale) - self.v1(_v1_features_t(p, b, px, scale, entry_px, entry_scale, hold_age))

    def delta1(self, p, px, scale, armed):
        return self._d1(p, px, scale, armed)

    def delta2(self, p, b, px, scale, entry_px, entry_scale, hold_age=None):
        return self._d2(p, b, px, scale, entry_px, entry_scale, hold_age)

    def diag(self, prefix=""):
        self._diag(prefix)

    def _collect_supervised(self, paths, n_pts=8):
        """
        Legacy supervised warm-start data collection.
        Three key differences from the earlier calibration path:
        - exit anchor sampled from a recovery band
        - linspace goes to xt inclusive so the exit point is supervised
        - a few post-exit states are anchored directly to G(p, b)
        """
        N, L = paths.shape[0], paths.shape[1] - 1
        idx = np.random.choice(N, min(64, N), replace=False)
        batch = paths[idx]; ps, bs, ts = [], [], []
        for i in range(len(batch)):
            signal_path = batch[i, :, 0]
            price_path = batch[i, :, 1]
            scale_path = batch[i, :, 2]
            et = _sample_entry_time(signal_path, max(1, L - cfg.min_hold))
            if et is None:
                continue
            bv = float(signal_path[et])
            entry_px = float(price_path[et])
            entry_scale = float(scale_path[et])
            xt = _sample_exit_time(signal_path, et, L)
            g_exit = _u_s(_trade_edge_s(float(price_path[xt]), entry_px, entry_scale))
            # Include xt itself so the exit state is anchored to terminal payoff.
            n = min(n_pts + 1, xt - et + 1)
            if n < 1: continue
            for t in np.linspace(et, xt, n, dtype=int):
                ps.append(float(signal_path[t])); bs.append(bv)
                ts.append(np.exp(-cfg.rho * max(0, xt - t)) * g_exit)
            # Add a few post-exit states anchored directly to G(p, b).
            for k in range(1, 4):
                tp = xt + k
                if tp > L: break
                pp = float(signal_path[tp])
                pxp = float(price_path[tp])
                ps.append(pp); bs.append(bv)
                ts.append(_u_s(_trade_edge_s(pxp, entry_px, entry_scale)))
        return ps, bs, ts

    def _collect_diag_supervised(self, paths):
        """Explicitly reinforce the diagonal entry slice V1(p, p)."""
        N, L = paths.shape[0], paths.shape[1] - 1
        idx = np.random.choice(N, min(cfg.diag_anchor_batch, N), replace=False)
        batch = paths[idx]; ps, ts = [], []
        for i in range(len(batch)):
            signal_path = batch[i, :, 0]
            price_path = batch[i, :, 1]
            scale_path = batch[i, :, 2]
            et = _sample_entry_time(signal_path, max(1, L - cfg.min_hold))
            if et is None:
                continue
            bv = float(signal_path[et])
            entry_px = float(price_path[et])
            entry_scale = float(scale_path[et])
            xt = _sample_exit_time(signal_path, et, L)
            g_exit = _u_s(_trade_edge_s(float(price_path[xt]), entry_px, entry_scale))
            ps.append(bv)
            ts.append(np.exp(-cfg.rho * max(0, xt - et)) * g_exit)
        return ps, ts

    def _diag_anchor_loss(self, paths):
        if cfg.diag_anchor_weight <= 0:
            return torch.tensor(0., device=self.dev)
        ps, ts = self._collect_diag_supervised(paths)
        if not ps:
            return torch.tensor(0., device=self.dev)
        p_t = torch.tensor(ps, dtype=torch.float32, device=self.dev)
        tgt = torch.tensor(ts, dtype=torch.float32, device=self.dev)
        px_t = torch.full_like(p_t, cfg.diag_price_ref)
        sc_t = torch.full_like(p_t, cfg.diag_scale_ref)
        pred = self.v1(_v1_features_t(p_t, p_t, px_t, sc_t, px_t, sc_t))
        return ((pred - tgt)**2).mean()

    def _v1_anchor_loss(self, paths):
        if cfg.v1_anchor_weight <= 0:
            return torch.tensor(0., device=self.dev)
        ps, bs, ts = self._collect_supervised(paths, n_pts=5)
        if not ps:
            return torch.tensor(0., device=self.dev)
        p_t = torch.tensor(ps, dtype=torch.float32, device=self.dev)
        b_t = torch.tensor(bs, dtype=torch.float32, device=self.dev)
        tgt = torch.tensor(ts, dtype=torch.float32, device=self.dev)
        px_t = torch.full_like(p_t, cfg.diag_price_ref)
        sc_t = torch.full_like(p_t, cfg.diag_scale_ref)
        pred = self.v1(_v1_features_t(p_t, b_t, px_t, sc_t, px_t, sc_t))
        return ((pred - tgt)**2).mean()

    def pretrain(self, paths):
        if cfg.n_pretrain <= 0:
            return
        opt = torch.optim.Adam(self.v1.parameters(), lr=1e-3)
        for it in range(cfg.n_pretrain):
            ps, bs, ts = self._collect_supervised(paths)
            if not ps: continue
            p_t = torch.tensor(ps, dtype=torch.float32, device=self.dev)
            b_t = torch.tensor(bs, dtype=torch.float32, device=self.dev)
            tgt = torch.tensor(ts, dtype=torch.float32, device=self.dev)
            px_t = torch.full_like(p_t, cfg.diag_price_ref)
            sc_t = torch.full_like(p_t, cfg.diag_scale_ref)
            loss = ((self.v1(_v1_features_t(p_t, b_t, px_t, sc_t, px_t, sc_t)) - tgt)**2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.v1.parameters(), 1.0); opt.step()
            if (it + 1) % 200 == 0:
                print(f"      V₁ pretrain {it+1}/{cfg.n_pretrain}  loss={loss.item():.5f}")
        self._diag("after pretrain")

    def _reanchor_v1(self, paths):
        if cfg.reanchor_steps <= 0:
            return
        for _ in range(cfg.reanchor_steps):
            ps, bs, ts = self._collect_supervised(paths, n_pts=5)
            if not ps: continue
            p_t = torch.tensor(ps, dtype=torch.float32, device=self.dev)
            b_t = torch.tensor(bs, dtype=torch.float32, device=self.dev)
            tgt = torch.tensor(ts, dtype=torch.float32, device=self.dev)
            px_t = torch.full_like(p_t, cfg.diag_price_ref)
            sc_t = torch.full_like(p_t, cfg.diag_scale_ref)
            loss = ((self.v1(_v1_features_t(p_t, b_t, px_t, sc_t, px_t, sc_t)) - tgt)**2).mean()
            self.opt_v1.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.v1.parameters(), 1.0); self.opt_v1.step()

    @torch.no_grad()
    def _sim_regime0(self, batch):
        B, L1, _ = batch.shape; L = L1 - 1; NS = B * cfg.n_sims
        base_signals = np.repeat(batch[:, :, 0], cfg.n_sims, 0)
        base_prices = np.repeat(batch[:, :, 1], cfg.n_sims, 0)
        base_scales = np.repeat(batch[:, :, 2], cfg.n_sims, 0)
        J = np.zeros(NS, dtype=np.int32)
        Armed = np.zeros(NS, dtype=bool)
        start_idx = np.zeros(NS, dtype=np.int64)
        start_prev = np.full(NS, np.inf, dtype=np.float32)
        for n in range(NS):
            path_sig = base_signals[n]
            start_idx[n] = _sample_regime0_start(path_sig, L)
            if start_idx[n] > 0:
                start_prev[n] = float(path_sig[start_idx[n] - 1])

        gather = np.clip(start_idx[:, None] + np.arange(L1, dtype=np.int64)[None, :], 0, L)
        row_ix = np.arange(NS)[:, None]
        signals = base_signals[row_ix, gather]
        prices = base_prices[row_ix, gather]
        scales = base_scales[row_ix, gather]

        R0 = []
        for l in range(L):
            if (J == 0).sum() == 0:
                break

            pl, pn = signals[:, l], signals[:, l + 1]
            pxl, pxn = prices[:, l], prices[:, l + 1]
            scl, scn = scales[:, l], scales[:, l + 1]
            J_now = J.copy()
            J_next = J_now.copy()
            Bsig_next = np.zeros(NS, dtype=np.float32)
            Bpx_next = np.zeros(NS, dtype=np.float32)
            Bscl_next = np.ones(NS, dtype=np.float32)
            Armed_next = Armed.copy()

            m0 = J_now == 0
            if m0.any():
                ix = np.where(m0)[0]
                prev_sig = start_prev[ix] if l == 0 else signals[ix, l - 1]
                _, armed_now, armed_next = _entry_admissibility_step_np(pl[ix], prev_sig, Armed[ix])
                pt = torch.tensor(pl[ix], dtype=torch.float32, device=self.dev)
                pxt = torch.tensor(pxl[ix], dtype=torch.float32, device=self.dev)
                sct = torch.tensor(scl[ix], dtype=torch.float32, device=self.dev)
                art = torch.tensor(armed_now.astype(np.float32), dtype=torch.float32, device=self.dev)
                lam = _entry_lam(self._d1(pt, pxt, sct, art), _signal_p=pt) * art
                q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
                ent = np.random.random(len(ix)) < q
                ent_ix = ix[ent]
                J_next[ent_ix] = 1
                Bsig_next[ent_ix] = pl[ent_ix]
                Bpx_next[ent_ix] = pxl[ent_ix]
                Bscl_next[ent_ix] = scl[ent_ix]
                armed_next[ent] = False
                Armed_next[ix] = armed_next
                # Record transitions for J=0 states in the decision-relevant
                # region: armed states (at/below gate) plus a thinned sample of
                # above-gate states for continuation coverage.  Recording every
                # far-above-gate state inflates V0 because the agent overfits to
                # "waiting forever eventually pays off", making Δ₁ always
                # negative and suppressing entry.
                for idx in ix:
                    loc = np.where(ix == idx)[0][0]
                    is_armed = bool(armed_now[loc])
                    # Always include armed (decision-relevant) states;
                    # thin above-gate states to ~25% to prevent V0 inflation
                    if not is_armed and np.random.random() > 0.25:
                        continue
                    next_hold_age = 1.0 if J_next[idx] == 1 else 0.0
                    R0.append(
                        (
                            float(pl[idx]),
                            float(pxl[idx]),
                            float(scl[idx]),
                            float(armed_now[loc]),
                            float(pn[idx]),
                            float(pxn[idx]),
                            float(scn[idx]),
                            float(armed_next[loc]),
                            int(J_next[idx]),
                            float(Bsig_next[idx]),
                            float(Bpx_next[idx]),
                            float(Bscl_next[idx]),
                            float(next_hold_age),
                        )
                    )

            J = J_next
            Armed = Armed_next
        return R0

    @torch.no_grad()
    def _sim_regime1(self, batch):
        """Simulate regime-1 (in-position) trajectories with actual Bernoulli
        exit draws, matching OU repro / Algorithm 1.

        Each R1 tuple now includes J_next so _loss_v1 can use the sampled
        next-state target instead of a probability-weighted mixture.
        """
        B, L1, _ = batch.shape; L = L1 - 1; NS = B * cfg.n_sims
        base_signals = np.repeat(batch[:, :, 0], cfg.n_sims, 0)
        base_prices = np.repeat(batch[:, :, 1], cfg.n_sims, 0)
        base_scales = np.repeat(batch[:, :, 2], cfg.n_sims, 0)
        Bsig = np.zeros(NS, dtype=np.float32)
        Bpx = np.zeros(NS, dtype=np.float32)
        Bscl = np.ones(NS, dtype=np.float32)
        start_idx = np.zeros(NS, dtype=np.int64)
        hold_age0 = np.zeros(NS, dtype=np.float32)
        active = np.zeros(NS, dtype=bool)
        for n in range(NS):
            path_sig = base_signals[n]
            path_px = base_prices[n]
            path_sc = base_scales[n]
            entry_idx, sim_start = _sample_regime1_anchor(path_sig, L)
            if entry_idx is None:
                continue
            active[n] = True
            start_idx[n] = sim_start
            Bsig[n] = float(path_sig[entry_idx])
            Bpx[n] = float(path_px[entry_idx])
            Bscl[n] = float(path_sc[entry_idx])
            hold_age0[n] = float(sim_start - entry_idx)

        gather = np.clip(start_idx[:, None] + np.arange(L1, dtype=np.int64)[None, :], 0, L)
        row_ix = np.arange(NS)[:, None]
        signals = base_signals[row_ix, gather]
        prices = base_prices[row_ix, gather]
        scales = base_scales[row_ix, gather]

        R1 = []
        for l in range(L):
            ix = np.where(active)[0]
            if len(ix) == 0:
                break

            pl, pn = signals[:, l], signals[:, l + 1]
            pxl, pxn = prices[:, l], prices[:, l + 1]
            scl, scn = scales[:, l], scales[:, l + 1]

            # Compute exit intensity and draw Bernoulli exits
            pt = torch.tensor(pl[ix], dtype=torch.float32, device=self.dev)
            bt = torch.tensor(Bsig[ix], dtype=torch.float32, device=self.dev)
            pxt = torch.tensor(pxl[ix], dtype=torch.float32, device=self.dev)
            sct = torch.tensor(scl[ix], dtype=torch.float32, device=self.dev)
            ept = torch.tensor(Bpx[ix], dtype=torch.float32, device=self.dev)
            est = torch.tensor(Bscl[ix], dtype=torch.float32, device=self.dev)
            hold_ages = torch.tensor(
                [hold_age0[idx] + l for idx in ix], dtype=torch.float32, device=self.dev
            )
            d2 = self._d2(pt, bt, pxt, sct, ept, est, hold_ages)
            lam = _exit_lam(d2, _signal_p=pt)
            q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
            exit_mask = np.random.random(len(ix)) < q

            for loc, idx in enumerate(ix):
                hold_age = float(hold_age0[idx] + l)
                exited = exit_mask[loc]
                j_next = 2 if exited else 1
                R1.append(
                    (
                        float(pl[idx]),
                        float(pxl[idx]),
                        float(scl[idx]),
                        float(Bsig[idx]),
                        float(Bpx[idx]),
                        float(Bscl[idx]),
                        hold_age,
                        float(pn[idx]),
                        float(pxn[idx]),
                        float(scn[idx]),
                        float(hold_age + 1.0),
                        int(j_next),
                    )
                )

            # Deactivate exited paths
            exit_ix = ix[exit_mask]
            active[exit_ix] = False

        return R1

    @torch.no_grad()
    def _sim(self, batch, regime1_batch=None):
        R0 = self._sim_regime0(batch)
        R1 = self._sim_regime1(regime1_batch if regime1_batch is not None else batch)
        return R0, R1

    def _build_nstep(self, traj):
        n = cfg.td_nstep; R0, R1 = [], []
        for pt in traj.values():
            T = len(pt)
            for t in range(T):
                p_t, j_t, b_t, hs_t, _ = pt[t]
                if j_t == 2: continue
                la = min(n, T - t - 1)
                if la < 1: continue
                bi = t + la
                for k in range(1, la + 1):
                    if t + k >= T: break
                    if pt[t+k][1] != j_t: bi = t + k; break
                steps = bi - t
                pb, jb, bb = (
                    (pt[bi][0], pt[bi][1], pt[bi][2]) if bi < T
                    else (pt[-1][4], pt[-1][1], pt[-1][2]))
                if j_t == 0:
                    R0.append((p_t, b_t, pb, int(jb), bb, steps))
                else:
                    R1.append((p_t, b_t, pb, int(jb), bb, steps, hs_t < cfg.min_hold))
        return R0, R1

    def _loss_v0(self, R0):
        if not R0: return torch.tensor(0., device=self.dev)
        disc = np.exp(-cfg.rho * cfg.dt)
        p  = torch.tensor([t[0] for t in R0], dtype=torch.float32, device=self.dev)
        px = torch.tensor([t[1] for t in R0], dtype=torch.float32, device=self.dev)
        s  = torch.tensor([t[2] for t in R0], dtype=torch.float32, device=self.dev)
        a  = torch.tensor([t[3] for t in R0], dtype=torch.float32, device=self.dev)
        pn = torch.tensor([t[4] for t in R0], dtype=torch.float32, device=self.dev)
        pxn = torch.tensor([t[5] for t in R0], dtype=torch.float32, device=self.dev)
        sn = torch.tensor([t[6] for t in R0], dtype=torch.float32, device=self.dev)
        an = torch.tensor([t[7] for t in R0], dtype=torch.float32, device=self.dev)
        jn = torch.tensor([t[8] for t in R0], dtype=torch.long, device=self.dev)
        bn = torch.tensor([t[9] for t in R0], dtype=torch.float32, device=self.dev)
        bpx = torch.tensor([t[10] for t in R0], dtype=torch.float32, device=self.dev)
        bs = torch.tensor([t[11] for t in R0], dtype=torch.float32, device=self.dev)
        hn = torch.tensor([t[12] for t in R0], dtype=torch.float32, device=self.dev)
        v0c = self.v0(_v0_features_t(p, px, s, a))
        ca = _ent_cost(self._d1(p, px, s, a)) * _entry_active_mask_t(p, a)
        with torch.no_grad():
            v0b = self.v0(_v0_features_t(pn, pxn, sn, an))
            v1b = self.v1(_v1_features_t(pn, bn, pxn, sn, bpx, bs, hn))
            vb = torch.where(jn == 0, v0b, torch.where(jn == 1, v1b, torch.zeros_like(v0b)))
        td = -ca.detach() * cfg.dt + disc * vb - v0c
        return (td**2).mean()

    def _loss_v1(self, R1):
        """TD loss for in-position regime, matching Algorithm 1 / OU repro.

        Uses the *sampled* next regime (J_next from Bernoulli draw) for the
        Bellman target, not a probability-weighted mixture:
          J_next=1 (stayed): target = V₁(p', b')
          J_next=2 (exited): target = G(p', b')
        """
        if not R1: return torch.tensor(0., device=self.dev)
        disc = np.exp(-cfg.rho * cfg.dt)
        p  = torch.tensor([t[0] for t in R1], dtype=torch.float32, device=self.dev)
        px = torch.tensor([t[1] for t in R1], dtype=torch.float32, device=self.dev)
        s  = torch.tensor([t[2] for t in R1], dtype=torch.float32, device=self.dev)
        b  = torch.tensor([t[3] for t in R1], dtype=torch.float32, device=self.dev)
        bpx = torch.tensor([t[4] for t in R1], dtype=torch.float32, device=self.dev)
        bs = torch.tensor([t[5] for t in R1], dtype=torch.float32, device=self.dev)
        h = torch.tensor([t[6] for t in R1], dtype=torch.float32, device=self.dev)
        pn = torch.tensor([t[7] for t in R1], dtype=torch.float32, device=self.dev)
        pxn = torch.tensor([t[8] for t in R1], dtype=torch.float32, device=self.dev)
        sn = torch.tensor([t[9] for t in R1], dtype=torch.float32, device=self.dev)
        hn = torch.tensor([t[10] for t in R1], dtype=torch.float32, device=self.dev)
        jn = torch.tensor([t[11] for t in R1], dtype=torch.long, device=self.dev)
        v1c = self.v1(_v1_features_t(p, b, px, s, bpx, bs, h))
        d2 = self._d2(p, b, px, s, bpx, bs, h)
        cb  = _ent_cost(d2)
        with torch.no_grad():
            v1b = self.v1(_v1_features_t(pn, b, pxn, sn, bpx, bs, hn))
            gb  = _G_trade_t(pxn, bpx, bs)
            # Paper-correct: use sampled J_next, not probability-weighted blend
            vb  = torch.where(jn == 1, v1b, gb)
        td = -cb.detach() * cfg.dt + disc * vb - v1c
        return (td**2).mean()

    def _safe_step(self, loss, opt):
        if torch.isnan(loss) or torch.isinf(loss): opt.zero_grad(); return False
        loss.backward()
        params = [p for g in opt.param_groups for p in g["params"]]
        if any(p.grad is not None and
               (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in params):
            opt.zero_grad(); return False
        nn.utils.clip_grad_norm_(params, 0.5); opt.step(); return True

    def train(self, paths, regime1_paths=None):
        print(f"    One-step offline policy iteration ({cfg.n_iter} iters) …")
        N = len(paths)
        N1 = len(regime1_paths) if regime1_paths is not None else N
        losses = []
        for it in range(cfg.n_iter):
            cfg.eta = cfg.eta_start + (cfg.eta_end - cfg.eta_start) * it / max(cfg.n_iter-1, 1)
            idx = np.random.choice(N, min(cfg.batch_size, N), replace=False)
            batch = paths[idx]
            if regime1_paths is None:
                regime1_batch = batch
            else:
                idx1 = np.random.choice(N1, min(cfg.batch_size, N1), replace=False)
                regime1_batch = regime1_paths[idx1]
            R0, R1 = self._sim(batch, regime1_batch=regime1_batch)
            if not (R0 or R1):
                continue
            n0, n1 = len(R0), len(R1)
            loss0 = self._loss_v0(R0)
            loss1 = self._loss_v1(R1)
            loss = (loss0 * n0 + loss1 * n1) / max(n0 + n1, 1)
            self.opt.zero_grad()
            self._safe_step(loss, self.opt)
            losses.append(float(loss.detach().item()))
            if (it + 1) % 250 == 0:
                avg = np.mean(losses[-250:]) if losses else 0
                print(f"      iter {it+1}/{cfg.n_iter}  loss={avg:.6f}  η={cfg.eta:.4f}")
                self._diag("  ")
        return losses

    def _diag(self, prefix=""):
        with torch.no_grad():
            ps = torch.tensor([-3.,-2.5,-2.,-1.5,-1.,0.,1.,1.5,2.], device=self.dev)
            scale_ref = torch.full_like(ps, cfg.diag_scale_ref)
            armed_ref = torch.ones_like(ps)
            trade_ref = _diag_trade_slice_t(ps, scale_ref)
            v0 = self.v0(_v0_features_t(ps, trade_ref, scale_ref, armed_ref)); d1 = self._d1(ps, trade_ref, scale_ref, armed_ref)
            qe = 1 - torch.exp(-_entry_lam(d1, _signal_p=ps) * cfg.dt)
            b2 = torch.full_like(ps, -2.)
            entry_trade = _diag_trade_slice_t(b2, scale_ref)
            hold_age_ref = torch.full_like(ps, cfg.diag_hold_age_ref)
            d2 = self._d2(ps, b2, trade_ref, scale_ref, entry_trade, scale_ref, hold_age_ref)
            qx = 1 - torch.exp(-_exit_lam(d2, _signal_p=ps) * cfg.dt)
            gv = _G_trade_t(trade_ref, entry_trade, scale_ref)
            zs = [-3.,-2.5,-2.,-1.5,-1.,0.,1.,1.5,2.]
            eq = " ".join(f"p{z:.1f}:{qe[i]:.2f}" for i,z in enumerate(zs))
            xq = " ".join(f"p{z:.1f}:{qx[i]:.2f}(G={gv[i]:.2f})" for i,z in enumerate(zs))
            print(f"    {prefix}V₀=[{v0.min():.2f},{v0.max():.2f}]  "
                  f"Δ₁=[{d1.min():.2f},{d1.max():.2f}]")
            print(f"    {prefix}Entry q: {eq}")
            print(f"    {prefix}Exit(b=-2): {xq}")

    @torch.no_grad()
    def backtest(self, signal, prices, scales, raw_prices, seed=0):
        np.random.seed(seed)
        trades, j, ei, es, ep, esc, raw_ep, cd = [], 0, 0, 0., 0., 1., 0., 0
        armed = False
        prev_signal = None
        eta_bt = cfg.eta_end
        for l in range(len(signal)):
            p = signal[l]
            px = prices[l]
            sc = scales[l]
            raw_px = raw_prices[l]
            admissible, armed_now, armed_next = _entry_admissibility_step_s(p, prev_signal, armed)
            if j == 0:
                if cd > 0:
                    cd -= 1
                else:
                    pt = torch.tensor([p], dtype=torch.float32, device=self.dev)
                    pxt = torch.tensor([px], dtype=torch.float32, device=self.dev)
                    sct = torch.tensor([sc], dtype=torch.float32, device=self.dev)
                    art = torch.tensor([float(armed_now)], dtype=torch.float32, device=self.dev)
                    lam = _entry_lam(self._d1(pt, pxt, sct, art), eta_bt, _signal_p=pt).item() * float(armed_now)
                    if admissible and np.random.random() < 1 - np.exp(-lam * cfg.dt):
                        j, ei, es, ep, esc, raw_ep = 1, l, p, px, sc, raw_px
                        armed_next = False
            elif j == 1:
                pt = torch.tensor([p], dtype=torch.float32, device=self.dev)
                bt = torch.tensor([es], dtype=torch.float32, device=self.dev)
                pxt = torch.tensor([px], dtype=torch.float32, device=self.dev)
                sct = torch.tensor([sc], dtype=torch.float32, device=self.dev)
                ept = torch.tensor([ep], dtype=torch.float32, device=self.dev)
                est = torch.tensor([esc], dtype=torch.float32, device=self.dev)
                hat = torch.tensor([float(l - ei)], dtype=torch.float32, device=self.dev)
                lam = _exit_lam(self._d2(pt, bt, pxt, sct, ept, est, hat), eta_bt, _signal_p=pt).item()
                if np.random.random() < 1 - np.exp(-lam * cfg.dt):
                    if l + 1 >= len(signal):
                        continue
                    xi = l + 1
                    exit_signal = signal[xi]
                    exit_trade_px = prices[xi]
                    exit_raw_px = raw_prices[xi]
                    edge = _trade_edge_s(exit_trade_px, ep, esc)
                    utility = _u_s(edge)
                    trades.append(dict(
                        ei=ei, xi=xi, ep=raw_ep, xp=exit_raw_px,
                        trade_ep=ep, trade_xp=exit_trade_px,
                        entry_scale=esc,
                        ret=(exit_raw_px-raw_ep)/max(abs(raw_ep), 1e-6),
                        edge=edge, utility=utility,
                        hd=xi-ei, spnl=exit_signal-es, ez=es, xz=exit_signal,
                        exit_reason="policy"))
                    if cfg.single_round_trip_eval:
                        return trades
                    j, cd = 0, cfg.cooldown
            armed_next = False if j == 1 else armed_next
            armed = armed_next
            prev_signal = p
        if j == 1:
            l = len(signal) - 1
            px = prices[l]
            raw_px = raw_prices[l]
            edge = _trade_edge_s(px, ep, esc)
            utility = _u_s(edge)
            trades.append(dict(
                ei=ei, xi=l, ep=raw_ep, xp=raw_px,
                trade_ep=ep, trade_xp=px,
                entry_scale=esc,
                ret=(raw_px-raw_ep)/max(abs(raw_ep), 1e-6),
                edge=edge, utility=utility,
                hd=l-ei, spnl=signal[l]-es, ez=es, xz=signal[l],
                exit_reason="terminal"))
        return trades


# ── Baseline ──────────────────────────────────────────────────────────────────
def _thresh_bt(signal, prices, scales, raw_prices, eth, xth):
    trades, j, ei, es, ep, esc, raw_ep = [], 0, 0, 0., 0., 1., 0.
    armed = False
    prev_signal = None
    for l in range(len(signal)):
        p = signal[l]
        px = prices[l]
        sc = scales[l]
        raw_px = raw_prices[l]
        admissible, _, armed_next = _entry_admissibility_step_s(p, prev_signal, armed, level=eth)
        if j == 0 and admissible:
            j, ei, es, ep, esc, raw_ep = 1, l, p, px, sc, raw_px
            armed_next = False
        elif j == 1 and p > xth:
            edge = _trade_edge_s(px, ep, esc)
            utility = _u_s(edge)
            trades.append(dict(ei=ei, xi=l, ep=raw_ep, xp=raw_px,
                               trade_ep=ep, trade_xp=px,
                               ret=(raw_px-raw_ep)/max(abs(raw_ep), 1e-6),
                               edge=edge, utility=utility,
                               hd=l-ei, spnl=p-es, ez=es, xz=p,
                               exit_reason="policy"))
            if cfg.single_round_trip_eval:
                return trades
            j = 0
        armed = False if j == 1 else armed_next
        prev_signal = p
    if j == 1:
        l = len(signal) - 1
        px = prices[l]
        raw_px = raw_prices[l]
        edge = _trade_edge_s(px, ep, esc)
        utility = _u_s(edge)
        trades.append(dict(ei=ei, xi=l, ep=raw_ep, xp=raw_px,
                           trade_ep=ep, trade_xp=px,
                           ret=(raw_px-raw_ep)/max(abs(raw_ep), 1e-6),
                           edge=edge, utility=utility,
                           hd=l-ei, spnl=signal[l]-es, ez=es, xz=signal[l],
                           exit_reason="terminal"))
    return trades


def _count_center_recoveries(signal, entry_gate, min_hold, exit_center):
    count = 0
    for ei in np.where(signal < entry_gate)[0]:
        lo = ei + min_hold
        if lo >= len(signal):
            continue
        if np.any(signal[lo:] > exit_center):
            count += 1
    return count

def optimize_thresholds(trn_z, trn_trade, trn_scale, trn_raw, tickers):
    best_sr, best = -1e10, (-1., .5)
    for eth in [-2.5,-2.,-1.5,-1.,-.5]:
        for xth in [-.5,0.,.5,1.,1.5]:
            utils = []
            for tk in tickers:
                tr = _thresh_bt(trn_z[tk].values.astype(np.float32),
                                trn_trade[tk].values.astype(np.float64),
                                trn_scale[tk].values.astype(np.float64),
                                trn_raw[tk].values.astype(np.float64),
                                eth, xth)
                utils.extend([t["utility"] for t in tr])
            if len(utils) > 10:
                u = np.array(utils); sr = u.mean() / (u.std() + 1e-8)
                if sr > best_sr: best_sr, best = sr, (eth, xth)
    return best


def optimize_exit_threshold(trn_z, trn_trade, trn_scale, trn_raw, tickers, entry_level):
    best_sr, best = -1e10, 0.0
    for xth in [-.5,0.,.5,1.,1.5]:
        utils = []
        for tk in tickers:
            tr = _thresh_bt(trn_z[tk].values.astype(np.float32),
                            trn_trade[tk].values.astype(np.float64),
                            trn_scale[tk].values.astype(np.float64),
                            trn_raw[tk].values.astype(np.float64),
                            entry_level, xth)
            utils.extend([t["utility"] for t in tr])
        if len(utils) > 10:
            u = np.array(utils)
            sr = u.mean() / (u.std() + 1e-8)
            if sr > best_sr:
                best_sr, best = sr, xth
    return best


def _build_trade_process(raw_prices, rolling_mean, rolling_std):
    mean = rolling_mean.abs().clip(lower=1e-6)
    std = rolling_std.abs().clip(lower=1e-6)
    if cfg.trade_process == "raw_price":
        return raw_prices.copy(), mean, "raw price / rolling-mean scale"
    if cfg.trade_process == "rolling_residual":
        return raw_prices - rolling_mean, std, "rolling residual / rolling-std scale"
    raise ValueError(f"Unsupported trade_process={cfg.trade_process!r}")


def _diag_trade_slice_t(signal, scale):
    if cfg.trade_process == "rolling_residual":
        return signal * scale
    return torch.full_like(signal, cfg.diag_price_ref)


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(trades, n_test_days=1, n_tickers=1, n_runs=1):
    if not trades:
        return dict(n=0,tot=0.,mu=0.,sr=0.,wr=0.,hd=0.,mdd=0.,
                    ann_edge=0.,inv_frac=0.,avg_ez=0.,avg_edge=0.,
                    ret_tot=0.,ret_mu=0.,ret_sr=0.)
    df = pd.DataFrame(trades)
    utility = df["utility"].values if "utility" in df.columns else df["ret"].values
    edge = df["edge"].values if "edge" in df.columns else df["ret"].values
    raw_ret = df["ret"].values
    cum = np.cumsum(utility); pk = np.maximum.accumulate(np.concatenate([[0],cum]))
    dd = pk[1:] - cum; ah = df["hd"].mean()
    util_sr = utility.mean()/(utility.std()+1e-8)*np.sqrt(252/max(ah,1)) if len(utility)>1 else 0.
    ret_sr = raw_ret.mean()/(raw_ret.std()+1e-8)*np.sqrt(252/max(ah,1)) if len(raw_ret)>1 else 0.
    tpy = (len(edge)/n_runs)/(n_test_days/252)
    # inv_frac is normalized per ticker.
    inv_frac = df["hd"].sum() / n_runs / (n_test_days * n_tickers)
    return dict(n=len(utility), tot=utility.sum(), mu=utility.mean(), sr=util_sr,
                wr=(edge>0).mean(), hd=ah,
                mdd=dd.max() if len(dd) else 0., ann_edge=edge.mean()*tpy,
                inv_frac=inv_frac,
                avg_ez=df["ez"].mean() if "ez" in df.columns else 0.,
                avg_edge=edge.mean(), ret_tot=raw_ret.sum(), ret_mu=raw_ret.mean(), ret_sr=ret_sr)


def trade_diagnostics(trades, entry_level=0.0, recovery_level=0.0):
    if not trades:
        return dict(n_trades=0., entry_below_level_frac=np.nan, exit_above_recovery_frac=np.nan,
                    policy_exit_frac=np.nan, terminal_exit_frac=np.nan, avg_xz=np.nan)
    df = pd.DataFrame(trades)
    exit_reason = df["exit_reason"] if "exit_reason" in df.columns else pd.Series(["policy"] * len(df))
    return dict(
        n_trades=float(len(df)),
        entry_below_level_frac=float((df["ez"] < entry_level).mean()) if "ez" in df.columns else 0.,
        exit_above_recovery_frac=float((df["xz"] > recovery_level).mean()) if "xz" in df.columns else 0.,
        policy_exit_frac=float((exit_reason == "policy").mean()),
        terminal_exit_frac=float((exit_reason == "terminal").mean()),
        avg_xz=float(df["xz"].mean()) if "xz" in df.columns else 0.,
    )


def continuation_gap_rows(agent, signal, prices, scales, trades, lookahead=None):
    if not trades:
        return []
    horizon = cfg.max_hold_pt if lookahead is None else max(int(lookahead), 1)
    rows = []
    with torch.no_grad():
        for trade in trades:
            ei = int(trade["ei"])
            xi = int(trade["xi"])
            if xi <= ei:
                continue
            entry_signal = float(trade["ez"])
            entry_price = float(trade["trade_ep"])
            entry_scale = float(trade.get("entry_scale", 1.0))
            actual_exit_utility = float(trade.get("utility", 0.0))
            for t in range(ei, xi):
                hi = min(len(signal) - 1, t + horizon)
                future_utilities = [
                    _u_s(_trade_edge_s(float(prices[u]), entry_price, entry_scale))
                    for u in range(t, hi + 1)
                ]
                immediate_utility = float(future_utilities[0])
                best_future_utility = float(max(future_utilities))
                hold_age = float(t - ei)
                pt = torch.tensor([float(signal[t])], dtype=torch.float32, device=agent.dev)
                bt = torch.tensor([entry_signal], dtype=torch.float32, device=agent.dev)
                pxt = torch.tensor([float(prices[t])], dtype=torch.float32, device=agent.dev)
                sct = torch.tensor([float(scales[t])], dtype=torch.float32, device=agent.dev)
                ept = torch.tensor([entry_price], dtype=torch.float32, device=agent.dev)
                est = torch.tensor([entry_scale], dtype=torch.float32, device=agent.dev)
                hat = torch.tensor([hold_age], dtype=torch.float32, device=agent.dev)
                hold_value = float(agent.v1(_v1_features_t(pt, bt, pxt, sct, ept, est, hat)).item())
                rows.append(dict(
                    signal=float(signal[t]),
                    hold_age=hold_age,
                    hold_value=hold_value,
                    immediate_utility=immediate_utility,
                    best_future_utility=best_future_utility,
                    actual_exit_utility=actual_exit_utility,
                    future_minus_hold=best_future_utility - hold_value,
                    future_minus_now=best_future_utility - immediate_utility,
                    future_minus_exit=best_future_utility - actual_exit_utility,
                ))
    return rows


def summarize_continuation_gap_rows(rows, tol=0.05):
    if not rows:
        return dict(
            n_states=np.nan,
            avg_hold_value=np.nan,
            avg_immediate_utility=np.nan,
            avg_best_future_utility=np.nan,
            avg_future_minus_hold=np.nan,
            avg_future_minus_now=np.nan,
            avg_future_minus_exit=np.nan,
            future_better_than_now_frac=np.nan,
            future_better_than_hold_frac=np.nan,
            midband_future_better_than_hold_frac=np.nan,
            midband_avg_future_minus_hold=np.nan,
        )
    df = pd.DataFrame(rows)
    midband = df[(df["signal"] >= -0.5) & (df["signal"] <= 1.5)]
    return dict(
        n_states=float(len(df)),
        avg_hold_value=float(df["hold_value"].mean()),
        avg_immediate_utility=float(df["immediate_utility"].mean()),
        avg_best_future_utility=float(df["best_future_utility"].mean()),
        avg_future_minus_hold=float(df["future_minus_hold"].mean()),
        avg_future_minus_now=float(df["future_minus_now"].mean()),
        avg_future_minus_exit=float(df["future_minus_exit"].mean()),
        future_better_than_now_frac=float((df["future_minus_now"] > tol).mean()),
        future_better_than_hold_frac=float((df["future_minus_hold"] > tol).mean()),
        midband_future_better_than_hold_frac=float((midband["future_minus_hold"] > tol).mean()) if len(midband) else np.nan,
        midband_avg_future_minus_hold=float(midband["future_minus_hold"].mean()) if len(midband) else np.nan,
    )


def summarize_metric_list(metric_list):
    if not metric_list:
        base = metrics([])
        return base, {k: 0.0 for k in base}
    keys = metric_list[0].keys()
    mean = {}
    std = {}
    for key in keys:
        vals = np.array([row[key] for row in metric_list], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mean[key] = 0.0
            std[key] = 0.0
            continue
        mean[key] = float(vals.mean())
        std[key] = float(vals.std(ddof=0))
    return mean, std


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time(); W = 80
    print("="*W)
    print(" Exploratory RL for Speculative Trading — NASDAQ 100  (V9)")
    print(" Zhao, Tse & Zheng (2026) · arXiv:2604.02035v1")
    print("="*W)

    print("\n[1/7] Downloading price data …")
    data = {}
    for tk in TICKERS:
        try:
            df = yf.download(tk, start="2018-01-01", end="2025-07-01",
                             auto_adjust=True, progress=False)
            if len(df) > 500:
                c = df["Close"]
                if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                data[tk] = c
        except (KeyError, ValueError, TypeError, IndexError, OSError):
            pass
    raw_prices_df = pd.DataFrame(data).sort_index().ffill().dropna()
    tickers = list(raw_prices_df.columns)
    print(f"      {len(tickers)} tickers, {len(raw_prices_df)} days")

    print("[2/7] Computing z-score signals …")
    rm = raw_prices_df.rolling(cfg.zscore_window).mean()
    rs = raw_prices_df.rolling(cfg.zscore_window).std()
    zdf = ((raw_prices_df - rm)/(rs+1e-8)).clip(-5,5).dropna()
    raw_prices_df = raw_prices_df.loc[zdf.index]
    trade_df, trade_scale_df, trade_process_label = _build_trade_process(
        raw_prices_df,
        rm.loc[zdf.index],
        rs.loc[zdf.index],
    )
    trn_z = zdf[zdf.index <= cfg.train_end]; tst_z = zdf[zdf.index >= cfg.test_start]
    trn_trade = trade_df.loc[trn_z.index]; tst_trade = trade_df.loc[tst_z.index]
    trn_scale = trade_scale_df.loc[trn_z.index]; tst_scale = trade_scale_df.loc[tst_z.index]
    trn_raw = raw_prices_df.loc[trn_z.index]; tst_raw = raw_prices_df.loc[tst_z.index]
    cfg.diag_price_ref = float(np.nanmedian(trn_trade.to_numpy()))
    cfg.diag_scale_ref = float(np.nanmedian(trn_scale.to_numpy()))
    n_test = len(tst_z)
    print(f"      Train: {len(trn_z)}d  Test: {n_test}d")
    print(f"      Trade process: {trade_process_label} | Entry mode: {cfg.entry_admissibility}")

    print("[3/7] Building training windows …")
    paths = []
    full_paths = []
    stride = cfg.window_len // 4
    for tk in tickers:
        s = trn_z[tk].values.astype(np.float32)
        p = trn_trade[tk].values.astype(np.float32)
        sc = trn_scale[tk].values.astype(np.float32)
        full_paths.append(np.stack([s, p, sc], axis=-1))
        for st in range(0, len(s)-cfg.window_len, stride):
            ws = s[st:st+cfg.window_len+1]
            wp = p[st:st+cfg.window_len+1]
            wsc = sc[st:st+cfg.window_len+1]
            if len(ws)==cfg.window_len+1 and np.all(np.isfinite(ws)) and np.all(np.isfinite(wp)) and np.all(np.isfinite(wsc)):
                paths.append(np.stack([ws, wp, wsc], axis=-1))
    paths = np.asarray(paths, dtype=np.float32)
    full_paths = np.asarray(full_paths, dtype=np.float32)
    n_gate = int(np.sum(np.any(paths[:, :, 0] < cfg.signal_gate, axis=1)))
    print(f"      {len(paths)} windows, {n_gate} ({100*n_gate/len(paths):.0f}%) "
          f"have z<{cfg.signal_gate}")
    print(f"      {len(full_paths)} full train paths reserved for regime-1 continuation")

    print(f"\n[4/7] Training  (entry_center={cfg.signal_gate}, exit_center={cfg.exit_hard}, "
          f"M={cfg.M}, Ψ={cfg.Psi}, min_hold={cfg.min_hold})")
    agent = Agent()
    losses = agent.train(paths, regime1_paths=full_paths)
    print("    Final:")
    agent.diag("  ")

    print("\n[5/7] Optimising threshold baseline …")
    best_ths = optimize_thresholds(trn_z, trn_trade, trn_scale, trn_raw, tickers)
    same_entry_exit = optimize_exit_threshold(trn_z, trn_trade, trn_scale, trn_raw, tickers, cfg.signal_gate)
    print(f"      Same-entry reference: entry<{cfg.signal_gate}, exit>{same_entry_exit}")
    print(f"      Entry-optimized reference: entry<{best_ths[0]}, exit>{best_ths[1]}")

    bnh = {tk: (tst_raw[tk].values[-1]-tst_raw[tk].values[0])/tst_raw[tk].values[0]
           for tk in tickers if len(tst_raw[tk])>1}
    bnh_avg = np.mean(list(bnh.values()))

    eval_mode = "single-round-trip" if cfg.single_round_trip_eval else "multi-round-trip"
    print(f"\n[6/7] Backtesting ({cfg.test_start} → end, {cfg.n_eval} RL runs, {eval_mode}) …")
    rl_res, bl_res, bl_same_res = {}, {}, {}
    rl_runs_all = [[] for _ in range(cfg.n_eval)]
    rl_gap_rows_all = [[] for _ in range(cfg.n_eval)]
    bl_all = []
    bl_same_all = []
    zero_trade_audit = []
    # Store per-ticker data for trade visualisation (run 0 only)
    ticker_plot_data = {}
    nt = 0
    for tk in tickers:
        sig = tst_z[tk].values.astype(np.float32)
        tp  = tst_trade[tk].values.astype(np.float64)
        sc  = tst_scale[tk].values.astype(np.float64)
        raw = tst_raw[tk].values.astype(np.float64)
        if not np.all(np.isfinite(sig)) or not np.all(np.isfinite(tp)) or not np.all(np.isfinite(sc)) or not np.all(np.isfinite(raw)):
            print(f"      Skip {tk}: non-finite test data")
            continue
        if len(sig) < 30: continue
        nt += 1
        run_tr = [agent.backtest(sig, tp, sc, raw, seed=r*7919+42) for r in range(cfg.n_eval)]
        run_m  = [metrics(tr, n_test, 1, 1) for tr in run_tr]
        rl_res[tk] = {k: np.mean([m[k] for m in run_m]) for k in run_m[0]}
        for run_idx, tr in enumerate(run_tr):
            rl_runs_all[run_idx].extend(tr)
            rl_gap_rows_all[run_idx].extend(continuation_gap_rows(agent, sig, tp, sc, tr))
        bl_tr = _thresh_bt(sig, tp, sc, raw, best_ths[0], best_ths[1])
        bl_same_tr = _thresh_bt(sig, tp, sc, raw, cfg.signal_gate, same_entry_exit)
        bl_res[tk] = metrics(bl_tr, n_test, 1, 1); bl_all.extend(bl_tr)
        bl_same_res[tk] = metrics(bl_same_tr, n_test, 1, 1)
        bl_same_all.extend(bl_same_tr)
        # Store first-run trades + price series for equity curve plots
        ticker_plot_data[tk] = dict(
            dates=tst_raw.index,
            raw_prices=raw,
            signal=sig,
            rl_trades=run_tr[0],
            bl_trades=bl_same_tr,
        )
        if rl_res[tk]["n"] == 0 and bl_same_res[tk]["n"] == 0:
            zero_trade_audit.append(dict(
                tk=tk,
                raw_nan=int(data[tk].isna().sum()),
                aligned_nan=int(raw_prices_df[tk].isna().sum()),
                train_min=float(trn_z[tk].min()),
                test_min=float(tst_z[tk].min()),
                test_lt_entry_center=int((sig < cfg.signal_gate).sum()),
                test_lt_bl=int((sig < best_ths[0]).sum()),
                test_gt_exit_center=int((sig > cfg.exit_hard).sum()),
                center_recoveries=_count_center_recoveries(sig, cfg.signal_gate,
                                                           cfg.min_hold, cfg.exit_hard),
            ))

    rl_run_metrics = [metrics(run_trades, n_test, nt, 1) for run_trades in rl_runs_all]
    ra, ra_std = summarize_metric_list(rl_run_metrics)
    rl_run_diags = [trade_diagnostics(run_trades, cfg.signal_gate, 0.0) for run_trades in rl_runs_all]
    rd, rd_std = summarize_metric_list(rl_run_diags)
    rl_gap_diags = [summarize_continuation_gap_rows(rows) for rows in rl_gap_rows_all]
    rg, rg_std = summarize_metric_list(rl_gap_diags)
    ba = metrics(bl_all, n_test, nt, 1)
    bsame = metrics(bl_same_all, n_test, nt, 1)
    bd = trade_diagnostics(bl_all, best_ths[0], 0.0)
    bdsame = trade_diagnostics(bl_same_all, cfg.signal_gate, 0.0)
    edge_label = "Edge"
    ann_edge_label = "AnnEdge"

    print(f"\n[7/7] Results  ({nt} tickers)\n")
    hdr = (f"{'Ticker':>8} │ {'Tr/r':>4} {'UΣ':>7} {'Uμ':>7} {'U-SR':>6} "
           f"{'Win%':>5} {'AvgH':>5} {edge_label:>6} │ {'#Tr':>4} {'UΣ':>7} "
           f"{'Uμ':>7} {'U-SR':>6} {'Win%':>5} {'AvgH':>5} │ {'B&H%':>5}")
    print(f"{'':>8}   {'── RL mean over runs ──':^46}   {'── Same-entry BL ──':^38}")
    print(hdr); print("─"*len(hdr))
    for tk in sorted(rl_res):
        r, b = rl_res[tk], bl_same_res.get(tk, metrics([]))
        bh = bnh.get(tk, 0)
        print(f"{tk:>8} │ {r['n']:>4.1f} {r['tot']:>7.2f} "
              f"{r['mu']:>7.4f} {r['sr']:>6.2f} {r['wr']*100:>4.0f}% "
              f"{r['hd']:>5.0f} {r['avg_edge']:>6.2f} │ {b['n']:>4.0f} "
              f"{b['tot']:>7.2f} {b['mu']:>7.4f} {b['sr']:>6.2f} "
              f"{b['wr']*100:>4.0f}% {b['hd']:>5.0f} │ {bh*100:>4.0f}%")
    print("─"*len(hdr))
    print("\n  Aggregate summary:")
    print(f"  RL mean±std over {cfg.n_eval} runs: "
          f"trades={ra['n']:.1f}±{ra_std['n']:.1f} | "
            f"UΣ={ra['tot']:.2f}±{ra_std['tot']:.2f} | "
            f"Uμ={ra['mu']:.4f}±{ra_std['mu']:.4f} | "
            f"U-SR={ra['sr']:.2f}±{ra_std['sr']:.2f} | "
          f"Win={ra['wr']*100:.1f}±{ra_std['wr']*100:.1f}% | "
          f"AvgH={ra['hd']:.0f}±{ra_std['hd']:.0f}d | "
            f"avg_edge={ra['avg_edge']:.2f}±{ra_std['avg_edge']:.2f}")
    print(f"  RL mean±std over {cfg.n_eval} runs: "
          f"invested={ra['inv_frac']*100:.1f}±{ra_std['inv_frac']*100:.1f}% | "
            f"{ann_edge_label}={ra['ann_edge']:.1f}±{ra_std['ann_edge']:.1f}/tk | "
            f"raw_tot={ra['ret_tot']*100:.1f}±{ra_std['ret_tot']*100:.1f}% | "
            f"raw_SR={ra['ret_sr']:.2f}±{ra_std['ret_sr']:.2f}")
    print(f"  BL same-entry reference: "
            f"trades={bsame['n']:.0f} | UΣ={bsame['tot']:.2f} | "
            f"Uμ={bsame['mu']:.4f} | U-SR={bsame['sr']:.2f} | "
            f"Win={bsame['wr']*100:.1f}% | AvgH={bsame['hd']:.0f}d | avg_edge={bsame['avg_edge']:.2f}")
    print(f"  BL same-entry reference: "
            f"invested={bsame['inv_frac']*100:.1f}% | {ann_edge_label}={bsame['ann_edge']:.1f}/tk | "
            f"raw_tot={bsame['ret_tot']*100:.1f}% | raw_SR={bsame['ret_sr']:.2f}")
    print(f"  BL deterministic reference: "
            f"trades={ba['n']:.0f} | UΣ={ba['tot']:.2f} | "
            f"Uμ={ba['mu']:.4f} | U-SR={ba['sr']:.2f} | "
            f"Win={ba['wr']*100:.1f}% | AvgH={ba['hd']:.0f}d | avg_edge={ba['avg_edge']:.2f}")
    print(f"  BL deterministic reference: "
            f"invested={ba['inv_frac']*100:.1f}% | {ann_edge_label}={ba['ann_edge']:.1f}/tk | "
            f"raw_tot={ba['ret_tot']*100:.1f}% | raw_SR={ba['ret_sr']:.2f}")
    print("\n  Trade diagnostics:")
    print(f"  RL mean±std over {cfg.n_eval} runs: "
          f"entry<{cfg.signal_gate:.1f}={rd['entry_below_level_frac']*100:.1f}±{rd_std['entry_below_level_frac']*100:.1f}% | "
          f"exit>0={rd['exit_above_recovery_frac']*100:.1f}±{rd_std['exit_above_recovery_frac']*100:.1f}% | "
          f"policy_exit={rd['policy_exit_frac']*100:.1f}±{rd_std['policy_exit_frac']*100:.1f}% | "
          f"terminal_exit={rd['terminal_exit_frac']*100:.1f}±{rd_std['terminal_exit_frac']*100:.1f}% | "
          f"avg_exit_z={rd['avg_xz']:.2f}±{rd_std['avg_xz']:.2f}")
    print(
          f"  BL same-entry reference: "
          f"entry<{cfg.signal_gate:.1f}={bdsame['entry_below_level_frac']*100:.1f}% | "
          f"exit>0={bdsame['exit_above_recovery_frac']*100:.1f}% | "
          f"policy_exit={bdsame['policy_exit_frac']*100:.1f}% | "
          f"terminal_exit={bdsame['terminal_exit_frac']*100:.1f}% | "
          f"avg_exit_z={bdsame['avg_xz']:.2f}")
    print(f"  BL deterministic reference: "
          f"entry<{best_ths[0]:.1f}={bd['entry_below_level_frac']*100:.1f}% | "
          f"exit>0={bd['exit_above_recovery_frac']*100:.1f}% | "
          f"policy_exit={bd['policy_exit_frac']*100:.1f}% | "
          f"terminal_exit={bd['terminal_exit_frac']*100:.1f}% | "
            f"avg_exit_z={bd['avg_xz']:.2f}")
    print("\n  Continuation gap diagnostic (RL only, max_hold horizon):")
    print(f"  RL mean±std over {cfg.n_eval} runs: "
          f"states={rg['n_states']:.1f}±{rg_std['n_states']:.1f} | "
          f"hold={rg['avg_hold_value']:.2f}±{rg_std['avg_hold_value']:.2f} | "
          f"now={rg['avg_immediate_utility']:.2f}±{rg_std['avg_immediate_utility']:.2f} | "
          f"best_future={rg['avg_best_future_utility']:.2f}±{rg_std['avg_best_future_utility']:.2f}")
    print(f"  RL mean±std over {cfg.n_eval} runs: "
          f"future-hold={rg['avg_future_minus_hold']:.2f}±{rg_std['avg_future_minus_hold']:.2f} | "
          f"future-now={rg['avg_future_minus_now']:.2f}±{rg_std['avg_future_minus_now']:.2f} | "
          f"future-exit={rg['avg_future_minus_exit']:.2f}±{rg_std['avg_future_minus_exit']:.2f} | "
          f"future>hold={rg['future_better_than_hold_frac']*100:.1f}±{rg_std['future_better_than_hold_frac']*100:.1f}%")
    print(f"  RL mid-band p∈[-0.5,1.5]: "
          f"future-hold={rg['midband_avg_future_minus_hold']:.2f}±{rg_std['midband_avg_future_minus_hold']:.2f} | "
          f"future>hold={rg['midband_future_better_than_hold_frac']*100:.1f}±{rg_std['midband_future_better_than_hold_frac']*100:.1f}%")
    print(f"  B&H : {bnh_avg*100:.1f}% total ({bnh_avg*252/n_test*100:.1f}% ann)")

    if zero_trade_audit:
        print("\n  Zero-trade audit (RL=0 and BL=0):")
        for row in zero_trade_audit:
            print(f"    {row['tk']}: raw_nan={row['raw_nan']} aligned_nan={row['aligned_nan']} "
                  f"train_min={row['train_min']:.2f} test_min={row['test_min']:.2f} "
                f"test<entry_center={row['test_lt_entry_center']} test<bl={row['test_lt_bl']} "
                f"test>exit_center={row['test_gt_exit_center']} center_recoveries={row['center_recoveries']}")

    print("\nGenerating plots → exploratory_rl_results.png")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0,0]
    ax.plot(losses, alpha=.15, lw=.5, color="steelblue")
    if len(losses)>30:
        ax.plot(pd.Series(losses).rolling(30).mean().values,"r-",lw=1.5,label="30-MA")
    ax.set(title="(a) Training Loss (TD²)",xlabel="Iteration",ylabel="Loss",yscale="log")
    ax.legend(fontsize=9)

    ax = axes[0,1]; ax2 = ax.twinx()
    # Keep tensor ops in one no_grad block before converting to NumPy.
    with torch.no_grad():
        pg = torch.linspace(-3.5, 3.5, 300).to(agent.dev)
        scale_ref = torch.full_like(pg, cfg.diag_scale_ref)
        armed_ref = torch.ones_like(pg)
        trade_ref = _diag_trade_slice_t(pg, scale_ref)
        v0v = agent.v0(_v0_features_t(pg, trade_ref, scale_ref, armed_ref)).cpu().numpy()
        d1v = agent.delta1(pg, trade_ref, scale_ref, armed_ref).cpu().numpy()
        qe  = (1-torch.exp(-_entry_lam(agent.delta1(pg, trade_ref, scale_ref, armed_ref),cfg.eta_end,_signal_p=pg)*cfg.dt)
               ).cpu().numpy()
        p_np = pg.cpu().numpy()
    ax.plot(p_np, v0v, "b-", lw=2, label="V₀(p)")
    ax.plot(p_np, d1v, "r--", lw=1.5, label="Δ₁(p)")
    ax2.fill_between(p_np, 0, qe, alpha=.12, color="green")
    ax2.plot(p_np, qe, "g-", lw=1, alpha=.6, label="q_α(p)")
    ax.axhline(0,color="gray",ls="--",alpha=.4)
    ax.axvline(cfg.signal_gate,color="orange",ls=":",lw=2,
               label=f"entry center z={cfg.signal_gate}")
    ax.axvline(cfg.exit_hard, color="red", ls=":", lw=1.5,
               label=f"exit center z={cfg.exit_hard}")
    ax.set(title="(b) V₀, Δ₁, Entry Probability",xlabel="Signal p (z-score)")
    ax.legend(fontsize=8,loc="upper right"); ax2.set_ylabel("q_α",color="green")
    ax2.set_ylim(-0.05,1.05)

    ax = axes[1,0]
    rl_all = [trade for run_trades in rl_runs_all for trade in run_trades]
    if rl_all:
        uu = [t["utility"] for t in rl_all]
        ax.hist(uu,bins=60,alpha=.55,color="steelblue",density=True,
            label=f"RL pooled {cfg.n_eval} runs (Uμ={np.mean(uu):.4f}, n={len(uu)})")
    if bl_all:
        bu = [t["utility"] for t in bl_all]
        ax.hist(bu,bins=40,alpha=.55,color="coral",density=True,
            label=f"BL deterministic (Uμ={np.mean(bu):.4f}, n={len(bu)})")
    ax.set(title="(c) Trade Utility Distribution",xlabel="Utility",ylabel="Density")
    ax.legend(fontsize=9)

    ax = axes[1,1]
    # Keep exit-probability tensor ops in one no_grad block.
    with torch.no_grad():
        pg2 = torch.linspace(-3.5, 3.5, 300).to(agent.dev)
        scale_ref = torch.full_like(pg2, cfg.diag_scale_ref)
        armed_ref = torch.ones_like(pg2)
        trade_ref = _diag_trade_slice_t(pg2, scale_ref)
        p2  = pg2.cpu().numpy()
        qe2 = (1-torch.exp(-_entry_lam(agent.delta1(pg2, trade_ref, scale_ref, armed_ref),cfg.eta_end,_signal_p=pg2)*cfg.dt)
               ).cpu().numpy()
        exit_curves = {}
        hold_age_ref = torch.full_like(pg2, cfg.diag_hold_age_ref)
        for bv in [-2.5, -2.0, -1.5]:
            bref = torch.full_like(pg2, bv)
            entry_trade = _diag_trade_slice_t(bref, scale_ref)
            exit_curves[bv] = (
                1-torch.exp(-_exit_lam(agent.delta2(pg2, bref, trade_ref, scale_ref, entry_trade, scale_ref, hold_age_ref),cfg.eta_end,_signal_p=pg2)*cfg.dt)
            ).cpu().numpy()
    for (bv,clr,ls) in [(-2.5,"darkred","-"),(-2.0,"red","--"),(-1.5,"salmon",":")]:
        ax.plot(p2, exit_curves[bv], ls, lw=1.5, color=clr, label=f"Exit q_β(b={bv})")
    ax.plot(p2, qe2, "b-", lw=2, label="Entry q_α(p)")
    ax.axvline(cfg.signal_gate,color="orange",ls=":",lw=1.5,alpha=.7)
    ax.axvline(cfg.exit_hard, color="red",  ls=":",lw=1.5,alpha=.7)
    ax.set(title="(d) Entry/Exit Probabilities",xlabel="Signal p",
           ylabel="Prob/day",ylim=(-0.05,1.05))
    ax.legend(fontsize=8,loc="center left")

    plt.tight_layout(pad=2.0)
    plt.savefig("exploratory_rl_results.png",dpi=150,bbox_inches="tight")
    plt.close(fig); print("Saved → exploratory_rl_results.png")

    # ── Per-ticker equity curve with buy/sell markers ─────────────────────
    # Pick up to 8 tickers that had RL trades, sorted by utility
    traded_tickers = [tk for tk in sorted(ticker_plot_data)
                      if ticker_plot_data[tk]["rl_trades"]]
    traded_tickers.sort(
        key=lambda tk: sum(t["utility"] for t in ticker_plot_data[tk]["rl_trades"]),
        reverse=True,
    )
    plot_tickers = traded_tickers[:8]
    if plot_tickers:
        n_panels = len(plot_tickers)
        ncols = min(4, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        if n_panels == 1:
            axes2 = np.array([axes2])
        axes2 = axes2.ravel()
        for i, tk in enumerate(plot_tickers):
            ax = axes2[i]
            pd_ = ticker_plot_data[tk]
            dates = pd_["dates"]
            prices = pd_["raw_prices"]
            rl_tr = pd_["rl_trades"]
            bl_tr = pd_["bl_trades"]
            sig = pd_["signal"]

            # Bollinger Bands (same window as z-score signal)
            price_s = pd.Series(prices, index=dates)
            bb_mid = price_s.rolling(cfg.zscore_window, min_periods=1).mean()
            bb_std = price_s.rolling(cfg.zscore_window, min_periods=1).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            ax.plot(dates, bb_mid.values, color="steelblue", lw=0.8, alpha=0.5, ls="--")
            ax.fill_between(dates, bb_lower.values, bb_upper.values,
                            alpha=0.07, color="steelblue", label="BB(60,2)")

            # Price line
            ax.plot(dates, prices, color="gray", lw=1.2, alpha=0.85, label="Price")

            # Z-score on twin axis
            ax2 = ax.twinx()
            ax2.fill_between(dates, sig, 0, alpha=0.08, color="blue")
            ax2.axhline(cfg.signal_gate, color="orange", ls=":", lw=0.8, alpha=0.5)
            ax2.set_ylabel("z-score", fontsize=7, color="blue")
            ax2.set_ylim(-4, 4)
            ax2.tick_params(labelsize=6, colors="blue")

            # RL trades: green ▲ = buy, red ▼ = sell, with price labels
            for tr in rl_tr:
                ei, xi = int(tr["ei"]), int(tr["xi"])
                ep = tr.get("ep", prices[ei] if ei < len(prices) else 0)
                xp = tr.get("xp", prices[xi] if xi < len(prices) else 0)
                ret_pct = tr.get("ret", 0) * 100
                if ei < len(dates):
                    ax.scatter(dates[ei], prices[ei], marker="^", s=90,
                               color="green", zorder=5, edgecolors="black", linewidths=0.5)
                    ax.annotate(f"${ep:.0f}", (dates[ei], prices[ei]),
                                textcoords="offset points", xytext=(5, -14),
                                fontsize=6, color="green", fontweight="bold")
                if xi < len(dates):
                    clr = "limegreen" if tr.get("utility", 0) > 0 else "red"
                    ax.scatter(dates[xi], prices[xi], marker="v", s=90,
                               color=clr, zorder=5, edgecolors="black", linewidths=0.5)
                    ax.annotate(f"${xp:.0f} ({ret_pct:+.1f}%)", (dates[xi], prices[xi]),
                                textcoords="offset points", xytext=(5, 8),
                                fontsize=6, color="darkgreen" if ret_pct >= 0 else "red",
                                fontweight="bold")
                    # Shade hold period
                    ei_c = max(0, min(ei, len(dates) - 1))
                    xi_c = max(0, min(xi, len(dates) - 1))
                    ax.axvspan(dates[ei_c], dates[xi_c], alpha=0.10,
                               color="green" if tr.get("utility", 0) > 0 else "red")

            # BL trades: small blue markers
            for tr in bl_tr:
                ei, xi = int(tr["ei"]), int(tr["xi"])
                if ei < len(dates):
                    ax.scatter(dates[ei], prices[ei], marker="^", s=35,
                               color="dodgerblue", zorder=4, alpha=0.7)
                if xi < len(dates):
                    ax.scatter(dates[xi], prices[xi], marker="v", s=35,
                               color="dodgerblue", zorder=4, alpha=0.7)

            # Annotation
            rl_u = sum(t["utility"] for t in rl_tr)
            rl_ret = sum(t.get("ret", 0) for t in rl_tr) * 100
            rl_hd = np.mean([t["hd"] for t in rl_tr]) if rl_tr else 0
            bl_u = sum(t["utility"] for t in bl_tr)
            ax.set_title(f"{tk}  RL:U={rl_u:.2f} ret={rl_ret:+.1f}% H={rl_hd:.0f}d  BL:U={bl_u:.2f}",
                         fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("Price", fontsize=7)

        # Hide unused axes
        for j in range(i + 1, len(axes2)):
            axes2[j].set_visible(False)

        # Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_els = [
            Line2D([0], [0], color="gray", lw=1.2, label="Price"),
            Patch(facecolor="steelblue", alpha=0.15, label=f"BB({cfg.zscore_window},2)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
                   markersize=8, markeredgecolor="black", label="RL Buy"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="limegreen",
                   markersize=8, markeredgecolor="black", label="RL Sell (win)"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="red",
                   markersize=8, markeredgecolor="black", label="RL Sell (loss)"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="dodgerblue",
                   markersize=6, label="BL Buy/Sell"),
        ]
        fig2.legend(handles=legend_els, loc="lower center", ncol=5, fontsize=8,
                    bbox_to_anchor=(0.5, -0.02))

        fig2.suptitle("Trade Visualisation — RL (green/red) vs Baseline (blue)",
                      fontsize=12, fontweight="bold")
        plt.tight_layout(pad=1.5, rect=[0, 0.03, 1, 0.97])
        plt.savefig("exploratory_rl_trades.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print("Saved → exploratory_rl_trades.png")

    elapsed = time.time()-t0
    print(f"\n{'='*W}")
    print(f"  RL Agent :  U-SR {ra['sr']:.2f} | Win {ra['wr']*100:.1f}% | "
            f"Uμ {ra['mu']:.4f} | {edge_label} {ra['avg_edge']:.2f} | RawSR {ra['ret_sr']:.2f}")
    print(f"  Baseline :  U-SR {ba['sr']:.2f} | Win {ba['wr']*100:.1f}% | "
            f"Uμ {ba['mu']:.4f} | {edge_label} {ba['avg_edge']:.2f} | RawSR {ba['ret_sr']:.2f}")
    print(f"  Buy&Hold :  {bnh_avg*100:.1f}% total | {bnh_avg*252/n_test*100:.1f}% ann")
    print(f"  Elapsed  :  {elapsed:.0f}s")
    print("="*W)


if __name__ == "__main__":
    main()
