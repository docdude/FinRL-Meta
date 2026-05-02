#!/usr/bin/env python3
"""
Exploratory RL for Speculative Trading on NASDAQ 100 Stocks  (V7)
=================================================================
Based on: Zhao, Tse & Zheng (2026), arXiv:2604.02035v1

V7 — Training/test distribution alignment (root-cause fix):
  1. signal_gate=-1.5 unified in BOTH training and backtest
  2. Gate-aware pretrain: entries ONLY at z<gate; realistic exit at first z>0.5 or 60d
     — removes optimistic "best-of-15" bias; teaches actual mean-reversion dynamics
  3. Deep-dip half-init: J=1 trajectories seeded from actual z<gate timestamps
  4. Hard entry gate: Δ₁>0 AND z<gate  (no sigmoid, no heuristic bonus)
  5. min_hold=35, Psi=0.20 (shift toward baseline's 55d holding regime)
  6. n_pretrain=1000, n_iter=1500 for V0 option-value convergence
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy, warnings, time

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
    gamma=1.0; iota=1.0; R=0.0; rho=2e-4
    M=1.5; varpi=0.5; k_loss=2.0; dt=1.0
    eta_start=0.05; eta_end=0.01
    Psi=0.20                  # V7: raised 0.15→0.20
    signal_gate=-1.5          # V7: unified gate (training = backtest)
    exit_target=0.5           # V7: pretrain exit when z crosses this
    max_hold_pt=60            # V7: max pretrain hold before forced exit
    zscore_window=60; window_len=200
    min_hold=35               # V7: raised 25→35
    cooldown=5
    n_sims=6; batch_size=32
    n_pretrain=1000           # V7: raised 800→1000
    n_iter=1500               # V7: raised 1200→1500
    freeze_K=4; tau=0.005
    td_nstep=5; reanchor_every=200; reanchor_steps=50
    lr=5e-4; hidden=64; taylor_thr=0.1
    train_end="2022-12-31"; test_start="2023-01-01"; n_eval=5

    @property
    def eta(self): return self._eta if hasattr(self,"_eta") else self.eta_start
    @eta.setter
    def eta(self, v): self._eta = v

cfg = Cfg()


# ── Utility / HJB ────────────────────────────────────────────────────────────
def _U_t(x):
    a = torch.abs(x) + 1e-8
    return torch.where(x >= 0, a.pow(cfg.varpi), -cfg.k_loss * a.pow(cfg.varpi))

def _G_t(p, b):
    return _U_t(cfg.gamma * p - cfg.iota * b - cfg.Psi - cfg.R)

def _u_s(x):
    a = abs(x) + 1e-8
    return a**cfg.varpi if x >= 0 else -cfg.k_loss * a**cfg.varpi

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

def _entry_lam(delta, eta=None, signal_p=None):
    """V7: Hard Δ₁>0 gate + hard signal gate. No sigmoid, no heuristic bonus."""
    eta = eta or cfg.eta
    lam = torch.where(delta > 0, _mean_lam(delta, eta), torch.zeros_like(delta))
    if signal_p is not None:
        lam = lam * (signal_p < cfg.signal_gate).float()
    return torch.clamp(lam, 0, cfg.M)


# ── Networks ─────────────────────────────────────────────────────────────────
class V0Net(nn.Module):
    def __init__(self):
        super().__init__(); h = cfg.hidden
        self.f = nn.Sequential(
            nn.Linear(1,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
    def forward(self, p): return self.f(p).squeeze(-1)

class V1Net(nn.Module):
    def __init__(self):
        super().__init__(); h = cfg.hidden
        self.f = nn.Sequential(
            nn.Linear(2,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, pb): return self.f(pb).squeeze(-1)


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v0 = V0Net().to(self.dev)
        self.v1 = V1Net().to(self.dev)
        self.v0_target = copy.deepcopy(self.v0)
        for p in self.v0_target.parameters(): p.requires_grad = False
        self.opt_v0 = torch.optim.Adam(self.v0.parameters(), lr=cfg.lr * 0.3)
        self.opt_v1 = torch.optim.Adam(self.v1.parameters(), lr=cfg.lr)

    def _update_target(self):
        for tp, sp in zip(self.v0_target.parameters(), self.v0.parameters()):
            tp.data.mul_(1 - cfg.tau).add_(sp.data, alpha=cfg.tau)

    def _d1(self, p):
        return self.v1(torch.stack([p, p], -1)) - self.v0(p.unsqueeze(-1))
    def _d1_t(self, p):
        return self.v1(torch.stack([p, p], -1)) - self.v0_target(p.unsqueeze(-1))
    def _d2(self, p, b):
        return _G_t(p, b) - self.v1(torch.stack([p, b], -1))

    def _collect_supervised(self, paths, n_pts=8):
        """V7: Gate-aware supervised targets — enter z<gate, exit at first z>exit_target."""
        N, L = paths.shape[0], paths.shape[1] - 1
        idx = np.random.choice(N, min(64, N), replace=False)
        batch = paths[idx]; ps, bs, ts = [], [], []
        for i in range(len(batch)):
            path = batch[i]
            # V7: ONLY entry points where z < signal_gate
            cands = [t for t in range(0, max(1, L - cfg.min_hold))
                     if path[t] < cfg.signal_gate]
            if not cands:
                continue
            et = np.random.choice(cands); bv = float(path[et])
            # V7: Realistic exit — first z > exit_target, or max_hold_pt days
            xt = min(et + cfg.min_hold, L)
            for xc in range(et + cfg.min_hold, min(et + cfg.max_hold_pt + 1, L + 1)):
                if xc > L: break
                if path[xc] > cfg.exit_target: xt = xc; break
            xt = min(xt, L)
            g = _u_s(cfg.gamma * float(path[xt]) - cfg.iota * bv - cfg.Psi)
            n = min(n_pts, xt - et)
            if n < 1: continue
            for t in np.linspace(et, xt - 1, n, dtype=int):
                ps.append(path[t]); bs.append(bv)
                ts.append(np.exp(-cfg.rho * (xt - t)) * g)
        return ps, bs, ts

    def pretrain(self, paths):
        opt = torch.optim.Adam(self.v1.parameters(), lr=1e-3)
        for it in range(cfg.n_pretrain):
            ps, bs, ts = self._collect_supervised(paths, n_pts=8)
            if not ps: continue
            p_t = torch.tensor(ps, dtype=torch.float32, device=self.dev)
            b_t = torch.tensor(bs, dtype=torch.float32, device=self.dev)
            tgt = torch.tensor(ts, dtype=torch.float32, device=self.dev)
            pred = self.v1(torch.stack([p_t, b_t], -1))
            loss = ((pred - tgt)**2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.v1.parameters(), 1.0); opt.step()
            if (it + 1) % 200 == 0:
                print(f"      V₁ pretrain {it+1}/{cfg.n_pretrain}  loss={loss.item():.5f}")
        self.v0_target.load_state_dict(self.v0.state_dict())
        self._diag("after pretrain")

    def _reanchor_v1(self, paths):
        """Periodic MC re-anchor with gate-consistent entries."""
        for _ in range(cfg.reanchor_steps):
            ps, bs, ts = self._collect_supervised(paths, n_pts=5)
            if not ps: continue
            p_t = torch.tensor(ps, dtype=torch.float32, device=self.dev)
            b_t = torch.tensor(bs, dtype=torch.float32, device=self.dev)
            tgt = torch.tensor(ts, dtype=torch.float32, device=self.dev)
            pred = self.v1(torch.stack([p_t, b_t], -1))
            loss = ((pred - tgt)**2).mean()
            self.opt_v1.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.v1.parameters(), 1.0); self.opt_v1.step()

    @torch.no_grad()
    def _sim(self, batch):
        B, L1 = batch.shape; L = L1 - 1; NS = B * cfg.n_sims
        P = np.repeat(batch, cfg.n_sims, 0)
        J = np.zeros(NS, dtype=np.int32)
        Br = np.zeros(NS, dtype=np.float32)
        Hs = np.zeros(NS, dtype=np.int32)
        half = NS // 2; J[half:] = 1; Hs[half:] = cfg.min_hold
        # V7: Deep-dip initialization — J=1 half gets entry at actual z<gate timestamps
        for n in range(half, NS):
            path = P[n]
            cands = [t for t in range(L // 2) if path[t] < cfg.signal_gate]
            Br[n] = float(path[np.random.choice(cands)]) if cands else float(np.min(path[:L//2]))
        traj = {n: [] for n in range(NS)}
        for l in range(L):
            if (J < 2).sum() == 0: break
            pl, pn = P[:, l], P[:, l + 1]
            for n in range(NS):
                if J[n] < 2:
                    traj[n].append((pl[n], int(J[n]), float(Br[n]), int(Hs[n]), pn[n]))
            m0 = J == 0
            if m0.any():
                ix = np.where(m0)[0]
                pt = torch.tensor(pl[ix], dtype=torch.float32, device=self.dev)
                lam = _entry_lam(self._d1_t(pt), signal_p=pt)
                q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
                ent = np.random.random(len(ix)) < q
                J[ix[ent]] = 1; Br[ix[ent]] = pl[ix[ent]]; Hs[ix[ent]] = 0
            m1r = (J == 1) & (Hs >= cfg.min_hold)
            if m1r.any():
                ix = np.where(m1r)[0]
                pt = torch.tensor(pl[ix], dtype=torch.float32, device=self.dev)
                bt = torch.tensor(Br[ix], dtype=torch.float32, device=self.dev)
                lam = _mean_lam(self._d2(pt, bt))
                q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
                ext = np.random.random(len(ix)) < q; J[ix[ext]] = 2
            Hs[J == 1] += 1
        return traj

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
                    if pt[t + k][1] != j_t: bi = t + k; break
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
        pb = torch.tensor([t[2] for t in R0], dtype=torch.float32, device=self.dev)
        jb = torch.tensor([t[3] for t in R0], dtype=torch.long, device=self.dev)
        bb = torch.tensor([t[4] for t in R0], dtype=torch.float32, device=self.dev)
        st = torch.tensor([t[5] for t in R0], dtype=torch.float32, device=self.dev)
        v0c = self.v0(p.unsqueeze(-1))
        ca = _ent_cost(self._d1(p))
        dn = torch.tensor(disc, device=self.dev) ** st
        with torch.no_grad():
            v0b = self.v0_target(pb.unsqueeze(-1))
            v1b = self.v1(torch.stack([pb, bb], -1))
            gb  = _G_t(pb, bb)
            vb  = torch.where(jb == 0, v0b, torch.where(jb == 1, v1b, gb))
        td = -ca.detach() * cfg.dt * st + dn * vb - v0c
        return (td**2).mean()

    def _loss_v1(self, R1):
        if not R1: return torch.tensor(0., device=self.dev)
        disc = np.exp(-cfg.rho * cfg.dt)
        p  = torch.tensor([t[0] for t in R1], dtype=torch.float32, device=self.dev)
        b  = torch.tensor([t[1] for t in R1], dtype=torch.float32, device=self.dev)
        pb = torch.tensor([t[2] for t in R1], dtype=torch.float32, device=self.dev)
        jb = torch.tensor([t[3] for t in R1], dtype=torch.long, device=self.dev)
        bb = torch.tensor([t[4] for t in R1], dtype=torch.float32, device=self.dev)
        st = torch.tensor([t[5] for t in R1], dtype=torch.float32, device=self.dev)
        forced = torch.tensor([t[6] for t in R1], dtype=torch.bool, device=self.dev)
        v1c = self.v1(torch.stack([p, b], -1))
        cb  = _ent_cost(self._d2(p, b))
        dn  = torch.tensor(disc, device=self.dev) ** st
        with torch.no_grad():
            v1b = self.v1(torch.stack([pb, bb], -1))
            gb  = _G_t(pb, bb)
            vb  = torch.where(jb == 1, v1b, gb)
        cost = torch.where(forced, torch.zeros_like(cb), cb.detach() * cfg.dt * st)
        td = -cost + dn * vb - v1c
        return (td**2).mean()

    def _safe_step(self, loss, opt):
        if torch.isnan(loss) or torch.isinf(loss): opt.zero_grad(); return False
        loss.backward()
        params = [p for g in opt.param_groups for p in g["params"]]
        if any(p.grad is not None and
               (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in params):
            opt.zero_grad(); return False
        nn.utils.clip_grad_norm_(params, 0.5); opt.step(); return True

    def train(self, paths):
        print("    Phase 1: Pre-training V₁ (gate-aware, realistic exits) …")
        self.pretrain(paths)
        print(f"    Phase 2: Policy iteration ({cfg.n_iter} iters) …")
        N = len(paths); losses = []
        for it in range(cfg.n_iter):
            cfg.eta = cfg.eta_start + (cfg.eta_end - cfg.eta_start) * it / max(cfg.n_iter - 1, 1)
            if it > 0 and it % cfg.reanchor_every == 0:
                self._reanchor_v1(paths)
            idx = np.random.choice(N, min(cfg.batch_size, N), replace=False)
            traj = self._sim(paths[idx]); R0, R1 = self._build_nstep(traj)
            phase = (it // cfg.freeze_K) % 2
            if phase == 0 and R1:
                self.opt_v1.zero_grad(); self._safe_step(self._loss_v1(R1), self.opt_v1)
            elif phase == 1 and R0:
                self.opt_v0.zero_grad(); self._safe_step(self._loss_v0(R0), self.opt_v0)
            if R0 or R1:
                with torch.no_grad():
                    l0 = self._loss_v0(R0).item() if R0 else 0.
                    l1 = self._loss_v1(R1).item() if R1 else 0.
                    losses.append((l0 + l1) / 2)
            self._update_target()
            if (it + 1) % 300 == 0:
                avg = np.mean(losses[-300:]) if losses else 0
                print(f"      iter {it+1}/{cfg.n_iter}  loss={avg:.6f}  η={cfg.eta:.4f}")
                self._diag("  ")
        return losses

    def _diag(self, prefix=""):
        with torch.no_grad():
            ps = torch.tensor([-3., -2.5, -2., -1.5, -1., 0., 1.], device=self.dev)
            v0 = self.v0(ps.unsqueeze(-1)); d1 = self._d1(ps)
            qe = 1 - torch.exp(-_entry_lam(d1, signal_p=ps) * cfg.dt)
            b2 = torch.full_like(ps, -2.)
            qx = 1 - torch.exp(-_mean_lam(self._d2(ps, b2)) * cfg.dt)
            print(f"    {prefix}V₀=[{v0.min():.2f},{v0.max():.2f}]  "
                  f"Δ₁=[{d1.min():.2f},{d1.max():.2f}]")
            labels_e = [-3., -2.5, -2., -1.5, -1., 0.]
            labels_x = [-2., -1.5, -1., 0., 1.]
            eq = " ".join(f"z{z:.1f}:{qe[i]:.2f}" for i, z in enumerate(labels_e))
            xq = " ".join(f"z{z:.1f}:{qx[i]:.2f}" for i, z in enumerate([0,1,2,3,4]))
            print(f"    {prefix}Entry q: {eq}")
            print(f"    {prefix}Exit(b=-2): {xq}")

    @torch.no_grad()
    def backtest(self, signal, prices, seed=0):
        np.random.seed(seed)
        trades, j, ei, es, hold, cd = [], 0, 0, 0., 0, 0
        for l in range(len(signal)):
            p = signal[l]
            if j == 0:
                if cd > 0: cd -= 1; continue
                if p >= cfg.signal_gate: continue          # V7: hard gate
                pt = torch.tensor([p], dtype=torch.float32, device=self.dev)
                lam = _entry_lam(self._d1(pt), cfg.eta_end, signal_p=pt).item()
                if np.random.random() < 1 - np.exp(-lam * cfg.dt):
                    j, ei, es, hold = 1, l, p, 0
            elif j == 1:
                hold += 1
                if hold < cfg.min_hold: continue
                pt = torch.tensor([p], dtype=torch.float32, device=self.dev)
                bt = torch.tensor([es], dtype=torch.float32, device=self.dev)
                lam = _mean_lam(self._d2(pt, bt), cfg.eta_end).item()
                if np.random.random() < 1 - np.exp(-lam * cfg.dt):
                    trades.append(dict(
                        ei=ei, xi=l, ep=prices[ei], xp=prices[l],
                        ret=(prices[l] - prices[ei]) / prices[ei],
                        hd=l - ei, spnl=p - es, ez=es))
                    j, cd = 0, cfg.cooldown
        if j == 1:
            l = len(signal) - 1
            trades.append(dict(
                ei=ei, xi=l, ep=prices[ei], xp=prices[l],
                ret=(prices[l] - prices[ei]) / prices[ei],
                hd=l - ei, spnl=signal[l] - es, ez=es))
        return trades


# ── Baseline ─────────────────────────────────────────────────────────────────
def _thresh_bt(signal, prices, eth, xth):
    trades, j, ei, es = [], 0, 0, 0.
    for l in range(len(signal)):
        p = signal[l]
        if j == 0 and p < eth: j, ei, es = 1, l, p
        elif j == 1 and p > xth:
            trades.append(dict(ei=ei, xi=l, ep=prices[ei], xp=prices[l],
                               ret=(prices[l]-prices[ei])/prices[ei],
                               hd=l-ei, spnl=p-es, ez=es)); j = 0
    if j == 1:
        l = len(signal) - 1
        trades.append(dict(ei=ei, xi=l, ep=prices[ei], xp=prices[l],
                           ret=(prices[l]-prices[ei])/prices[ei],
                           hd=l-ei, spnl=signal[l]-es, ez=es))
    return trades

def optimize_thresholds(trn_z, trn_p, tickers):
    best_sr, best = -1e10, (-1., .5)
    for eth in [-2.5, -2., -1.5, -1., -.5]:
        for xth in [-.5, 0., .5, 1., 1.5]:
            rets = []
            for tk in tickers:
                tr = _thresh_bt(trn_z[tk].values.astype(np.float32),
                                trn_p[tk].values.astype(np.float64), eth, xth)
                rets.extend([t["ret"] for t in tr])
            if len(rets) > 10:
                r = np.array(rets); sr = r.mean() / (r.std() + 1e-8)
                if sr > best_sr: best_sr, best = sr, (eth, xth)
    return best


# ── Metrics ──────────────────────────────────────────────────────────────────
def metrics(trades, n_test_days=1, n_runs=1):
    if not trades:
        return dict(n=0, tot=0., mu=0., sr=0., wr=0., hd=0., mdd=0.,
                    ann_ret=0., inv_frac=0., avg_ez=0.)
    df = pd.DataFrame(trades); r = df["ret"].values
    cum = np.cumsum(r); pk = np.maximum.accumulate(np.concatenate([[0], cum]))
    dd = pk[1:] - cum; ah = df["hd"].mean()
    sr = r.mean() / (r.std() + 1e-8) * np.sqrt(252 / max(ah, 1)) if len(r) > 1 else 0.
    trades_per_yr = (len(r) / n_runs) / (n_test_days / 252)
    return dict(
        n=len(r), tot=r.sum(), mu=r.mean(), sr=sr, wr=(r > 0).mean(), hd=ah,
        mdd=dd.max() if len(dd) else 0.,
        ann_ret=r.mean() * trades_per_yr,
        inv_frac=df["hd"].sum() / n_runs / n_test_days,
        avg_ez=df["ez"].mean() if "ez" in df.columns else 0.)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time(); W = 80
    print("=" * W)
    print(" Exploratory RL for Speculative Trading — NASDAQ 100  (V7)")
    print(" Zhao, Tse & Zheng (2026) · arXiv:2604.02035v1")
    print("=" * W)

    print("\n[1/7] Downloading price data …")
    data = {}
    for tk in TICKERS:
        try:
            df = yf.download(tk, start="2018-01-01", end="2025-07-01",
                             auto_adjust=True, progress=False)
            if len(df) > 500:
                c = df["Close"]
                if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
                data[tk] = c
        except: pass
    prices_df = pd.DataFrame(data).sort_index().ffill().dropna()
    tickers = list(prices_df.columns)
    print(f"      {len(tickers)} tickers, {len(prices_df)} days")

    print("[2/7] Computing z-score signals …")
    rm = prices_df.rolling(cfg.zscore_window).mean()
    rs = prices_df.rolling(cfg.zscore_window).std()
    zdf = ((prices_df - rm) / (rs + 1e-8)).clip(-5, 5).dropna()
    prices_df = prices_df.loc[zdf.index]
    trn_z = zdf[zdf.index <= cfg.train_end]; tst_z = zdf[zdf.index >= cfg.test_start]
    trn_p = prices_df.loc[trn_z.index]; tst_p = prices_df.loc[tst_z.index]
    n_test = len(tst_z)
    print(f"      Train: {len(trn_z)}d  Test: {n_test}d")

    print("[3/7] Building training windows …")
    paths = []
    stride = cfg.window_len // 4
    for tk in tickers:
        s = trn_z[tk].values.astype(np.float32)
        for st in range(0, len(s) - cfg.window_len, stride):
            w = s[st:st + cfg.window_len + 1]
            if len(w) == cfg.window_len + 1 and np.all(np.isfinite(w)):
                paths.append(w)
    paths = np.array(paths)
    # Log how many windows actually have z<gate (affects training quality)
    n_with_gate = sum(1 for w in paths if np.any(w < cfg.signal_gate))
    print(f"      {len(paths)} windows, {n_with_gate} ({100*n_with_gate/len(paths):.0f}%) "
          f"have z<{cfg.signal_gate}")

    print(f"\n[4/7] Training  (gate={cfg.signal_gate}, M={cfg.M}, "
          f"η={cfg.eta_start}→{cfg.eta_end}, Ψ={cfg.Psi}, min_hold={cfg.min_hold})")
    agent = Agent()
    losses = agent.train(paths)
    print("    Final:"); agent._diag("  ")

    print("\n[5/7] Optimising threshold baseline …")
    best_ths = optimize_thresholds(trn_z, trn_p, tickers)
    print(f"      Best: entry<{best_ths[0]}, exit>{best_ths[1]}")

    bnh_rets = {tk: (tst_p[tk].values[-1] - tst_p[tk].values[0]) / tst_p[tk].values[0]
                for tk in tickers if len(tst_p[tk]) > 1}
    bnh_avg = np.mean(list(bnh_rets.values()))

    print(f"\n[6/7] Backtesting ({cfg.test_start} → end, {cfg.n_eval} RL runs) …")
    rl_res, bl_res = {}, {}; rl_all, bl_all = [], []
    for tk in tickers:
        sig = tst_z[tk].values.astype(np.float32)
        px = tst_p[tk].values.astype(np.float64)
        if len(sig) < 30: continue
        run_tr = [agent.backtest(sig, px, seed=r * 7919 + 42) for r in range(cfg.n_eval)]
        run_m = [metrics(tr, n_test, 1) for tr in run_tr]
        rl_res[tk] = {k: np.mean([m[k] for m in run_m]) for k in run_m[0]}
        for tr in run_tr: rl_all.extend(tr)
        bl_tr = _thresh_bt(sig, px, best_ths[0], best_ths[1])
        bl_res[tk] = metrics(bl_tr, n_test, 1); bl_all.extend(bl_tr)

    nt = len(rl_res)
    ra = metrics(rl_all, n_test, cfg.n_eval)
    ba = metrics(bl_all, n_test, 1)

    print(f"\n[7/7] Results  ({nt} tickers)\n")
    hdr = (f"{'Ticker':>8} │ {'#Tr':>4} {'Tot%':>7} {'μ%':>7} {'SR':>6} "
           f"{'Win%':>5} {'AvgH':>5} {'ez':>5} │ {'#Tr':>4} {'Tot%':>7} "
           f"{'μ%':>7} {'SR':>6} {'Win%':>5} {'AvgH':>5} │ {'B&H%':>5}")
    print(f"{'':>8}   {'── Exploratory RL (V7) ──':^44}   {'── Threshold BL ──':^38}")
    print(hdr); print("─" * len(hdr))
    for tk in sorted(rl_res):
        r, b = rl_res[tk], bl_res.get(tk, metrics([]))
        bh = bnh_rets.get(tk, 0)
        print(f"{tk:>8} │ {r['n']:>4.0f} {r['tot']*100:>6.1f}% "
              f"{r['mu']*100:>6.2f}% {r['sr']:>6.2f} {r['wr']*100:>4.0f}% "
              f"{r['hd']:>5.0f} {r['avg_ez']:>5.1f} │ {b['n']:>4} "
              f"{b['tot']*100:>6.1f}% {b['mu']*100:>6.2f}% {b['sr']:>6.2f} "
              f"{b['wr']*100:>4.0f}% {b['hd']:>5.0f} │ {bh*100:>4.0f}%")
    print("─" * len(hdr))
    print(f"{'ALL':>8} │ {ra['n']:>4} {ra['tot']*100:>6.1f}% "
          f"{ra['mu']*100:>6.2f}% {ra['sr']:>6.2f} {ra['wr']*100:>4.0f}% "
          f"{ra['hd']:>5.0f} {ra['avg_ez']:>5.1f} │ {ba['n']:>4} "
          f"{ba['tot']*100:>6.1f}% {ba['mu']*100:>6.2f}% {ba['sr']:>6.2f} "
          f"{ba['wr']*100:>4.0f}% {ba['hd']:>5.0f} │ {bnh_avg*100:>4.0f}%")
    print(f"\n  RL  : {ra['n']} trades, avg_ez={ra['avg_ez']:.2f}, "
          f"invested={ra['inv_frac']*100:.0f}%, ann={ra['ann_ret']*100:.1f}%/tk")
    print(f"  BL  : {ba['n']} trades, avg_ez={ba['avg_ez']:.2f}, "
          f"invested={ba['inv_frac']*100:.0f}%, ann={ba['ann_ret']*100:.1f}%/tk")
    print(f"  B&H : {bnh_avg*100:.1f}% total ({bnh_avg*252/n_test*100:.1f}% annualised)")

    print("\nGenerating plots → exploratory_rl_results.png")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.plot(losses, alpha=.15, lw=.5, color="steelblue")
    if len(losses) > 30:
        ax.plot(pd.Series(losses).rolling(30).mean().values, "r-", lw=1.5, label="30-MA")
    ax.set(title="(a) Training Loss (TD²)", xlabel="Iteration", ylabel="Loss", yscale="log")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax2 = ax.twinx()
    with torch.no_grad():
        pg = torch.linspace(-3.5, 3.5, 300).to(agent.dev)
        v0v = agent.v0(pg.unsqueeze(-1)).cpu().numpy()
        d1v = agent._d1(pg).cpu().numpy()
        qe = (1 - torch.exp(-_entry_lam(agent._d1(pg), cfg.eta_end, signal_p=pg) * cfg.dt)
              ).cpu().numpy()
    p_np = pg.cpu().numpy()
    ax.plot(p_np, v0v, "b-", lw=2, label="V₀(p)")
    ax.plot(p_np, d1v, "r--", lw=1.5, label="Δ₁(p)")
    ax2.fill_between(p_np, 0, qe, alpha=.12, color="green")
    ax2.plot(p_np, qe, "g-", lw=1, alpha=.6, label="q_α(p)")
    ax.axhline(0, color="gray", ls="--", alpha=.4)
    ax.axvline(cfg.signal_gate, color="orange", ls=":", lw=2, label=f"gate z<{cfg.signal_gate}")
    ax.set(title="(b) V₀, Δ₁, Entry Probability", xlabel="Signal p (z-score)")
    ax.legend(fontsize=8, loc="upper right"); ax2.set_ylabel("q_α", color="green")
    ax2.set_ylim(-0.05, 1.05)

    ax = axes[1, 0]
    if rl_all:
        rr = [t["ret"] * 100 for t in rl_all]
        ax.hist(rr, bins=60, alpha=.55, color="steelblue", density=True,
                label=f"RL (μ={np.mean(rr):.2f}%, n={len(rr)})")
    if bl_all:
        br = [t["ret"] * 100 for t in bl_all]
        ax.hist(br, bins=40, alpha=.55, color="coral", density=True,
                label=f"BL (μ={np.mean(br):.2f}%, n={len(br)})")
    ax.set(title="(c) Trade Return Distribution", xlabel="Return %", ylabel="Density")
    ax.set_xlim(-40, 50); ax.legend(fontsize=9)

    ax = axes[1, 1]
    with torch.no_grad():
        pg2 = torch.linspace(-3.5, 3.5, 300).to(agent.dev)
        for bv, clr, ls in [(-2.5,"darkred","-"),(-1.5,"red","--"),(0.,"salmon",":")]:
            bref = torch.full_like(pg2, bv)
            qx = (1 - torch.exp(-_mean_lam(agent._d2(pg2, bref), cfg.eta_end) * cfg.dt)
                  ).cpu().numpy()
            ax.plot(pg2.cpu().numpy(), qx, ls, lw=1.5, color=clr, label=f"Exit q_β(b={bv})")
        qe2 = (1 - torch.exp(-_entry_lam(agent._d1(pg2), cfg.eta_end, signal_p=pg2) * cfg.dt)
               ).cpu().numpy()
    ax.plot(pg2.cpu().numpy(), qe2, "b-", lw=2, label="Entry q_α(p)")
    ax.axvline(cfg.signal_gate, color="orange", ls=":", lw=1.5, alpha=.7)
    ax.set(title="(d) Entry/Exit Probabilities", xlabel="Signal p", ylabel="Prob/day",
           ylim=(-0.05, 1.05))
    ax.legend(fontsize=8, loc="center left")

    plt.tight_layout(pad=2.0)
    plt.savefig("exploratory_rl_results.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Saved → exploratory_rl_results.png")

    elapsed = time.time() - t0
    print(f"\n{'='*W}")
    print(f"  RL Agent :  SR {ra['sr']:.2f} | Win {ra['wr']*100:.1f}% | "
          f"μ {ra['mu']*100:.2f}% | AvgH {ra['hd']:.0f}d | avg_ez {ra['avg_ez']:.2f}")
    print(f"  Baseline :  SR {ba['sr']:.2f} | Win {ba['wr']*100:.1f}% | "
          f"μ {ba['mu']*100:.2f}% | AvgH {ba['hd']:.0f}d | avg_ez {ba['avg_ez']:.2f}")
    print(f"  Buy&Hold :  {bnh_avg*100:.1f}% total | {bnh_avg*252/n_test*100:.1f}% ann")
    print(f"  Elapsed  :  {elapsed:.0f}s")
    print("=" * W)


if __name__ == "__main__":
    main()
