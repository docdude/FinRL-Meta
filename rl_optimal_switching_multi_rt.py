#!/usr/bin/env python3
"""
Multi-round-trip optimal switching extension of the exploratory RL framework.

Extends the single round-trip model in Zhao, Tse & Zheng (2026), arXiv:2604.02035v1,
to the general optimal switching problem described in their Section 5 conclusion:

  "Future work includes extending the model to multiple round-trip trades where
   one needs to analyze a more general optimal switching problem."

Key difference from the single-round-trip formulation:
  - The regime J no longer absorbs at J=2. After exit (J: 1→0), the agent
    re-enters the pre-entry state and can initiate another round trip.
  - The value function V₀ now accounts for unlimited future entry-exit cycles.
  - V₁(p, b) still represents the in-position value, but upon exit, the agent
    receives G(p, b) *plus* e^{-ρ·0}·V₀(p) (the continuation value from the
    next cycle, discounted to exit time).
  - The HJB system becomes fully coupled through this cycling link.

Mathematical formulation:
  ρ V₀(p) - L_P V₀(p) = η ln((η(e^{MΔ₁/η} - 1))/(MΔ₁))
  ρ V₁(p,b) - L_P V₁(p,b) = η ln((η(e^{MΔ₂/η} - 1))/(MΔ₂))

  where:
    Δ₁(p) = V₁(p,p) - V₀(p)        [entry advantage, same as single-RT]
    Δ₂(p,b) = G(p,b) + V₀(p) - V₁(p,b)  [exit advantage: now includes V₀(p)]

The exit advantage Δ₂ now includes V₀(p) because upon exiting at signal p,
the agent receives G(p,b) from the closed trade *and* enters a new pre-entry
state with continuation value V₀(p).
"""

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Cfg:
    gamma = 1.0
    iota = 1.0
    Psi = 0.0
    R = 1.0
    rho = 0.05

    theta = 0.1
    pbar = 0.0
    sigma = 0.2

    M = 50.0
    varpi = 0.5
    k_loss = 2.0
    dt = 0.1
    eta = 1e-5

    path_steps = 100
    eval_path_steps = 100
    long_eval_path_steps = 300
    train_paths = 512
    eval_paths = 256
    n_sims = 6
    batch_size = 32
    n_iter = 1200
    lr = 1e-3
    hidden = 32
    grad_clip = 1.0
    td_loss_mode = "joint_active"

    grid_lo = -3.2
    grid_hi = 3.2
    grid_size = 257
    b_lo = -4.0
    b_hi = 4.0
    hjb_p_min = -4.0
    hjb_p_max = 4.0
    hjb_b_min = -4.0
    hjb_b_max = 4.0
    hjb_step = 0.05
    hjb_tol = 1e-6
    hjb_max_iter = 600
    eval_b_values = (-1.0, 0.0, 1.0)
    seed = 42


cfg = Cfg()

PAPER_TAN_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_tan_blue",
    ["#1f5aa6", "#74a9cf", "#eef4f8", "#efe0cf"],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-round-trip optimal switching RL (extension of arXiv:2604.02035v1)"
    )
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke configuration")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--train-paths", type=int, default=None)
    parser.add_argument("--eval-paths", type=int, default=None)
    parser.add_argument(
        "--td-loss-mode",
        choices=("joint_active", "balanced_regimes"),
        default=None,
    )
    parser.add_argument("--out", default="multi_rt_results.png")
    return parser.parse_args()


def apply_args(args):
    if args.quick:
        cfg.train_paths = 128
        cfg.eval_paths = 64
        cfg.batch_size = 16
        cfg.n_iter = 120
        cfg.hjb_step = 0.1
        cfg.hjb_tol = 1e-5
        cfg.hjb_max_iter = 300
        cfg.long_eval_path_steps = 200
    if args.iters is not None:
        cfg.n_iter = args.iters
    if args.train_paths is not None:
        cfg.train_paths = args.train_paths
    if args.eval_paths is not None:
        cfg.eval_paths = args.eval_paths
    if args.td_loss_mode is not None:
        cfg.td_loss_mode = args.td_loss_mode
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


# ---------------------------------------------------------------------------
# Utility and payoff helpers (identical to single-RT)
# ---------------------------------------------------------------------------
def _U_t(x):
    a = torch.abs(x) + 1e-8
    return torch.where(x >= 0, a.pow(cfg.varpi), -cfg.k_loss * a.pow(cfg.varpi))


def _G_t(p, b):
    return _U_t(cfg.gamma * p - cfg.iota * b - cfg.Psi - cfg.R)


def _u_s(x):
    a = abs(x) + 1e-8
    return a ** cfg.varpi if x >= 0 else -cfg.k_loss * a ** cfg.varpi


def _U_np(x):
    a = np.abs(x) + 1e-8
    return np.where(x >= 0, a ** cfg.varpi, -cfg.k_loss * a ** cfg.varpi)


def _G_np(p, b):
    return _U_np(cfg.gamma * p - cfg.iota * b - cfg.Psi - cfg.R)


# ---------------------------------------------------------------------------
# HJB source term and mean intensity (same formulas, parameterized by M/eta)
# ---------------------------------------------------------------------------
def _hjb_src(delta, M=None, eta=None):
    M = cfg.M if M is None else M
    eta = cfg.eta if eta is None else eta
    z = M * delta / eta
    az = torch.abs(z)
    sa = az + 1e-6
    exact = eta * (torch.relu(z) + torch.log1p(-torch.exp(-sa)) - torch.log(sa))
    return torch.where(az < 0.1, M * delta / 2.0, exact)


def _mean_lam(delta, M=None, eta=None):
    M = cfg.M if M is None else M
    eta = cfg.eta if eta is None else eta
    z = M * delta / eta
    az = torch.abs(z)
    safe = torch.where(az > 0.1, z, torch.ones_like(z))
    exact = torch.clamp(M / (1 - torch.exp(-safe) + 1e-30) - M / safe, 0, M)
    taylor = torch.clamp(M / 2 + M * z / 12, 0, M)
    return torch.where(az > 0.1, exact, taylor)


def _ent_cost(delta):
    return torch.clamp(_mean_lam(delta) * delta - _hjb_src(delta), min=0)


def _hjb_src_np(delta, M=None, eta=None):
    M = cfg.M if M is None else M
    eta = cfg.eta if eta is None else eta
    z = M * delta / eta
    az = np.abs(z)
    sa = az + 1e-6
    exact = eta * (np.maximum(z, 0.0) + np.log1p(-np.exp(-sa)) - np.log(sa))
    return np.where(az < 0.1, M * delta / 2.0, exact)


def _mean_lam_np(delta, M=None, eta=None):
    M = cfg.M if M is None else M
    eta = cfg.eta if eta is None else eta
    z = M * delta / eta
    az = np.abs(z)
    safe = np.where(az > 0.1, z, np.ones_like(z))
    exact = np.clip(M / (1 - np.exp(-safe) + 1e-30) - M / safe, 0.0, M)
    taylor = np.clip(M / 2 + M * z / 12, 0.0, M)
    return np.where(az > 0.1, exact, taylor)


# ---------------------------------------------------------------------------
# OU path generation
# ---------------------------------------------------------------------------
def generate_ou_paths(n_paths, steps=None):
    steps = cfg.path_steps if steps is None else int(steps)
    paths = np.zeros((n_paths, steps + 1), dtype=np.float32)
    paths[:, 0] = np.random.uniform(cfg.b_lo, cfg.b_hi, size=n_paths).astype(np.float32)
    decay = np.exp(-cfg.theta * cfg.dt)
    innov_std = np.sqrt(
        (cfg.sigma ** 2) * (1 - np.exp(-2 * cfg.theta * cfg.dt)) / (2 * cfg.theta)
    )
    for step in range(steps):
        mean = cfg.pbar + (paths[:, step] - cfg.pbar) * decay
        paths[:, step + 1] = mean + innov_std * np.random.randn(n_paths)
    return paths.astype(np.float32)


# ---------------------------------------------------------------------------
# Tridiagonal solver and FD coefficients (same as single-RT)
# ---------------------------------------------------------------------------
def _solve_tridiagonal(lower, diag, upper, rhs):
    lower = lower.astype(np.float64).copy()
    diag = diag.astype(np.float64).copy()
    upper = upper.astype(np.float64).copy()
    rhs = rhs.astype(np.float64).copy()
    n = len(diag)
    for idx in range(1, n):
        factor = lower[idx] / diag[idx - 1]
        diag[idx] -= factor * upper[idx - 1]
        rhs[idx] -= factor * rhs[idx - 1]
    sol = np.empty(n, dtype=np.float64)
    sol[-1] = rhs[-1] / diag[-1]
    for idx in range(n - 2, -1, -1):
        sol[idx] = (rhs[idx] - upper[idx] * sol[idx + 1]) / diag[idx]
    return sol


def _ou_fd_coefficients(p_grid, step):
    drift = cfg.theta * (cfg.pbar - p_grid)
    diff = 0.5 * (cfg.sigma ** 2) / (step ** 2)
    conv = drift / (2 * step)
    lower = conv - diff
    diag = np.full_like(p_grid, cfg.rho + 2 * diff, dtype=np.float64)
    upper = -conv - diff
    return lower, diag, upper


def _solve_ou_line(lower_base, diag, upper_base, rhs, left_bc, right_bc):
    lower = lower_base.copy()
    diag = diag.copy()
    upper = upper_base.copy()
    rhs = rhs.copy()

    left_kind, left_value = left_bc
    if left_kind == "dirichlet":
        lower[0] = 0.0
        diag[0] = 1.0
        upper[0] = 0.0
        rhs[0] = left_value
    else:
        lower[0] = 0.0
        diag[0] = 1.0
        upper[0] = -1.0
        rhs[0] = left_value

    right_kind, right_value = right_bc
    if right_kind == "dirichlet":
        lower[-1] = 0.0
        diag[-1] = 1.0
        upper[-1] = 0.0
        rhs[-1] = right_value
    else:
        lower[-1] = -1.0
        diag[-1] = 1.0
        upper[-1] = 0.0
        rhs[-1] = right_value

    return _solve_tridiagonal(lower, diag, upper, rhs)


# ---------------------------------------------------------------------------
# Find boundary via linear interpolation
# ---------------------------------------------------------------------------
def find_boundary(grid, values, mode):
    if mode == "pos_to_neg":
        idx = np.where((values[:-1] >= 0) & (values[1:] < 0))[0]
    else:
        idx = np.where((values[:-1] < 0) & (values[1:] >= 0))[0]
    if len(idx) == 0:
        valid = np.where(values >= 0)[0]
        return float(grid[valid[-1]]) if len(valid) else np.nan
    i = int(idx[0])
    x0, x1 = float(grid[i]), float(grid[i + 1])
    y0, y1 = float(values[i]), float(values[i + 1])
    if abs(y1 - y0) < 1e-8:
        return x0
    return x0 - y0 * (x1 - x0) / (y1 - y0)


# ---------------------------------------------------------------------------
# Multi-round-trip HJB solver
#
# Key difference from single-RT:
#   Δ₂(p,b) = G(p,b) + V₀(p) - V₁(p,b)
#
# V₁ and V₀ are now fully coupled because:
#   - V₀ depends on V₁(p,p) through Δ₁
#   - V₁ depends on V₀(p) through Δ₂
#
# We solve by outer iteration, alternating V₁ and V₀ sweeps until joint
# convergence.
# ---------------------------------------------------------------------------
def solve_hjb_multi_rt():
    step = cfg.hjb_step
    p_grid = np.arange(cfg.hjb_p_min, cfg.hjb_p_max + 0.5 * step, step, dtype=np.float64)
    b_grid = np.arange(cfg.hjb_b_min, cfg.hjb_b_max + 0.5 * step, step, dtype=np.float64)
    lower_base, diag_base, upper_base = _ou_fd_coefficients(p_grid, step)

    p_mesh, b_mesh = np.meshgrid(p_grid, b_grid)
    payoff = _G_np(p_mesh, b_mesh)

    # Initialize V1 and V0
    v1 = np.maximum(payoff, 0.0)
    diag_v1 = np.diag(v1)  # V1(p, p)
    v0 = np.maximum(diag_v1, 0.0)

    outer_converged = False
    outer_error = np.inf
    total_inner_v1 = 0
    total_inner_v0 = 0

    for outer_iter in range(cfg.hjb_max_iter):
        prev_v0 = v0.copy()
        prev_v1 = v1.copy()

        # --- Inner sweep for V1 given current V0 ---
        # Δ₂(p,b) = G(p,b) + V₀(p) - V₁(p,b)  [multi-RT: adds V₀(p)]
        for inner_it in range(cfg.hjb_max_iter):
            prev_v1_inner = v1.copy()
            for row in range(len(b_grid)):
                # The exit advantage now includes V0(p) at each p
                delta = payoff[row] + v0 - v1[row]
                lam = _mean_lam_np(delta)
                rhs = _hjb_src_np(delta) + lam * v1[row]
                diag = diag_base + lam
                v1[row] = _solve_ou_line(
                    lower_base,
                    diag,
                    upper_base,
                    rhs,
                    left_bc=("neumann", 0.0),
                    right_bc=("dirichlet", float(payoff[row, -1] + v0[-1])),
                )
            v1_error = float(np.max(np.abs(v1 - prev_v1_inner)))
            total_inner_v1 += 1
            if v1_error < cfg.hjb_tol:
                break

        # Extract diagonal V1(p,p) for entry advantage
        if not (len(p_grid) == len(b_grid) and np.allclose(p_grid, b_grid)):
            raise ValueError("HJB p-grid and b-grid must align")
        diag_v1 = np.diag(v1)

        # --- Inner sweep for V0 given current V1 ---
        # Δ₁(p) = V₁(p,p) - V₀(p)  [same as single-RT]
        for inner_it in range(cfg.hjb_max_iter):
            prev_v0_inner = v0.copy()
            delta = diag_v1 - v0
            lam = _mean_lam_np(delta)
            rhs = _hjb_src_np(delta) + lam * v0
            diag = diag_base + lam
            v0 = _solve_ou_line(
                lower_base,
                diag,
                upper_base,
                rhs,
                left_bc=("dirichlet", float(diag_v1[0])),
                right_bc=("neumann", 0.0),
            )
            v0_error = float(np.max(np.abs(v0 - prev_v0_inner)))
            total_inner_v0 += 1
            if v0_error < cfg.hjb_tol:
                break

        # Check outer convergence
        outer_error = max(
            float(np.max(np.abs(v0 - prev_v0))),
            float(np.max(np.abs(v1 - prev_v1))),
        )
        if outer_error < cfg.hjb_tol:
            outer_converged = True
            break

    outer_iterations = outer_iter + 1

    # Compute advantages
    delta1 = diag_v1 - v0
    delta2 = payoff + v0[None, :] - v1  # multi-RT: G(p,b) + V0(p) - V1(p,b)

    entry_boundary = find_boundary(p_grid, delta1, mode="pos_to_neg")
    full_exit_boundary = np.array(
        [find_boundary(p_grid, row, mode="neg_to_pos") for row in delta2],
        dtype=np.float64,
    )
    slice_indices = {
        b_val: int(np.argmin(np.abs(b_grid - b_val))) for b_val in cfg.eval_b_values
    }
    exit_boundaries = {
        b_val: float(full_exit_boundary[idx]) for b_val, idx in slice_indices.items()
    }

    return {
        "p_grid": p_grid,
        "b_grid": b_grid,
        "payoff": payoff,
        "v0": v0,
        "v1": v1,
        "delta1": delta1,
        "delta2": delta2,
        "entry_boundary": float(entry_boundary),
        "full_exit_boundary": full_exit_boundary,
        "exit_boundaries": exit_boundaries,
        "slice_indices": slice_indices,
        "outer_iterations": outer_iterations,
        "outer_error": outer_error,
        "outer_converged": outer_converged,
        "total_inner_v1": total_inner_v1,
        "total_inner_v0": total_inner_v0,
    }


# ---------------------------------------------------------------------------
# Neural network value approximators
# ---------------------------------------------------------------------------
class V0Net(nn.Module):
    def __init__(self):
        super().__init__()
        h = cfg.hidden
        self.f = nn.Sequential(
            nn.Linear(1, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, p):
        return self.f(p).squeeze(-1)


class V1Net(nn.Module):
    def __init__(self):
        super().__init__()
        h = cfg.hidden
        self.f = nn.Sequential(
            nn.Linear(2, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, pb):
        return self.f(pb).squeeze(-1)


# ---------------------------------------------------------------------------
# Multi-round-trip RL Agent
# ---------------------------------------------------------------------------
class MultiRTAgent:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v0 = V0Net().to(self.dev)
        self.v1 = V1Net().to(self.dev)
        self.opt = torch.optim.Adam(
            list(self.v0.parameters()) + list(self.v1.parameters()),
            lr=cfg.lr,
        )

    def delta1(self, p):
        return self.v1(torch.stack([p, p], -1)) - self.v0(p.unsqueeze(-1))

    def delta2(self, p, b):
        """Exit advantage: G(p,b) + V0(p) - V1(p,b) [multi-RT coupling]."""
        return _G_t(p, b) + self.v0(p.unsqueeze(-1)) - self.v1(torch.stack([p, b], -1))

    @torch.no_grad()
    def _simulate_transitions(self, batch):
        """Simulate (J, B) trajectories allowing re-entry after exit.

        In the multi-RT formulation, J cycles: 0→1→0→1→...
        Each exit event records a completed round trip and resets J to 0.
        """
        num_paths, path_len = batch.shape
        steps = path_len - 1
        total = num_paths * cfg.n_sims
        paths = np.repeat(batch, cfg.n_sims, axis=0)

        J = np.zeros(total, dtype=np.int32)
        B = np.zeros(total, dtype=np.float32)

        # Half-init per Remark 4.1 (same rationale applies to multi-RT)
        half = total // 2
        J[half:] = 1
        B[half:] = np.random.uniform(cfg.b_lo, cfg.b_hi, size=total - half).astype(
            np.float32
        )

        R0 = []  # entry transitions
        R1 = []  # exit transitions

        for step in range(steps):
            p_now = paths[:, step]
            p_next = paths[:, step + 1]
            J_now = J.copy()
            B_now = B.copy()
            J_next = J_now.copy()
            B_next = B_now.copy()

            # Pre-entry regime
            m0 = J_now == 0
            if m0.any():
                ix = np.where(m0)[0]
                pt = torch.tensor(p_now[ix], dtype=torch.float32, device=self.dev)
                lam = _mean_lam(self.delta1(pt))
                q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
                enter = np.random.random(len(ix)) < q
                enter_ix = ix[enter]
                J_next[enter_ix] = 1
                B_next[enter_ix] = p_now[enter_ix]
                for idx in ix:
                    # Multi-RT: after a no-entry step, next state is still J=0 or J=1
                    R0.append(
                        (
                            float(p_now[idx]),
                            float(p_next[idx]),
                            int(J_next[idx]),
                            float(B_next[idx]),
                        )
                    )

            # In-position regime
            m1 = J_now == 1
            if m1.any():
                ix = np.where(m1)[0]
                pt = torch.tensor(p_now[ix], dtype=torch.float32, device=self.dev)
                bt = torch.tensor(B_now[ix], dtype=torch.float32, device=self.dev)
                lam = _mean_lam(self.delta2(pt, bt))
                q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
                exit_mask = np.random.random(len(ix)) < q
                exit_ix = ix[exit_mask]
                # Multi-RT: after exit, J returns to 0 (not absorbed at 2)
                J_next[exit_ix] = 0
                B_next[exit_ix] = 0.0  # reset B
                for idx in ix:
                    R1.append(
                        (
                            float(p_now[idx]),
                            float(B_now[idx]),
                            float(p_next[idx]),
                            int(J_next[idx]),
                            float(B_next[idx]),
                        )
                    )

            J = J_next
            B = B_next

        return R0, R1

    def _loss_v0(self, transitions):
        """TD error for pre-entry regime.

        Same structure as single-RT: the Bellman target for J=0 is
        V₀(p') if stayed out, or V₁(p', b') if entered.
        """
        if not transitions:
            return torch.tensor(0.0, device=self.dev)
        p = torch.tensor([r[0] for r in transitions], dtype=torch.float32, device=self.dev)
        pn = torch.tensor([r[1] for r in transitions], dtype=torch.float32, device=self.dev)
        jn = torch.tensor([r[2] for r in transitions], dtype=torch.long, device=self.dev)
        bn = torch.tensor([r[3] for r in transitions], dtype=torch.float32, device=self.dev)

        v0_now = self.v0(p.unsqueeze(-1))
        ca = _ent_cost(self.delta1(p))
        discount = np.exp(-cfg.rho * cfg.dt)

        with torch.no_grad():
            v0_next = self.v0(pn.unsqueeze(-1))
            v1_next = self.v1(torch.stack([pn, bn], -1))
            target = torch.where(jn == 0, v0_next, v1_next)

        td = -ca.detach() * cfg.dt + discount * target - v0_now
        return (td ** 2).mean()

    def _loss_v1(self, transitions):
        """TD error for in-position regime.

        Multi-RT key change: upon exit (J_next=0), the Bellman target is
          G(p', b') + V₀(p')
        rather than just G(p', b'), because the agent continues from
        the pre-entry state.
        """
        if not transitions:
            return torch.tensor(0.0, device=self.dev)
        p = torch.tensor([r[0] for r in transitions], dtype=torch.float32, device=self.dev)
        b = torch.tensor([r[1] for r in transitions], dtype=torch.float32, device=self.dev)
        pn = torch.tensor([r[2] for r in transitions], dtype=torch.float32, device=self.dev)
        jn = torch.tensor([r[3] for r in transitions], dtype=torch.long, device=self.dev)
        bn = torch.tensor([r[4] for r in transitions], dtype=torch.float32, device=self.dev)

        v1_now = self.v1(torch.stack([p, b], -1))
        cb = _ent_cost(self.delta2(p, b))
        discount = np.exp(-cfg.rho * cfg.dt)

        with torch.no_grad():
            v1_next = self.v1(torch.stack([pn, bn], -1))
            g_next = _G_t(pn, bn)
            v0_next = self.v0(pn.unsqueeze(-1))
            # Multi-RT: exit target = G(p', b') + V0(p')
            exit_target = g_next + v0_next
            target = torch.where(jn == 1, v1_next, exit_target)

        td = -cb.detach() * cfg.dt + discount * target - v1_now
        return (td ** 2).mean()

    def _combine_td_losses(self, loss0, loss1, n0, n1):
        if cfg.td_loss_mode == "balanced_regimes":
            if n0 > 0 and n1 > 0:
                return 0.5 * (loss0 + loss1)
            if n0 > 0:
                return loss0
            if n1 > 0:
                return loss1
            return loss0 + loss1
        active_td = n0 + n1
        if active_td > 0:
            return (loss0 * n0 + loss1 * n1) / active_td
        return loss0 + loss1

    def train(self, paths):
        losses = []
        total_paths = len(paths)
        print(
            "Training multi-round-trip optimal switching "
            f"(paths={total_paths}, steps={cfg.path_steps}, M={cfg.M}, "
            f"eta={cfg.eta}, td_loss_mode={cfg.td_loss_mode})"
        )
        for it in range(cfg.n_iter):
            idx = np.random.choice(total_paths, min(cfg.batch_size, total_paths), replace=False)
            R0, R1 = self._simulate_transitions(paths[idx])
            loss0 = self._loss_v0(R0)
            loss1 = self._loss_v1(R1)
            loss = self._combine_td_losses(loss0, loss1, len(R0), len(R1))
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.v0.parameters()) + list(self.v1.parameters()), cfg.grad_clip
            )
            self.opt.step()
            losses.append(float(loss.item()))
            if (it + 1) % max(1, cfg.n_iter // 4) == 0:
                print(f"  iter {it+1}/{cfg.n_iter} loss={np.mean(losses[-25:]):.6f}")
        return losses

    @torch.no_grad()
    def boundary_summary(self):
        grid = torch.linspace(cfg.grid_lo, cfg.grid_hi, cfg.grid_size, device=self.dev)
        delta1 = self.delta1(grid).cpu().numpy()
        entry_boundary = find_boundary(grid.cpu().numpy(), delta1, mode="pos_to_neg")
        exit_boundaries = {}
        for b_val in cfg.eval_b_values:
            b = torch.full_like(grid, b_val)
            delta2 = self.delta2(grid, b).cpu().numpy()
            exit_boundaries[b_val] = find_boundary(
                grid.cpu().numpy(), delta2, mode="neg_to_pos"
            )
        return entry_boundary, exit_boundaries

    @torch.no_grad()
    def rollout_path(self, path, rng):
        """Roll out the multi-RT policy on a single path.

        The agent can complete multiple round trips within one path.
        """
        state = 0  # J: 0 = pre-entry, 1 = in-position
        entry_signal = 0.0
        completed_trades = []
        entries = []
        horizon_steps = len(path) - 1

        for step in range(horizon_steps):
            p_now = float(path[step])
            p_next = float(path[step + 1])

            if state == 0:
                pt = torch.tensor([p_now], dtype=torch.float32, device=self.dev)
                lam = float(_mean_lam(self.delta1(pt)).item())
                q = 1 - np.exp(-lam * cfg.dt)
                if rng.random() < q:
                    state = 1
                    entry_signal = p_now
                    entries.append({"step": step, "signal": p_now})

            elif state == 1:
                pt = torch.tensor([p_now], dtype=torch.float32, device=self.dev)
                bt = torch.tensor([entry_signal], dtype=torch.float32, device=self.dev)
                lam = float(_mean_lam(self.delta2(pt, bt)).item())
                q = 1 - np.exp(-lam * cfg.dt)
                if rng.random() < q:
                    utility = _u_s(
                        cfg.gamma * p_next - cfg.iota * entry_signal - cfg.Psi - cfg.R
                    )
                    completed_trades.append(
                        {
                            "entry_step": entries[-1]["step"],
                            "exit_step": step + 1,
                            "entry_signal": entry_signal,
                            "exit_signal": p_next,
                            "hold_steps": step + 1 - entries[-1]["step"],
                            "utility": utility,
                        }
                    )
                    state = 0  # re-enter pre-entry regime

        open_at_horizon = state == 1
        total_utility = sum(t["utility"] for t in completed_trades)

        return {
            "n_entries": len(entries),
            "n_completed": len(completed_trades),
            "open_at_horizon": int(open_at_horizon),
            "total_utility": total_utility,
            "avg_utility_per_trade": (
                total_utility / len(completed_trades) if completed_trades else 0.0
            ),
            "avg_hold_steps": (
                np.mean([t["hold_steps"] for t in completed_trades])
                if completed_trades
                else 0.0
            ),
            "trades": completed_trades,
        }

    @torch.no_grad()
    def evaluate(self, paths):
        rng = np.random.default_rng(cfg.seed + 7)
        rows = [self.rollout_path(path, rng) for path in paths]

        n_entries = np.array([r["n_entries"] for r in rows], dtype=float)
        n_completed = np.array([r["n_completed"] for r in rows], dtype=float)
        open_at_horizon = np.array([r["open_at_horizon"] for r in rows], dtype=float)
        total_utilities = np.array([r["total_utility"] for r in rows], dtype=float)

        all_trades = [t for r in rows for t in r["trades"]]
        holds = np.array([t["hold_steps"] for t in all_trades], dtype=float) if all_trades else np.array([])
        trade_utils = np.array([t["utility"] for t in all_trades], dtype=float) if all_trades else np.array([])

        return {
            "avg_entries_per_path": float(n_entries.mean()),
            "avg_completed_per_path": float(n_completed.mean()),
            "open_at_horizon_rate": float(open_at_horizon.mean()),
            "avg_total_utility": float(total_utilities.mean()),
            "avg_utility_per_trade": float(trade_utils.mean()) if len(trade_utils) else 0.0,
            "avg_hold_steps": float(holds.mean()) if len(holds) else 0.0,
            "total_trades": len(all_trades),
            "paths_with_trades": int((n_completed > 0).sum()),
            "max_trades_per_path": int(n_completed.max()) if len(n_completed) else 0,
        }


# ---------------------------------------------------------------------------
# Compare RL to HJB
# ---------------------------------------------------------------------------
@torch.no_grad()
def compare_agent_to_hjb(agent, benchmark):
    p_grid = benchmark["p_grid"]
    b_grid = benchmark["b_grid"]
    p_idx = np.where((p_grid >= cfg.grid_lo) & (p_grid <= cfg.grid_hi))[0]
    b_idx = np.where((b_grid >= cfg.grid_lo) & (b_grid <= cfg.grid_hi))[0]

    p_t = torch.tensor(p_grid, dtype=torch.float32, device=agent.dev)
    v0_rl = agent.v0(p_t.unsqueeze(-1)).cpu().numpy()
    diag_v1_rl = agent.v1(torch.stack([p_t, p_t], -1)).cpu().numpy()
    delta1_rl = diag_v1_rl - v0_rl

    p_mesh, b_mesh = np.meshgrid(p_grid, b_grid)
    pb = np.column_stack([p_mesh.ravel(), b_mesh.ravel()])
    pb_t = torch.tensor(pb, dtype=torch.float32, device=agent.dev)
    v1_rl = agent.v1(pb_t).cpu().numpy().reshape(len(b_grid), len(p_grid))

    # Multi-RT exit advantage: G(p,b) + V0(p) - V1(p,b)
    delta2_rl = benchmark["payoff"] + v0_rl[None, :] - v1_rl

    v0_diff = np.abs(v0_rl - benchmark["v0"])
    v1_diff = np.abs(v1_rl - benchmark["v1"])

    exit_boundaries_rl = {}
    for b_val, row in benchmark["slice_indices"].items():
        exit_boundaries_rl[b_val] = float(
            find_boundary(p_grid, delta2_rl[row], mode="neg_to_pos")
        )

    return {
        "v0_rl": v0_rl,
        "v1_rl": v1_rl,
        "delta1_rl": delta1_rl,
        "delta2_rl": delta2_rl,
        "v0_mae": float(v0_diff[p_idx].mean()),
        "v0_max": float(v0_diff[p_idx].max()),
        "v1_mae": float(v1_diff[np.ix_(b_idx, p_idx)].mean()),
        "v1_max": float(v1_diff[np.ix_(b_idx, p_idx)].max()),
        "entry_boundary_rl": float(find_boundary(p_grid, delta1_rl, mode="pos_to_neg")),
        "exit_boundaries_rl": exit_boundaries_rl,
        "v1_abs_error": v1_diff,
        "interior_p_idx": p_idx,
        "interior_b_idx": b_idx,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _with_suffix(path, suffix):
    stem, ext = os.path.splitext(path)
    return f"{stem}{suffix}{ext or '.png'}"


def plot_results(losses, benchmark, comparison, out_path):
    p_grid = benchmark["p_grid"]
    b_grid = benchmark["b_grid"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multi-Round-Trip Optimal Switching", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(losses, color="steelblue", alpha=0.35, lw=1.0)
    if len(losses) > 10:
        window = min(25, len(losses))
        kernel = np.ones(window) / window
        smooth = np.convolve(losses, kernel, mode="valid")
        ax.plot(np.arange(window - 1, len(losses)), smooth, color="darkred", lw=1.5)
    ax.set(title="(a) Training loss", xlabel="Iteration", ylabel="TD loss", yscale="log")

    ax = axes[0, 1]
    ax.plot(p_grid, benchmark["v0"], color="black", ls="--", lw=2.0, label="HJB V0")
    ax.plot(p_grid, comparison["v0_rl"], color="steelblue", lw=1.8, label="RL V0")
    ax.set(title="(b) V0 comparison (multi-RT)", xlabel="Signal p", ylabel="Value")
    ax.set_xlim(cfg.grid_lo, cfg.grid_hi)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 2]
    ax.plot(p_grid, benchmark["delta1"], color="black", ls="--", lw=2.0, label="HJB Δ₁")
    ax.plot(p_grid, comparison["delta1_rl"], color="crimson", lw=1.6, label="RL Δ₁")
    ax.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax.axvline(benchmark["entry_boundary"], color="black", ls=":", lw=1.5, label="HJB entry bdy")
    ax.axvline(comparison["entry_boundary_rl"], color="green", ls=":", lw=1.5, label="RL entry bdy")
    ax.set(title="(c) Entry advantage and boundary", xlabel="Signal p")
    ax.set_xlim(cfg.grid_lo, cfg.grid_hi)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 0]
    for b_val, color, style in [(-1.0, "darkred", "-"), (0.0, "coral", "--"), (1.0, "purple", ":")]:
        row = benchmark["slice_indices"][b_val]
        ax.plot(
            p_grid, benchmark["delta2"][row],
            linestyle=style, color=color, lw=1.8, alpha=0.9,
            label=f"HJB Δ₂(p, b={b_val:.1f})",
        )
        ax.plot(
            p_grid, comparison["delta2_rl"][row],
            linestyle=style, color=color, lw=1.2, alpha=0.45,
        )
    ax.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax.set(title="(d) Exit advantage slices (multi-RT)", xlabel="Signal p")
    ax.set_xlim(cfg.grid_lo, cfg.grid_hi)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 1]
    ax.plot(b_grid, benchmark["full_exit_boundary"], color="black", ls="--", lw=2.0, label="HJB p*(b)")
    hjb_x = list(benchmark["exit_boundaries"].keys())
    hjb_y = list(benchmark["exit_boundaries"].values())
    rl_x = list(comparison["exit_boundaries_rl"].keys())
    rl_y = list(comparison["exit_boundaries_rl"].values())
    ax.scatter(hjb_x, hjb_y, color="black", s=50, marker="x", label="HJB sampled")
    ax.scatter(rl_x, rl_y, color="royalblue", s=55, label="RL sampled")
    ax.set(
        title="(e) Exit free boundary (multi-RT)",
        xlabel="Entry signal b",
        ylabel="Exit boundary p*(b)",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    entry_gap = abs(comparison["entry_boundary_rl"] - benchmark["entry_boundary"])
    ax.text(
        0.02, 0.98,
        f"V0 MAE={comparison['v0_mae']:.4f}\nV1 MAE={comparison['v1_mae']:.4f}\nentry gap={entry_gap:.4f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 2]
    p_idx = comparison["interior_p_idx"]
    b_idx = comparison["interior_b_idx"]
    p_lo = p_grid[p_idx[0]]
    p_hi = p_grid[p_idx[-1]]
    error_map = (comparison["v1_rl"] - benchmark["v1"])[np.ix_(b_idx, p_idx)]
    im = ax.imshow(
        error_map.T,
        origin="lower", aspect="auto",
        extent=[b_grid[b_idx[0]], b_grid[b_idx[-1]], p_lo, p_hi],
        cmap=PAPER_TAN_BLUE_CMAP,
    )
    boundary_curve = benchmark["full_exit_boundary"][b_idx]
    boundary_mask = (
        np.isfinite(boundary_curve)
        & (boundary_curve >= p_lo)
        & (boundary_curve <= p_hi)
    )
    ax.plot(
        b_grid[b_idx][boundary_mask], boundary_curve[boundary_mask],
        color="black", ls="--", lw=1.8, alpha=0.95,
    )
    ax.set(title="(f) V1 RL−HJB on interior", xlabel="Entry signal b", ylabel="Signal p")
    ax.set_ylim(p_lo, p_hi)
    fig.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_single_vs_multi_comparison(single_bench, multi_bench, out_path):
    """Compare single-RT and multi-RT HJB value functions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Single vs Multi Round-Trip Value Functions", fontsize=14, fontweight="bold")

    p_grid = multi_bench["p_grid"]
    p_idx = np.where((p_grid >= cfg.grid_lo) & (p_grid <= cfg.grid_hi))[0]
    p_view = p_grid[p_idx]

    # V0 comparison
    ax = axes[0]
    ax.plot(p_view, single_bench["v0"][p_idx], color="black", ls="--", lw=2.0, label="Single-RT V₀")
    ax.plot(p_view, multi_bench["v0"][p_idx], color="royalblue", lw=1.8, label="Multi-RT V₀")
    ax.set(title="(a) V₀ comparison", xlabel="Signal p", ylabel="Value")
    ax.legend(loc="best", fontsize=9)

    # V0 difference
    ax = axes[1]
    v0_diff = multi_bench["v0"][p_idx] - single_bench["v0"][p_idx]
    ax.plot(p_view, v0_diff, color="darkgreen", lw=1.8)
    ax.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax.set(title="(b) V₀(multi) − V₀(single)", xlabel="Signal p", ylabel="Value difference")
    ax.text(
        0.02, 0.98,
        f"max diff = {v0_diff.max():.4f}\nmean diff = {v0_diff.mean():.4f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    # Exit boundary comparison
    ax = axes[2]
    b_grid = multi_bench["b_grid"]
    b_idx = np.where((b_grid >= cfg.grid_lo) & (b_grid <= cfg.grid_hi))[0]
    b_view = b_grid[b_idx]
    ax.plot(b_view, single_bench["full_exit_boundary"][b_idx], color="black", ls="--", lw=2.0, label="Single-RT p*(b)")
    ax.plot(b_view, multi_bench["full_exit_boundary"][b_idx], color="royalblue", lw=1.8, label="Multi-RT p*(b)")
    ax.set(
        title="(c) Exit free boundary comparison",
        xlabel="Entry signal b",
        ylabel="Exit boundary p*(b)",
    )
    ax.legend(loc="best", fontsize=9)

    single_entry = single_bench["entry_boundary"]
    multi_entry = multi_bench["entry_boundary"]
    ax.text(
        0.02, 0.02,
        f"Entry bdy: single={single_entry:.3f}, multi={multi_entry:.3f}",
        transform=ax.transAxes, va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-RT HJB solver (imported logic for comparison)
# ---------------------------------------------------------------------------
def solve_hjb_single_rt():
    """Solve the original single-round-trip HJB for comparison."""
    step = cfg.hjb_step
    p_grid = np.arange(cfg.hjb_p_min, cfg.hjb_p_max + 0.5 * step, step, dtype=np.float64)
    b_grid = np.arange(cfg.hjb_b_min, cfg.hjb_b_max + 0.5 * step, step, dtype=np.float64)
    lower_base, diag_base, upper_base = _ou_fd_coefficients(p_grid, step)

    p_mesh, b_mesh = np.meshgrid(p_grid, b_grid)
    payoff = _G_np(p_mesh, b_mesh)

    v1 = np.maximum(payoff, 0.0)
    for iteration in range(cfg.hjb_max_iter):
        prev = v1.copy()
        for row in range(len(b_grid)):
            delta = payoff[row] - prev[row]
            lam = _mean_lam_np(delta)
            rhs = _hjb_src_np(delta) + lam * prev[row]
            diag = diag_base + lam
            v1[row] = _solve_ou_line(
                lower_base, diag, upper_base, rhs,
                left_bc=("neumann", 0.0),
                right_bc=("dirichlet", float(payoff[row, -1])),
            )
        if float(np.max(np.abs(v1 - prev))) < cfg.hjb_tol:
            break

    diag_v1 = np.diag(v1)
    v0 = np.maximum(diag_v1, 0.0)
    for iteration in range(cfg.hjb_max_iter):
        prev = v0.copy()
        delta = diag_v1 - prev
        lam = _mean_lam_np(delta)
        rhs = _hjb_src_np(delta) + lam * prev
        diag = diag_base + lam
        v0 = _solve_ou_line(
            lower_base, diag, upper_base, rhs,
            left_bc=("dirichlet", float(diag_v1[0])),
            right_bc=("neumann", 0.0),
        )
        if float(np.max(np.abs(v0 - prev))) < cfg.hjb_tol:
            break

    delta1 = diag_v1 - v0
    delta2 = payoff - v1
    entry_boundary = find_boundary(p_grid, delta1, mode="pos_to_neg")
    full_exit_boundary = np.array(
        [find_boundary(p_grid, row, mode="neg_to_pos") for row in delta2],
        dtype=np.float64,
    )
    slice_indices = {b_val: int(np.argmin(np.abs(b_grid - b_val))) for b_val in cfg.eval_b_values}
    exit_boundaries = {b_val: float(full_exit_boundary[idx]) for b_val, idx in slice_indices.items()}

    return {
        "p_grid": p_grid, "b_grid": b_grid, "payoff": payoff,
        "v0": v0, "v1": v1, "delta1": delta1, "delta2": delta2,
        "entry_boundary": float(entry_boundary),
        "full_exit_boundary": full_exit_boundary,
        "exit_boundaries": exit_boundaries,
        "slice_indices": slice_indices,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    apply_args(args)

    t0 = time.time()
    print("=" * 80)
    print(" Multi-round-trip optimal switching RL")
    print(" Extension of Zhao, Tse & Zheng (2026) · Section 5 future work")
    print("=" * 80)
    print(
        f"Config: dt={cfg.dt}, steps={cfg.path_steps}, train_paths={cfg.train_paths}, "
        f"eval_paths={cfg.eval_paths}, M={cfg.M}, eta={cfg.eta}"
    )

    train_paths = generate_ou_paths(cfg.train_paths)
    eval_paths = generate_ou_paths(cfg.eval_paths, steps=cfg.eval_path_steps)

    # --- Solve both HJB systems ---
    print("\n[1/5] Solving single-RT HJB benchmark …")
    single_t0 = time.time()
    single_bench = solve_hjb_single_rt()
    print(f"  Single-RT entry boundary p* ≈ {single_bench['entry_boundary']:.3f}")
    for b_val, bdy in single_bench["exit_boundaries"].items():
        print(f"  Single-RT exit boundary for b={b_val:.1f}: p*(b) ≈ {bdy:.3f}")
    print(f"  Elapsed: {time.time() - single_t0:.1f}s")

    print("\n[2/5] Solving multi-RT HJB benchmark …")
    multi_t0 = time.time()
    multi_bench = solve_hjb_multi_rt()
    print(
        f"  Outer iterations: {multi_bench['outer_iterations']} "
        f"(max change {multi_bench['outer_error']:.2e}, converged={multi_bench['outer_converged']})"
    )
    print(
        f"  Total inner V1 sweeps: {multi_bench['total_inner_v1']}, "
        f"V0 sweeps: {multi_bench['total_inner_v0']}"
    )
    print(f"  Multi-RT entry boundary p* ≈ {multi_bench['entry_boundary']:.3f}")
    for b_val, bdy in multi_bench["exit_boundaries"].items():
        print(f"  Multi-RT exit boundary for b={b_val:.1f}: p*(b) ≈ {bdy:.3f}")
    print(f"  Elapsed: {time.time() - multi_t0:.1f}s")

    # --- Compare single vs multi HJB ---
    print("\n[3/5] Comparing single-RT vs multi-RT HJB solutions …")
    p_idx = np.where(
        (multi_bench["p_grid"] >= cfg.grid_lo) & (multi_bench["p_grid"] <= cfg.grid_hi)
    )[0]
    v0_uplift = multi_bench["v0"][p_idx] - single_bench["v0"][p_idx]
    print(f"  V0 uplift (multi − single): mean={v0_uplift.mean():.4f}, max={v0_uplift.max():.4f}")
    print(
        f"  Entry boundary shift: {multi_bench['entry_boundary']:.3f} vs "
        f"{single_bench['entry_boundary']:.3f} "
        f"(delta={multi_bench['entry_boundary'] - single_bench['entry_boundary']:.4f})"
    )
    for b_val in cfg.eval_b_values:
        s = single_bench["exit_boundaries"][b_val]
        m = multi_bench["exit_boundaries"][b_val]
        print(f"  Exit boundary shift b={b_val:.1f}: {m:.3f} vs {s:.3f} (delta={m-s:.4f})")

    # --- Train multi-RT RL agent ---
    print("\n[4/5] Training multi-RT offline policy iteration …")
    agent = MultiRTAgent()
    losses = agent.train(train_paths)
    entry_boundary, exit_boundaries = agent.boundary_summary()
    comparison = compare_agent_to_hjb(agent, multi_bench)

    print("\n[5/5] Evaluating multi-RT RL agent …")
    evaluation = agent.evaluate(eval_paths)
    long_eval_paths = generate_ou_paths(cfg.eval_paths, steps=cfg.long_eval_path_steps)
    long_eval = agent.evaluate(long_eval_paths)

    print("\nRL boundary summary (multi-RT):")
    print(f"  RL entry boundary p* ≈ {entry_boundary:.3f}")
    for b_val, bdy in exit_boundaries.items():
        print(f"  RL exit boundary for b={b_val:.1f}: p*(b) ≈ {bdy:.3f}")

    print("\nRL vs multi-RT HJB comparison on interior:")
    print(f"  V0 MAE: {comparison['v0_mae']:.5f}, max: {comparison['v0_max']:.5f}")
    print(f"  V1 MAE: {comparison['v1_mae']:.5f}, max: {comparison['v1_max']:.5f}")
    print(
        f"  entry boundary gap: "
        f"{abs(comparison['entry_boundary_rl'] - multi_bench['entry_boundary']):.5f}"
    )
    for b_val in cfg.eval_b_values:
        gap = abs(
            comparison["exit_boundaries_rl"][b_val] - multi_bench["exit_boundaries"][b_val]
        )
        print(f"  exit boundary gap for b={b_val:.1f}: {gap:.5f}")

    print(f"\nMonte Carlo policy summary ({cfg.eval_path_steps} steps):")
    print(f"  avg entries/path  : {evaluation['avg_entries_per_path']:.2f}")
    print(f"  avg completed/path: {evaluation['avg_completed_per_path']:.2f}")
    print(f"  open at horizon   : {evaluation['open_at_horizon_rate']*100:.1f}%")
    print(f"  avg total utility : {evaluation['avg_total_utility']:.4f}")
    print(f"  avg util/trade    : {evaluation['avg_utility_per_trade']:.4f}")
    print(f"  avg hold steps    : {evaluation['avg_hold_steps']:.2f}")
    print(f"  total trades      : {evaluation['total_trades']}")
    print(f"  max trades/path   : {evaluation['max_trades_per_path']}")

    print(f"\nLong-horizon summary ({cfg.long_eval_path_steps} steps):")
    print(f"  avg entries/path  : {long_eval['avg_entries_per_path']:.2f}")
    print(f"  avg completed/path: {long_eval['avg_completed_per_path']:.2f}")
    print(f"  avg total utility : {long_eval['avg_total_utility']:.4f}")
    print(f"  avg util/trade    : {long_eval['avg_utility_per_trade']:.4f}")
    print(f"  total trades      : {long_eval['total_trades']}")
    print(f"  max trades/path   : {long_eval['max_trades_per_path']}")

    # --- Plots ---
    plot_results(losses, multi_bench, comparison, args.out)
    comparison_out = _with_suffix(args.out, "_single_vs_multi")
    plot_single_vs_multi_comparison(single_bench, multi_bench, comparison_out)

    print(f"\nSaved multi-RT results -> {args.out}")
    print(f"Saved single vs multi comparison -> {comparison_out}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
