#!/usr/bin/env python3
"""
OU reproduction of Section 4 in Zhao, Tse & Zheng (2026), arXiv:2604.02035v1.

This script keeps the paper-oriented pieces separate from the Nasdaq adaptation:
- offline OU signal paths instead of downloaded market data
- paper baseline parameters by default
- finite-difference HJB benchmark for the OU model
- pure Gibbs mean intensities with no signal-side heuristics
- one-step TD policy iteration on simulated (J, B) trajectories
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
    density_plot_eta = 1e-4

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
    density_lam_points = 5001
    b_lo = -4.0
    b_hi = 4.0
    hjb_p_min = -4.0
    hjb_p_max = 4.0
    hjb_b_min = -4.0
    hjb_b_max = 4.0
    hjb_step = 0.05
    hjb_tol = 1e-6
    hjb_max_iter = 400
    eval_b_values = (-1.0, 0.0, 1.0)
    sweep_M_values = (1.0, 5.0, 10.0, 50.0)
    sweep_eta_values = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    sweep_theta_values = (0.05, 0.1, 0.2, 0.4)
    sweep_sigma_values = (0.1, 0.2, 0.3, 0.4)
    seed = 42


cfg = Cfg()

PAPER_TAN_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_tan_blue",
    ["#1f5aa6", "#74a9cf", "#eef4f8", "#efe0cf"],
)


def parse_args():
    parser = argparse.ArgumentParser(description="OU reproduction of the optimal stopping paper")
    parser.add_argument("--quick", action="store_true", help="Run a smaller smoke configuration")
    parser.add_argument("--iters", type=int, default=None, help="Override training iterations")
    parser.add_argument("--train-paths", type=int, default=None, help="Override number of training paths")
    parser.add_argument("--eval-paths", type=int, default=None, help="Override number of evaluation paths")
    parser.add_argument(
        "--td-loss-mode",
        choices=("joint_active", "balanced_regimes"),
        default=None,
        help="Loss aggregation for entry/exit TD errors",
    )
    parser.add_argument("--out", default="ou_reproduction_results.png", help="Output plot path")
    return parser.parse_args()


def apply_args(args):
    if args.quick:
        cfg.train_paths = 128
        cfg.eval_paths = 64
        cfg.batch_size = 16
        cfg.n_iter = 120
        cfg.hjb_step = 0.1
        cfg.hjb_tol = 1e-5
        cfg.hjb_max_iter = 200
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


def _U_t(x):
    a = torch.abs(x) + 1e-8
    return torch.where(x >= 0, a.pow(cfg.varpi), -cfg.k_loss * a.pow(cfg.varpi))


def _G_t(p, b):
    return _U_t(cfg.gamma * p - cfg.iota * b - cfg.Psi - cfg.R)


def _u_s(x):
    a = abs(x) + 1e-8
    return a ** cfg.varpi if x >= 0 else -cfg.k_loss * a ** cfg.varpi


def _hjb_src(delta):
    z = cfg.M * delta / cfg.eta
    az = torch.abs(z)
    sa = az + 1e-6
    exact = cfg.eta * (torch.relu(z) + torch.log1p(-torch.exp(-sa)) - torch.log(sa))
    return torch.where(az < 0.1, cfg.M * delta / 2.0, exact)


def _mean_lam(delta):
    z = cfg.M * delta / cfg.eta
    az = torch.abs(z)
    safe = torch.where(az > 0.1, z, torch.ones_like(z))
    exact = torch.clamp(cfg.M / (1 - torch.exp(-safe) + 1e-30) - cfg.M / safe, 0, cfg.M)
    taylor = torch.clamp(cfg.M / 2 + cfg.M * z / 12, 0, cfg.M)
    return torch.where(az > 0.1, exact, taylor)


def _ent_cost(delta):
    return torch.clamp(_mean_lam(delta) * delta - _hjb_src(delta), min=0)


def _U_np(x):
    a = np.abs(x) + 1e-8
    return np.where(x >= 0, a ** cfg.varpi, -cfg.k_loss * a ** cfg.varpi)


def _G_np(p, b):
    return _U_np(cfg.gamma * p - cfg.iota * b - cfg.Psi - cfg.R)


def _hjb_src_np(delta):
    z = cfg.M * delta / cfg.eta
    az = np.abs(z)
    sa = az + 1e-6
    exact = cfg.eta * (np.maximum(z, 0.0) + np.log1p(-np.exp(-sa)) - np.log(sa))
    return np.where(az < 0.1, cfg.M * delta / 2.0, exact)


def _mean_lam_np(delta):
    z = cfg.M * delta / cfg.eta
    az = np.abs(z)
    safe = np.where(az > 0.1, z, np.ones_like(z))
    exact = np.clip(cfg.M / (1 - np.exp(-safe) + 1e-30) - cfg.M / safe, 0.0, cfg.M)
    taylor = np.clip(cfg.M / 2 + cfg.M * z / 12, 0.0, cfg.M)
    return np.where(az > 0.1, exact, taylor)


def _gibbs_density_np(delta, lam_grid, eta=None):
    eta = cfg.eta if eta is None else eta
    delta = np.asarray(delta, dtype=np.float64)
    lam_grid = np.asarray(lam_grid, dtype=np.float64)
    scores = np.outer(delta / eta, lam_grid)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    norm = np.trapezoid(weights, lam_grid, axis=1)
    return weights / np.clip(norm[:, None], 1e-300, None)


def _density_ridge(density, lam_grid):
    return lam_grid[np.argmax(density, axis=1)]


def _with_suffix(path, suffix):
    stem, ext = os.path.splitext(path)
    return f"{stem}{suffix}{ext or '.png'}"


def _density_display(density, floor=1e-30, ceiling=None):
    vmax = max(float(ceiling), floor * 10) if ceiling is not None else max(float(density.max()), floor * 10)
    clipped = np.clip(density, floor, vmax)
    return clipped, LogNorm(vmin=floor, vmax=vmax)


def _set_log_colorbar_ticks(cbar, top_exp, bottom_exp=-30, step=-5):
    exponents = np.arange(top_exp, bottom_exp - 1, step)
    cbar.set_ticks(10.0 ** exponents)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())


def _density_lam_grid(max_lam, points=None):
    points = cfg.density_lam_points if points is None else int(points)
    return np.linspace(0.0, max_lam, points)


def _entry_density_case(result, M_val, eta_val):
    p_idx, _ = _interior_indices(result)
    p_view = result["p_grid"][p_idx]
    delta = result["delta1"][p_idx]
    lam_grid = _density_lam_grid(M_val)
    density = _gibbs_density_np(delta, lam_grid, eta=eta_val)
    ridge = _density_ridge(density, lam_grid)

    z = M_val * delta / eta_val
    az = np.abs(z)
    safe = np.where(az > 0.1, z, np.ones_like(z))
    exact = np.clip(M_val / (1 - np.exp(-safe) + 1e-30) - M_val / safe, 0.0, M_val)
    mean_lam = np.where(az > 0.1, exact, np.clip(M_val / 2 + M_val * z / 12, 0.0, M_val))
    return p_view, lam_grid, density, ridge, mean_lam


def _support_start_curve(density, lam_grid, floor=1e-30):
    starts = np.full(density.shape[0], np.nan, dtype=np.float64)
    mask = density >= floor
    valid = mask.any(axis=1)
    if np.any(valid):
        starts[valid] = lam_grid[np.argmax(mask[valid], axis=1)]
    return starts


def _infer_eta_for_visible_start(delta, lam_grid, target_start, floor=1e-30, eta_lo=1e-8, eta_hi=1.0, steps=80):
    if not np.isfinite(delta) or delta <= 0.0 or not np.isfinite(target_start):
        return np.nan

    candidates = np.geomspace(eta_lo, eta_hi, int(steps), dtype=np.float64)
    starts = np.empty_like(candidates)
    for idx, eta in enumerate(candidates):
        density = _gibbs_density_np(np.array([delta], dtype=np.float64), lam_grid, eta=eta)[0]
        starts[idx] = _support_start_curve(density[None, :], lam_grid, floor=floor)[0]

    valid = np.isfinite(starts)
    if not np.any(valid):
        return np.nan

    best = np.argmin(np.abs(starts[valid] - target_start))
    return float(candidates[valid][best])


def summarize_density_temperature_scale(
    benchmark,
    b_values=(-1.0, 1.0),
    p_targets=(0.5, 1.5, 2.5),
    floors=(1e-30, 1e-10),
    onset_targets=(46.0,),
):
    p_idx, _ = _interior_indices(benchmark)
    p_view = benchmark["p_grid"][p_idx]
    lam_grid = _density_lam_grid(cfg.M)
    floor_list = tuple(float(floor) for floor in floors)
    onset_list = tuple(float(onset) for onset in onset_targets)
    diagnostics = []

    for b_val in b_values:
        row = benchmark["slice_indices"][b_val]
        delta_slice = benchmark["delta2"][row][p_idx]
        density_slice = _gibbs_density_np(delta_slice, lam_grid, eta=cfg.eta)

        for p_target in p_targets:
            p_row = int(np.argmin(np.abs(p_view - p_target)))
            entry = {
                "kind": "exit",
                "b": float(b_val),
                "p": float(p_view[p_row]),
                "delta": float(delta_slice[p_row]),
                "delta_over_eta": float(delta_slice[p_row] / cfg.eta),
            }
            density_row = density_slice[p_row]
            for floor in floor_list:
                label = f"visible_start_{floor:.0e}"
                entry[label] = float(_support_start_curve(density_row[None, :], lam_grid, floor=floor)[0])
            for onset in onset_list:
                for floor in floor_list:
                    label = f"eta_for_start_{onset:.2f}_at_{floor:.0e}"
                    entry[label] = _infer_eta_for_visible_start(delta_slice[p_row], lam_grid, onset, floor=floor)
            diagnostics.append(entry)

    return {
        "lam_grid": lam_grid,
        "floors": floor_list,
        "onset_targets": onset_list,
        "entries": diagnostics,
    }


def _format_density_temperature_scale_report(summary):
    lines = ["Density temperature-scale diagnostics (HJB benchmark):"]
    for entry in summary["entries"]:
        line = (
            f"  {entry['kind']} slice b={entry['b']:.1f}, p~{entry['p']:.2f}: "
            f"Delta={entry['delta']:.6g}, Delta/eta={entry['delta_over_eta']:.3f}"
        )
        lines.append(line)
        for floor in summary["floors"]:
            lines.append(
                f"    visible start @ {floor:.0e}: "
                f"lambda~{entry[f'visible_start_{floor:.0e}']:.2f}"
            )
        for onset in summary["onset_targets"]:
            for floor in summary["floors"]:
                eta_eff = entry[f"eta_for_start_{onset:.2f}_at_{floor:.0e}"]
                ratio = eta_eff / cfg.eta if np.isfinite(eta_eff) else np.nan
                lines.append(
                    f"    eta_eff for start lambda~{onset:.2f} @ {floor:.0e}: "
                    f"{eta_eff:.6g} (x{ratio:.2f} current eta)"
                )
    return "\n".join(lines)


def _benchmark_key(**overrides):
    return (
        float(overrides.get("M", cfg.M)),
        float(overrides.get("eta", cfg.eta)),
        float(overrides.get("theta", cfg.theta)),
        float(overrides.get("sigma", cfg.sigma)),
    )


def solve_hjb_with_overrides(**overrides):
    original = {name: getattr(cfg, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(cfg, name, value)
        return solve_hjb_benchmark()
    finally:
        for name, value in original.items():
            setattr(cfg, name, value)


def get_hjb_benchmark(cache, **overrides):
    key = _benchmark_key(**overrides)
    if key not in cache:
        cache[key] = solve_hjb_with_overrides(**overrides)
    return cache[key]


def _interior_indices(benchmark):
    p_grid = benchmark["p_grid"]
    b_grid = benchmark["b_grid"]
    p_idx = np.where((p_grid >= cfg.grid_lo) & (p_grid <= cfg.grid_hi))[0]
    b_idx = np.where((b_grid >= cfg.grid_lo) & (b_grid <= cfg.grid_hi))[0]
    return p_idx, b_idx


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


def solve_hjb_benchmark():
    step = cfg.hjb_step
    p_grid = np.arange(cfg.hjb_p_min, cfg.hjb_p_max + 0.5 * step, step, dtype=np.float64)
    b_grid = np.arange(cfg.hjb_b_min, cfg.hjb_b_max + 0.5 * step, step, dtype=np.float64)
    lower_base, diag_base, upper_base = _ou_fd_coefficients(p_grid, step)

    p_mesh, b_mesh = np.meshgrid(p_grid, b_grid)
    payoff = _G_np(p_mesh, b_mesh)

    v1 = np.maximum(payoff, 0.0)
    v1_error = np.inf
    v1_converged = False
    for iteration in range(cfg.hjb_max_iter):
        prev = v1.copy()
        for row in range(len(b_grid)):
            delta = payoff[row] - prev[row]
            lam = _mean_lam_np(delta)
            rhs = _hjb_src_np(delta) + lam * prev[row]
            diag = diag_base + lam
            v1[row] = _solve_ou_line(
                lower_base,
                diag,
                upper_base,
                rhs,
                left_bc=("neumann", 0.0),
                right_bc=("dirichlet", float(payoff[row, -1])),
            )
        v1_error = float(np.max(np.abs(v1 - prev)))
        if v1_error < cfg.hjb_tol:
            v1_converged = True
            break
    v1_iterations = iteration + 1

    if not (len(p_grid) == len(b_grid) and np.allclose(p_grid, b_grid)):
        raise ValueError("HJB p-grid and b-grid must align to read V1(p, p) on the diagonal")

    diag_v1 = np.diag(v1)
    v0 = np.maximum(diag_v1, 0.0)
    v0_error = np.inf
    v0_converged = False
    for iteration in range(cfg.hjb_max_iter):
        prev = v0.copy()
        delta = diag_v1 - prev
        lam = _mean_lam_np(delta)
        rhs = _hjb_src_np(delta) + lam * prev
        diag = diag_base + lam
        v0 = _solve_ou_line(
            lower_base,
            diag,
            upper_base,
            rhs,
            left_bc=("dirichlet", float(diag_v1[0])),
            right_bc=("neumann", 0.0),
        )
        v0_error = float(np.max(np.abs(v0 - prev)))
        if v0_error < cfg.hjb_tol:
            v0_converged = True
            break
    v0_iterations = iteration + 1

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
        "v1_iterations": v1_iterations,
        "v1_error": v1_error,
        "v1_converged": v1_converged,
        "v0_iterations": v0_iterations,
        "v0_error": v0_error,
        "v0_converged": v0_converged,
    }


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
    delta2_rl = benchmark["payoff"] - v1_rl

    entry_intensity_hjb = _mean_lam_np(benchmark["delta1"])
    entry_intensity_rl = _mean_lam_np(delta1_rl)
    exit_intensity_hjb = _mean_lam_np(benchmark["delta2"])
    exit_intensity_rl = _mean_lam_np(delta2_rl)

    entry_region_hjb = benchmark["delta1"] >= 0.0
    entry_region_rl = delta1_rl >= 0.0
    exit_region_hjb = benchmark["delta2"] >= 0.0
    exit_region_rl = delta2_rl >= 0.0
    entry_region_mismatch = entry_region_rl != entry_region_hjb
    exit_region_mismatch = exit_region_rl != exit_region_hjb

    entry_interior = np.zeros_like(entry_region_hjb, dtype=bool)
    entry_interior[p_idx] = True
    exit_interior = np.zeros_like(exit_region_hjb, dtype=bool)
    exit_interior[np.ix_(b_idx, p_idx)] = True

    entry_boundary_band = np.zeros_like(entry_region_hjb, dtype=bool)
    if np.isfinite(benchmark["entry_boundary"]):
        entry_boundary_band = np.abs(p_grid - benchmark["entry_boundary"]) <= 3 * cfg.hjb_step

    exit_boundary_band = np.zeros_like(exit_region_hjb, dtype=bool)
    finite_exit = np.isfinite(benchmark["full_exit_boundary"])
    if np.any(finite_exit):
        exit_boundary_band[finite_exit] = (
            np.abs(p_grid[None, :] - benchmark["full_exit_boundary"][finite_exit, None]) <= 3 * cfg.hjb_step
        )

    entry_intensity_gap = np.abs(entry_intensity_rl - entry_intensity_hjb)
    exit_intensity_gap = np.abs(exit_intensity_rl - exit_intensity_hjb)

    v0_diff = np.abs(v0_rl - benchmark["v0"])
    v1_diff = np.abs(v1_rl - benchmark["v1"])
    exit_boundaries_rl = {}
    for b_val, row in benchmark["slice_indices"].items():
        exit_boundaries_rl[b_val] = float(find_boundary(p_grid, delta2_rl[row], mode="neg_to_pos"))

    return {
        "v0_rl": v0_rl,
        "v1_rl": v1_rl,
        "delta1_rl": delta1_rl,
        "delta2_rl": delta2_rl,
        "entry_intensity_hjb": entry_intensity_hjb,
        "entry_intensity_rl": entry_intensity_rl,
        "exit_intensity_hjb": exit_intensity_hjb,
        "exit_intensity_rl": exit_intensity_rl,
        "v0_mae": float(v0_diff[p_idx].mean()),
        "v0_max": float(v0_diff[p_idx].max()),
        "v1_mae": float(v1_diff[np.ix_(b_idx, p_idx)].mean()),
        "v1_max": float(v1_diff[np.ix_(b_idx, p_idx)].max()),
        "entry_intensity_mae": float(entry_intensity_gap[p_idx].mean()),
        "entry_intensity_max": float(entry_intensity_gap[p_idx].max()),
        "exit_intensity_mae": float(exit_intensity_gap[np.ix_(b_idx, p_idx)].mean()),
        "exit_intensity_max": float(exit_intensity_gap[np.ix_(b_idx, p_idx)].max()),
        "entry_intensity_mae_near_boundary": float(entry_intensity_gap[entry_boundary_band & entry_interior].mean())
        if np.any(entry_boundary_band & entry_interior)
        else np.nan,
        "exit_intensity_mae_near_boundary": float(exit_intensity_gap[exit_boundary_band & exit_interior].mean())
        if np.any(exit_boundary_band & exit_interior)
        else np.nan,
        "entry_region_hjb": entry_region_hjb,
        "entry_region_rl": entry_region_rl,
        "exit_region_hjb": exit_region_hjb,
        "exit_region_rl": exit_region_rl,
        "entry_region_mismatch": entry_region_mismatch,
        "exit_region_mismatch": exit_region_mismatch,
        "entry_region_mismatch_rate": float(entry_region_mismatch[p_idx].mean()),
        "exit_region_mismatch_rate": float(exit_region_mismatch[np.ix_(b_idx, p_idx)].mean()),
        "entry_region_mismatch_near_boundary": float(entry_region_mismatch[entry_boundary_band & entry_interior].mean())
        if np.any(entry_boundary_band & entry_interior)
        else np.nan,
        "exit_region_mismatch_near_boundary": float(exit_region_mismatch[exit_boundary_band & exit_interior].mean())
        if np.any(exit_boundary_band & exit_interior)
        else np.nan,
        "entry_boundary_rl": float(find_boundary(p_grid, delta1_rl, mode="pos_to_neg")),
        "exit_boundaries_rl": exit_boundaries_rl,
        "v1_abs_error": v1_diff,
        "interior_p_idx": p_idx,
        "interior_b_idx": b_idx,
    }


def audit_rl_paper_alignment():
    return {
        "matches": [
            "one-step Bernoulli regime updates use q=1-exp(-mean_intensity*dt), matching Algorithm 1",
            "half of simulated trajectories start in J=1 with B0 sampled uniformly, matching Remark 4.1",
            "training loss averages entry and exit TD errors jointly by active sample count, matching Algorithm 1",
        ],
        "mismatches": [],
        "notes": [
            "default reporting focused on value and boundary errors; low-eta control mismatch can be materially larger",
        ],
    }


def generate_ou_paths(n_paths, steps=None):
    steps = cfg.path_steps if steps is None else int(steps)
    paths = np.zeros((n_paths, steps + 1), dtype=np.float32)
    paths[:, 0] = np.random.uniform(cfg.b_lo, cfg.b_hi, size=n_paths).astype(np.float32)
    decay = np.exp(-cfg.theta * cfg.dt)
    innov_std = np.sqrt((cfg.sigma ** 2) * (1 - np.exp(-2 * cfg.theta * cfg.dt)) / (2 * cfg.theta))
    for step in range(steps):
        mean = cfg.pbar + (paths[:, step] - cfg.pbar) * decay
        paths[:, step + 1] = mean + innov_std * np.random.randn(n_paths)
    return paths.astype(np.float32)


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


class Agent:
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
        return _G_t(p, b) - self.v1(torch.stack([p, b], -1))

    @torch.no_grad()
    def _simulate_transitions(self, batch):
        num_paths, path_len = batch.shape
        steps = path_len - 1
        total = num_paths * cfg.n_sims
        paths = np.repeat(batch, cfg.n_sims, axis=0)
        J = np.zeros(total, dtype=np.int32)
        B = np.zeros(total, dtype=np.float32)

        half = total // 2
        J[half:] = 1
        B[half:] = np.random.uniform(cfg.b_lo, cfg.b_hi, size=total - half).astype(np.float32)

        R0 = []
        R1 = []
        for step in range(steps):
            p_now = paths[:, step]
            p_next = paths[:, step + 1]
            J_now = J.copy()
            B_now = B.copy()
            J_next = J_now.copy()
            B_next = B_now.copy()

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
                    R0.append((float(p_now[idx]), float(p_next[idx]), int(J_next[idx]), float(B_next[idx])))

            m1 = J_now == 1
            if m1.any():
                ix = np.where(m1)[0]
                pt = torch.tensor(p_now[ix], dtype=torch.float32, device=self.dev)
                bt = torch.tensor(B_now[ix], dtype=torch.float32, device=self.dev)
                lam = _mean_lam(self.delta2(pt, bt))
                q = (1 - torch.exp(-lam * cfg.dt)).cpu().numpy()
                exit_mask = np.random.random(len(ix)) < q
                exit_ix = ix[exit_mask]
                J_next[exit_ix] = 2
                for idx in ix:
                    R1.append((float(p_now[idx]), float(B_now[idx]), float(p_next[idx]), int(J_next[idx]), float(B_next[idx])))

            J = J_next
            B = B_next
        return R0, R1

    def _loss_v0(self, transitions):
        if not transitions:
            return torch.tensor(0.0, device=self.dev)
        p = torch.tensor([row[0] for row in transitions], dtype=torch.float32, device=self.dev)
        pn = torch.tensor([row[1] for row in transitions], dtype=torch.float32, device=self.dev)
        jn = torch.tensor([row[2] for row in transitions], dtype=torch.long, device=self.dev)
        bn = torch.tensor([row[3] for row in transitions], dtype=torch.float32, device=self.dev)

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
        if not transitions:
            return torch.tensor(0.0, device=self.dev)
        p = torch.tensor([row[0] for row in transitions], dtype=torch.float32, device=self.dev)
        b = torch.tensor([row[1] for row in transitions], dtype=torch.float32, device=self.dev)
        pn = torch.tensor([row[2] for row in transitions], dtype=torch.float32, device=self.dev)
        jn = torch.tensor([row[3] for row in transitions], dtype=torch.long, device=self.dev)
        bn = torch.tensor([row[4] for row in transitions], dtype=torch.float32, device=self.dev)

        v1_now = self.v1(torch.stack([p, b], -1))
        cb = _ent_cost(self.delta2(p, b))
        discount = np.exp(-cfg.rho * cfg.dt)

        with torch.no_grad():
            v1_next = self.v1(torch.stack([pn, bn], -1))
            g_next = _G_t(pn, bn)
            target = torch.where(jn == 1, v1_next, g_next)

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
            "Training OU reproduction "
            f"(paths={total_paths}, steps={cfg.path_steps}, M={cfg.M}, eta={cfg.eta}, td_loss_mode={cfg.td_loss_mode})"
        )
        for it in range(cfg.n_iter):
            idx = np.random.choice(total_paths, min(cfg.batch_size, total_paths), replace=False)
            R0, R1 = self._simulate_transitions(paths[idx])
            loss0 = self._loss_v0(R0)
            loss1 = self._loss_v1(R1)
            loss = self._combine_td_losses(loss0, loss1, len(R0), len(R1))
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.v0.parameters()) + list(self.v1.parameters()), cfg.grad_clip)
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
            exit_boundaries[b_val] = find_boundary(grid.cpu().numpy(), delta2, mode="neg_to_pos")
        return entry_boundary, exit_boundaries

    @torch.no_grad()
    def rollout_path(self, path, rng):
        state = 0
        entry_step = None
        entry_signal = 0.0
        horizon_steps = len(path) - 1
        for step in range(len(path) - 1):
            p_now = float(path[step])
            p_next = float(path[step + 1])
            if state == 0:
                pt = torch.tensor([p_now], dtype=torch.float32, device=self.dev)
                lam = float(_mean_lam(self.delta1(pt)).item())
                q = 1 - np.exp(-lam * cfg.dt)
                if rng.random() < q:
                    state = 1
                    entry_step = step
                    entry_signal = p_now
            elif state == 1:
                pt = torch.tensor([p_now], dtype=torch.float32, device=self.dev)
                bt = torch.tensor([entry_signal], dtype=torch.float32, device=self.dev)
                lam = float(_mean_lam(self.delta2(pt, bt)).item())
                q = 1 - np.exp(-lam * cfg.dt)
                if rng.random() < q:
                    utility = _u_s(cfg.gamma * p_next - cfg.iota * entry_signal - cfg.Psi - cfg.R)
                    return {
                        "entered": 1,
                        "completed": 1,
                        "open_at_horizon": 0,
                        "entry_step": entry_step,
                        "entry_signal": entry_signal,
                        "exit_signal": p_next,
                        "hold_steps": step + 1 - entry_step,
                        "remaining_steps_after_entry": horizon_steps - entry_step,
                        "utility": utility,
                    }
        if state == 1:
            return {
                "entered": 1,
                "completed": 0,
                "open_at_horizon": 1,
                "entry_step": entry_step,
                "entry_signal": entry_signal,
                "exit_signal": np.nan,
                "hold_steps": len(path) - 1 - entry_step,
                "remaining_steps_after_entry": horizon_steps - entry_step,
                "utility": 0.0,
            }
        return {
            "entered": 0,
            "completed": 0,
            "open_at_horizon": 0,
            "entry_step": np.nan,
            "entry_signal": np.nan,
            "exit_signal": np.nan,
            "hold_steps": 0,
            "remaining_steps_after_entry": np.nan,
            "utility": 0.0,
        }

    @torch.no_grad()
    def evaluate(self, paths):
        rng = np.random.default_rng(cfg.seed + 7)
        rows = [self.rollout_path(path, rng) for path in paths]
        entered = np.array([row["entered"] for row in rows], dtype=float)
        completed = np.array([row["completed"] for row in rows], dtype=float)
        open_at_horizon = np.array([row["open_at_horizon"] for row in rows], dtype=float)
        utilities = np.array([row["utility"] for row in rows], dtype=float)
        holds = np.array([row["hold_steps"] for row in rows], dtype=float)
        entry_steps = np.array([row["entry_step"] for row in rows if row["entered"]], dtype=float)
        entry_vals = np.array([row["entry_signal"] for row in rows if row["entered"]], dtype=float)
        exit_vals = np.array([row["exit_signal"] for row in rows if row["completed"]], dtype=float)
        remaining = np.array([row["remaining_steps_after_entry"] for row in rows if row["entered"]], dtype=float)
        completion_given_entry = float(completed.sum() / entered.sum()) if entered.sum() else 0.0
        return {
            "entry_rate": float(entered.mean()),
            "completion_rate": float(completed.mean()),
            "completion_given_entry": completion_given_entry,
            "open_at_horizon_rate": float(open_at_horizon.mean()),
            "avg_utility": float(utilities.mean()),
            "avg_hold_steps": float(holds[completed == 1].mean()) if completed.sum() else 0.0,
            "avg_entry_step": float(entry_steps.mean()) if len(entry_steps) else np.nan,
            "avg_remaining_steps_after_entry": float(remaining.mean()) if len(remaining) else np.nan,
            "avg_entry_signal": float(entry_vals.mean()) if len(entry_vals) else np.nan,
            "avg_exit_signal": float(exit_vals.mean()) if len(exit_vals) else np.nan,
        }


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


def plot_results(losses, benchmark, comparison, out_path):
    p_grid = benchmark["p_grid"]
    b_grid = benchmark["b_grid"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

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
    ax.set(title="(b) V0 comparison", xlabel="Signal p", ylabel="Value")
    ax.set_xlim(cfg.grid_lo, cfg.grid_hi)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 2]
    ax.plot(p_grid, benchmark["delta1"], color="black", ls="--", lw=2.0, label="HJB Delta1")
    ax.plot(p_grid, comparison["delta1_rl"], color="crimson", lw=1.6, label="RL Delta1")
    ax.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax.axvline(benchmark["entry_boundary"], color="black", ls=":", lw=1.5, label="HJB entry boundary")
    ax.axvline(comparison["entry_boundary_rl"], color="green", ls=":", lw=1.5, label="RL entry boundary")
    ax.set(title="(c) Entry advantage and boundary", xlabel="Signal p")
    ax.set_xlim(cfg.grid_lo, cfg.grid_hi)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 0]
    for b_val, color, style in [(-1.0, "darkred", "-"), (0.0, "coral", "--"), (1.0, "purple", ":")]:
        row = benchmark["slice_indices"][b_val]
        ax.plot(
            p_grid,
            benchmark["delta2"][row],
            linestyle=style,
            color=color,
            lw=1.8,
            alpha=0.9,
            label=f"HJB Delta2(p, b={b_val:.1f})",
        )
        ax.plot(
            p_grid,
            comparison["delta2_rl"][row],
            linestyle=style,
            color=color,
            lw=1.2,
            alpha=0.45,
        )
    ax.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax.set(title="(d) Exit advantage slices", xlabel="Signal p")
    ax.set_xlim(cfg.grid_lo, cfg.grid_hi)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 1]
    ax.plot(b_grid, benchmark["full_exit_boundary"], color="black", ls="--", lw=2.0, label="HJB p*(b)")
    hjb_x = list(benchmark["exit_boundaries"].keys())
    hjb_y = list(benchmark["exit_boundaries"].values())
    rl_x = list(comparison["exit_boundaries_rl"].keys())
    rl_y = list(comparison["exit_boundaries_rl"].values())
    ax.scatter(hjb_x, hjb_y, color="black", s=50, marker="x", label="HJB sampled boundaries")
    ax.scatter(rl_x, rl_y, color="royalblue", s=55, label="RL sampled boundaries")
    ax.set(
        title="(e) Exit free boundary comparison",
        xlabel="Entry signal b",
        ylabel="Exit boundary p*(b)",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    entry_gap = abs(comparison["entry_boundary_rl"] - benchmark["entry_boundary"])
    ax.text(
        0.02,
        0.98,
        (
            f"V0 MAE={comparison['v0_mae']:.4f}\n"
            f"V1 MAE={comparison['v1_mae']:.4f}\n"
            f"entry gap={entry_gap:.4f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
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
        origin="lower",
        aspect="auto",
        extent=[b_grid[b_idx[0]], b_grid[b_idx[-1]], p_lo, p_hi],
        cmap=PAPER_TAN_BLUE_CMAP,
        vmin=-0.150,
        vmax=0.025,
    )
    boundary_curve = benchmark["full_exit_boundary"][b_idx]
    boundary_mask = np.isfinite(boundary_curve) & (boundary_curve >= p_lo) & (boundary_curve <= p_hi)
    ax.plot(
        b_grid[b_idx][boundary_mask],
        boundary_curve[boundary_mask],
        color="black",
        ls="--",
        lw=1.8,
        alpha=0.95,
    )
    ax.set(title="(f) V1 RL - HJB on interior", xlabel="Entry signal b", ylabel="Signal p")
    ax.set_ylim(p_lo, p_hi)
    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.85,
        ticks=[-0.150, -0.125, -0.100, -0.075, -0.050, -0.025, 0.0, 0.025],
    )
    cbar.set_ticklabels(["-0.150", "-0.125", "-0.100", "-0.075", "-0.050", "-0.025", "0.00", "0.025"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_diagnostics(benchmark, out_path):
    cache = {_benchmark_key(): benchmark}
    plot_benchmark = get_hjb_benchmark(cache, eta=cfg.density_plot_eta)
    p_grid = plot_benchmark["p_grid"]
    p_mask = (p_grid >= cfg.grid_lo) & (p_grid <= cfg.grid_hi)
    p_view = p_grid[p_mask]
    lam_grid = _density_lam_grid(cfg.M, points=241)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    panels = [

        (
            axes[0],
            plot_benchmark["delta2"][plot_benchmark["slice_indices"][-1.0]][p_mask],
            f"(a) Exit density, b=-1, eta={cfg.density_plot_eta:g}",
            "Signal p",
        ),
        (
            axes[1],
            plot_benchmark["delta2"][plot_benchmark["slice_indices"][1.0]][p_mask],
            f"(b) Exit density, b=1, eta={cfg.density_plot_eta:g}",
            "Signal p",
        ),
    ]

    for ax, delta, title, xlabel in panels:
        density = _gibbs_density_np(delta, lam_grid, eta=cfg.density_plot_eta)
        ridge = _density_ridge(density, lam_grid)
        display, norm = _density_display(density, floor=1e-30, ceiling=1.0)
        im = ax.imshow(
            display,
            origin="lower",
            aspect="auto",
            extent=[lam_grid[0], lam_grid[-1], p_view[0], p_view[-1]],
            cmap="Blues",
            norm=norm,
        )
        ax.plot(ridge, p_view, color="red", lw=1.8, label=r"$\mathrm{argmax}_{\lambda}\,\pi^{\beta,*}(\lambda; p, b)$")
        ax.set(title=title, xlabel="Intensity lambda", ylabel=xlabel)
        ax.set_xlim(-0.5, cfg.M + 0.5)
        ax.legend(loc="upper left", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        _set_log_colorbar_ticks(cbar, top_exp=0)
        cbar.set_label("Density")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entry_density_high_eta_regimes(benchmark, out_path):
    cache = {_benchmark_key(): benchmark}
    cases = [
        (1.0, 1e-1, "(a) Entry density, M=1, eta=1e-1"),
        (cfg.M, 1e-1, f"(b) Entry density, M={cfg.M:.0f}, eta=1e-1"),
    ]
    scales = [
        (0.5, 3.0, np.arange(0.5, 3.0 + 0.5, 0.5)),
        (0.01, 0.05, np.arange(0.01, 0.05 + 0.01, 0.01)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, (M_val, eta_val, title), (vmin, vmax, ticks) in zip(axes.ravel(), cases, scales):
        result = get_hjb_benchmark(cache, M=M_val, eta=eta_val)
        p_view, lam_grid, density, ridge, mean_lam = _entry_density_case(result, M_val, eta_val)
        display = np.clip(density, vmin, vmax)

        im = ax.imshow(
            display,
            origin="lower",
            aspect="auto",
            extent=[lam_grid[0], lam_grid[-1], p_view[0], p_view[-1]],
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
        )
        ax.plot(mean_lam, p_view, color="black", lw=1.8, label="Mean intensity")
        ax.plot(ridge, p_view, color="dimgray", lw=1.1, alpha=0.9, label="Ridge")
        if np.isfinite(result["entry_boundary"]):
            ax.axhline(result["entry_boundary"], color="crimson", ls="--", lw=1.5, label="Free boundary")
        ax.set(title=title, xlabel="Intensity lambda", ylabel="Entry signal p")
        ax.set_xlim(-0.02 * M_val, 1.02 * M_val)
        ax.legend(loc="upper left", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88, ticks=ticks)
        cbar.set_label("Density")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entry_density_low_eta_regimes(benchmark, out_path):
    cache = {_benchmark_key(): benchmark}
    cases = [
        (1.0, cfg.density_plot_eta, f"(c) Entry density, M=1, eta={cfg.density_plot_eta:g}"),
        (cfg.M, cfg.density_plot_eta, f"(d) Entry density, M={cfg.M:.0f}, eta={cfg.density_plot_eta:g}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, (M_val, eta_val, title) in zip(axes.ravel(), cases):
        result = get_hjb_benchmark(cache, M=M_val, eta=eta_val)
        p_view, lam_grid, density, ridge, _ = _entry_density_case(result, M_val, eta_val)
        display, norm = _density_display(density, floor=1e-30, ceiling=1.0)

        im = ax.imshow(
            display,
            origin="lower",
            aspect="auto",
            extent=[lam_grid[0], lam_grid[-1], p_view[0], p_view[-1]],
            cmap="Blues",
            norm=norm,
        )
        ax.plot(ridge, p_view, color="red", lw=1.8, label=r"$\mathrm{argmax}_{\lambda}\,\pi^{\alpha,*}(\lambda; p)$")
        if np.isfinite(result["entry_boundary"]):
            ax.scatter(
                [0.0],
                [result["entry_boundary"]],
                color="forestgreen",
                marker="D",
                s=65,
                zorder=5,
                label=rf"$\Delta_1(p)=0:\ p\approx {result['entry_boundary']:.2f}$",
            )
        ax.set(title=title, xlabel="Intensity lambda", ylabel="Entry signal p")
        ax.set_xlim(-0.02 * M_val, 1.02 * M_val)
        ax.legend(loc="upper left", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        _set_log_colorbar_ticks(cbar, top_exp=0)
        cbar.set_label("Density")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_concentration_diagnostics(benchmark, out_path):
    cache = {_benchmark_key(): benchmark}
    plot_benchmark = get_hjb_benchmark(cache, eta=cfg.density_plot_eta)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    entry_panels = [
        (axes[0, 0], 1.0, "(a) Entry concentration, M=1"),
        (axes[0, 1], cfg.M, f"(b) Entry concentration, M={cfg.M:.0f}"),
    ]
    for ax, M_val, title in entry_panels:
        result = get_hjb_benchmark(cache, M=M_val, eta=cfg.density_plot_eta)
        p_view, lam_grid, density, _, _ = _entry_density_case(result, M_val, cfg.density_plot_eta)
        support_start = _support_start_curve(density, lam_grid)
        boundary = result["entry_boundary"]

        ax.plot(p_view, support_start, color="royalblue", lw=2.0, label="Support start")
        ax.axvline(boundary, color="forestgreen", ls="--", lw=1.3, label=rf"$p^*\approx {boundary:.3f}$")
        ax.set(
            title=title,
            xlabel="Signal p",
            ylabel="Visible lambda start",
            xlim=(cfg.grid_lo, cfg.grid_hi),
            ylim=(-0.02 * M_val, 1.02 * M_val),
        )

        ax2 = ax.twinx()
        ax2.plot(p_view, result["delta1"][np.where((result["p_grid"] >= cfg.grid_lo) & (result["p_grid"] <= cfg.grid_hi))[0]], color="darkred", lw=1.4, alpha=0.9, label="Delta1")
        ax2.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax2.set_ylabel("Delta1")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    p_idx, _ = _interior_indices(plot_benchmark)
    p_view = plot_benchmark["p_grid"][p_idx]
    lam_grid = _density_lam_grid(cfg.M)
    exit_panels = [
        (axes[1, 0], -1.0, "(c) Exit concentration, b=-1"),
        (axes[1, 1], 1.0, "(d) Exit concentration, b=1"),
    ]
    for ax, b_val, title in exit_panels:
        row = plot_benchmark["slice_indices"][b_val]
        delta = plot_benchmark["delta2"][row][p_idx]
        density = _gibbs_density_np(delta, lam_grid, eta=cfg.density_plot_eta)
        support_start = _support_start_curve(density, lam_grid)
        boundary = plot_benchmark["exit_boundaries"][b_val]

        ax.plot(p_view, support_start, color="royalblue", lw=2.0, label="Support start")
        ax.axvline(boundary, color="forestgreen", ls="--", lw=1.3, label=rf"$p^*(b)\approx {boundary:.3f}$")
        ax.set(
            title=title,
            xlabel="Signal p",
            ylabel="Visible lambda start",
            xlim=(cfg.grid_lo, cfg.grid_hi),
            ylim=(-1.0, cfg.M + 1.0),
        )

        ax2 = ax.twinx()
        ax2.plot(p_view, delta, color="darkred", lw=1.4, alpha=0.9, label="Delta2")
        ax2.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax2.set_ylabel("Delta2")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_control_diagnostics(benchmark, comparison, out_path):
    p_grid = benchmark["p_grid"]
    b_grid = benchmark["b_grid"]
    p_idx = comparison["interior_p_idx"]
    b_idx = comparison["interior_b_idx"]
    p_view = p_grid[p_idx]
    b_view = b_grid[b_idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(p_view, comparison["entry_intensity_hjb"][p_idx], color="black", ls="--", lw=2.0, label="HJB mean intensity")
    ax.plot(p_view, comparison["entry_intensity_rl"][p_idx], color="royalblue", lw=1.8, label="RL mean intensity")
    ax.axvline(benchmark["entry_boundary"], color="black", ls=":", lw=1.3, label="HJB boundary")
    ax.axvline(comparison["entry_boundary_rl"], color="royalblue", ls=":", lw=1.3, label="RL boundary")
    ax.set(
        title="(a) Entry mean intensity",
        xlabel="Signal p",
        ylabel="Mean intensity",
        xlim=(cfg.grid_lo, cfg.grid_hi),
        ylim=(-1.0, cfg.M + 1.0),
    )
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    entry_gap = np.abs(comparison["entry_intensity_rl"][p_idx] - comparison["entry_intensity_hjb"][p_idx])
    entry_mismatch = comparison["entry_region_mismatch"][p_idx].astype(float)
    ax.plot(p_view, entry_gap, color="darkorange", lw=1.8, label="|mean intensity gap|")
    ax.set(
        title="(b) Entry control disagreement",
        xlabel="Signal p",
        ylabel="Intensity gap",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    ax2 = ax.twinx()
    ax2.step(p_view, entry_mismatch, where="mid", color="crimson", lw=1.2, label="Stop-region mismatch")
    ax2.set_ylabel("Mismatch indicator")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0.0, 1.0])
    ax.text(
        0.02,
        0.98,
        (
            f"MAE={comparison['entry_intensity_mae']:.3f}\n"
            f"near-boundary MAE={comparison['entry_intensity_mae_near_boundary']:.3f}\n"
            f"near-boundary mismatch={comparison['entry_region_mismatch_near_boundary'] * 100:.1f}%"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax = axes[1, 0]
    for b_val, color in [(-1.0, "darkred"), (0.0, "coral"), (1.0, "purple")]:
        row = benchmark["slice_indices"][b_val]
        ax.plot(
            p_view,
            comparison["exit_intensity_hjb"][row][p_idx],
            color=color,
            ls="--",
            lw=2.0,
            alpha=0.95,
            label=f"HJB b={b_val:.1f}",
        )
        ax.plot(
            p_view,
            comparison["exit_intensity_rl"][row][p_idx],
            color=color,
            lw=1.3,
            alpha=0.8,
            label=f"RL b={b_val:.1f}",
        )
    ax.set(
        title="(c) Exit mean intensity slices",
        xlabel="Signal p",
        ylabel="Mean intensity",
        xlim=(cfg.grid_lo, cfg.grid_hi),
        ylim=(-1.0, cfg.M + 1.0),
    )
    ax.legend(loc="best", fontsize=8, ncol=2)

    ax = axes[1, 1]
    mismatch_map = comparison["exit_region_mismatch"][np.ix_(b_idx, p_idx)].astype(float)
    im = ax.imshow(
        mismatch_map.T,
        origin="lower",
        aspect="auto",
        extent=[b_view[0], b_view[-1], p_view[0], p_view[-1]],
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
    )
    boundary_curve = benchmark["full_exit_boundary"][b_idx]
    boundary_mask = np.isfinite(boundary_curve) & (boundary_curve >= p_view[0]) & (boundary_curve <= p_view[-1])
    ax.plot(b_view[boundary_mask], boundary_curve[boundary_mask], color="black", ls="--", lw=1.6, alpha=0.95)
    ax.set(
        title="(d) Exit stopping-region disagreement",
        xlabel="Entry signal b",
        ylabel="Signal p",
        xlim=(cfg.grid_lo, cfg.grid_hi),
        ylim=(cfg.grid_lo, cfg.grid_hi),
    )
    ax.text(
        0.02,
        0.98,
        (
            f"MAE={comparison['exit_intensity_mae']:.3f}\n"
            f"near-boundary MAE={comparison['exit_intensity_mae_near_boundary']:.3f}\n"
            f"near-boundary mismatch={comparison['exit_region_mismatch_near_boundary'] * 100:.1f}%"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[0.0, 1.0])
    cbar.set_ticklabels(["match", "mismatch"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hjb_parameter_diagnostics(benchmark, out_path):
    cache = {_benchmark_key(): benchmark}

    eta_results = [(eta, get_hjb_benchmark(cache, M=cfg.M, eta=eta)) for eta in cfg.sweep_eta_values]
    m_results = [(M, get_hjb_benchmark(cache, M=M, eta=cfg.eta)) for M in cfg.sweep_M_values]
    v1_low_m = get_hjb_benchmark(cache, M=1.0, eta=cfg.eta)
    v1_high_m = get_hjb_benchmark(cache, M=cfg.M, eta=cfg.eta)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    for eta, result in eta_results:
        p_idx, _ = _interior_indices(result)
        ax.plot(result["p_grid"][p_idx], result["v0"][p_idx], lw=1.8, label=f"eta={eta:g}")
    ax.set(
        title=f"(a) V0 with fixed M={cfg.M:.0f}",
        xlabel="Signal p",
        ylabel="Value",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    for M, result in m_results:
        p_idx, _ = _interior_indices(result)
        ax.plot(result["p_grid"][p_idx], result["v0"][p_idx], lw=1.8, label=f"M={M:g}")
    ax.set(
        title=f"(b) V0 with fixed eta={cfg.eta:g}",
        xlabel="Signal p",
        ylabel="Value",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    ax.legend(loc="best", fontsize=8)

    low_p_idx, low_b_idx = _interior_indices(v1_low_m)
    high_p_idx, high_b_idx = _interior_indices(v1_high_m)
    low_v1 = v1_low_m["v1"][np.ix_(low_b_idx, low_p_idx)]
    high_v1 = v1_high_m["v1"][np.ix_(high_b_idx, high_p_idx)]
    vmin = min(float(low_v1.min()), float(high_v1.min()))
    vmax = max(float(low_v1.max()), float(high_v1.max()))

    panels = [
        (axes[1, 0], v1_low_m, low_p_idx, low_b_idx, "(c) V1 heatmap, M=1, eta=1e-5"),
        (axes[1, 1], v1_high_m, high_p_idx, high_b_idx, f"(d) V1 heatmap, M={cfg.M:.0f}, eta={cfg.eta:g}"),
    ]
    images = []
    for ax, result, p_idx, b_idx, title in panels:
        image = ax.imshow(
            result["v1"][np.ix_(b_idx, p_idx)].T,
            origin="lower",
            aspect="auto",
            extent=[
                result["b_grid"][b_idx[0]],
                result["b_grid"][b_idx[-1]],
                result["p_grid"][p_idx[0]],
                result["p_grid"][p_idx[-1]],
            ],
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set(title=title, xlabel="Entry signal b", ylabel="Signal p")
        images.append(image)

    fig.colorbar(images[-1], ax=axes[1, :], shrink=0.88)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparative_statics_diagnostics(benchmark, out_path):
    cache = {_benchmark_key(): benchmark}
    theta_results = [(theta, get_hjb_benchmark(cache, theta=theta, sigma=cfg.sigma)) for theta in cfg.sweep_theta_values]
    sigma_results = [(sigma, get_hjb_benchmark(cache, theta=cfg.theta, sigma=sigma)) for sigma in cfg.sweep_sigma_values]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    ax = axes[0]
    for theta, result in theta_results:
        p_idx, _ = _interior_indices(result)
        ax.plot(result["p_grid"][p_idx], result["v0"][p_idx], lw=1.8, label=f"theta={theta:g}")
    ax.axvline(cfg.pbar, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax.set(
        title=f"(a) V0 with fixed sigma={cfg.sigma:g}",
        xlabel="Signal p",
        ylabel="Value",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for sigma, result in sigma_results:
        p_idx, _ = _interior_indices(result)
        ax.plot(result["p_grid"][p_idx], result["v0"][p_idx], lw=1.8, label=f"sigma={sigma:g}")
    ax.axvline(cfg.pbar, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax.set(
        title=f"(b) V0 with fixed theta={cfg.theta:g}",
        xlabel="Signal p",
        ylabel="Value",
        xlim=(cfg.grid_lo, cfg.grid_hi),
    )
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    apply_args(args)
    audit = audit_rl_paper_alignment()

    t0 = time.time()
    print("=" * 80)
    print(" OU reproduction of exploratory optimal stopping")
    print(" Zhao, Tse & Zheng (2026) · Section 4 offline policy iteration")
    print("=" * 80)
    print(
        f"Config: dt={cfg.dt}, steps={cfg.path_steps}, train_paths={cfg.train_paths}, "
        f"eval_paths={cfg.eval_paths}, M={cfg.M}, eta={cfg.eta}"
    )

    train_paths = generate_ou_paths(cfg.train_paths)
    eval_paths = generate_ou_paths(cfg.eval_paths, steps=cfg.eval_path_steps)

    print("\n[1/3] Solving HJB benchmark by finite differences …")
    hjb_t0 = time.time()
    benchmark = solve_hjb_benchmark()
    print(
        f"  V1 iterations: {benchmark['v1_iterations']} "
        f"(max change {benchmark['v1_error']:.2e}, converged={benchmark['v1_converged']})"
    )
    print(
        f"  V0 iterations: {benchmark['v0_iterations']} "
        f"(max change {benchmark['v0_error']:.2e}, converged={benchmark['v0_converged']})"
    )
    print(f"  HJB entry boundary p* ≈ {benchmark['entry_boundary']:.3f}")
    for b_val, boundary in benchmark["exit_boundaries"].items():
        print(f"  HJB exit boundary for b={b_val:.1f}: p*(b) ≈ {boundary:.3f}")
    print(f"  HJB elapsed: {time.time() - hjb_t0:.1f}s")

    print("\n[2/3] Training offline policy iteration …")

    agent = Agent()
    losses = agent.train(train_paths)
    entry_boundary, exit_boundaries = agent.boundary_summary()
    evaluation = agent.evaluate(eval_paths)
    long_eval_paths = generate_ou_paths(cfg.eval_paths, steps=cfg.long_eval_path_steps)
    long_horizon_evaluation = agent.evaluate(long_eval_paths)
    comparison = compare_agent_to_hjb(agent, benchmark)

    print("\n[3/3] Comparing RL policy iteration against HJB …")

    print("\nRL boundary summary:")
    print(f"  RL entry boundary p* ≈ {entry_boundary:.3f}")
    for b_val, boundary in exit_boundaries.items():
        print(f"  RL exit boundary for b={b_val:.1f}: p*(b) ≈ {boundary:.3f}")

    print("\nRL vs HJB comparison on the interior benchmark region:")
    print(f"  V0 mean abs error : {comparison['v0_mae']:.5f}")
    print(f"  V0 max abs error  : {comparison['v0_max']:.5f}")
    print(f"  V1 mean abs error : {comparison['v1_mae']:.5f}")
    print(f"  V1 max abs error  : {comparison['v1_max']:.5f}")
    print(f"  entry boundary gap: {abs(comparison['entry_boundary_rl'] - benchmark['entry_boundary']):.5f}")
    for b_val in cfg.eval_b_values:
        gap = abs(comparison['exit_boundaries_rl'][b_val] - benchmark['exit_boundaries'][b_val])
        print(f"  exit boundary gap for b={b_val:.1f}: {gap:.5f}")

    print("\nRL control comparison on the same benchmark grid:")
    print(f"  entry mean intensity MAE     : {comparison['entry_intensity_mae']:.5f}")
    print(f"  entry mean intensity max gap : {comparison['entry_intensity_max']:.5f}")
    print(f"  exit mean intensity MAE      : {comparison['exit_intensity_mae']:.5f}")
    print(f"  exit mean intensity max gap  : {comparison['exit_intensity_max']:.5f}")
    print(f"  entry mismatch rate          : {comparison['entry_region_mismatch_rate'] * 100:.1f}%")
    print(f"  exit mismatch rate           : {comparison['exit_region_mismatch_rate'] * 100:.1f}%")
    print(f"  entry mismatch near boundary : {comparison['entry_region_mismatch_near_boundary'] * 100:.1f}%")
    print(f"  exit mismatch near boundary  : {comparison['exit_region_mismatch_near_boundary'] * 100:.1f}%")

    density_temperature_summary = summarize_density_temperature_scale(benchmark)
    density_temperature_report = _format_density_temperature_scale_report(density_temperature_summary)
    density_temperature_out = _with_suffix(args.out, "_density_temperature_scale.txt")
    with open(density_temperature_out, "w", encoding="utf-8") as handle:
        handle.write(density_temperature_report)
        handle.write("\n")
    print(f"\n{density_temperature_report}")

    print("\nRL implementation audit against Section 4.3.2:")
    for item in audit["matches"]:
        print(f"  match    : {item}")
    for item in audit["mismatches"]:
        print(f"  mismatch : {item}")
    for item in audit["notes"]:
        print(f"  note     : {item}")

    print(f"\nMonte Carlo policy summary on OU evaluation paths ({cfg.eval_path_steps} steps):")
    print(f"  entry rate      : {evaluation['entry_rate'] * 100:.1f}%")
    print(f"  completion rate : {evaluation['completion_rate'] * 100:.1f}%")
    print(f"  completion|entry: {evaluation['completion_given_entry'] * 100:.1f}%")
    print(f"  open at horizon : {evaluation['open_at_horizon_rate'] * 100:.1f}%")
    print(f"  avg utility     : {evaluation['avg_utility']:.4f}")
    print(f"  avg hold steps  : {evaluation['avg_hold_steps']:.2f}")
    print(f"  avg entry step  : {evaluation['avg_entry_step']:.2f}")
    print(f"  rem steps entry : {evaluation['avg_remaining_steps_after_entry']:.2f}")
    print(f"  avg entry signal: {evaluation['avg_entry_signal']:.3f}")
    print(f"  avg exit signal : {evaluation['avg_exit_signal']:.3f}")

    print(f"\nLong-horizon policy summary ({cfg.long_eval_path_steps} steps):")
    print(f"  entry rate      : {long_horizon_evaluation['entry_rate'] * 100:.1f}%")
    print(f"  completion rate : {long_horizon_evaluation['completion_rate'] * 100:.1f}%")
    print(f"  completion|entry: {long_horizon_evaluation['completion_given_entry'] * 100:.1f}%")
    print(f"  open at horizon : {long_horizon_evaluation['open_at_horizon_rate'] * 100:.1f}%")
    print(f"  avg utility     : {long_horizon_evaluation['avg_utility']:.4f}")
    print(f"  avg hold steps  : {long_horizon_evaluation['avg_hold_steps']:.2f}")
    print(f"  avg entry step  : {long_horizon_evaluation['avg_entry_step']:.2f}")
    print(f"  rem steps entry : {long_horizon_evaluation['avg_remaining_steps_after_entry']:.2f}")

    plot_results(losses, benchmark, comparison, args.out)
    density_out = _with_suffix(args.out, "_density_diagnostics")
    density_concentration_out = _with_suffix(args.out, "_density_concentration_diagnostics")
    control_out = _with_suffix(args.out, "_control_diagnostics")
    entry_high_eta_out = _with_suffix(args.out, "_entry_density_high_eta")
    entry_low_eta_out = _with_suffix(args.out, "_entry_density_low_eta")
    hjb_sweep_out = _with_suffix(args.out, "_hjb_parameter_diagnostics")
    comparative_out = _with_suffix(args.out, "_comparative_statics")
    plot_density_diagnostics(benchmark, density_out)
    plot_density_concentration_diagnostics(benchmark, density_concentration_out)
    plot_control_diagnostics(benchmark, comparison, control_out)
    plot_entry_density_high_eta_regimes(benchmark, entry_high_eta_out)
    plot_entry_density_low_eta_regimes(benchmark, entry_low_eta_out)
    plot_hjb_parameter_diagnostics(benchmark, hjb_sweep_out)
    plot_comparative_statics_diagnostics(benchmark, comparative_out)
    print(f"\nSaved plot -> {args.out}")
    print(f"Saved density diagnostics -> {density_out}")
    print(f"Saved density concentration diagnostics -> {density_concentration_out}")
    print(f"Saved control diagnostics -> {control_out}")
    print(f"Saved entry density high eta -> {entry_high_eta_out}")
    print(f"Saved entry density low eta -> {entry_low_eta_out}")
    print(f"Saved HJB parameter diagnostics -> {hjb_sweep_out}")
    print(f"Saved comparative statics -> {comparative_out}")
    print(f"Saved density temperature scale -> {density_temperature_out}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()