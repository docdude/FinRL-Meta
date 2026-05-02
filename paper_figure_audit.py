#!/usr/bin/env python3
"""Automated audit of paper Figure 1/2/4/5 panels against the local OU benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


PAPER_PANEL_URLS = {
    "fig1_v0_fixed_M50.png": "https://arxiv.org/html/2604.02035v1/graphs/compare_fixed_M=50_V0.png",
    "fig1_v0_fixed_eta1e-05.png": "https://arxiv.org/html/2604.02035v1/graphs/compare_fixed_eta=1e-05_V0.png",
    "fig2_v1_M1_eta1e-05.png": "https://arxiv.org/html/2604.02035v1/graphs/plot_V1_M=1_eta=1e-05.png",
    "fig2_v1_M50_eta1e-05.png": "https://arxiv.org/html/2604.02035v1/graphs/plot_V1_M=50_eta=1e-05.png",
    "fig4_entry_M50_eta1e-05.png": "https://arxiv.org/html/2604.02035v1/graphs/optimal_density_heatmap_entry_M=50_eta=1e-05.png",
    "fig5_exit_bneg1.png": "https://arxiv.org/html/2604.02035v1/graphs/optimal_density_heatmap_exit_M=50_eta=1e-05_b-1.png",
    "fig5_exit_b1.png": "https://arxiv.org/html/2604.02035v1/graphs/optimal_density_heatmap_exit_M=50_eta=1e-05_b1.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper figures against the local HJB benchmark")
    parser.add_argument("--refresh-paper", action="store_true", help="Re-download paper panels even if cached")
    parser.add_argument(
        "--out-prefix",
        default="paper_figure_audit",
        help="Prefix for generated audit artifacts",
    )
    return parser.parse_args()


def load_ou_module(root: Path):
    module_path = root / "rl_optimal_stopping_ou_repro.py"
    spec = importlib.util.spec_from_file_location("ou_repro", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_paper_panels(root: Path, refresh: bool) -> dict[str, Path]:
    paper_dir = root / ".paper_figs"
    paper_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, url in PAPER_PANEL_URLS.items():
        path = paper_dir / filename
        if refresh or not path.exists():
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        paths[filename] = path
    return paths


def load_rgb(path: Path) -> np.ndarray:
    rgb = mpimg.imread(path)[..., :3]
    if rgb.dtype.kind != "f":
        rgb = rgb.astype(np.float64) / 255.0
    return np.asarray(rgb, dtype=np.float64)


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def longest_run(runs: list[tuple[int, int]]) -> tuple[int, int]:
    if not runs:
        raise ValueError("No contiguous runs found")
    return max(runs, key=lambda item: item[1] - item[0])


def find_heatmap_plot_box(rgb: np.ndarray, coverage: float = 0.18) -> tuple[int, int, int, int]:
    nonwhite = np.any(rgb < 0.98, axis=2)
    row_counts = nonwhite.sum(axis=1)
    col_counts = nonwhite.sum(axis=0)
    row_runs = contiguous_runs(np.where(row_counts > rgb.shape[1] * coverage)[0])
    col_runs = contiguous_runs(np.where(col_counts > rgb.shape[0] * coverage)[0])
    row0, row1 = longest_run(row_runs)
    col0, col1 = longest_run(col_runs)
    pad = 6
    row0 = min(max(row0 + pad, 0), rgb.shape[0] - 2)
    row1 = max(min(row1 - pad, rgb.shape[0] - 1), row0 + 1)
    col0 = min(max(col0 + pad, 0), rgb.shape[1] - 2)
    col1 = max(min(col1 - pad, rgb.shape[1] - 1), col0 + 1)
    return row0, row1, col0, col1


def crop_box(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    row0, row1, col0, col1 = box
    return rgb[row0 : row1 + 1, col0 : col1 + 1]


def darkness_field(rgb: np.ndarray) -> np.ndarray:
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return 1.0 - luminance


def robust_minmax(field: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    lo, hi = np.nanpercentile(field, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        lo = float(np.nanmin(field))
        hi = float(np.nanmax(field))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(field, dtype=np.float64)
    scaled = (field - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def resize_bilinear(field: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_shape
    in_h, in_w = field.shape
    x_old = np.linspace(0.0, 1.0, in_w)
    x_new = np.linspace(0.0, 1.0, out_w)
    tmp = np.vstack([np.interp(x_new, x_old, row) for row in field])
    y_old = np.linspace(0.0, 1.0, in_h)
    y_new = np.linspace(0.0, 1.0, out_h)
    resized = np.vstack([np.interp(y_new, y_old, tmp[:, column]) for column in range(out_w)]).T
    return resized


def correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_vec = lhs.ravel()
    rhs_vec = rhs.ravel()
    if np.std(lhs_vec) < 1e-12 or np.std(rhs_vec) < 1e-12:
        return np.nan
    return float(np.corrcoef(lhs_vec, rhs_vec)[0, 1])


def profile_slopes(field: np.ndarray) -> tuple[float, float]:
    p_profile = field.mean(axis=1)[::-1]
    b_profile = field.mean(axis=0)
    p_axis = np.linspace(-1.0, 1.0, len(p_profile))
    b_axis = np.linspace(-1.0, 1.0, len(b_profile))
    p_slope = float(np.polyfit(p_axis, p_profile, deg=1)[0])
    b_slope = float(np.polyfit(b_axis, b_profile, deg=1)[0])
    return p_slope, b_slope


def interior_indices(ou, benchmark) -> tuple[np.ndarray, np.ndarray]:
    p_grid = benchmark["p_grid"]
    b_grid = benchmark["b_grid"]
    p_idx = np.where((p_grid >= ou.cfg.grid_lo) & (p_grid <= ou.cfg.grid_hi))[0]
    b_idx = np.where((b_grid >= ou.cfg.grid_lo) & (b_grid <= ou.cfg.grid_hi))[0]
    return p_idx, b_idx


def density_lam_grid(ou, max_lam: float, points: int | None = None) -> np.ndarray:
    lam_points = ou.cfg.density_lam_points if points is None else int(points)
    return np.linspace(0.0, max_lam, lam_points)


def gibbs_density_np(ou, delta, lam_grid, eta: float | None = None) -> np.ndarray:
    eta_value = ou.cfg.eta if eta is None else eta
    delta = np.asarray(delta, dtype=np.float64)
    lam_grid = np.asarray(lam_grid, dtype=np.float64)
    scores = np.outer(delta / eta_value, lam_grid)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    norm = np.trapezoid(weights, lam_grid, axis=1)
    return weights / np.clip(norm[:, None], 1e-300, None)


def density_display(density: np.ndarray, floor: float = 1e-30, ceiling: float | None = None) -> np.ndarray:
    vmax = max(float(ceiling), floor * 10) if ceiling is not None else max(float(density.max()), floor * 10)
    clipped = np.clip(density, floor, vmax)
    log_min = np.log(floor)
    log_max = np.log(vmax)
    return np.clip((np.log(clipped) - log_min) / (log_max - log_min), 0.0, 1.0)


def entry_density_case(ou, result, M_value: float, eta_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_idx, _ = interior_indices(ou, result)
    p_view = result["p_grid"][p_idx]
    delta = result["delta1"][p_idx]
    lam_grid = density_lam_grid(ou, M_value)
    density = gibbs_density_np(ou, delta, lam_grid, eta=eta_value)
    return p_view, lam_grid, density


def compare_heatmap(paper_path: Path, local_field: np.ndarray) -> dict[str, object]:
    paper_rgb = load_rgb(paper_path)
    box = find_heatmap_plot_box(paper_rgb)
    paper_crop = crop_box(paper_rgb, box)
    paper_field = robust_minmax(darkness_field(paper_crop))
    local_resized = robust_minmax(resize_bilinear(local_field, paper_field.shape))
    p_slope_paper, b_slope_paper = profile_slopes(paper_field)
    p_slope_local, b_slope_local = profile_slopes(local_resized)
    return {
        "paper_path": paper_path,
        "paper_box": box,
        "paper_field": paper_field,
        "local_field": local_resized,
        "correlation": correlation(paper_field, local_resized),
        "mae": float(np.mean(np.abs(paper_field - local_resized))),
        "paper_p_slope": p_slope_paper,
        "paper_b_slope": b_slope_paper,
        "local_p_slope": p_slope_local,
        "local_b_slope": b_slope_local,
    }


def extract_visible_start(field: np.ndarray, max_lam: float, threshold: float = 0.08) -> np.ndarray:
    mask = field >= threshold
    starts = np.full(field.shape[0], np.nan, dtype=np.float64)
    valid = mask.any(axis=1)
    if np.any(valid):
        starts[valid] = max_lam * np.argmax(mask[valid], axis=1) / max(field.shape[1] - 1, 1)
    return starts


def sample_profile_at_p(values: np.ndarray, p_lo: float, p_hi: float, p_target: float) -> float:
    p_axis = np.linspace(p_hi, p_lo, len(values))
    index = int(np.argmin(np.abs(p_axis - p_target)))
    return float(values[index])


def make_v1_image_field(ou, result) -> np.ndarray:
    p_idx, b_idx = interior_indices(ou, result)
    field = result["v1"][np.ix_(b_idx, p_idx)].T
    return robust_minmax(field[::-1])


def make_entry_density_image_field(ou, result, M_value: float, eta_value: float) -> np.ndarray:
    _, _, density = entry_density_case(ou, result, M_value, eta_value)
    return density_display(density, floor=1e-30, ceiling=1.0)[::-1]


def make_exit_density_image_field(ou, result, b_value: float) -> np.ndarray:
    p_idx, _ = interior_indices(ou, result)
    delta = result["delta2"][result["slice_indices"][b_value]][p_idx]
    lam_grid = density_lam_grid(ou, ou.cfg.M)
    density = gibbs_density_np(ou, delta, lam_grid, eta=ou.cfg.eta)
    return density_display(density, floor=1e-30, ceiling=1.0)[::-1]


def audit_figure1(ou) -> dict[str, object]:
    eta_values = sorted(float(value) for value in ou.cfg.sweep_eta_values)
    m_values = sorted(float(value) for value in ou.cfg.sweep_M_values)
    p_targets = (-1.0, 0.0, 1.0)

    eta_curves = {p_target: [] for p_target in p_targets}
    for eta in eta_values:
        result = ou.solve_hjb_with_overrides(M=ou.cfg.M, eta=eta)
        for p_target in p_targets:
            p_index = int(np.argmin(np.abs(result["p_grid"] - p_target)))
            eta_curves[p_target].append(float(result["v0"][p_index]))

    m_curves = {p_target: [] for p_target in p_targets}
    for m_value in m_values:
        result = ou.solve_hjb_with_overrides(M=m_value, eta=ou.cfg.eta)
        for p_target in p_targets:
            p_index = int(np.argmin(np.abs(result["p_grid"] - p_target)))
            m_curves[p_target].append(float(result["v0"][p_index]))

    eta_decreasing = {p_target: bool(np.all(np.diff(eta_curves[p_target]) < 0.0)) for p_target in p_targets}
    m_increasing = {p_target: bool(np.all(np.diff(m_curves[p_target]) > 0.0)) for p_target in p_targets}
    return {
        "eta_values": eta_values,
        "m_values": m_values,
        "eta_curves": eta_curves,
        "m_curves": m_curves,
        "eta_decreasing": eta_decreasing,
        "m_increasing": m_increasing,
        "paper_eta_text_claim": "V0 is increasing as eta increases",
        "paper_m_text_claim": "V0 is increasing as M increases",
    }


def audit_figure2(ou, paper_paths: dict[str, Path]) -> dict[str, object]:
    low_m = ou.solve_hjb_with_overrides(M=1.0, eta=ou.cfg.eta)
    high_m = ou.solve_hjb_with_overrides(M=ou.cfg.M, eta=ou.cfg.eta)
    eta_alt = ou.solve_hjb_with_overrides(M=ou.cfg.M, eta=1e-4)

    p_idx, b_idx = interior_indices(ou, high_m)
    diff = eta_alt["v1"][np.ix_(b_idx, p_idx)] - high_m["v1"][np.ix_(b_idx, p_idx)]

    return {
        "M=1": compare_heatmap(paper_paths["fig2_v1_M1_eta1e-05.png"], make_v1_image_field(ou, low_m)),
        "M=50": compare_heatmap(paper_paths["fig2_v1_M50_eta1e-05.png"], make_v1_image_field(ou, high_m)),
        "eta_sensitivity_mae": float(np.mean(np.abs(diff))),
        "eta_sensitivity_max": float(np.max(np.abs(diff))),
        "entry_boundary_eta1e-5": float(high_m["entry_boundary"]),
        "entry_boundary_eta1e-4": float(eta_alt["entry_boundary"]),
    }


def audit_figure4(ou, paper_paths: dict[str, Path]) -> dict[str, object]:
    result = ou.solve_hjb_with_overrides(M=ou.cfg.M, eta=ou.cfg.eta)
    comparison = compare_heatmap(
        paper_paths["fig4_entry_M50_eta1e-05.png"],
        make_entry_density_image_field(ou, result, ou.cfg.M, ou.cfg.eta),
    )
    paper_starts = extract_visible_start(comparison["paper_field"], ou.cfg.M)
    local_starts = extract_visible_start(comparison["local_field"], ou.cfg.M)
    return {
        "comparison": comparison,
        "paper_start_p=-1.0": sample_profile_at_p(paper_starts, ou.cfg.grid_lo, ou.cfg.grid_hi, -1.0),
        "local_start_p=-1.0": sample_profile_at_p(local_starts, ou.cfg.grid_lo, ou.cfg.grid_hi, -1.0),
        "paper_start_p=-0.5": sample_profile_at_p(paper_starts, ou.cfg.grid_lo, ou.cfg.grid_hi, -0.5),
        "local_start_p=-0.5": sample_profile_at_p(local_starts, ou.cfg.grid_lo, ou.cfg.grid_hi, -0.5),
        "entry_boundary": float(result["entry_boundary"]),
    }


def audit_figure5(ou, paper_paths: dict[str, Path], b_value: float) -> dict[str, object]:
    result = ou.solve_hjb_with_overrides(M=ou.cfg.M, eta=ou.cfg.eta)
    paper_name = "fig5_exit_bneg1.png" if b_value < 0 else "fig5_exit_b1.png"
    comparison = compare_heatmap(paper_paths[paper_name], make_exit_density_image_field(ou, result, b_value))
    paper_starts = extract_visible_start(comparison["paper_field"], ou.cfg.M)
    local_starts = extract_visible_start(comparison["local_field"], ou.cfg.M)
    samples = {}
    for p_target in (0.5, 1.5, 2.5):
        samples[f"paper_start_p={p_target:.1f}"] = sample_profile_at_p(paper_starts, ou.cfg.grid_lo, ou.cfg.grid_hi, p_target)
        samples[f"local_start_p={p_target:.1f}"] = sample_profile_at_p(local_starts, ou.cfg.grid_lo, ou.cfg.grid_hi, p_target)
    return {
        "comparison": comparison,
        "exit_boundary": float(result["exit_boundaries"][b_value]),
        **samples,
    }


def format_report(fig1: dict[str, object], fig2: dict[str, object], fig4: dict[str, object], fig5_neg: dict[str, object], fig5_pos: dict[str, object]) -> str:
    lines = []
    lines.append("Paper figure audit against local OU HJB benchmark")
    lines.append("")
    lines.append("Figure 1 audit:")
    lines.append(f"  paper text claim: {fig1['paper_eta_text_claim']}")
    for p_target, values in fig1["eta_curves"].items():
        direction = "decreasing" if fig1["eta_decreasing"][p_target] else "not monotone decreasing"
        series = ", ".join(f"{value:.4f}" for value in values)
        lines.append(f"  local V0(p={p_target:.1f}) across eta {fig1['eta_values']}: {series} -> {direction}")
    lines.append(f"  paper text claim: {fig1['paper_m_text_claim']}")
    for p_target, values in fig1["m_curves"].items():
        direction = "increasing" if fig1["m_increasing"][p_target] else "not monotone increasing"
        series = ", ".join(f"{value:.4f}" for value in values)
        lines.append(f"  local V0(p={p_target:.1f}) across M {fig1['m_values']}: {series} -> {direction}")
    lines.append("")
    lines.append("Figure 2 audit:")
    for label in ("M=1", "M=50"):
        entry = fig2[label]
        lines.append(
            f"  {label}: heatmap correlation={entry['correlation']:.4f}, MAE={entry['mae']:.4f}, "
            f"paper slopes (p={entry['paper_p_slope']:.4f}, b={entry['paper_b_slope']:.4f}), "
            f"local slopes (p={entry['local_p_slope']:.4f}, b={entry['local_b_slope']:.4f})"
        )
    lines.append(
        f"  eta sensitivity at M=50: V1 MAE(1e-4 vs 1e-5)={fig2['eta_sensitivity_mae']:.6f}, "
        f"max={fig2['eta_sensitivity_max']:.6f}, "
        f"entry boundary shift={fig2['entry_boundary_eta1e-4'] - fig2['entry_boundary_eta1e-5']:.6f}"
    )
    lines.append("")
    lines.append("Figure 4 audit:")
    entry = fig4["comparison"]
    lines.append(
        f"  heatmap correlation={entry['correlation']:.4f}, MAE={entry['mae']:.4f}, entry boundary={fig4['entry_boundary']:.6f}"
    )
    lines.append(
        f"  visible start at p=-1.0: paper~{fig4['paper_start_p=-1.0']:.2f}, local~{fig4['local_start_p=-1.0']:.2f}"
    )
    lines.append(
        f"  visible start at p=-0.5: paper~{fig4['paper_start_p=-0.5']:.2f}, local~{fig4['local_start_p=-0.5']:.2f}"
    )
    lines.append("")
    lines.append("Figure 5 audit:")
    for label, payload in (("b=-1", fig5_neg), ("b=1", fig5_pos)):
        comparison = payload["comparison"]
        lines.append(
            f"  {label}: heatmap correlation={comparison['correlation']:.4f}, MAE={comparison['mae']:.4f}, "
            f"exit boundary={payload['exit_boundary']:.6f}"
        )
        for p_target in (0.5, 1.5, 2.5):
            paper_key = f"paper_start_p={p_target:.1f}"
            local_key = f"local_start_p={p_target:.1f}"
            lines.append(
                f"    visible start at p={p_target:.1f}: paper~{payload[paper_key]:.2f}, local~{payload[local_key]:.2f}"
            )
    return "\n".join(lines)


def save_heatmap_sheet(
    out_path: Path,
    fig2: dict[str, object],
    fig4: dict[str, object],
    fig5_neg: dict[str, object],
    fig5_pos: dict[str, object],
) -> None:
    rows = [
        ("Figure 2, M=1", fig2["M=1"]),
        ("Figure 2, M=50", fig2["M=50"]),
        ("Figure 4, Entry M=50", fig4["comparison"]),
        ("Figure 5, Exit b=-1", fig5_neg["comparison"]),
        ("Figure 5, Exit b=1", fig5_pos["comparison"]),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.1 * len(rows)), constrained_layout=True)
    for row_index, (title, payload) in enumerate(rows):
        paper_ax, local_ax, diff_ax = axes[row_index]
        paper_ax.imshow(payload["paper_field"], cmap="gray", vmin=0.0, vmax=1.0)
        paper_ax.set_title(f"{title}\nPaper")
        local_ax.imshow(payload["local_field"], cmap="gray", vmin=0.0, vmax=1.0)
        local_ax.set_title(
            f"Local\nCorr={payload['correlation']:.3f}, MAE={payload['mae']:.3f}"
        )
        difference = payload["local_field"] - payload["paper_field"]
        diff_image = diff_ax.imshow(difference, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        diff_ax.set_title("Local - Paper")
        for axis in (paper_ax, local_ax, diff_ax):
            axis.set_xticks([])
            axis.set_yticks([])
    fig.colorbar(diff_image, ax=axes[:, 2], shrink=0.96)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    ou = load_ou_module(root)
    paper_paths = ensure_paper_panels(root, refresh=args.refresh_paper)

    fig1 = audit_figure1(ou)
    fig2 = audit_figure2(ou, paper_paths)
    fig4 = audit_figure4(ou, paper_paths)
    fig5_neg = audit_figure5(ou, paper_paths, -1.0)
    fig5_pos = audit_figure5(ou, paper_paths, 1.0)

    report = format_report(fig1, fig2, fig4, fig5_neg, fig5_pos)
    report_path = root / f"{args.out_prefix}_report.txt"
    sheet_path = root / f"{args.out_prefix}_heatmaps.png"
    report_path.write_text(report + "\n", encoding="utf-8")
    save_heatmap_sheet(sheet_path, fig2, fig4, fig5_neg, fig5_pos)

    print(report)
    print(f"\nSaved report -> {report_path}")
    print(f"Saved heatmap sheet -> {sheet_path}")


if __name__ == "__main__":
    main()