#!/usr/bin/env python3
"""
Quick ρ sweep for rl_optimal_stopping_v5.py

Runs 5 values of ρ with M=5.0, η_end=0.005 fixed.
Outputs a summary table and saves results to sweep_rho_results.csv.
"""
import subprocess
import sys
import re
import csv
import time

RHO_VALUES = [0.001, 0.002, 0.003, 0.006, 0.008]

# Patterns to extract from output
PATTERNS = {
    "trades": r"trades=([\d.]+)±",
    "U_sum": r"UΣ=([\d.-]+)±",
    "U_mu": r"Uμ=([\d.-]+)±([\d.]+)",
    "U_SR": r"U-SR=([\d.-]+)±",
    "win": r"Win=([\d.]+)±",
    "avg_hold": r"AvgH=(\d+)±",
    "avg_edge": r"avg_edge=([\d.-]+)±",
    "raw_SR": r"raw_SR=([\d.-]+)±",
    "exit_z": r"avg_exit_z=([\d.-]+)±",
    "policy_exit": r"policy_exit=([\d.]+)±",
    "inv_frac": r"invested=([\d.]+)±",
}


def patch_rho(rho_val):
    """Temporarily patch cfg.rho in v5 by editing the file."""
    with open("rl_optimal_stopping_v5.py", "r") as f:
        content = f.read()

    # Match the rho= assignment in the Cfg class
    patched = re.sub(
        r"(gamma=1\.0; iota=1\.0; R=0\.0; rho=)[\d.eE-]+",
        rf"\g<1>{rho_val}",
        content,
    )
    with open("rl_optimal_stopping_v5.py", "w") as f:
        f.write(patched)


def extract_metrics(output):
    """Extract aggregate RL metrics from stdout."""
    result = {"rho": None}
    lines = output.split("\n")

    # Find the RL aggregate lines (first occurrence of "RL mean±std")
    rl_lines = [l for l in lines if "RL mean±std" in l]
    # Find trade diagnostic RL lines
    diag_lines = [l for l in lines if "RL mean±std" in l and "entry<" in l]

    search_text = "\n".join(rl_lines + diag_lines)

    for key, pattern in PATTERNS.items():
        m = re.search(pattern, search_text)
        if m:
            result[key] = float(m.group(1))
        else:
            result[key] = None

    return result


def run_single(rho_val, run_idx):
    """Run v5 with a specific ρ and capture output."""
    print(f"\n{'='*60}")
    print(f"  Sweep run {run_idx+1}/{len(RHO_VALUES)}: ρ = {rho_val}")
    print(f"{'='*60}")

    patch_rho(rho_val)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "rl_optimal_stopping_v5.py"],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min safety cap
    )
    elapsed = time.time() - t0

    # Print abbreviated output
    stdout = proc.stdout
    for line in stdout.split("\n"):
        if any(k in line for k in ["iter ", "Final:", "Aggregate", "Trade diag",
                                     "RL mean", "RL Agent", "Baseline", "Elapsed"]):
            print(f"  {line.strip()}")

    result = extract_metrics(stdout)
    result["rho"] = rho_val
    result["elapsed_s"] = int(elapsed)

    print(f"  → ρ={rho_val}: trades={result.get('trades')}, "
          f"Uμ={result.get('U_mu')}, edge={result.get('avg_edge')}, "
          f"exit_z={result.get('exit_z')}, hold={result.get('avg_hold')}d, "
          f"raw_SR={result.get('raw_SR')}, elapsed={elapsed:.0f}s")

    return result


def main():
    print("ρ sweep for rl_optimal_stopping_v5.py")
    print(f"Values: {RHO_VALUES}")
    print(f"Fixed: M=5.0, η_end=0.005")

    # Save original file
    with open("rl_optimal_stopping_v5.py", "r") as f:
        original = f.read()

    results = []
    try:
        for i, rho in enumerate(RHO_VALUES):
            result = run_single(rho, i)
            results.append(result)
    finally:
        # Restore original file
        with open("rl_optimal_stopping_v5.py", "w") as f:
            f.write(original)
        print("\nRestored original rl_optimal_stopping_v5.py")

    # Summary table
    print(f"\n{'='*90}")
    print("  ρ SWEEP SUMMARY")
    print(f"{'='*90}")
    hdr = f"{'ρ':>8} {'Trades':>7} {'Uμ':>7} {'Edge':>7} {'ExitZ':>7} {'AvgH':>6} {'Win%':>6} {'PolEx%':>7} {'RawSR':>7} {'Time':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['rho']:>8.4f} "
              f"{r.get('trades', 0) or 0:>7.1f} "
              f"{r.get('U_mu', 0) or 0:>7.4f} "
              f"{r.get('avg_edge', 0) or 0:>7.2f} "
              f"{r.get('exit_z', 0) or 0:>7.2f} "
              f"{r.get('avg_hold', 0) or 0:>6.0f} "
              f"{r.get('win', 0) or 0:>5.1f}% "
              f"{r.get('policy_exit', 0) or 0:>6.1f}% "
              f"{r.get('raw_SR', 0) or 0:>7.2f} "
              f"{r.get('elapsed_s', 0):>5}s")
    print("-" * len(hdr))

    # Save CSV
    csv_path = "sweep_rho_results.csv"
    fieldnames = ["rho", "trades", "U_mu", "U_sum", "U_SR", "avg_edge",
                  "exit_z", "avg_hold", "win", "policy_exit", "inv_frac",
                  "raw_SR", "elapsed_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved → {csv_path}")

    # Best run
    valid = [r for r in results if r.get("avg_edge") is not None]
    if valid:
        best = max(valid, key=lambda r: r["avg_edge"])
        print(f"\nBest by edge: ρ={best['rho']} → edge={best['avg_edge']:.2f}, "
              f"Uμ={best.get('U_mu'):.4f}, exit_z={best.get('exit_z'):.2f}")


if __name__ == "__main__":
    main()
