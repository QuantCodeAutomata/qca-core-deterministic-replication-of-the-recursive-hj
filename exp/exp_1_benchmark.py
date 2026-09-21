"""
Experiment 1: Core deterministic replication of the recursive HJB system.

Reproduces benchmark value multipliers varphi(t,i,h), optimal portfolio weights
pi*(t,i,h), and optimal consumption rates c*(t,i,h) for all regimes and default
states. Validates the recursive solution and benchmark point values from the paper.

Custom HJB implementation – no library directly provides this recursive
default-contagion regime-switching model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.model_params import benchmark_params, SOLVE_ORDER
from src.hjb_solver import HJBSolver

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

STATE_LABELS = {(1, 1): "h=(1,1)", (0, 1): "h=(0,1)", (1, 0): "h=(1,0)", (0, 0): "h=(0,0)"}
COLORS = ["steelblue", "darkorange"]


def run_exp1() -> dict:
    """
    Run experiment 1: benchmark HJB replication.

    Returns
    -------
    dict with keys 'table', 'terminal_checks', 'positivity',
    'hjb_residuals', 'benchmark_values'.
    """
    params = benchmark_params()
    t_grid = np.linspace(0.0, 1.0, 201)
    solver = HJBSolver(params, t_eval=t_grid)
    solver.solve()

    # ── Terminal condition check ──────────────────────────────────────────────
    term_checks = solver.check_terminal_condition(tol=1e-5)
    print("\n=== Terminal Condition Checks ===")
    for key, v in term_checks.items():
        print(
            f"  h={key[0]}, regime={key[1]+1}: phi(T)={v['phi_T']:.8f}, "
            f"error={v['error']:.2e}, pass={v['pass']}"
        )

    # ── Positivity check ──────────────────────────────────────────────────────
    pos_ok = solver.check_positivity(t_grid)
    print(f"\nPositivity of phi across all states/times: {pos_ok}")

    # ── HJB residuals ─────────────────────────────────────────────────────────
    t_check = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
    residuals = {}
    for h in SOLVE_ORDER:
        r = solver.compute_hjb_residuals(t_check, h=h)
        residuals[h] = r
        print(f"  HJB residuals h={h}: max={r.max():.2e}")

    # ── Tabulate controls ─────────────────────────────────────────────────────
    t_report = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    rows = []
    for h in SOLVE_ORDER:
        rows.extend(solver.tabulate_controls(t_report, h=h))
    df = pd.DataFrame(rows)
    print("\n=== Controls Table (h=(0,0)) ===")
    print(df[df["h"] == str((0, 0))].to_string(index=False))

    # ── Benchmark point values at t=0.3 ──────────────────────────────────────
    t_bm = 0.3
    bm = {}
    for i in range(2):
        for h in SOLVE_ORDER:
            phi_val = solver.phi(t_bm, i, h)
            pi = solver.optimal_pi(t_bm, i, h)
            c_star = solver.optimal_c(t_bm, i, h)
            bm[(i, h)] = {"phi": phi_val, "pi": pi, "c": c_star}
            print(
                f"  t=0.3, regime={i+1}, h={h}: "
                f"phi={phi_val:.4f}, pi={np.round(pi, 4)}, c={c_star:.4f}"
            )

    # Paper benchmark:
    #   regime 1, h=(0,0): pi=(0.3056, 0.1315), c=0.3966
    #   regime 2, h=(0,0): pi=(0.1676, 0.0533), c=0.3742
    val_r1 = bm[(0, (0, 0))]
    val_r2 = bm[(1, (0, 0))]
    print("\nBenchmark check (regime 1, h=(0,0), t=0.3):")
    print(
        f"  pi1={val_r1['pi'][0]:.4f} (paper: 0.3056), "
        f"pi2={val_r1['pi'][1]:.4f} (paper: 0.1315), "
        f"c={val_r1['c']:.4f} (paper: 0.3966)"
    )
    print("Benchmark check (regime 2, h=(0,0), t=0.3):")
    print(
        f"  pi1={val_r2['pi'][0]:.4f} (paper: 0.1676), "
        f"pi2={val_r2['pi'][1]:.4f} (paper: 0.0533), "
        f"c={val_r2['c']:.4f} (paper: 0.3742)"
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_benchmark(solver, t_grid)
    df.to_csv(RESULTS / "exp1_controls_table.csv", index=False)

    return {
        "table": df,
        "terminal_checks": term_checks,
        "positivity": pos_ok,
        "hjb_residuals": residuals,
        "benchmark_values": bm,
    }


def _plot_benchmark(solver: HJBSolver, t_grid: np.ndarray) -> None:
    """Generate benchmark time-series plots for all states and regimes."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, h in enumerate(SOLVE_ORDER):
        for i in range(2):
            phi_arr = np.array([solver.phi(t, i, h) for t in t_grid])
            pi_arr = np.array([solver.optimal_pi(t, i, h) for t in t_grid])
            axes[0, col].plot(t_grid, phi_arr, color=COLORS[i], label=f"R{i+1}")
            axes[1, col].plot(t_grid, pi_arr[:, 0], "--", color=COLORS[i], label=f"pi1 R{i+1}")
            if h == (0, 0):
                axes[1, col].plot(t_grid, pi_arr[:, 1], "-.", color=COLORS[i], label=f"pi2 R{i+1}")
        axes[0, col].set_title(f"phi(t) {STATE_LABELS[h]}")
        axes[0, col].set_xlabel("t")
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_title(f"pi*(t) {STATE_LABELS[h]}")
        axes[1, col].set_xlabel("t")
        axes[1, col].legend(fontsize=7)
    fig.suptitle("Exp 1: Benchmark HJB Solution - phi and pi*", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp1_benchmark_controls.png", dpi=120)
    plt.close()

    # Consumption plot
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
    for col, h in enumerate(SOLVE_ORDER):
        for i in range(2):
            c_arr = np.array([solver.optimal_c(t, i, h) for t in t_grid])
            axes2[col].plot(t_grid, c_arr, color=COLORS[i], label=f"R{i+1}")
        axes2[col].set_title(f"c*(t) {STATE_LABELS[h]}")
        axes2[col].set_xlabel("t")
        axes2[col].legend(fontsize=8)
    fig2.suptitle("Exp 1: Benchmark HJB Solution - c*", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp1_benchmark_consumption.png", dpi=120)
    plt.close()
    print("Saved: exp1_benchmark_controls.png, exp1_benchmark_consumption.png")


if __name__ == "__main__":
    run_exp1()
