"""
Experiment 4: Regime-switching frequency effect on value multipliers.

Tests that faster switching (Q = s*Q0 for s in {1,2,3,4}) compresses the
gap varphi(t,1,h) - varphi(t,2,h) across all default states.
Corresponds to Figure 5.

Custom implementation – no library covers this model.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.model_params import benchmark_params
from src.hjb_solver import HJBSolver

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

Q0 = np.array([[-0.5, 0.5], [1.0, -1.0]])
SCALES = [1, 2, 3, 4]
COLORS_SCALE = ["navy", "steelblue", "darkorange", "firebrick"]
DEFAULT_STATES = [(1, 1), (0, 1), (1, 0), (0, 0)]


def run_exp4() -> dict:
    """Run experiment 4: regime-switching frequency sweep."""
    base = benchmark_params()
    t_grid = np.linspace(0.0, 1.0, 201)
    t_check = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

    solvers = {}
    rows = []

    print("\n=== Exp 4: Regime-switching frequency effect ===")
    for s in SCALES:
        p = copy.deepcopy(base)
        p.Q = s * Q0
        p._validate()
        p._precompute()
        solver = HJBSolver(p)
        solver.solve()
        solvers[s] = solver

        for h in DEFAULT_STATES:
            for t in t_check:
                phi1 = solver.phi(t, 0, h)
                phi2 = solver.phi(t, 1, h)
                rows.append({
                    "s": s, "h": str(h), "t": t,
                    "phi_r1": phi1, "phi_r2": phi2, "gap": phi1 - phi2,
                })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "exp4_regime_gap.csv", index=False)

    # Benchmark validation at t=0
    print("\nBenchmark check at t=0:")
    for h_tup, h_str in [((0, 0), "(0,0)"), ((1, 1), "(1,1)")]:
        for s in SCALES:
            sub = df[(df["s"] == s) & (df["h"] == str(h_tup)) & (df["t"] == 0.0)]
            if not sub.empty:
                print(f"  s={s}, h={h_str}, t=0: gap={sub['gap'].values[0]:.4f}")
    # Paper: (0,0) gap ~0.4019→0.1746; (1,1) gap ~0.0983→0.0366

    _plot_exp4(df, t_grid, solvers)
    return {"df": df, "solvers": solvers}


def _plot_exp4(df, t_grid, solvers):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for col, h in enumerate(DEFAULT_STATES):
        for ci, s in enumerate(SCALES):
            solver = solvers[s]
            gap_arr = np.array([
                solver.phi(t, 0, h) - solver.phi(t, 1, h) for t in t_grid
            ])
            axes[col].plot(t_grid, gap_arr, color=COLORS_SCALE[ci], label=f"s={s}")
        axes[col].set_title(f"Delta_phi(t) {str(h)}")
        axes[col].set_xlabel("t")
        axes[col].axhline(0, color="k", linestyle="--", linewidth=0.8)
        axes[col].legend(fontsize=8)
    fig.suptitle("Exp 4: Regime gap Delta_phi = phi(t,1,h)-phi(t,2,h) vs switching speed", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp4_regime_gap.png", dpi=120)
    plt.close()
    print("Saved: exp4_regime_gap.png")


if __name__ == "__main__":
    run_exp4()
