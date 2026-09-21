"""
Experiment 3: Before-default direct and indirect default-intensity effects.

Tests how changing one asset's current default intensity in h=(0,0) affects
both its own optimal portfolio weight, the other asset's weight, and total
risky allocation. Replicates Figures 3 and 4.

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
COLORS = ["steelblue", "darkorange"]


def _solve_and_extract(params, t_bm=0.3):
    """Run full recursive solver and extract h=(0,0) controls."""
    solver = HJBSolver(params)
    solver.solve()
    rows = []
    for i in range(params.n_regimes):
        pi = solver.optimal_pi(t_bm, i, (0, 0))
        c = solver.optimal_c(t_bm, i, (0, 0))
        phi = solver.phi(t_bm, i, (0, 0))
        rows.append({
            "regime": i + 1,
            "pi_1": pi[0], "pi_2": pi[1],
            "pi_total": pi[0] + pi[1], "c_star": c, "phi": phi,
        })
    return rows


def run_exp3_scenario_a() -> pd.DataFrame:
    """
    Scenario A: Vary lambda_2^{(0,0)}(1) – asset 2's pre-default intensity
    in regime 1 – and record h=(0,0) controls at t=0.3.
    """
    base = benchmark_params()
    lam_grid = np.linspace(0.2, 2.5, 30)
    rows = []
    print("  Scenario A sweep...")
    for lam_val in lam_grid:
        p = copy.deepcopy(base)
        arr = p.lam[((0, 0), 0)].copy()
        arr[1] = lam_val   # asset 2 intensity in regime 1
        p.lam[((0, 0), 0)] = arr
        p._precompute()
        ctrl = _solve_and_extract(p)
        for row in ctrl:
            row.update({"scenario": "A", "lam_val": lam_val, "t": 0.3,
                        "varied": "lambda_2_(0,0)_regime1"})
            rows.append(row)
    return pd.DataFrame(rows)


def run_exp3_scenario_b() -> pd.DataFrame:
    """
    Scenario B: Vary lambda_1^{(0,0)}(2) – asset 1's pre-default intensity
    in regime 2 – and record h=(0,0) controls at t=0.3.
    """
    base = benchmark_params()
    lam_grid = np.linspace(0.2, 2.5, 30)
    rows = []
    print("  Scenario B sweep...")
    for lam_val in lam_grid:
        p = copy.deepcopy(base)
        arr = p.lam[((0, 0), 1)].copy()
        arr[0] = lam_val   # asset 1 intensity in regime 2
        p.lam[((0, 0), 1)] = arr
        p._precompute()
        ctrl = _solve_and_extract(p)
        for row in ctrl:
            row.update({"scenario": "B", "lam_val": lam_val, "t": 0.3,
                        "varied": "lambda_1_(0,0)_regime2"})
            rows.append(row)
    return pd.DataFrame(rows)


def run_exp3() -> dict:
    """Run experiment 3: before-default intensity comparative statics."""
    print("\n=== Exp 3: Scenario A – vary lambda_2^{(0,0)} in regime 1 ===")
    df_a = run_exp3_scenario_a()
    print(df_a.to_string(index=False))
    df_a.to_csv(RESULTS / "exp3_scenario_a.csv", index=False)

    print("\n=== Exp 3: Scenario B – vary lambda_1^{(0,0)} in regime 2 ===")
    df_b = run_exp3_scenario_b()
    print(df_b.to_string(index=False))
    df_b.to_csv(RESULTS / "exp3_scenario_b.csv", index=False)

    _plot_exp3(df_a, df_b)
    return {"scenario_a": df_a, "scenario_b": df_b}


def _plot_exp3(df_a, df_b):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    datasets = [
        (df_a, "Scenario A: lambda_2^{(0,0)} (Regime 1)"),
        (df_b, "Scenario B: lambda_1^{(0,0)} (Regime 2)"),
    ]
    for row_idx, (df, label) in enumerate(datasets):
        for i, regime in enumerate([1, 2]):
            s = df[df["regime"] == regime]
            axes[row_idx, 0].plot(s["lam_val"], s["pi_1"], color=COLORS[i], label=f"R{regime}")
            axes[row_idx, 1].plot(s["lam_val"], s["pi_2"], color=COLORS[i], label=f"R{regime}")
            axes[row_idx, 2].plot(s["lam_val"], s["pi_total"], color=COLORS[i], label=f"R{regime}")
        axes[row_idx, 0].set_title(f"pi_1* – {label}")
        axes[row_idx, 0].set_xlabel("lambda")
        axes[row_idx, 0].legend()
        axes[row_idx, 1].set_title(f"pi_2* – {label}")
        axes[row_idx, 1].set_xlabel("lambda")
        axes[row_idx, 1].legend()
        axes[row_idx, 2].set_title(f"pi_total – {label}")
        axes[row_idx, 2].set_xlabel("lambda")
        axes[row_idx, 2].legend()
    fig.suptitle("Exp 3: Before-default intensity comparative statics (h=(0,0))", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp3_before_default_intensity.png", dpi=120)
    plt.close()
    print("Saved: exp3_before_default_intensity.png")


if __name__ == "__main__":
    run_exp3()
