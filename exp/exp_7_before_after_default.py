"""
Experiment 7: Before-vs-after default comparison and consumption expenditure drop.

Compares optimal controls immediately before and after a default event,
including the surviving asset's weight and consumption rate, and computes
the actual consumption expenditure drop due to the simultaneous wealth jump.
Corresponds to Figure 11.

Custom implementation – no library covers this model.
"""

from __future__ import annotations

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


def compare_before_after_default(
    defaulted_asset: int,
    t: float,
    regime: int,
    solver: HJBSolver,
    X_pre: float = 1.0,
) -> dict:
    """
    Compare controls immediately before and after one asset defaults.

    Parameters
    ----------
    defaulted_asset : int
        0 for asset 1, 1 for asset 2.
    t : float
        Forward time of the comparison.
    regime : int
        Regime index (0-based).
    solver : HJBSolver
        Already-solved benchmark model.
    X_pre : float
        Pre-default wealth (normalised to 1.0).

    Returns
    -------
    dict with before/after controls and expenditure drop.
    """
    h_before = (0, 0)
    # If asset 1 defaults → h=(1,0); if asset 2 defaults → h=(0,1)
    h_after = (1, 0) if defaulted_asset == 0 else (0, 1)
    surv_asset = 1 - defaulted_asset  # 1 if asset 1 defaults, 0 otherwise

    pi_before = solver.optimal_pi(t, regime, h_before)
    c_before = solver.optimal_c(t, regime, h_before)

    pi_after = solver.optimal_pi(t, regime, h_after)
    c_after = solver.optimal_c(t, regime, h_after)

    # Wealth jump: X_after = (1 - pi_defaulted) * X_pre
    pi_defaulted = pi_before[defaulted_asset]
    X_after = (1.0 - pi_defaulted) * X_pre

    exp_before = c_before * X_pre
    exp_after = c_after * X_after
    exp_drop = 1.0 - exp_after / exp_before if exp_before > 0 else np.nan

    return {
        "defaulted_asset": defaulted_asset + 1,
        "t": t,
        "regime": regime + 1,
        "pi_surv_before": float(pi_before[surv_asset]),
        "pi_surv_after": float(pi_after[surv_asset]),
        "pi_defaulted_before": float(pi_defaulted),
        "c_before": float(c_before),
        "c_after": float(c_after),
        "X_pre": float(X_pre),
        "X_after": float(X_after),
        "exp_before": float(exp_before),
        "exp_after": float(exp_after),
        "exp_drop_pct": float(100.0 * exp_drop),
    }


def run_exp7() -> dict:
    """Run experiment 7: before-vs-after default comparison."""
    print("\n=== Exp 7: Before-vs-After Default Comparison ===")
    params = benchmark_params()
    solver = HJBSolver(params)
    solver.solve()

    # Benchmark: default of asset 1 at t=0.3, regime 1
    # Paper: pi_surv_before≈0.1315, pi_surv_after≈0.1248,
    #        c_before≈0.3966, c_after≈0.3670, exp_drop≈35.8%
    res_main = compare_before_after_default(
        defaulted_asset=0, t=0.3, regime=0, solver=solver
    )
    print("\nBenchmark: default of asset 1 at t=0.3, regime 1")
    for k, v in res_main.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Full comparison: both assets, both regimes, t=0.3
    rows = []
    for da in [0, 1]:
        for i in range(2):
            rows.append(compare_before_after_default(da, 0.3, i, solver))
    df = pd.DataFrame(rows)
    print("\n=== Controls at t=0.3 (all cases) ===")
    print(df.to_string(index=False))
    df.to_csv(RESULTS / "exp7_before_after.csv", index=False)

    # Time series of expenditure drop
    t_grid = np.linspace(0.0, 0.99, 100)
    exp_series = {}
    for da in [0, 1]:
        for i in range(2):
            series = []
            for t in t_grid:
                try:
                    r = compare_before_after_default(da, t, i, solver)
                    series.append(r["exp_drop_pct"])
                except Exception:
                    series.append(np.nan)
            exp_series[(da, i)] = series

    _plot_exp7(df, t_grid, exp_series)
    return {"table": df, "benchmark": res_main, "exp_drop_series": exp_series}


def _plot_exp7(df, t_grid, exp_series):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(df))
    # Surviving asset pi* before vs after
    axes[0].bar(x - 0.2, df["pi_surv_before"], width=0.4, color="steelblue", label="Before")
    axes[0].bar(x + 0.2, df["pi_surv_after"], width=0.4, color="darkorange", label="After")
    axes[0].set_xticks(x)
    labels = [f"DA={r['defaulted_asset']},R{r['regime']}" for _, r in df.iterrows()]
    axes[0].set_xticklabels(labels, rotation=30, fontsize=7)
    axes[0].set_title("Surviving asset pi* before/after default")
    axes[0].legend()

    # Consumption before vs after
    axes[1].bar(x - 0.2, df["c_before"], width=0.4, color="steelblue", label="Before")
    axes[1].bar(x + 0.2, df["c_after"], width=0.4, color="darkorange", label="After")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, fontsize=7)
    axes[1].set_title("c* before/after default")
    axes[1].legend()

    # Expenditure drop over time
    colors = ["steelblue", "darkorange", "green", "red"]
    for ci, (key, series) in enumerate(exp_series.items()):
        da, i = key
        axes[2].plot(t_grid, series, color=colors[ci], label=f"DA={da+1},R{i+1}")
    axes[2].set_title("Expenditure drop % over time")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("Drop %")
    axes[2].legend(fontsize=8)

    fig.suptitle("Exp 7: Before-vs-After Default", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp7_before_after_default.png", dpi=120)
    plt.close()
    print("Saved: exp7_before_after_default.png")


if __name__ == "__main__":
    run_exp7()
