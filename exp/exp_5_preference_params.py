"""
Experiment 5: Preference-parameter sweeps – eta and gamma.

Sweeps the Cobb-Douglas weight eta and risk aversion gamma and records
optimal controls at t=0.3, regime 1, h=(0,0). Replicates Figures 6 and 7.

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

T_BM = 0.3
REGIME_IDX = 0  # regime 1 (0-based)
H_BM = (0, 0)


def _solve_and_get_controls(params, t_bm=T_BM, regime_idx=REGIME_IDX, h=H_BM):
    """Solve full recursive system and return controls."""
    solver = HJBSolver(params)
    solver.solve()
    pi = solver.optimal_pi(t_bm, regime_idx, h)
    c = solver.optimal_c(t_bm, regime_idx, h)
    phi = solver.phi(t_bm, regime_idx, h)
    return pi, c, phi


def run_exp5_eta() -> pd.DataFrame:
    """Eta sweep: eta in [0.2, 1.0], gamma=2.5 fixed."""
    base = benchmark_params()
    eta_grid = np.linspace(0.2, 1.0, 25)
    rows = []
    print("  eta sweep...")
    for eta_val in eta_grid:
        p = copy.deepcopy(base)
        p.eta = eta_val
        p._precompute()
        try:
            pi, c, phi = _solve_and_get_controls(p)
            rows.append({
                "eta": eta_val, "gamma": p.gamma,
                "pi_1": pi[0], "pi_2": pi[1],
                "pi_total": pi[0] + pi[1], "c_star": c, "phi": phi,
            })
        except Exception as exc:
            print(f"  eta={eta_val:.3f} failed: {exc}")
    df = pd.DataFrame(rows)
    # Paper: c rises ~0.1722→0.6494; pi_1 slight decrease ~0.3061→0.3042
    return df


def run_exp5_gamma() -> pd.DataFrame:
    """Gamma sweep: gamma in [1.2, 5.0], eta=0.5 fixed."""
    base = benchmark_params()
    gamma_grid = np.linspace(1.2, 5.0, 25)
    rows = []
    print("  gamma sweep...")
    for gamma_val in gamma_grid:
        p = copy.deepcopy(base)
        p.gamma = gamma_val
        p._precompute()
        try:
            pi, c, phi = _solve_and_get_controls(p)
            rows.append({
                "eta": p.eta, "gamma": gamma_val,
                "pi_1": pi[0], "pi_2": pi[1],
                "pi_total": pi[0] + pi[1], "c_star": c, "phi": phi,
            })
        except Exception as exc:
            print(f"  gamma={gamma_val:.3f} failed: {exc}")
    df = pd.DataFrame(rows)
    # Paper: pi_1 decreases ~0.5771→0.1584; c* increases ~0.3213→0.4300
    return df


def run_exp5() -> dict:
    """Run experiment 5: preference parameter sweeps."""
    print("\n=== Exp 5: Eta Sweep ===")
    df_eta = run_exp5_eta()
    print(df_eta.to_string(index=False))
    df_eta.to_csv(RESULTS / "exp5_eta_sweep.csv", index=False)

    print("\n=== Exp 5: Gamma Sweep ===")
    df_gamma = run_exp5_gamma()
    print(df_gamma.to_string(index=False))
    df_gamma.to_csv(RESULTS / "exp5_gamma_sweep.csv", index=False)

    _plot_exp5(df_eta, df_gamma)
    return {"eta_sweep": df_eta, "gamma_sweep": df_gamma}


def _plot_exp5(df_eta, df_gamma):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if not df_eta.empty:
        axes[0, 0].plot(df_eta["eta"], df_eta["c_star"], "steelblue")
        axes[0, 0].set_title("c* vs eta (gamma=2.5, h=(0,0), t=0.3, R1)")
        axes[0, 0].set_xlabel("eta")
        axes[0, 1].plot(df_eta["eta"], df_eta["pi_1"], "steelblue", label="pi_1")
        axes[0, 1].plot(df_eta["eta"], df_eta["pi_2"], "darkorange", label="pi_2")
        axes[0, 1].set_title("pi* vs eta")
        axes[0, 1].set_xlabel("eta")
        axes[0, 1].legend()
    if not df_gamma.empty:
        axes[1, 0].plot(df_gamma["gamma"], df_gamma["c_star"], "steelblue")
        axes[1, 0].set_title("c* vs gamma (eta=0.5, h=(0,0), t=0.3, R1)")
        axes[1, 0].set_xlabel("gamma")
        axes[1, 1].plot(df_gamma["gamma"], df_gamma["pi_1"], "steelblue", label="pi_1")
        axes[1, 1].plot(df_gamma["gamma"], df_gamma["pi_2"], "darkorange", label="pi_2")
        axes[1, 1].set_title("pi* vs gamma")
        axes[1, 1].set_xlabel("gamma")
        axes[1, 1].legend()
    fig.suptitle("Exp 5: Preference-parameter sensitivity (h=(0,0), t=0.3, Regime 1)", fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp5_preference_params.png", dpi=120)
    plt.close()
    print("Saved: exp5_preference_params.png")


if __name__ == "__main__":
    run_exp5()
