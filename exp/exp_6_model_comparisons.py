"""
Experiment 6: Model comparisons – contagion vs no contagion, regime
comparison, and full utility vs terminal-wealth-only.

Corresponds to Figures 8, 9, 10.

Custom implementation – no library covers this model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.integrate import solve_ivp

from src.model_params import benchmark_params, no_contagion_params
from src.hjb_solver import (
    HJBSolver,
    _rhs_h11,
    _optimise_pi_1d, _portfolio_hamiltonian_1d,
    _optimise_pi_2d, _portfolio_hamiltonian_2d,
)

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def _solve_and_extract(params, t_bm=0.3):
    """Run full recursive solver and extract h=(0,0) controls at t_bm."""
    solver = HJBSolver(params)
    solver.solve()
    out = {}
    for i in range(params.n_regimes):
        pi = solver.optimal_pi(t_bm, i, (0, 0))
        c = solver.optimal_c(t_bm, i, (0, 0))
        phi = solver.phi(t_bm, i, (0, 0))
        out[i + 1] = {"pi": pi, "c": c, "phi": phi}
    return out


# ── Module A: Contagion vs no contagion ──────────────────────────────────────

def run_module_a() -> pd.DataFrame:
    """Contagion vs no-contagion comparison at t=0.3, h=(0,0)."""
    print("\n=== Exp 6 Module A: Contagion vs No-Contagion ===")
    rows = []
    for label, p in [("contagion", benchmark_params()), ("no_contagion", no_contagion_params())]:
        ctrl = _solve_and_extract(p)
        for i in [1, 2]:
            rows.append({
                "model": label, "regime": i,
                "pi_1": ctrl[i]["pi"][0], "pi_2": ctrl[i]["pi"][1],
                "pi_total": ctrl[i]["pi"][0] + ctrl[i]["pi"][1],
                "c_star": ctrl[i]["c"], "phi": ctrl[i]["phi"],
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # Paper: no-contagion ~(0.3050,0.1358) c=0.3981; contagion ~(0.3056,0.1315) c=0.3966
    return df


# ── Module B: Favorable vs adverse regime ────────────────────────────────────

def run_module_b() -> pd.DataFrame:
    """Compare regime 1 (favorable) vs regime 2 (adverse)."""
    print("\n=== Exp 6 Module B: Regime Comparison ===")
    ctrl = _solve_and_extract(benchmark_params())
    rows = []
    for i in [1, 2]:
        rows.append({
            "regime": i,
            "pi_1": ctrl[i]["pi"][0], "pi_2": ctrl[i]["pi"][1],
            "pi_total": ctrl[i]["pi"][0] + ctrl[i]["pi"][1],
            "c_star": ctrl[i]["c"], "phi": ctrl[i]["phi"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ── Module C: Full utility vs terminal-wealth-only ────────────────────────────

def _rhs_terminal_h11(t_bar, psi, params):
    """h=(1,1) ODE without consumption (terminal-only criterion)."""
    m = params.n_regimes
    dpsi = np.zeros(m)
    for i in range(m):
        dpsi[i] = (1.0 - params.gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
    return dpsi


def _rhs_terminal_h01(t_bar, psi, params, sol_h11):
    """h=(0,1) ODE without consumption."""
    m = params.n_regimes
    psi_11 = np.maximum(sol_h11(t_bar), 1e-15)
    dpsi = np.zeros(m)
    for i in range(m):
        linear = (1.0 - params.gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        pi_star = _optimise_pi_1d(i, (0, 1), 0, psi[i], psi_11[i], params)
        H_val = -_portfolio_hamiltonian_1d(pi_star, i, (0, 1), 0, psi[i], psi_11[i], params)
        dpsi[i] = linear + H_val
    return dpsi


def _rhs_terminal_h10(t_bar, psi, params, sol_h11):
    """h=(1,0) ODE without consumption."""
    m = params.n_regimes
    psi_11 = np.maximum(sol_h11(t_bar), 1e-15)
    dpsi = np.zeros(m)
    for i in range(m):
        linear = (1.0 - params.gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        pi_star = _optimise_pi_1d(i, (1, 0), 1, psi[i], psi_11[i], params)
        H_val = -_portfolio_hamiltonian_1d(pi_star, i, (1, 0), 1, psi[i], psi_11[i], params)
        dpsi[i] = linear + H_val
    return dpsi


def _rhs_terminal_h00(t_bar, psi, params, sol_h01, sol_h10):
    """h=(0,0) ODE without consumption."""
    m = params.n_regimes
    psi_01 = np.maximum(sol_h01(t_bar), 1e-15)
    psi_10 = np.maximum(sol_h10(t_bar), 1e-15)
    dpsi = np.zeros(m)
    for i in range(m):
        linear = (1.0 - params.gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        pi_star = _optimise_pi_2d(i, psi[i], psi_01[i], psi_10[i], params)
        H_val = -_portfolio_hamiltonian_2d(pi_star, i, psi[i], psi_01[i], psi_10[i], params)
        dpsi[i] = linear + H_val
    return dpsi


def run_module_c() -> pd.DataFrame:
    """Terminal-wealth-only vs full utility comparison."""
    print("\n=== Exp 6 Module C: Full utility vs Terminal-wealth-only ===")
    params = benchmark_params()
    ivp_kw = dict(method="Radau", rtol=1e-9, atol=1e-11, dense_output=True)
    y0 = np.ones(params.n_regimes)
    T = params.T

    sol_11 = solve_ivp(
        lambda t, y: _rhs_terminal_h11(t, y, params),
        t_span=(0, T), y0=y0.copy(), **ivp_kw)
    assert sol_11.success
    sol_01 = solve_ivp(
        lambda t, y: _rhs_terminal_h01(t, y, params, sol_11.sol),
        t_span=(0, T), y0=y0.copy(), **ivp_kw)
    assert sol_01.success
    sol_10 = solve_ivp(
        lambda t, y: _rhs_terminal_h10(t, y, params, sol_11.sol),
        t_span=(0, T), y0=y0.copy(), **ivp_kw)
    assert sol_10.success
    sol_00 = solve_ivp(
        lambda t, y: _rhs_terminal_h00(t, y, params, sol_01.sol, sol_10.sol),
        t_span=(0, T), y0=y0.copy(), **ivp_kw)
    assert sol_00.success

    t_bm = 0.3
    t_bar = T - t_bm
    rows = []
    for i in range(params.n_regimes):
        psi_h = float(sol_00.sol(t_bar)[i])
        psi_01_i = float(sol_01.sol(t_bar)[i])
        psi_10_i = float(sol_10.sol(t_bar)[i])
        pi_to = _optimise_pi_2d(i, psi_h, psi_01_i, psi_10_i, params)
        rows.append({
            "model": "terminal_only", "regime": i + 1,
            "pi_1": pi_to[0], "pi_2": pi_to[1],
            "pi_total": pi_to[0] + pi_to[1], "c_star": 0.0, "phi": psi_h,
        })

    full = _solve_and_extract(params)
    for i in [1, 2]:
        rows.append({
            "model": "full_utility", "regime": i,
            "pi_1": full[i]["pi"][0], "pi_2": full[i]["pi"][1],
            "pi_total": full[i]["pi"][0] + full[i]["pi"][1],
            "c_star": full[i]["c"], "phi": full[i]["phi"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # Paper: terminal-only R1 ~(0.2972,0.1287); full R1 ~(0.3056,0.1315)
    #        terminal-only R2 ~(0.1632,0.0523); full R2 ~(0.1676,0.0533)
    return df


def run_exp6() -> dict:
    """Run experiment 6: all three model-comparison modules."""
    df_a = run_module_a()
    df_b = run_module_b()
    df_c = run_module_c()
    _plot_exp6(df_a, df_b, df_c)
    df_a.to_csv(RESULTS / "exp6_module_a.csv", index=False)
    df_b.to_csv(RESULTS / "exp6_module_b.csv", index=False)
    df_c.to_csv(RESULTS / "exp6_module_c.csv", index=False)
    return {"module_a": df_a, "module_b": df_b, "module_c": df_c}


def _plot_exp6(df_a, df_b, df_c):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Module A: contagion vs no-contagion
    for idx, (model, color) in enumerate([("contagion", "steelblue"), ("no_contagion", "darkorange")]):
        sub = df_a[df_a["model"] == model]
        if not sub.empty:
            axes[0].bar(
                np.array([1, 2]) + idx * 0.3 - 0.15,
                sub["pi_total"].values, width=0.25, color=color, label=model,
            )
    axes[0].set_title("Module A: Total risky share (t=0.3, h=(0,0))")
    axes[0].set_xlabel("Regime")
    axes[0].set_xticks([1, 2])
    axes[0].legend()

    # Module B: regime comparison
    axes[1].bar([1, 2], df_b["pi_1"].values, width=0.3, color="steelblue", label="pi_1")
    axes[1].bar([1.3, 2.3], df_b["pi_2"].values, width=0.3, color="darkorange", label="pi_2")
    axes[1].set_title("Module B: Regime comparison (h=(0,0), t=0.3)")
    axes[1].set_xlabel("Regime")
    axes[1].legend()

    # Module C: full vs terminal-only
    for idx, (model, color) in enumerate([("full_utility", "steelblue"), ("terminal_only", "darkorange")]):
        sub = df_c[df_c["model"] == model]
        if not sub.empty:
            axes[2].bar(
                np.array([1, 2]) + idx * 0.3 - 0.15,
                sub["pi_total"].values, width=0.25, color=color, label=model,
            )
    axes[2].set_title("Module C: Full vs Terminal-only utility")
    axes[2].set_xlabel("Regime")
    axes[2].set_xticks([1, 2])
    axes[2].legend()

    fig.suptitle("Exp 6: Model Comparisons (t=0.3, h=(0,0))", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "exp6_model_comparisons.png", dpi=120)
    plt.close()
    print("Saved: exp6_model_comparisons.png")


if __name__ == "__main__":
    run_exp6()
