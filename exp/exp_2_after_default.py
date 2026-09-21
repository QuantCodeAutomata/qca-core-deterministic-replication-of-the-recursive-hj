"""
Experiment 2: After-default comparative statics.

Studies how the surviving asset's optimal portfolio weight and consumption
respond to changes in its own default intensity and its own diffusion
volatility in post-default states h=(0,1) and h=(1,0).

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

from scipy.integrate import solve_ivp

from src.model_params import ModelParams, benchmark_params
from src.hjb_solver import (
    _rhs_h11, _rhs_h01, _rhs_h10,
    _optimise_pi_1d, _optimal_c,
)

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def _solve_h11(params: ModelParams):
    """Solve h=(1,1) ODE and return dense-output solution object."""
    sol = solve_ivp(
        lambda t, y: _rhs_h11(t, y, params),
        t_span=(0.0, params.T),
        y0=np.ones(params.n_regimes),
        method="Radau", rtol=1e-9, atol=1e-11, dense_output=True,
    )
    assert sol.success, sol.message
    return sol.sol


def _solve_post_default(params: ModelParams, sol_h11, h: tuple) -> object:
    """Solve one-survivor ODE for h=(0,1) or h=(1,0)."""
    rhs = _rhs_h01 if h == (0, 1) else _rhs_h10
    sol = solve_ivp(
        lambda t, y: rhs(t, y, params, sol_h11),
        t_span=(0.0, params.T),
        y0=np.ones(params.n_regimes),
        method="Radau", rtol=1e-9, atol=1e-11, dense_output=True,
    )
    assert sol.success, sol.message
    return sol.sol


def _get_controls_post(sol_h, sol_h11, params, h, j_asset, t_fwd):
    """Recover pi*, c* for one-survivor state at forward time t_fwd."""
    t_bar = params.T - t_fwd
    results = []
    for i in range(params.n_regimes):
        psi_h_i = float(sol_h(t_bar)[i])
        psi_11_i = float(sol_h11(t_bar)[i])
        pi_star = _optimise_pi_1d(i, h, j_asset, psi_h_i, psi_11_i, params)
        c_star = _optimal_c(psi_h_i, params)
        results.append({"regime": i + 1, "pi": pi_star, "c": c_star, "phi": psi_h_i})
    return results


def run_exp2_intensity() -> pd.DataFrame:
    """Own default-intensity comparative static for post-default states."""
    base = benchmark_params()
    n_grid = 30
    rows = []

    # h=(0,1): vary lambda_1(i,(0,1)) – asset 1's post-default intensity
    lam_grid = np.linspace(0.3, 2.0, n_grid)
    for lam_val in lam_grid:
        p = copy.deepcopy(base)
        # Update surviving asset intensity in both regimes for state (0,1)
        p.lam[((0, 1), 0)] = np.array([lam_val, p.lam[((0, 1), 0)][1]])
        p.lam[((0, 1), 1)] = np.array([lam_val, p.lam[((0, 1), 1)][1]])
        p._precompute()
        sol_11 = _solve_h11(p)
        sol_01 = _solve_post_default(p, sol_11, (0, 1))
        for c_info in _get_controls_post(sol_01, sol_11, p, (0, 1), 0, 0.3):
            rows.append({
                "state": "(0,1)", "varied": "lambda_1_(0,1)",
                "lam_val": lam_val, "t": 0.3,
                "regime": c_info["regime"],
                "pi_surv": c_info["pi"], "c_star": c_info["c"],
                "phi": c_info["phi"],
            })

    # h=(1,0): vary lambda_2(i,(1,0)) – asset 2's post-default intensity
    for lam_val in lam_grid:
        p = copy.deepcopy(base)
        p.lam[((1, 0), 0)] = np.array([p.lam[((1, 0), 0)][0], lam_val])
        p.lam[((1, 0), 1)] = np.array([p.lam[((1, 0), 1)][0], lam_val])
        p._precompute()
        sol_11 = _solve_h11(p)
        sol_10 = _solve_post_default(p, sol_11, (1, 0))
        for c_info in _get_controls_post(sol_10, sol_11, p, (1, 0), 1, 0.3):
            rows.append({
                "state": "(1,0)", "varied": "lambda_2_(1,0)",
                "lam_val": lam_val, "t": 0.3,
                "regime": c_info["regime"],
                "pi_surv": c_info["pi"], "c_star": c_info["c"],
                "phi": c_info["phi"],
            })

    df = pd.DataFrame(rows)
    _plot_exp2(df, "intensity", "lam_val", "Default Intensity", "exp2_intensity_sweep.png")
    return df


def run_exp2_volatility() -> pd.DataFrame:
    """Own volatility comparative static for post-default states."""
    base = benchmark_params()
    vol_grid = np.linspace(0.3, 2.5, 30)
    rows = []

    # h=(0,1): vary sigma diagonal entry for asset 1
    for vol_val in vol_grid:
        p = copy.deepcopy(base)
        p.sigma = p.sigma.copy()
        p.sigma[0, 0, 0] = vol_val  # regime 0, asset 1
        p.sigma[1, 0, 0] = vol_val  # regime 1, asset 1
        p._precompute()
        sol_11 = _solve_h11(p)
        sol_01 = _solve_post_default(p, sol_11, (0, 1))
        for c_info in _get_controls_post(sol_01, sol_11, p, (0, 1), 0, 0.3):
            rows.append({
                "state": "(0,1)", "varied": "sigma_1",
                "vol_val": vol_val, "t": 0.3,
                "regime": c_info["regime"],
                "pi_surv": c_info["pi"], "c_star": c_info["c"],
                "phi": c_info["phi"],
            })

    # h=(1,0): vary sigma diagonal entry for asset 2
    for vol_val in vol_grid:
        p = copy.deepcopy(base)
        p.sigma = p.sigma.copy()
        p.sigma[0, 1, 1] = vol_val
        p.sigma[1, 1, 1] = vol_val
        p._precompute()
        sol_11 = _solve_h11(p)
        sol_10 = _solve_post_default(p, sol_11, (1, 0))
        for c_info in _get_controls_post(sol_10, sol_11, p, (1, 0), 1, 0.3):
            rows.append({
                "state": "(1,0)", "varied": "sigma_2",
                "vol_val": vol_val, "t": 0.3,
                "regime": c_info["regime"],
                "pi_surv": c_info["pi"], "c_star": c_info["c"],
                "phi": c_info["phi"],
            })

    df = pd.DataFrame(rows)
    _plot_exp2(df, "volatility", "vol_val", "Volatility", "exp2_volatility_sweep.png")
    return df


def _plot_exp2(df, exp_type, x_col, x_label, fname):
    states = df["state"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_idx, state in enumerate(states[:2]):
        sub = df[df["state"] == state]
        for i, regime in enumerate([1, 2]):
            s = sub[sub["regime"] == regime]
            axes[row_idx, 0].plot(s[x_col], s["pi_surv"], label=f"R{regime}")
            axes[row_idx, 1].plot(s[x_col], s["c_star"], label=f"R{regime}")
        axes[row_idx, 0].set_title(f"pi* survivor – state {state}")
        axes[row_idx, 0].set_xlabel(x_label)
        axes[row_idx, 0].legend()
        axes[row_idx, 1].set_title(f"c* – state {state}")
        axes[row_idx, 1].set_xlabel(x_label)
        axes[row_idx, 1].legend()
    fig.suptitle(f"Exp 2: After-default {exp_type} comparative statics", fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / fname, dpi=120)
    plt.close()
    print(f"Saved: {fname}")


def run_exp2() -> dict:
    """Run experiment 2: after-default comparative statics."""
    print("\n=== Exp 2: After-default Intensity Sweep ===")
    df_int = run_exp2_intensity()
    print(df_int.to_string(index=False))
    df_int.to_csv(RESULTS / "exp2_intensity_sweep.csv", index=False)

    print("\n=== Exp 2: After-default Volatility Sweep ===")
    df_vol = run_exp2_volatility()
    print(df_vol.to_string(index=False))
    df_vol.to_csv(RESULTS / "exp2_volatility_sweep.csv", index=False)

    return {"intensity": df_int, "volatility": df_vol}


if __name__ == "__main__":
    run_exp2()
