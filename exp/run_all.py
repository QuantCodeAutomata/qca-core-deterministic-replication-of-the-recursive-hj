"""
Master runner: executes all seven experiments and writes RESULTS.md.
"""

from __future__ import annotations

import time
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    summary_lines = [
        "# RESULTS.md – Recursive HJB Default-Contagion Portfolio Model\n",
        "## Overview\n",
        "Experiments replicate the recursive HJB / ODE system for a two-asset, "
        "two-regime, default-contagion portfolio/consumption model.\n",
        "All experiments use synthetic benchmark parameters (no external data).\n\n",
        "---\n",
    ]

    # Experiment 1
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Core Benchmark Replication")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_1_benchmark import run_exp1
    r1 = run_exp1()
    t1 = time.time() - t0
    bm = r1["benchmark_values"]
    val_r1 = bm[(0, (0, 0))]
    val_r2 = bm[(1, (0, 0))]
    term_pass = all(v["pass"] for v in r1["terminal_checks"].values())
    summary_lines += [
        "## Exp 1: Core Benchmark Replication\n",
        f"- Terminal condition phi(T)=1 satisfied: **{term_pass}**\n",
        f"- Positivity of phi: **{r1['positivity']}**\n",
        f"- Regime 1, h=(0,0), t=0.3: pi1={val_r1['pi'][0]:.4f} (paper: 0.3056), "
        f"pi2={val_r1['pi'][1]:.4f} (paper: 0.1315), c*={val_r1['c']:.4f} (paper: 0.3966)\n",
        f"- Regime 2, h=(0,0), t=0.3: pi1={val_r2['pi'][0]:.4f} (paper: 0.1676), "
        f"pi2={val_r2['pi'][1]:.4f} (paper: 0.0533), c*={val_r2['c']:.4f} (paper: 0.3742)\n",
        f"- Runtime: {t1:.1f}s\n\n",
    ]

    # Experiment 2
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: After-Default Comparative Statics")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_2_after_default import run_exp2
    r2 = run_exp2()
    t2 = time.time() - t0
    df_int = r2["intensity"]
    df_vol = r2["volatility"]
    # Monotonicity checks
    int_mono_r1 = df_int[(df_int["state"] == "(0,1)") & (df_int["regime"] == 1)]
    mono_ok_int = bool(int_mono_r1["pi_surv"].is_monotonic_decreasing) if len(int_mono_r1) > 1 else True
    summary_lines += [
        "## Exp 2: After-Default Comparative Statics\n",
        f"- Intensity sweep (h=(0,1), R1): pi_surv monotone decreasing = **{mono_ok_int}**\n",
        f"- Grid points: intensity={len(df_int)//4}, volatility={len(df_vol)//4}\n",
        f"- Runtime: {t2:.1f}s\n\n",
    ]

    # Experiment 3
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Before-Default Intensity Effects")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_3_before_default import run_exp3
    r3 = run_exp3()
    t3 = time.time() - t0
    df_a = r3["scenario_a"]
    df_b = r3["scenario_b"]
    # Check: own asset pi decreases, other asset pi may increase
    r1_a = df_a[df_a["regime"] == 1].sort_values("lam_val")
    own_mono_a = bool(r1_a["pi_2"].is_monotonic_decreasing) if len(r1_a) > 1 else True
    total_mono_a = bool(r1_a["pi_total"].is_monotonic_decreasing) if len(r1_a) > 1 else True
    summary_lines += [
        "## Exp 3: Before-Default Intensity Effects (h=(0,0))\n",
        f"- Scenario A: own asset (pi_2) monotone decreasing in lambda_2 = **{own_mono_a}**\n",
        f"- Scenario A: total risky share monotone decreasing = **{total_mono_a}**\n",
        f"- Runtime: {t3:.1f}s\n\n",
    ]

    # Experiment 4
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Regime Switching Frequency")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_4_regime_switching import run_exp4
    r4 = run_exp4()
    t4 = time.time() - t0
    df4 = r4["df"]
    gap_00 = {s: df4[(df4["s"] == s) & (df4["h"] == str((0, 0))) & (df4["t"] == 0.0)]["gap"].values[0]
              for s in [1, 2, 3, 4]}
    gap_11 = {s: df4[(df4["s"] == s) & (df4["h"] == str((1, 1))) & (df4["t"] == 0.0)]["gap"].values[0]
              for s in [1, 2, 3, 4]}
    summary_lines += [
        "## Exp 4: Regime Switching Frequency\n",
        f"- h=(0,0) gap at t=0: s=1 → {gap_00[1]:.4f}, s=4 → {gap_00[4]:.4f} "
        f"(paper: ~0.4019 → ~0.1746)\n",
        f"- h=(1,1) gap at t=0: s=1 → {gap_11[1]:.4f}, s=4 → {gap_11[4]:.4f} "
        f"(paper: ~0.0983 → ~0.0366)\n",
        f"- Gap compression monotone in s: "
        f"**{all(gap_00[s] > gap_00[s+1] for s in [1, 2, 3])}**\n",
        f"- Runtime: {t4:.1f}s\n\n",
    ]

    # Experiment 5
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Preference Parameter Sweeps")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_5_preference_params import run_exp5
    r5 = run_exp5()
    t5 = time.time() - t0
    df_eta = r5["eta_sweep"]
    df_gam = r5["gamma_sweep"]
    eta_c_mono = bool(df_eta["c_star"].is_monotonic_increasing) if len(df_eta) > 1 else True
    gam_pi_mono = bool(df_gam["pi_1"].is_monotonic_decreasing) if len(df_gam) > 1 else True
    c_range = f"[{df_eta['c_star'].min():.4f}, {df_eta['c_star'].max():.4f}]"
    pi_range = f"[{df_gam['pi_1'].min():.4f}, {df_gam['pi_1'].max():.4f}]"
    summary_lines += [
        "## Exp 5: Preference Parameter Sweeps\n",
        f"- Eta sweep: c* monotone increasing = **{eta_c_mono}**, range {c_range} "
        f"(paper: ~[0.1722, 0.6494])\n",
        f"- Gamma sweep: pi_1* monotone decreasing = **{gam_pi_mono}**, range {pi_range} "
        f"(paper: ~[0.1584, 0.5771])\n",
        f"- Runtime: {t5:.1f}s\n\n",
    ]

    # Experiment 6
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Model Comparisons")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_6_model_comparisons import run_exp6
    r6 = run_exp6()
    t6 = time.time() - t0
    df_a = r6["module_a"]
    df_c = r6["module_c"]
    # Module A: contagion vs no-contagion comparison
    cont_r1 = df_a[(df_a["model"] == "contagion") & (df_a["regime"] == 1)]
    nc_r1 = df_a[(df_a["model"] == "no_contagion") & (df_a["regime"] == 1)]
    # Module C: full vs terminal-only
    full_r1 = df_c[(df_c["model"] == "full_utility") & (df_c["regime"] == 1)]
    to_r1 = df_c[(df_c["model"] == "terminal_only") & (df_c["regime"] == 1)]
    summary_lines += [
        "## Exp 6: Model Comparisons\n",
        "### Module A: Contagion vs No-Contagion (R1, t=0.3, h=(0,0))\n",
    ]
    if not cont_r1.empty and not nc_r1.empty:
        summary_lines += [
            f"  - Contagion: pi_total={cont_r1['pi_total'].values[0]:.4f}, "
            f"c*={cont_r1['c_star'].values[0]:.4f} (paper: ~0.4371, 0.3966)\n",
            f"  - No-contagion: pi_total={nc_r1['pi_total'].values[0]:.4f}, "
            f"c*={nc_r1['c_star'].values[0]:.4f} (paper: ~0.4408, 0.3981)\n",
        ]
    summary_lines += [
        "### Module C: Full vs Terminal-only (R1, t=0.3, h=(0,0))\n",
    ]
    if not full_r1.empty and not to_r1.empty:
        summary_lines += [
            f"  - Full utility: pi1={full_r1['pi_1'].values[0]:.4f}, "
            f"pi2={full_r1['pi_2'].values[0]:.4f} (paper: 0.3056, 0.1315)\n",
            f"  - Terminal-only: pi1={to_r1['pi_1'].values[0]:.4f}, "
            f"pi2={to_r1['pi_2'].values[0]:.4f} (paper: 0.2972, 0.1287)\n",
        ]
    summary_lines += [f"- Runtime: {t6:.1f}s\n\n"]

    # Experiment 7
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Before-vs-After Default")
    print("=" * 60)
    t0 = time.time()
    from exp.exp_7_before_after_default import run_exp7
    r7 = run_exp7()
    t7 = time.time() - t0
    bm7 = r7["benchmark"]
    summary_lines += [
        "## Exp 7: Before-vs-After Default\n",
        "### Asset 1 default, t=0.3, Regime 1\n",
        f"  - pi_surv_before={bm7['pi_surv_before']:.4f} (paper: ~0.1315), "
        f"pi_surv_after={bm7['pi_surv_after']:.4f} (paper: ~0.1248)\n",
        f"  - c_before={bm7['c_before']:.4f} (paper: ~0.3966), "
        f"c_after={bm7['c_after']:.4f} (paper: ~0.3670)\n",
        f"  - Expenditure drop: {bm7['exp_drop_pct']:.2f}% (paper: ~35.8%)\n",
        f"- Runtime: {t7:.1f}s\n\n",
        "---\n",
        "## Files Generated\n",
        "- `exp1_benchmark_controls.png`, `exp1_benchmark_consumption.png`\n",
        "- `exp2_intensity_sweep.png`, `exp2_volatility_sweep.png`\n",
        "- `exp3_before_default_intensity.png`\n",
        "- `exp4_regime_gap.png`\n",
        "- `exp5_preference_params.png`\n",
        "- `exp6_model_comparisons.png`\n",
        "- `exp7_before_after_default.png`\n",
        "- CSV tables for each experiment\n",
    ]

    md_path = RESULTS / "RESULTS.md"
    md_path.write_text("".join(summary_lines))
    print(f"\nResults written to {md_path}")


if __name__ == "__main__":
    main()
