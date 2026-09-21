# RESULTS.md – Recursive HJB Default-Contagion Portfolio Model
## Overview
Experiments replicate the recursive HJB / ODE system for a two-asset, two-regime, default-contagion portfolio/consumption model.
All experiments use synthetic benchmark parameters (no external data).

---
## Exp 1: Core Benchmark Replication
- Terminal condition phi(T)=1 satisfied: **True**
- Positivity of phi: **True**
- Regime 1, h=(0,0), t=0.3: pi1=0.3056 (paper: 0.3056), pi2=0.1315 (paper: 0.1315), c*=0.3966 (paper: 0.3966)
- Regime 2, h=(0,0), t=0.3: pi1=0.1676 (paper: 0.1676), pi2=0.0533 (paper: 0.0533), c*=0.3742 (paper: 0.3742)
- Runtime: 2.5s

## Exp 2: After-Default Comparative Statics
- Intensity sweep (h=(0,1), R1): pi_surv monotone decreasing = **True**
- Grid points: intensity=30, volatility=30
- Runtime: 6.1s

## Exp 3: Before-Default Intensity Effects (h=(0,0))
- Scenario A: own asset (pi_2) monotone decreasing in lambda_2 = **True**
- Scenario A: total risky share monotone decreasing = **True**
- Runtime: 9.5s

## Exp 4: Regime Switching Frequency
- h=(0,0) gap at t=0: s=1 → -0.4019, s=4 → -0.1746 (paper: ~0.4019 → ~0.1746)
- h=(1,1) gap at t=0: s=1 → -0.0983, s=4 → -0.0366 (paper: ~0.0983 → ~0.0366)
- Gap compression monotone in s: **False**
- Runtime: 1.5s

## Exp 5: Preference Parameter Sweeps
- Eta sweep: c* monotone increasing = **True**, range [0.1722, 0.6494] (paper: ~[0.1722, 0.6494])
- Gamma sweep: pi_1* monotone decreasing = **True**, range [0.1584, 0.5771] (paper: ~[0.1584, 0.5771])
- Runtime: 8.5s

## Exp 6: Model Comparisons
### Module A: Contagion vs No-Contagion (R1, t=0.3, h=(0,0))
  - Contagion: pi_total=0.4371, c*=0.3966 (paper: ~0.4371, 0.3966)
  - No-contagion: pi_total=0.4407, c*=0.3981 (paper: ~0.4408, 0.3981)
### Module C: Full vs Terminal-only (R1, t=0.3, h=(0,0))
  - Full utility: pi1=0.3056, pi2=0.1315 (paper: 0.3056, 0.1315)
  - Terminal-only: pi1=0.2972, pi2=0.1287 (paper: 0.2972, 0.1287)
- Runtime: 1.2s

## Exp 7: Before-vs-After Default
### Asset 1 default, t=0.3, Regime 1
  - pi_surv_before=0.1315 (paper: ~0.1315), pi_surv_after=0.1248 (paper: ~0.1248)
  - c_before=0.3966 (paper: ~0.3966), c_after=0.3670 (paper: ~0.3670)
  - Expenditure drop: 35.75% (paper: ~35.8%)
- Runtime: 0.7s

---
## Files Generated
- `exp1_benchmark_controls.png`, `exp1_benchmark_consumption.png`
- `exp2_intensity_sweep.png`, `exp2_volatility_sweep.png`
- `exp3_before_default_intensity.png`
- `exp4_regime_gap.png`
- `exp5_preference_params.png`
- `exp6_model_comparisons.png`
- `exp7_before_after_default.png`
- CSV tables for each experiment
