# Recursive HJB Default-Contagion Portfolio Model

Numerical replication of a two-asset, two-regime, default-contagion
portfolio/consumption model solved via a recursive HJB/ODE system.

## Overview

The project implements and validates the core numerical engine for a
stochastic control problem in which an investor allocates wealth between
a risk-free asset and two defaultable risky assets, under Markov regime
switching and default contagion.

The HJB equation is solved by:
1. Exploiting power-utility homogeneity to reduce to a scalar multiplier φ(t,i,h).
2. Reversing time so the terminal condition becomes an initial condition.
3. Solving the four default-state ODEs recursively:
   **h=(1,1) → h=(0,1) → h=(1,0) → h=(0,0)**.

## Repository Structure

```
├── data/               # Data directory (synthetic only; no external data required)
├── src/
│   ├── model_params.py # Benchmark parameters and model validation
│   └── hjb_solver.py   # Core recursive HJB solver
├── exp/
│   ├── exp_1_benchmark.py         # Core replication & benchmark values
│   ├── exp_2_after_default.py     # After-default comparative statics
│   ├── exp_3_before_default.py    # Before-default intensity effects
│   ├── exp_4_regime_switching.py  # Regime-switching frequency effects
│   ├── exp_5_preference_params.py # η and γ preference sweeps
│   ├── exp_6_model_comparisons.py # Contagion/regime/utility comparisons
│   ├── exp_7_before_after_default.py # Before-vs-after default comparison
│   └── run_all.py                 # Master runner → generates RESULTS.md
├── tests/
│   └── test_hjb_solver.py         # Comprehensive pytest suite
├── results/
│   ├── RESULTS.md                 # Summary of all experiment outcomes
│   └── *.png, *.csv               # Generated figures and tables
└── README.md
```

## Benchmark Parameters

| Parameter | Value |
|-----------|-------|
| Horizon T | 1.0 |
| Risk aversion γ | 2.5 |
| Cobb-Douglas weight η | 0.5 |
| μ(regime 1) | (1.4, 0.8) |
| μ(regime 2) | (1.0, 0.55) |
| r(regime 1) | 0.1 |
| r(regime 2) | 0.06 |
| Q₀ | [[-0.5, 0.5], [1, -1]] |
| σ(regime 1) | diag(0.7, 1.0) |
| σ(regime 2) | diag(1.0, 1.5) |

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run all experiments and generate RESULTS.md
python -m exp.run_all

# Run individual experiments
python -m exp.exp_1_benchmark
python -m exp.exp_2_after_default
# ...

# Run tests
pytest tests/ -v
```

## Key Results

At t=0.3, regime 1, h=(0,0), the model reproduces paper benchmark values:
- **π₁\*** ≈ 0.3056, **π₂\*** ≈ 0.1315, **c\*** ≈ 0.3966
- After asset-1 default: **expenditure drop** ≈ 35.8%

## Dependencies

- numpy, scipy (ODE solver, optimisation)
- pandas (tabulation)
- matplotlib (visualisation)

No external market data is used. All inputs are synthetic model parameters.
