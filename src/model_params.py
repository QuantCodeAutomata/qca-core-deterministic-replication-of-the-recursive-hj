"""
Model parameters and scenario manager for the two-asset, two-regime,
default-contagion portfolio/consumption model.

Implements the benchmark parameter set from the paper and provides
parameter validation utilities.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional


# Default states: (h1, h2) where h_j=1 means asset j has defaulted
DEFAULT_STATES: List[Tuple[int, int]] = [(1, 1), (0, 1), (1, 0), (0, 0)]
# Recursive solve order (from all-default to none-default)
SOLVE_ORDER: List[Tuple[int, int]] = [(1, 1), (0, 1), (1, 0), (0, 0)]


@dataclass
class ModelParams:
    """
    Holds all model parameters for the recursive HJB solver.

    Parameters
    ----------
    T : float
        Investment horizon.
    gamma : float
        Relative risk aversion (γ > 1 for the benchmark).
    eta : float
        Cobb-Douglas weight on consumption (η ∈ (0,1)).
    mu : np.ndarray
        Shape (m, n). Drift vectors mu[i] for regime i (0-indexed), n assets.
    r : np.ndarray
        Shape (m,). Risk-free rates r[i] for regime i.
    Q : np.ndarray
        Shape (m, m). Generator of the Markov regime chain (row sums = 0,
        off-diagonal ≥ 0).
    sigma : np.ndarray
        Shape (m, n, d). Volatility matrices sigma[i] for regime i.
    lam : dict
        Default intensities keyed by (h, i) tuples.
        lam[(h, i)] = array of length n with intensities for surviving assets.
        h is a tuple (h1, h2), i is the regime index (0-based).
    """

    T: float = 1.0
    gamma: float = 2.5
    eta: float = 0.5
    mu: np.ndarray = field(default_factory=lambda: np.array([
        [1.4, 0.8],   # regime 1 (index 0)
        [1.0, 0.55],  # regime 2 (index 1)
    ]))
    r: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.06]))
    Q: np.ndarray = field(default_factory=lambda: np.array([
        [-0.5, 0.5],
        [1.0, -1.0],
    ]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([
        [[0.7, 0.0], [0.0, 1.0]],     # regime 1: σ(1)
        [[1.0, 0.0], [0.0, 1.5]],     # regime 2: σ(2)
    ]))
    lam: Dict = field(default_factory=lambda: {
        # h=(0,0): both alive.
        # λ_1^{(0,0)} = (0.50, 0.75): asset-1 intensity in (regime-1, regime-2)
        # λ_2^{(0,0)} = (0.75, 1.10): asset-2 intensity in (regime-1, regime-2)
        # → per-regime vectors [λ_asset1, λ_asset2]
        ((0, 0), 0): np.array([0.50, 0.75]),   # regime 1
        ((0, 0), 1): np.array([0.75, 1.10]),   # regime 2
        # h=(0,1): only asset 1 survives.
        # λ_1^{(0,1)} = (0.70, 1.00): asset-1 intensity in (regime-1, regime-2)
        # asset-2 is dead → its intensity entry is irrelevant (set 0)
        ((0, 1), 0): np.array([0.70, 0.00]),   # regime 1: λ_1=0.70
        ((0, 1), 1): np.array([1.00, 0.00]),   # regime 2: λ_1=1.00
        # h=(1,0): only asset 2 survives.
        # λ_2^{(1,0)} = (0.90, 1.30): asset-2 intensity in (regime-1, regime-2)
        # asset-1 is dead → its entry is 0
        ((1, 0), 0): np.array([0.00, 0.90]),   # regime 1: λ_2=0.90
        ((1, 0), 1): np.array([0.00, 1.30]),   # regime 2: λ_2=1.30
        # h=(1,1): no assets survive
        ((1, 1), 0): np.array([0.0, 0.0]),
        ((1, 1), 1): np.array([0.0, 0.0]),
    })

    def __post_init__(self) -> None:
        self._validate()
        self._precompute()

    def _validate(self) -> None:
        """Validate parameter shapes and financial constraints."""
        m = self.Q.shape[0]
        n = self.mu.shape[1]

        assert self.Q.shape == (m, m), "Q must be square"
        assert np.allclose(self.Q.sum(axis=1), 0.0, atol=1e-10), \
            "Generator Q: rows must sum to zero"
        assert np.all(self.Q - np.diag(np.diag(self.Q)) >= -1e-12), \
            "Generator Q: off-diagonals must be non-negative"

        assert self.mu.shape == (m, n), f"mu must be ({m},{n})"
        assert self.r.shape == (m,), f"r must be ({m},)"
        assert self.sigma.shape[0] == m, "sigma dim 0 must match m"

        for i in range(m):
            sig_i = self.sigma[i]
            Sigma_i = sig_i @ sig_i.T
            eigvals = np.linalg.eigvalsh(Sigma_i)
            assert np.all(eigvals > -1e-10), \
                f"σ(i={i})σ(i={i})^T must be positive semi-definite"

        assert 0.0 < self.eta <= 1.0, "eta must be in (0,1]"
        assert self.gamma > 1.0, "gamma must be > 1 for the benchmark"
        assert self.T > 0.0, "T must be positive"

    def _precompute(self) -> None:
        """Cache derived quantities used repeatedly in the solver."""
        m = self.Q.shape[0]
        n = self.mu.shape[1]
        self.n_assets: int = n
        self.n_regimes: int = m

        # Covariance matrices Σ(i) = σ(i) σ(i)^T, shape (m, n, n)
        self.Sigma: np.ndarray = np.array([
            self.sigma[i] @ self.sigma[i].T for i in range(m)
        ])

        # Exponents for consumption term
        alpha = self.eta * (1.0 - self.gamma)          # η(1-γ)
        k = alpha - 1.0                                  # η(1-γ) - 1
        self.alpha: float = alpha
        self.k: float = k
        self.p: float = alpha / k                        # η(1-γ)/(η(1-γ)-1)
        self.inv_k: float = 1.0 / k                      # 1/(η(1-γ)-1)

        # Coefficient for Φ_i: (η^{-p} - (1-γ)η^{-1/k})
        A = self.eta ** (-self.p) - (1.0 - self.gamma) * self.eta ** (-self.inv_k)
        self.phi_coeff: float = A                        # scalar coefficient

    def compensated_excess_return(self, h: Tuple[int, int], i: int) -> np.ndarray:
        """
        Compute compensated excess return vector μ̃(i, h) for regime i and
        default state h.

        μ̃_j(i,h) = μ_j(i) - r(i) + λ_j(i,h) for surviving assets (h_j=0),
        μ̃_j = 0 for defaulted assets (h_j=1).

        Parameters
        ----------
        h : tuple (h1, h2)
            Default state.
        i : int
            Regime index (0-based).

        Returns
        -------
        np.ndarray of shape (n,)
        """
        mu_tilde = np.zeros(self.n_assets)
        lam_vec = self.lam.get((h, i), np.zeros(self.n_assets))
        for j in range(self.n_assets):
            if h[j] == 0:  # asset j survives
                mu_tilde[j] = self.mu[i, j] - self.r[i] + lam_vec[j]
        return mu_tilde

    def surviving_assets(self, h: Tuple[int, int]) -> List[int]:
        """Return list of asset indices (0-based) that survive in state h."""
        return [j for j in range(self.n_assets) if h[j] == 0]

    def next_state(self, h: Tuple[int, int], defaulted_asset: int) -> Tuple[int, int]:
        """Return the default state after asset `defaulted_asset` defaults."""
        h_list = list(h)
        h_list[defaulted_asset] = 1
        return tuple(h_list)

    def copy_with(self, **kwargs) -> "ModelParams":
        """
        Return a new ModelParams with selected fields overridden.
        Useful for sensitivity / comparative-statics sweeps.
        """
        import copy
        new = copy.deepcopy(self)
        for key, val in kwargs.items():
            setattr(new, key, val)
        new._validate()
        new._precompute()
        return new


def benchmark_params() -> ModelParams:
    """Return the standard benchmark parameterization from the paper."""
    return ModelParams()


def no_contagion_params() -> ModelParams:
    """
    Return parameters for the no-contagion specification:
    post-default intensities equal pre-default intensities.
    """
    p = ModelParams()
    lam_nc = dict(p.lam)
    # λ^{(0,1)} = λ^{(0,0)} for asset 1 (surviving when h=(0,1))
    lam_nc[((0, 1), 0)] = p.lam[((0, 0), 0)].copy()
    lam_nc[((0, 1), 1)] = p.lam[((0, 0), 1)].copy()
    # λ^{(1,0)} = λ^{(0,0)} for asset 2 (surviving when h=(1,0))
    lam_nc[((1, 0), 0)] = p.lam[((0, 0), 0)].copy()
    lam_nc[((1, 0), 1)] = p.lam[((0, 0), 1)].copy()
    p.lam = lam_nc
    return p
