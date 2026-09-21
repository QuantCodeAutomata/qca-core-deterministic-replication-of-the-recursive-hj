"""
Core recursive HJB solver for the two-asset, two-regime, default-contagion
portfolio/consumption model.

The solver implements the time-reversal trick: defining
    φ̄(t, i, h) = φ(T - t, i, h),
so that the HJB PDE becomes a forward-in-time IVP with
    φ̄(0, i, h) = φ(T, i, h) = 1  (terminal condition).

Solve order (required by recursive state dependence):
    1. h=(1,1) – both defaulted, no portfolio choice
    2. h=(0,1) – only asset 1 survives
    3. h=(1,0) – only asset 2 survives
    4. h=(0,0) – both alive (uses solutions from all three above)

Custom implementation – Context7 confirmed no library directly supports
this recursive HJB with default contagion and regime switching.

Uses:
    scipy.integrate.solve_ivp  (dense_output=True for interpolation)
    scipy.optimize.brentq       (fast monotone FOC root-finding for pi*)
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src.model_params import ModelParams, SOLVE_ORDER


# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _phi_i(y: float, params: ModelParams) -> float:
    """
    Closed-form contribution of optimal consumption to the ODE RHS for regime i.

    After optimising over c (consumption-fraction of wealth), the HJB adds:
        Φ_i(y) = (η^{-p} − (1-γ) η^{-1/k}) · y^p
    where p = η(1-γ)/(η(1-γ)-1) and k = η(1-γ)-1.

    Parameters
    ----------
    y : float
        Current value of φ̄_{i,h} for regime i.
    params : ModelParams
        Holds pre-computed scalars p, phi_coeff.

    Returns
    -------
    float
    """
    if y <= 0.0:
        return 0.0
    return params.phi_coeff * (y ** params.p)


def _optimal_c(phi_val: float, params: ModelParams) -> float:
    """
    Optimal consumption rate (fraction of wealth):
        c*(t,i,h) = (φ(t,i,h) / η)^{1/(η(1-γ)-1)}

    Parameters
    ----------
    phi_val : float
        Value of φ at (t, i, h).
    params : ModelParams

    Returns
    -------
    float
    """
    if phi_val <= 0.0:
        return 0.0
    return (phi_val / params.eta) ** params.inv_k


def _portfolio_hamiltonian_1d(
    pi1: float,
    i: int,
    h: Tuple[int, int],
    j_asset: int,
    psi_h: float,
    psi_next: float,
    params: ModelParams,
) -> float:
    """
    One-dimensional portfolio contribution to the ODE RHS for a single
    surviving asset j_asset in state h.

    H_port(π_j) = (1-γ) π_j μ̃_j φ̄ - (1-γ)γ/2 (π_j)^2 Σ^{jj} φ̄
                  + λ_j [(1-π_j)^{1-γ} φ̄_{next} - φ̄]

    We MAXIMISE this over π_j < 1, so we pass -H_port to scipy as the
    objective to minimise.

    Parameters
    ----------
    pi1 : float
        Portfolio weight for surviving asset j_asset.
    i : int
        Regime index (0-based).
    h : tuple
        Current default state.
    j_asset : int
        Asset index (0-based) of the surviving asset.
    psi_h : float
        φ̄_{i,h} at current time.
    psi_next : float
        φ̄_{i, h+e_{j_asset}} at current time (from already-solved state).
    params : ModelParams

    Returns
    -------
    float  – value of -H_port (to be minimised).
    """
    gamma = params.gamma
    lam_vec = params.lam.get((h, i), np.zeros(params.n_assets))
    lam_j = lam_vec[j_asset]
    Sigma_jj = params.Sigma[i, j_asset, j_asset]
    mu_tilde_j = params.compensated_excess_return(h, i)[j_asset]

    # Guard: boundary barrier at pi1 → 1
    if 1.0 - pi1 <= 1e-14:
        return 1e30

    H = ((1.0 - gamma) * pi1 * mu_tilde_j * psi_h
         - (1.0 - gamma) * gamma / 2.0 * pi1**2 * Sigma_jj * psi_h
         + lam_j * ((1.0 - pi1) ** (1.0 - gamma) * psi_next - psi_h))
    return -H  # negate for minimisation


def _portfolio_hamiltonian_2d(
    pi: np.ndarray,
    i: int,
    psi_00: float,
    psi_01: float,
    psi_10: float,
    params: ModelParams,
) -> float:
    """
    Two-dimensional portfolio Hamiltonian for the all-alive state h=(0,0).

    H_port(π^1, π^2) = (1-γ)(π^1 μ̃^1 + π^2 μ̃^2) φ̄_{00}
                       - (1-γ)γ/2 π^T Σ π φ̄_{00}
                       + λ_1[(1-π^1)^{1-γ} φ̄_{01} - φ̄_{00}]
                       + λ_2[(1-π^2)^{1-γ} φ̄_{10} - φ̄_{00}]

    Parameters
    ----------
    pi : np.ndarray of shape (2,)
    i : int
        Regime index (0-based).
    psi_00, psi_01, psi_10 : float
        φ̄ values at regime i for states (0,0), (0,1), (1,0).
    params : ModelParams

    Returns
    -------
    float  – value of -H_port (to be minimised).
    """
    gamma = params.gamma
    pi1, pi2 = pi[0], pi[1]

    if (1.0 - pi1) <= 1e-14 or (1.0 - pi2) <= 1e-14:
        return 1e30

    h00 = (0, 0)
    mu_tilde = params.compensated_excess_return(h00, i)
    lam_vec = params.lam.get((h00, i), np.zeros(params.n_assets))
    lam1, lam2 = lam_vec[0], lam_vec[1]
    Sigma_i = params.Sigma[i]

    quad_term = pi[0] ** 2 * Sigma_i[0, 0] + pi[1] ** 2 * Sigma_i[1, 1] + \
                2.0 * pi[0] * pi[1] * Sigma_i[0, 1]

    # Asset 1 default event: h=(0,0) → h=(1,0), so use psi_10.
    # Asset 2 default event: h=(0,0) → h=(0,1), so use psi_01.
    H = ((1.0 - gamma) * (pi1 * mu_tilde[0] + pi2 * mu_tilde[1]) * psi_00
         - (1.0 - gamma) * gamma / 2.0 * quad_term * psi_00
         + lam1 * ((1.0 - pi1) ** (1.0 - gamma) * psi_10 - psi_00)
         + lam2 * ((1.0 - pi2) ** (1.0 - gamma) * psi_01 - psi_00))
    return -H  # negate for minimisation


def _foc_1d(
    pi: float,
    gamma: float,
    mu_tilde_j: float,
    Sigma_jj: float,
    lam_j: float,
    ratio: float,
) -> float:
    """
    First-order condition for the 1D portfolio problem (after dividing by ψ_h > 0).

    g(π) = μ̃_j - γ Σ_{jj} π - λ_j (1-π)^{-γ} · (ψ_{next}/ψ_h) = 0

    g is strictly DECREASING in π (see docstring of _optimise_pi_1d),
    enabling reliable brentq root-finding.

    Parameters
    ----------
    pi : float
        Portfolio weight for the surviving asset.
    gamma : float
        Risk aversion coefficient.
    mu_tilde_j : float
        Compensated excess return of the surviving asset.
    Sigma_jj : float
        Diagonal variance of the surviving asset.
    lam_j : float
        Default intensity of the surviving asset.
    ratio : float
        ψ_{next} / ψ_h (ratio of neighbouring-state φ̄ to current φ̄).
    """
    return mu_tilde_j - gamma * Sigma_jj * pi - lam_j * (1.0 - pi) ** (-gamma) * ratio


def _optimise_pi_1d(
    i: int,
    h: Tuple[int, int],
    j_asset: int,
    psi_h: float,
    psi_next: float,
    params: ModelParams,
    pi_init: float = 0.0,
) -> float:
    """
    Optimal portfolio weight for the single-survivor state via brentq.

    The FOC  g(π) = μ̃_j - γΣ_{jj}π - λ_j(1-π)^{-γ}·(ψ_next/ψ_h) = 0
    is strictly decreasing in π ∈ (-∞, 1), guaranteeing a unique root
    which brentq finds in O(log(1/tol)) evaluations — typically ~50×
    faster than minimize_scalar.

    Custom – Context7 found no library equivalent (paper Eq. for H_port).
    """
    gamma = params.gamma
    lam_vec = params.lam.get((h, i), np.zeros(params.n_assets))
    lam_j = float(lam_vec[j_asset])
    Sigma_jj = float(params.Sigma[i, j_asset, j_asset])
    mu_tilde_j = float(params.compensated_excess_return(h, i)[j_asset])
    ratio = psi_next / psi_h if psi_h > 0.0 else 1.0

    ub = 1.0 - 1e-10
    lo = -1000.0
    # If lambda=0 the FOC is linear: pi = mu_tilde / (gamma * Sigma_jj)
    if lam_j == 0.0:
        return mu_tilde_j / (gamma * Sigma_jj) if Sigma_jj > 0.0 else 0.0

    return float(brentq(_foc_1d, lo, ub,
                        args=(gamma, mu_tilde_j, Sigma_jj, lam_j, ratio),
                        xtol=1e-12, rtol=1e-12, maxiter=200))


def _foc_2d_j(
    pi_j: float,
    j: int,
    gamma: float,
    mu_tilde: np.ndarray,
    Sigma: np.ndarray,
    lam: np.ndarray,
    pi_other: float,
    ratio_j: float,
) -> float:
    """
    FOC for asset j in the 2D problem, given the other asset's weight.

    g_j(π_j) = μ̃_j - γ[Σ_{jj}π_j + Σ_{jk}π_k] - λ_j(1-π_j)^{-γ}·ratio_j = 0

    For diagonal Σ (Σ_{jk}=0) this decouples from π_k.
    """
    k = 1 - j
    cross = Sigma[j, k] * pi_other  # 0 when Σ is diagonal
    return (mu_tilde[j] - gamma * (Sigma[j, j] * pi_j + cross)
            - lam[j] * (1.0 - pi_j) ** (-gamma) * ratio_j)


def _optimise_pi_2d(
    i: int,
    psi_00: float,
    psi_01: float,
    psi_10: float,
    params: ModelParams,
    pi_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Optimal (π^1, π^2) for the all-alive state h=(0,0).

    When Σ is diagonal (standard benchmark), the two FOC equations decouple
    and each is solved independently via brentq.  For non-diagonal Σ, we run
    a fixed-point Newton iteration (outer loop over the cross-term) that
    converges in a few iterations.

    Custom – Context7 found no library equivalent (paper Eq. for H_port 2D).
    """
    gamma = params.gamma
    h00 = (0, 0)
    mu_tilde = params.compensated_excess_return(h00, i)
    lam_vec = params.lam.get((h00, i), np.zeros(params.n_assets))
    Sigma_i = params.Sigma[i]

    # ratio[j] = φ̄_{next_j} / φ̄_{00} where next_j is the state after asset j defaults.
    # Asset 0 default: (0,0)→(1,0) → ratio[0] = psi_10/psi_00
    # Asset 1 default: (0,0)→(0,1) → ratio[1] = psi_01/psi_00
    psi_next = np.array([psi_10, psi_01])
    ratio = psi_next / psi_00 if psi_00 > 0.0 else np.ones(2)

    ub = 1.0 - 1e-10
    lo = -1000.0

    if abs(Sigma_i[0, 1]) < 1e-15:
        # Diagonal Σ: decouple into two independent 1D brentq problems.
        pi_out = np.zeros(2)
        for j in range(2):
            if lam_vec[j] == 0.0:
                pi_out[j] = mu_tilde[j] / (gamma * Sigma_i[j, j])
            else:
                pi_out[j] = brentq(
                    _foc_2d_j, lo, ub,
                    args=(j, gamma, mu_tilde, Sigma_i, lam_vec, 0.0, ratio[j]),
                    xtol=1e-12, rtol=1e-12, maxiter=200,
                )
        return pi_out

    # Non-diagonal Σ: alternating-variable Newton (converges in ~5 iterations).
    pi_out = np.array(pi_init if pi_init is not None else [0.3, 0.1], dtype=float)
    for _ in range(40):
        pi_prev = pi_out.copy()
        for j in range(2):
            pi_other = pi_out[1 - j]
            if lam_vec[j] == 0.0:
                k = 1 - j
                pi_out[j] = (mu_tilde[j] - gamma * Sigma_i[j, k] * pi_other) / (gamma * Sigma_i[j, j])
            else:
                pi_out[j] = brentq(
                    _foc_2d_j, lo, ub,
                    args=(j, gamma, mu_tilde, Sigma_i, lam_vec, pi_other, ratio[j]),
                    xtol=1e-12, rtol=1e-12, maxiter=200,
                )
        if np.max(np.abs(pi_out - pi_prev)) < 1e-12:
            break
    return pi_out


# ──────────────────────────────────────────────────────────────────────────────
#  Per-state ODE right-hand sides
# ──────────────────────────────────────────────────────────────────────────────

def _rhs_h11(
    t_bar: float,
    psi: np.ndarray,
    params: ModelParams,
) -> np.ndarray:
    """
    ODE RHS for the all-default state h=(1,1).

    No portfolio choice (π*=0). Only regime switching, risk-free drift,
    and consumption contribution.

    d/dt̄ φ̄_{i,(1,1)} = Σ_k Q_{ik} φ̄_{k,(1,1)}
                        + (1-γ) r(i) φ̄_{i,(1,1)}
                        + Φ_i(φ̄_{i,(1,1)})

    Equivalently: d/dt̄ φ̄ = Λ^(2) φ̄ + Φ^(2)(φ̄)
    where Λ^(2) = diag((1-γ)r) + Q.

    Parameters
    ----------
    t_bar : float  (not used; autonomous ODE)
    psi : np.ndarray of shape (m,)  – φ̄_{:,(1,1)}
    params : ModelParams
    """
    m = params.n_regimes
    gamma = params.gamma
    dpsi = np.zeros(m)
    for i in range(m):
        # Regime-switching + interest-rate linear term
        linear = (1.0 - gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        # Consumption contribution
        cons = _phi_i(psi[i], params)
        dpsi[i] = linear + cons
    return dpsi


def _rhs_h01(
    t_bar: float,
    psi: np.ndarray,
    params: ModelParams,
    sol_h11: object,  # dense_output ODE solution object for h=(1,1)
) -> np.ndarray:
    """
    ODE RHS for state h=(0,1): only asset 1 (index 0) survives.

    Requires the already-solved φ̄_{:,(1,1)} evaluated at t_bar.
    """
    m = params.n_regimes
    gamma = params.gamma
    h = (0, 1)
    j_asset = 0  # asset 1 (index 0) survives

    psi_11 = sol_h11(t_bar)  # shape (m,)
    psi_11 = np.maximum(psi_11, 1e-15)

    dpsi = np.zeros(m)
    for i in range(m):
        linear = (1.0 - gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        cons = _phi_i(psi[i], params)

        pi_star = _optimise_pi_1d(i, h, j_asset, psi[i], psi_11[i], params)
        H_val = -_portfolio_hamiltonian_1d(pi_star, i, h, j_asset, psi[i], psi_11[i], params)
        dpsi[i] = linear + cons + H_val
    return dpsi


def _rhs_h10(
    t_bar: float,
    psi: np.ndarray,
    params: ModelParams,
    sol_h11: object,
) -> np.ndarray:
    """
    ODE RHS for state h=(1,0): only asset 2 (index 1) survives.
    """
    m = params.n_regimes
    gamma = params.gamma
    h = (1, 0)
    j_asset = 1  # asset 2 (index 1) survives

    psi_11 = sol_h11(t_bar)
    psi_11 = np.maximum(psi_11, 1e-15)

    dpsi = np.zeros(m)
    for i in range(m):
        linear = (1.0 - gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        cons = _phi_i(psi[i], params)

        pi_star = _optimise_pi_1d(i, h, j_asset, psi[i], psi_11[i], params)
        H_val = -_portfolio_hamiltonian_1d(pi_star, i, h, j_asset, psi[i], psi_11[i], params)
        dpsi[i] = linear + cons + H_val
    return dpsi


def _rhs_h00(
    t_bar: float,
    psi: np.ndarray,
    params: ModelParams,
    sol_h01: object,
    sol_h10: object,
    pi_cache: Optional[Dict] = None,
) -> np.ndarray:
    """
    ODE RHS for state h=(0,0): both assets alive.

    Requires φ̄_{:,(0,1)} and φ̄_{:,(1,0)} evaluated at t_bar.
    """
    m = params.n_regimes
    gamma = params.gamma

    psi_01 = sol_h01(t_bar)
    psi_10 = sol_h10(t_bar)
    psi_01 = np.maximum(psi_01, 1e-15)
    psi_10 = np.maximum(psi_10, 1e-15)

    dpsi = np.zeros(m)
    for i in range(m):
        linear = (1.0 - gamma) * params.r[i] * psi[i] + np.dot(params.Q[i], psi)
        cons = _phi_i(psi[i], params)

        pi_init = None
        if pi_cache is not None and i in pi_cache:
            pi_init = pi_cache[i]

        pi_star = _optimise_pi_2d(i, psi[i], psi_01[i], psi_10[i], params, pi_init=pi_init)
        if pi_cache is not None:
            pi_cache[i] = pi_star

        H_val = -_portfolio_hamiltonian_2d(pi_star, i, psi[i], psi_01[i], psi_10[i], params)
        dpsi[i] = linear + cons + H_val
    return dpsi


# ──────────────────────────────────────────────────────────────────────────────
#  Main recursive solver
# ──────────────────────────────────────────────────────────────────────────────

class HJBSolver:
    """
    Recursive solver for the time-reversed HJB ODE system.

    After calling `.solve()`, provides forward-time interpolation of
    φ(t, i, h) and optimal controls π*(t, i, h), c*(t, i, h).

    Custom implementation – no library directly solves this recursive
    default-contagion regime-switching HJB problem.
    """

    def __init__(self, params: ModelParams, t_eval: Optional[np.ndarray] = None) -> None:
        self.params = params
        self.T = params.T
        self.t_eval = t_eval  # forward-time evaluation points
        self._sols: Dict[Tuple[int, int], object] = {}   # dense ODE solutions (rev. time)
        self._solved = False

    # ── Solve ────────────────────────────────────────────────────────────────

    def solve(self) -> None:
        """
        Solve the recursive ODE system in the mandatory order:
        h=(1,1) → h=(0,1) → h=(1,0) → h=(0,0).

        Uses scipy.integrate.solve_ivp with dense_output=True and tight
        tolerances (rtol=1e-9, atol=1e-11).
        """
        ivp_kwargs = dict(
            t_span=(0.0, self.T),
            method="Radau",
            rtol=1e-9,
            atol=1e-11,
            dense_output=True,
        )

        m = self.params.n_regimes
        y0 = np.ones(m)  # φ̄(0, i, h) = 1 for all i, h

        # 1) h = (1,1) ──────────────────────────────────────────────────────
        sol_11 = solve_ivp(
            lambda t, y: _rhs_h11(t, y, self.params),
            y0=y0.copy(),
            **ivp_kwargs,
        )
        assert sol_11.success, f"ODE h=(1,1) failed: {sol_11.message}"
        self._sols[(1, 1)] = sol_11.sol

        # 2) h = (0,1) ──────────────────────────────────────────────────────
        sol_01 = solve_ivp(
            lambda t, y: _rhs_h01(t, y, self.params, sol_11.sol),
            y0=y0.copy(),
            **ivp_kwargs,
        )
        assert sol_01.success, f"ODE h=(0,1) failed: {sol_01.message}"
        self._sols[(0, 1)] = sol_01.sol

        # 3) h = (1,0) ──────────────────────────────────────────────────────
        sol_10 = solve_ivp(
            lambda t, y: _rhs_h10(t, y, self.params, sol_11.sol),
            y0=y0.copy(),
            **ivp_kwargs,
        )
        assert sol_10.success, f"ODE h=(1,0) failed: {sol_10.message}"
        self._sols[(1, 0)] = sol_10.sol

        # 4) h = (0,0) ──────────────────────────────────────────────────────
        pi_cache: Dict[int, np.ndarray] = {}
        sol_00 = solve_ivp(
            lambda t, y: _rhs_h00(t, y, self.params, sol_01.sol, sol_10.sol, pi_cache),
            y0=y0.copy(),
            **ivp_kwargs,
        )
        assert sol_00.success, f"ODE h=(0,0) failed: {sol_00.message}"
        self._sols[(0, 0)] = sol_00.sol

        self._solved = True

    # ── Forward-time access ───────────────────────────────────────────────────

    def phi(self, t: float, i: int, h: Tuple[int, int]) -> float:
        """
        Forward-time value multiplier φ(t, i, h) = φ̄(T-t, i, h).

        Parameters
        ----------
        t : float  – forward time in [0, T]
        i : int    – regime index (0-based)
        h : tuple  – default state (h1, h2)
        """
        assert self._solved, "Call solve() first."
        t_bar = self.T - t
        return float(self._sols[h](t_bar)[i])

    def phi_array(self, t_arr: np.ndarray, h: Tuple[int, int]) -> np.ndarray:
        """
        Return φ(t, :, h) for all regimes at each time in t_arr.

        Returns
        -------
        np.ndarray of shape (len(t_arr), m)
        """
        assert self._solved
        t_bar_arr = self.T - t_arr
        vals = np.column_stack([self._sols[h](tb) for tb in t_bar_arr])
        return vals.T  # shape (len(t_arr), m)

    def optimal_pi(
        self,
        t: float,
        i: int,
        h: Tuple[int, int],
    ) -> np.ndarray:
        """
        Optimal portfolio weight vector π*(t, i, h).

        For h=(1,1): π* = zeros(n).
        For one-survivor states: scalar optimisation.
        For h=(0,0): two-dimensional optimisation.

        Returns np.ndarray of shape (n,).
        """
        assert self._solved
        params = self.params
        t_bar = self.T - t
        n = params.n_assets
        psi_h_i = float(self._sols[h](t_bar)[i])

        if h == (1, 1):
            return np.zeros(n)

        if h == (0, 1):
            j_asset = 0
            psi_next_i = float(self._sols[(1, 1)](t_bar)[i])
            pi_j = _optimise_pi_1d(i, h, j_asset, psi_h_i, psi_next_i, params)
            pi = np.zeros(n)
            pi[j_asset] = pi_j
            return pi

        if h == (1, 0):
            j_asset = 1
            psi_next_i = float(self._sols[(1, 1)](t_bar)[i])
            pi_j = _optimise_pi_1d(i, h, j_asset, psi_h_i, psi_next_i, params)
            pi = np.zeros(n)
            pi[j_asset] = pi_j
            return pi

        # h = (0, 0)
        psi_01_i = float(self._sols[(0, 1)](t_bar)[i])
        psi_10_i = float(self._sols[(1, 0)](t_bar)[i])
        pi = _optimise_pi_2d(i, psi_h_i, psi_01_i, psi_10_i, params)
        return pi

    def optimal_c(self, t: float, i: int, h: Tuple[int, int]) -> float:
        """
        Optimal consumption rate c*(t, i, h) = (φ(t,i,h)/η)^{1/(η(1-γ)-1)}.
        """
        phi_val = self.phi(t, i, h)
        return _optimal_c(phi_val, self.params)

    # ── Validation / diagnostics ──────────────────────────────────────────────

    def check_terminal_condition(self, tol: float = 1e-6) -> Dict:
        """
        Verify φ(T, i, h) = 1 for all regimes i and states h.
        """
        results = {}
        for h in SOLVE_ORDER:
            for i in range(self.params.n_regimes):
                val = self.phi(self.T, i, h)
                err = abs(val - 1.0)
                results[(h, i)] = {"phi_T": val, "error": err, "pass": err < tol}
        return results

    def check_positivity(self, t_arr: np.ndarray) -> bool:
        """Check that φ̄ > 0 for all states, regimes, and times."""
        for h in SOLVE_ORDER:
            sol = self._sols[h]
            for tb in (self.T - t_arr):
                vals = sol(tb)
                if np.any(vals <= 0):
                    return False
        return True

    def compute_hjb_residuals(
        self,
        t_arr: np.ndarray,
        h: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """
        Estimate HJB residuals for state h at selected forward times.

        Residual = |d/dt φ̄ - ODE_RHS| computed via finite differences.
        Returns array of shape (len(t_arr), m).
        """
        m = self.params.n_regimes
        residuals = np.zeros((len(t_arr), m))
        dt = 1e-5
        sol = self._sols[h]

        for k, t in enumerate(t_arr):
            t_bar = self.T - t
            psi = sol(t_bar)
            if t_bar + dt <= self.T:
                psi_fwd = sol(t_bar + dt)
            else:
                psi_fwd = sol(t_bar - dt)
                dt_sign = -1
            dpsi_fd = (psi_fwd - psi) / dt

            if h == (1, 1):
                dpsi_ode = _rhs_h11(t_bar, psi, self.params)
            elif h == (0, 1):
                dpsi_ode = _rhs_h01(t_bar, psi, self.params, self._sols[(1, 1)])
            elif h == (1, 0):
                dpsi_ode = _rhs_h10(t_bar, psi, self.params, self._sols[(1, 1)])
            else:
                dpsi_ode = _rhs_h00(t_bar, psi, self.params, self._sols[(0, 1)], self._sols[(1, 0)])

            residuals[k] = np.abs(dpsi_fd - dpsi_ode)
        return residuals

    def tabulate_controls(
        self,
        t_arr: np.ndarray,
        h: Tuple[int, int] = (0, 0),
    ) -> list:
        """
        Tabulate varphi, pi*, c*, total risky share for all regimes.

        Returns list of dicts.
        """
        rows = []
        for t in t_arr:
            for i in range(self.params.n_regimes):
                phi_val = self.phi(t, i, h)
                pi = self.optimal_pi(t, i, h)
                c_star = self.optimal_c(t, i, h)
                rows.append({
                    "t": t,
                    "regime": i + 1,
                    "h": str(h),
                    "phi": phi_val,
                    "pi_1": pi[0],
                    "pi_2": pi[1],
                    "pi_total": pi[0] + pi[1],
                    "c_star": c_star,
                })
        return rows
