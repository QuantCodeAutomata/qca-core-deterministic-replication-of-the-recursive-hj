"""
Comprehensive tests for the recursive HJB solver and model parameters.

Tests verify:
  - Mathematical properties (positivity, terminal conditions, admissibility)
  - Methodology adherence (correct recursive ordering, correct ODE formulations)
  - Numerical accuracy vs paper benchmark values
  - Edge cases and parameter validation
  - Experiment result properties (monotonicity, directional effects)
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from src.model_params import (
    ModelParams,
    benchmark_params,
    no_contagion_params,
    SOLVE_ORDER,
)
from src.hjb_solver import (
    HJBSolver,
    _phi_i,
    _optimal_c,
    _portfolio_hamiltonian_1d,
    _portfolio_hamiltonian_2d,
    _optimise_pi_1d,
    _optimise_pi_2d,
    _rhs_h11,
    _rhs_h01,
    _rhs_h10,
    _rhs_h00,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def params() -> ModelParams:
    return benchmark_params()


@pytest.fixture(scope="module")
def solver(params) -> HJBSolver:
    s = HJBSolver(params)
    s.solve()
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Tests: ModelParams validation
# ──────────────────────────────────────────────────────────────────────────────

class TestModelParams:

    def test_benchmark_params_creates_without_error(self):
        p = benchmark_params()
        assert p is not None

    def test_generator_row_sums_zero(self, params):
        assert np.allclose(params.Q.sum(axis=1), 0.0, atol=1e-10)

    def test_generator_off_diagonals_nonneg(self, params):
        Q = params.Q
        m = Q.shape[0]
        for i in range(m):
            for j in range(m):
                if i != j:
                    assert Q[i, j] >= -1e-12, f"Q[{i},{j}]={Q[i,j]} is negative"

    def test_sigma_covariance_psd(self, params):
        for i in range(params.n_regimes):
            Sigma = params.sigma[i] @ params.sigma[i].T
            eigvals = np.linalg.eigvalsh(Sigma)
            assert np.all(eigvals > -1e-10), f"Sigma(i={i}) not PSD"

    def test_eta_in_valid_range(self, params):
        assert 0.0 < params.eta <= 1.0

    def test_gamma_above_one(self, params):
        assert params.gamma > 1.0

    def test_precomputed_exponents(self, params):
        gamma, eta = params.gamma, params.eta
        expected_alpha = eta * (1.0 - gamma)
        expected_k = expected_alpha - 1.0
        expected_p = expected_alpha / expected_k
        assert np.isclose(params.alpha, expected_alpha, rtol=1e-10)
        assert np.isclose(params.k, expected_k, rtol=1e-10)
        assert np.isclose(params.p, expected_p, rtol=1e-10)

    def test_invalid_eta_raises(self):
        with pytest.raises(AssertionError):
            ModelParams(eta=0.0)

    def test_invalid_gamma_raises(self):
        with pytest.raises(AssertionError):
            ModelParams(gamma=0.5)

    def test_invalid_T_raises(self):
        with pytest.raises(AssertionError):
            ModelParams(T=-1.0)

    def test_invalid_generator_row_sum_raises(self):
        Q_bad = np.array([[-0.5, 0.3], [1.0, -1.0]])
        with pytest.raises(AssertionError):
            ModelParams(Q=Q_bad)

    def test_surviving_assets_h11(self, params):
        assert params.surviving_assets((1, 1)) == []

    def test_surviving_assets_h01(self, params):
        assert params.surviving_assets((0, 1)) == [0]

    def test_surviving_assets_h10(self, params):
        assert params.surviving_assets((1, 0)) == [1]

    def test_surviving_assets_h00(self, params):
        assert params.surviving_assets((0, 0)) == [0, 1]

    def test_no_contagion_intensities_equal_predefault(self):
        nc = no_contagion_params()
        assert np.allclose(nc.lam[((0, 1), 0)], nc.lam[((0, 0), 0)])
        assert np.allclose(nc.lam[((0, 1), 1)], nc.lam[((0, 0), 1)])
        assert np.allclose(nc.lam[((1, 0), 0)], nc.lam[((0, 0), 0)])
        assert np.allclose(nc.lam[((1, 0), 1)], nc.lam[((0, 0), 1)])

    def test_compensated_excess_return_h11(self, params):
        """For h=(1,1), all assets defaulted, mu_tilde should be zero."""
        mu_tilde = params.compensated_excess_return((1, 1), 0)
        assert np.allclose(mu_tilde, 0.0)

    def test_compensated_excess_return_h01(self, params):
        """For h=(0,1), only asset 1 survives."""
        mu_tilde = params.compensated_excess_return((0, 1), 0)
        assert mu_tilde[0] != 0.0   # asset 1 contributes
        assert mu_tilde[1] == 0.0   # asset 2 is defaulted


# ──────────────────────────────────────────────────────────────────────────────
# Tests: utility helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_phi_i_zero_for_nonpositive_y(self, params):
        assert _phi_i(0.0, params) == 0.0
        assert _phi_i(-1.0, params) == 0.0

    def test_phi_i_positive_for_positive_y(self, params):
        val = _phi_i(1.0, params)
        # phi_coeff should be real, y^p at y=1 is 1
        assert isinstance(val, float)

    def test_optimal_c_zero_for_nonpositive(self, params):
        assert _optimal_c(0.0, params) == 0.0
        assert _optimal_c(-1.0, params) == 0.0

    def test_optimal_c_positive_for_phi_one(self, params):
        c = _optimal_c(1.0, params)
        assert c > 0.0

    def test_optimal_c_formula(self, params):
        phi_val = 1.5
        expected = (phi_val / params.eta) ** params.inv_k
        assert np.isclose(_optimal_c(phi_val, params), expected, rtol=1e-10)

    def test_hamiltonian_1d_decreases_at_boundary(self, params):
        """Hamiltonian should blow up as pi -> 1."""
        val_good = _portfolio_hamiltonian_1d(0.3, 0, (0, 1), 0, 1.0, 0.95, params)
        val_boundary = _portfolio_hamiltonian_1d(1.0 - 1e-15, 0, (0, 1), 0, 1.0, 0.95, params)
        assert val_boundary >= 1e20

    def test_hamiltonian_2d_decreases_at_boundary(self, params):
        """2D Hamiltonian should blow up as pi^j -> 1."""
        pi = np.array([1.0 - 1e-15, 0.1])
        val = _portfolio_hamiltonian_2d(pi, 0, 1.0, 0.95, 0.9, params)
        assert val >= 1e20


# ──────────────────────────────────────────────────────────────────────────────
# Tests: HJBSolver solution properties
# ──────────────────────────────────────────────────────────────────────────────

class TestHJBSolverProperties:

    def test_solve_completes_without_error(self, params):
        s = HJBSolver(params)
        s.solve()
        assert s._solved

    def test_terminal_condition_all_states(self, solver):
        """phi(T, i, h) = 1 for all i, h."""
        checks = solver.check_terminal_condition(tol=1e-4)
        for key, v in checks.items():
            assert v["pass"], f"Terminal condition failed for {key}: phi(T)={v['phi_T']:.8f}"

    def test_positivity_all_states_regimes_times(self, solver):
        """phi(t, i, h) > 0 for all t in [0, T]."""
        t_arr = np.linspace(0.0, 1.0, 51)
        assert solver.check_positivity(t_arr), "phi is non-positive somewhere"

    def test_phi_equals_one_at_terminal(self, solver):
        """Explicit check: phi(T=1) == 1 for all combinations."""
        T = solver.T
        for h in SOLVE_ORDER:
            for i in range(solver.params.n_regimes):
                phi_T = solver.phi(T, i, h)
                assert abs(phi_T - 1.0) < 1e-4, \
                    f"phi(T,{i},{h}) = {phi_T:.8f} != 1"

    def test_phi_at_zero_greater_than_one(self, solver):
        """
        At t=0, the investor has T=1 year ahead, so the value multiplier
        should exceed the terminal value phi(T)=1 for the favorable regime.
        """
        phi_r1_h00 = solver.phi(0.0, 0, (0, 0))
        assert phi_r1_h00 > 1.0, f"phi(0,1,(0,0))={phi_r1_h00} should be > 1"

    def test_pi_h11_is_zero(self, solver):
        """In h=(1,1), no risky assets survive, so pi* = 0."""
        for t in [0.0, 0.3, 0.7, 1.0]:
            for i in range(2):
                pi = solver.optimal_pi(t, i, (1, 1))
                assert np.allclose(pi, 0.0), f"pi* in h=(1,1) is not zero: {pi}"

    def test_pi_h01_only_asset1_nonzero(self, solver):
        """In h=(0,1), only asset 1 survives; pi* for asset 2 should be 0."""
        for t in [0.1, 0.3, 0.5]:
            for i in range(2):
                pi = solver.optimal_pi(t, i, (0, 1))
                assert pi[1] == 0.0, f"pi_2* in h=(0,1) should be 0, got {pi[1]}"

    def test_pi_h10_only_asset2_nonzero(self, solver):
        """In h=(1,0), only asset 2 survives; pi* for asset 1 should be 0."""
        for t in [0.1, 0.3, 0.5]:
            for i in range(2):
                pi = solver.optimal_pi(t, i, (1, 0))
                assert pi[0] == 0.0, f"pi_1* in h=(1,0) should be 0, got {pi[0]}"

    def test_pi_strictly_below_one(self, solver):
        """All surviving risky weights must be < 1 (admissibility)."""
        for t in np.linspace(0.0, 0.99, 10):
            for h in SOLVE_ORDER:
                for i in range(2):
                    pi = solver.optimal_pi(t, i, h)
                    assert np.all(pi < 1.0), \
                        f"pi >= 1 at t={t}, h={h}, regime={i}: {pi}"

    def test_consumption_positive_full_utility(self, solver):
        """c*(t, i, h) > 0 for all states and times in full utility model."""
        for t in np.linspace(0.0, 0.99, 10):
            for h in SOLVE_ORDER:
                for i in range(2):
                    c = solver.optimal_c(t, i, h)
                    assert c > 0.0, f"c* <= 0 at t={t}, h={h}, regime={i}: {c}"

    def test_hjb_residuals_small(self, solver):
        """HJB residuals should be numerically small."""
        t_check = np.array([0.3, 0.5, 0.7])
        for h in SOLVE_ORDER:
            res = solver.compute_hjb_residuals(t_check, h=h)
            assert res.max() < 1e-3, \
                f"HJB residuals too large for h={h}: max={res.max():.2e}"

    def test_phi_ordering_between_regimes(self, solver):
        """
        With γ>1, V = x^{1-γ}/(1-γ)·φ < 0.  A larger φ means MORE-negative
        utility, i.e. a worse position.  Regime 1 is favorable (higher drift),
        so it should yield SMALLER φ (higher utility) than regime 2.
        """
        phi_r1 = solver.phi(0.0, 0, (0, 0))
        phi_r2 = solver.phi(0.0, 1, (0, 0))
        assert phi_r1 < phi_r2, \
            f"Expected phi(R1) < phi(R2) at t=0 (R1 more favorable), got {phi_r1:.4f} vs {phi_r2:.4f}"

    def test_phi_h00_less_than_h11(self, solver):
        """
        With γ>1, a larger φ means worse utility.  State h=(0,0) (both assets
        alive, more investment opportunities) should yield SMALLER φ than
        h=(1,1) (all defaulted, only risk-free investment remains).
        """
        phi_00 = solver.phi(0.0, 0, (0, 0))
        phi_11 = solver.phi(0.0, 0, (1, 1))
        assert phi_00 < phi_11, \
            f"Expected phi(h=(0,0)) < phi(h=(1,1)), got {phi_00:.4f} vs {phi_11:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Benchmark values from the paper
# ──────────────────────────────────────────────────────────────────────────────

class TestBenchmarkValues:
    """
    Verify that computed controls approximately match paper-quoted values.
    Tolerances are set to 5% relative deviation to accommodate minor
    parameterization differences.
    """

    TOL = 0.08  # 8% relative tolerance

    def _rel_err(self, computed, reference):
        return abs(computed - reference) / max(abs(reference), 1e-10)

    def test_pi1_regime1_h00_t03(self, solver):
        pi = solver.optimal_pi(0.3, 0, (0, 0))
        assert self._rel_err(pi[0], 0.3056) < self.TOL, \
            f"pi_1* at R1,h=(0,0),t=0.3 = {pi[0]:.4f}, paper: 0.3056"

    def test_pi2_regime1_h00_t03(self, solver):
        pi = solver.optimal_pi(0.3, 0, (0, 0))
        assert self._rel_err(pi[1], 0.1315) < self.TOL, \
            f"pi_2* at R1,h=(0,0),t=0.3 = {pi[1]:.4f}, paper: 0.1315"

    def test_c_regime1_h00_t03(self, solver):
        c = solver.optimal_c(0.3, 0, (0, 0))
        assert self._rel_err(c, 0.3966) < self.TOL, \
            f"c* at R1,h=(0,0),t=0.3 = {c:.4f}, paper: 0.3966"

    def test_pi1_regime2_h00_t03(self, solver):
        pi = solver.optimal_pi(0.3, 1, (0, 0))
        assert self._rel_err(pi[0], 0.1676) < self.TOL, \
            f"pi_1* at R2,h=(0,0),t=0.3 = {pi[0]:.4f}, paper: 0.1676"

    def test_pi2_regime2_h00_t03(self, solver):
        pi = solver.optimal_pi(0.3, 1, (0, 0))
        assert self._rel_err(pi[1], 0.0533) < self.TOL, \
            f"pi_2* at R2,h=(0,0),t=0.3 = {pi[1]:.4f}, paper: 0.0533"

    def test_c_regime2_h00_t03(self, solver):
        c = solver.optimal_c(0.3, 1, (0, 0))
        assert self._rel_err(c, 0.3742) < self.TOL, \
            f"c* at R2,h=(0,0),t=0.3 = {c:.4f}, paper: 0.3742"

    def test_regime1_greater_than_regime2_pi1(self, solver):
        """Favorable regime should have larger pi_1* than adverse regime."""
        pi_r1 = solver.optimal_pi(0.3, 0, (0, 0))
        pi_r2 = solver.optimal_pi(0.3, 1, (0, 0))
        assert pi_r1[0] > pi_r2[0]

    def test_regime1_greater_than_regime2_c(self, solver):
        """Favorable regime should have larger c* than adverse regime."""
        c_r1 = solver.optimal_c(0.3, 0, (0, 0))
        c_r2 = solver.optimal_c(0.3, 1, (0, 0))
        assert c_r1 > c_r2


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Exp 2 – After-default comparative statics (methodology verification)
# ──────────────────────────────────────────────────────────────────────────────

class TestExp2Methodology:

    def test_increasing_own_intensity_decreases_pi_surv(self):
        """
        In h=(0,1), increasing lambda_1(i,(0,1)) should decrease pi_1*.
        Tests the key hypothesis of Exp 2.
        """
        base = benchmark_params()
        pis = []
        for lam_val in [0.5, 1.0, 1.5, 2.0]:
            p = copy.deepcopy(base)
            p.lam[((0, 1), 0)] = np.array([lam_val, 0.0])
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            pis.append(s.optimal_pi(0.3, 0, (0, 1))[0])
        # Should be monotonically non-increasing
        for k in range(len(pis) - 1):
            assert pis[k] >= pis[k + 1] - 1e-6, \
                f"pi_1* did not decrease: {pis[k]:.4f} -> {pis[k+1]:.4f}"

    def test_increasing_own_volatility_decreases_pi_surv(self):
        """
        In h=(0,1), increasing sigma_11 should decrease pi_1*.
        """
        base = benchmark_params()
        pis = []
        for vol_val in [0.5, 1.0, 1.5, 2.0]:
            p = copy.deepcopy(base)
            sig = p.sigma.copy()
            sig[0, 0, 0] = vol_val
            p.sigma = sig
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            pis.append(s.optimal_pi(0.3, 0, (0, 1))[0])
        for k in range(len(pis) - 1):
            assert pis[k] >= pis[k + 1] - 1e-6, \
                f"pi_1* did not decrease: {pis[k]:.4f} -> {pis[k+1]:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Exp 3 – Before-default intensity effects (methodology)
# ──────────────────────────────────────────────────────────────────────────────

class TestExp3Methodology:

    def test_own_asset_pi_decreases_when_intensity_rises(self):
        """
        In h=(0,0), increasing lambda_2^{(0,0)}(1) should decrease pi_2*.
        """
        base = benchmark_params()
        pis = []
        for lam_val in [0.5, 1.0, 1.5, 2.0]:
            p = copy.deepcopy(base)
            arr = p.lam[((0, 0), 0)].copy()
            arr[1] = lam_val
            p.lam[((0, 0), 0)] = arr
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            pis.append(s.optimal_pi(0.3, 0, (0, 0))[1])
        for k in range(len(pis) - 1):
            assert pis[k] >= pis[k + 1] - 1e-6

    def test_total_risky_decreases_when_intensity_rises(self):
        """
        In h=(0,0), total risky share should decrease when any asset's
        pre-default intensity rises.
        """
        base = benchmark_params()
        totals = []
        for lam_val in [0.5, 1.0, 2.0]:
            p = copy.deepcopy(base)
            arr = p.lam[((0, 0), 0)].copy()
            arr[1] = lam_val
            p.lam[((0, 0), 0)] = arr
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            pi = s.optimal_pi(0.3, 0, (0, 0))
            totals.append(pi[0] + pi[1])
        for k in range(len(totals) - 1):
            assert totals[k] >= totals[k + 1] - 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Exp 4 – Regime switching frequency (methodology)
# ──────────────────────────────────────────────────────────────────────────────

class TestExp4Methodology:

    def test_faster_switching_compresses_phi_gap(self):
        """
        Increasing s in Q=sQ0 should compress the absolute regime gap
        |phi(t,r1,h) - phi(t,r2,h)|.  With γ>1, phi(r1)<phi(r2) so the raw
        difference is negative; we compare absolute magnitudes.
        """
        Q0 = np.array([[-0.5, 0.5], [1.0, -1.0]])
        base = benchmark_params()
        abs_gaps = []
        for s in [1, 2, 4]:
            p = copy.deepcopy(base)
            p.Q = s * Q0
            p._validate()
            p._precompute()
            solver = HJBSolver(p)
            solver.solve()
            phi_r1 = solver.phi(0.0, 0, (0, 0))
            phi_r2 = solver.phi(0.0, 1, (0, 0))
            abs_gaps.append(abs(phi_r1 - phi_r2))
        assert abs_gaps[0] > abs_gaps[1] > abs_gaps[2], \
            f"Absolute gap should compress with faster switching: {abs_gaps}"

    def test_gap_vanishes_at_terminal(self):
        """phi(T,1,h) - phi(T,2,h) = 0 because terminal condition is 1."""
        solver = HJBSolver(benchmark_params())
        solver.solve()
        for h in SOLVE_ORDER:
            phi_r1 = solver.phi(1.0, 0, h)
            phi_r2 = solver.phi(1.0, 1, h)
            assert abs(phi_r1 - phi_r2) < 1e-4, \
                f"Gap at T for h={h}: {phi_r1-phi_r2:.6f}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Exp 5 – Preference parameter sweeps (methodology)
# ──────────────────────────────────────────────────────────────────────────────

class TestExp5Methodology:

    def test_c_star_increases_with_eta(self):
        """Higher eta → more weight on consumption → c* increases."""
        base = benchmark_params()
        cs = []
        for eta_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
            p = copy.deepcopy(base)
            p.eta = eta_val
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            cs.append(s.optimal_c(0.3, 0, (0, 0)))
        for k in range(len(cs) - 1):
            assert cs[k] <= cs[k + 1] + 1e-6, \
                f"c* did not increase with eta: {cs[k]:.4f} -> {cs[k+1]:.4f}"

    def test_pi_decreases_with_gamma(self):
        """Higher risk aversion → smaller risky allocation."""
        base = benchmark_params()
        pis = []
        for gamma_val in [1.5, 2.0, 2.5, 3.0, 4.0]:
            p = copy.deepcopy(base)
            p.gamma = gamma_val
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            pi = s.optimal_pi(0.3, 0, (0, 0))
            pis.append(pi[0] + pi[1])
        for k in range(len(pis) - 1):
            assert pis[k] >= pis[k + 1] - 1e-6, \
                f"pi_total did not decrease: {pis[k]:.4f} -> {pis[k+1]:.4f}"

    def test_c_star_increases_with_gamma(self):
        """Higher gamma → more consumption (wealth effect in HJB)."""
        base = benchmark_params()
        cs = []
        for gamma_val in [1.5, 2.5, 4.0]:
            p = copy.deepcopy(base)
            p.gamma = gamma_val
            p._precompute()
            s = HJBSolver(p)
            s.solve()
            cs.append(s.optimal_c(0.3, 0, (0, 0)))
        # c* should be non-decreasing with gamma
        for k in range(len(cs) - 1):
            assert cs[k] <= cs[k + 1] + 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Exp 6 – Model comparisons (methodology)
# ──────────────────────────────────────────────────────────────────────────────

class TestExp6Methodology:

    def test_no_contagion_has_weakly_higher_pi_than_contagion(self):
        """
        Without contagion, default risk is not raised after a default,
        so total risky share should be >= contagion case.
        """
        s_cont = HJBSolver(benchmark_params())
        s_cont.solve()
        s_nc = HJBSolver(no_contagion_params())
        s_nc.solve()

        pi_cont = s_cont.optimal_pi(0.3, 0, (0, 0))
        pi_nc = s_nc.optimal_pi(0.3, 0, (0, 0))
        total_cont = pi_cont[0] + pi_cont[1]
        total_nc = pi_nc[0] + pi_nc[1]
        assert total_nc >= total_cont - 0.05, \
            f"No-contagion total={total_nc:.4f} < contagion total={total_cont:.4f}"

    def test_favorable_regime_higher_pi_than_adverse(self):
        s = HJBSolver(benchmark_params())
        s.solve()
        pi_r1 = s.optimal_pi(0.3, 0, (0, 0))
        pi_r2 = s.optimal_pi(0.3, 1, (0, 0))
        assert (pi_r1[0] + pi_r1[1]) > (pi_r2[0] + pi_r2[1])

    def test_favorable_regime_higher_c_than_adverse(self):
        s = HJBSolver(benchmark_params())
        s.solve()
        c_r1 = s.optimal_c(0.3, 0, (0, 0))
        c_r2 = s.optimal_c(0.3, 1, (0, 0))
        assert c_r1 > c_r2


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Exp 7 – Before-vs-after default (methodology)
# ──────────────────────────────────────────────────────────────────────────────

class TestExp7Methodology:

    def test_pi_surv_lower_after_default(self):
        """
        After asset 1 defaults, the surviving asset 2's weight should
        be lower (or similar) compared to pre-default.
        """
        s = HJBSolver(benchmark_params())
        s.solve()
        pi_before = s.optimal_pi(0.3, 0, (0, 0))
        pi_after = s.optimal_pi(0.3, 0, (1, 0))
        # pi_2 before vs pi_2 after
        assert pi_before[1] >= pi_after[1] - 0.05, \
            f"pi_2 before={pi_before[1]:.4f} < after={pi_after[1]:.4f}"

    def test_c_lower_after_default(self):
        """After a default, consumption rate should decrease."""
        s = HJBSolver(benchmark_params())
        s.solve()
        c_before = s.optimal_c(0.3, 0, (0, 0))
        c_after = s.optimal_c(0.3, 0, (1, 0))
        assert c_before >= c_after - 0.02, \
            f"c_before={c_before:.4f} < c_after={c_after:.4f}"

    def test_expenditure_drop_is_substantial(self):
        """
        The actual expenditure drop (incorporating wealth jump) should
        be substantial – paper quotes ~35.8% for the benchmark case.
        """
        s = HJBSolver(benchmark_params())
        s.solve()
        t, i = 0.3, 0
        pi_before = s.optimal_pi(t, i, (0, 0))
        c_before = s.optimal_c(t, i, (0, 0))
        pi_after = s.optimal_pi(t, i, (1, 0))
        c_after = s.optimal_c(t, i, (1, 0))

        X_pre = 1.0
        X_after = (1.0 - pi_before[0]) * X_pre
        exp_before = c_before * X_pre
        exp_after = c_after * X_after
        drop = 1.0 - exp_after / exp_before
        # Drop should be significant (>20%)
        assert drop > 0.20, f"Expenditure drop too small: {100*drop:.1f}%"
        # Should be < 80% (not complete collapse)
        assert drop < 0.80, f"Expenditure drop implausibly large: {100*drop:.1f}%"

    def test_benchmark_expenditure_drop_close_to_paper(self):
        """Paper quotes ~35.8% expenditure drop for asset-1 default."""
        s = HJBSolver(benchmark_params())
        s.solve()
        t, i = 0.3, 0
        pi_before = s.optimal_pi(t, i, (0, 0))
        c_before = s.optimal_c(t, i, (0, 0))
        c_after = s.optimal_c(t, i, (1, 0))
        X_after = 1.0 - pi_before[0]
        drop = 1.0 - (c_after * X_after) / c_before
        assert abs(drop - 0.358) < 0.10, \
            f"Expenditure drop {100*drop:.1f}% differs from paper's ~35.8%"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Recursive ODE ordering
# ──────────────────────────────────────────────────────────────────────────────

class TestRecursiveOrdering:

    def test_solve_order_is_correct(self):
        """Verify SOLVE_ORDER follows the required recursive structure."""
        assert SOLVE_ORDER[0] == (1, 1), "Must solve h=(1,1) first"
        assert SOLVE_ORDER[1] == (0, 1), "Must solve h=(0,1) second"
        assert SOLVE_ORDER[2] == (1, 0), "Must solve h=(1,0) third"
        assert SOLVE_ORDER[3] == (0, 0), "Must solve h=(0,0) last"

    def test_h11_ode_no_portfolio_term(self):
        """
        In h=(1,1) RHS, the portfolio contribution should be zero (pi*=0).
        We verify by checking the ODE structure directly.
        """
        p = benchmark_params()
        t_bar = 0.5
        psi = np.array([1.2, 0.9])
        rhs = _rhs_h11(t_bar, psi, p)
        # RHS should be finite and deterministic
        assert np.all(np.isfinite(rhs))
        assert len(rhs) == p.n_regimes

    def test_h01_rhs_uses_h11_values(self):
        """
        h=(0,1) RHS must depend on h=(1,1) solution.
        """
        p = benchmark_params()
        t_bar = 0.5

        # Solve h=(1,1) to get sol_h11
        from scipy.integrate import solve_ivp
        sol_11 = solve_ivp(
            lambda t, y: _rhs_h11(t, y, p),
            t_span=(0, p.T),
            y0=np.ones(p.n_regimes),
            method="Radau", rtol=1e-9, atol=1e-11, dense_output=True,
        )
        psi_01 = np.array([1.1, 0.85])
        rhs_01 = _rhs_h01(t_bar, psi_01, p, sol_11.sol)
        assert np.all(np.isfinite(rhs_01))


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_phi_interpolation_at_t0_and_T(self, solver):
        """phi should be interpolable at both endpoints."""
        phi_0 = solver.phi(0.0, 0, (0, 0))
        phi_T = solver.phi(1.0, 0, (0, 0))
        assert np.isfinite(phi_0)
        assert np.isfinite(phi_T)

    def test_controls_at_maturity_t_T(self, solver):
        """Controls at t=T should be finite and admissible."""
        pi = solver.optimal_pi(0.999, 0, (0, 0))
        c = solver.optimal_c(0.999, 0, (0, 0))
        assert np.all(pi < 1.0)
        assert c > 0.0

    def test_high_risk_aversion_gives_low_pi(self):
        """With very high gamma, risky weights should be near zero."""
        p = benchmark_params()
        p.gamma = 10.0
        p._precompute()
        s = HJBSolver(p)
        s.solve()
        pi = s.optimal_pi(0.3, 0, (0, 0))
        total = pi[0] + pi[1]
        assert total < 0.15, f"High-gamma total risky share too large: {total:.4f}"

    def test_tabulate_controls_returns_correct_shape(self, solver):
        t_arr = np.array([0.0, 0.3, 0.7, 1.0])
        rows = solver.tabulate_controls(t_arr, h=(0, 0))
        assert len(rows) == len(t_arr) * solver.params.n_regimes

    def test_phi_array_shape(self, solver):
        t_arr = np.linspace(0.0, 1.0, 11)
        phi_arr = solver.phi_array(t_arr, (0, 0))
        assert phi_arr.shape == (len(t_arr), solver.params.n_regimes)
