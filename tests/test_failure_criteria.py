"""
tests/test_failure_criteria.py
===============================
Unit tests for beam_fea.failure_criteria

Covers:
- Scalar inputs (hand-calc validation)
- Array inputs (vectorisation)
- MoS = SF − 1 identity
- Exact failure boundaries (SF ≈ 1)
- Validation of material axes keyword arguments
- Removal of FI key
"""

import pytest
import numpy as np
from beam_fea.failure_criteria import (
    VonMisesCriterion,
    TrescaCriterion,
    MaxPrincipalStressCriterion,
    MaximumStressCriterion,
    TsaiHillCriterion,
    TsaiWuCriterion,
    MaximumStrainCriterion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_mos_identity(result):
    """MoS = SF − 1 must hold everywhere."""
    SF = result['SF']
    MoS = result['MoS']
    np.testing.assert_allclose(MoS, SF - 1.0, rtol=1e-9)
    assert 'FI' not in result


# ---------------------------------------------------------------------------
# VonMisesCriterion
# ---------------------------------------------------------------------------

class TestVonMises:
    def test_uniaxial_at_yield(self):
        """Pure uniaxial tension at exactly yield: SF = 1."""
        crit = VonMisesCriterion(yield_strength=250.0)
        result = crit.evaluate(sigma_x=250.0, sigma_y=0.0, tau_xy=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)
        assert result['stress'] == pytest.approx(250.0, rel=1e-6)
        assert result['passed'] == True

    def test_below_yield(self):
        crit = VonMisesCriterion(yield_strength=250.0)
        result = crit.evaluate(sigma_x=100.0, sigma_y=0.0, tau_xy=0.0)
        assert result['SF'] > 1.0
        assert result['passed'] == True

    def test_pure_shear(self):
        """Pure shear at τ_y = σ_y / √3 yields SF = 1."""
        sigma_y = 250.0
        tau_y = sigma_y / np.sqrt(3.0)
        crit = VonMisesCriterion(yield_strength=sigma_y)
        result = crit.evaluate(sigma_x=0.0, sigma_y=0.0, tau_xy=tau_y)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_scalar_array_equivalence(self):
        """Array input must produce same result as looping scalars."""
        crit = VonMisesCriterion(yield_strength=300.0)
        sx = np.array([100.0, 200.0, 300.0])
        r_arr  = crit.evaluate(sigma_x=sx, sigma_y=0.0, tau_xy=0.0)
        r_loop = [crit.evaluate(sigma_x=v, sigma_y=0.0, tau_xy=0.0)['SF'] for v in sx]
        np.testing.assert_allclose(r_arr['SF'], r_loop, rtol=1e-9)

    def test_mos_identity(self):
        crit = VonMisesCriterion(yield_strength=250.0)
        result = crit.evaluate(
            sigma_x=np.linspace(10, 300, 20),
            sigma_y=0.0, tau_xy=0.0,
        )
        assert_mos_identity(result)

    def test_invalid_yield(self):
        with pytest.raises(ValueError):
            VonMisesCriterion(yield_strength=-100.0)
            
    def test_invalid_kwargs(self):
        with pytest.raises(TypeError):
            VonMisesCriterion(yield_strength=250).evaluate(sigma_1=100)


# ---------------------------------------------------------------------------
# TrescaCriterion
# ---------------------------------------------------------------------------

class TestTresca:
    def test_uniaxial_at_yield(self):
        """Uniaxial tension at yield: σ₁ − σ₂ = σ_y − 0 → SF = 1."""
        crit = TrescaCriterion(yield_strength=250.0)
        result = crit.evaluate(sigma_x=250.0, sigma_y=0.0, tau_xy=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_tresca_vs_von_mises_pure_shear(self):
        """At pure shear, Tresca yields at τ_y = σ_y/2, VM at σ_y/√3."""
        sy = 300.0
        crit_t = TrescaCriterion(yield_strength=sy)
        crit_v = VonMisesCriterion(yield_strength=sy)
        tau = sy / 2.0  # Tresca shear limit
        r_t = crit_t.evaluate(sigma_x=0.0, sigma_y=0.0, tau_xy=tau)
        r_v = crit_v.evaluate(sigma_x=0.0, sigma_y=0.0, tau_xy=tau)
        assert r_t['SF'] == pytest.approx(1.0, rel=1e-6)
        # VM is less conservative: SF > 1 at Tresca limit
        assert r_v['SF'] > 1.0


# ---------------------------------------------------------------------------
# MaxPrincipalStressCriterion
# ---------------------------------------------------------------------------

class TestMaxPrincipal:
    def test_uniaxial_at_Ftu(self):
        crit = MaxPrincipalStressCriterion(Ftu=500.0)
        result = crit.evaluate(sigma_x=500.0, sigma_y=0.0, tau_xy=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_compression_uses_Fcu(self):
        crit = MaxPrincipalStressCriterion(Ftu=500.0, Fcu=300.0)
        result = crit.evaluate(sigma_x=0.0, sigma_y=-300.0, tau_xy=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# MaximumStressCriterion
# ---------------------------------------------------------------------------

class TestMaximumStress:
    def test_longitudinal_tension(self):
        crit = MaximumStressCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(sigma_1=1500.0, sigma_2=0.0, tau_12=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_shear_dominant(self):
        crit = MaximumStressCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(sigma_1=0.0, sigma_2=0.0, tau_12=70.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_safe_under_all(self):
        crit = MaximumStressCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(sigma_1=100.0, sigma_2=10.0, tau_12=5.0)
        assert result['SF'] > 1.0

    def test_vectorized(self):
        crit = MaximumStressCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        sigma_1 = np.array([0, 750.0, 1500.0])
        result = crit.evaluate(sigma_1=sigma_1, sigma_2=0.0, tau_12=0.0)
        np.testing.assert_allclose(result['stress'], [0.0, 0.5, 1.0], atol=1e-9)

    def test_invalid_kwargs(self):
        crit = MaximumStressCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        with pytest.raises(TypeError):
            crit.evaluate(sigma_x=100)


# ---------------------------------------------------------------------------
# TsaiHillCriterion
# ---------------------------------------------------------------------------

class TestTsaiHill:
    def test_pure_longitudinal_tension(self):
        crit = TsaiHillCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(sigma_1=1500.0, sigma_2=0.0, tau_12=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_pure_shear(self):
        crit = TsaiHillCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(sigma_1=0.0, sigma_2=0.0, tau_12=70.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_mos_identity(self):
        crit = TsaiHillCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(
            sigma_1=np.linspace(100, 1400, 10),
            sigma_2=5.0,
            tau_12=10.0,
        )
        assert_mos_identity(result)


# ---------------------------------------------------------------------------
# TsaiWuCriterion
# ---------------------------------------------------------------------------

class TestTsaiWu:
    def test_safe_state(self):
        crit = TsaiWuCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        result = crit.evaluate(sigma_1=100.0, sigma_2=5.0, tau_12=5.0)
        assert result['SF'] > 1.0

    def test_longitudinal_tension_only(self):
        """At σ₁ = Xt, F11*σ₁² + F1*σ₁ = 1 → SF = 1 only when F1 ≈ 0."""
        Xt, Xc = 1500.0, 1500.0  # For symmetric: F1 = 0
        crit = TsaiWuCriterion(Xt=Xt, Xc=Xc, Yt=500.0, Yc=500.0, S=500.0, F12=0.0)
        result = crit.evaluate(sigma_1=Xt, sigma_2=0.0, tau_12=0.0)
        # F11*Xt^2 = 1/(Xt*Xc) * Xt^2 = Xt/Xc = 1
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_vectorized(self):
        crit = TsaiWuCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        SFs = crit.evaluate(
            sigma_1=np.array([0.0, 300.0, 750.0]),
            sigma_2=0.0,
            tau_12=0.0,
        )['SF']
        assert SFs.shape == (3,)
        assert SFs[0] == np.inf
        assert SFs[1] > SFs[2]


# ---------------------------------------------------------------------------
# MaximumStrainCriterion
# ---------------------------------------------------------------------------

class TestMaximumStrain:
    def test_longitudinal_tension(self):
        crit = MaximumStrainCriterion(
            eps_Xt=0.01, eps_Xc=0.008,
            eps_Yt=0.005, eps_Yc=0.025,
            gamma_S=0.02,
        )
        result = crit.evaluate(eps_1=0.01, eps_2=0.0, gamma_12=0.0)
        assert result['SF'] == pytest.approx(1.0, rel=1e-6)

    def test_safe_state(self):
        crit = MaximumStrainCriterion(
            eps_Xt=0.01, eps_Xc=0.008,
            eps_Yt=0.005, eps_Yc=0.025,
            gamma_S=0.02,
        )
        result = crit.evaluate(eps_1=0.005, eps_2=0.001, gamma_12=0.005)
        assert result['SF'] > 1.0
