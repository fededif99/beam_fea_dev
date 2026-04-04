"""
failure_criteria.py
===================
Unified Failure Criteria for metals and composite laminates.

All criteria accept scalars or NumPy arrays and return a dict containing:
  - 'stress' : Governing equivalent stress (or dimensionless ratio/index).
  - 'SF'     : Safety Factor = allowable / stress. SF >= 1 → safe; SF < 1 → failed.
  - 'MoS'    : Margin of Safety = SF - 1. MoS >= 0 → safe.
  - 'passed' : Boolean array. True where SF >= 1.

Usage
-----
Isotropic (metal)
>>> from beam_fea.failure_criteria import VonMisesCriterion
>>> crit = VonMisesCriterion(yield_strength=250.0)  # MPa
>>> result = crit.evaluate(sigma_x=180.0, sigma_y=50.0, tau_xy=30.0)
>>> result['SF']   # → > 1.0 (safe)

Composite ply (material axes)
>>> from beam_fea.failure_criteria import TsaiWuCriterion
>>> crit = TsaiWuCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
>>> result = crit.evaluate(sigma_1=300.0, sigma_2=10.0, tau_12=5.0)
"""

from __future__ import annotations

import enum
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_Array = Union[float, np.ndarray]
_Result = Dict[str, np.ndarray]

class _MaterialType(enum.Enum):
    ISOTROPIC = "isotropic"
    COMPOSITE = "composite"

# ---------------------------------------------------------------------------
# Shared Internal Helpers
# ---------------------------------------------------------------------------

def _principal_stresses_3d(
    sigma_x: _Array, sigma_y: _Array, sigma_z: _Array,
    tau_xy: _Array, tau_yz: _Array, tau_xz: _Array
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 3D principal stresses (s1, s2, s3) and max shear stress (t_max)."""
    # Broadcast to common shape
    sh = np.broadcast(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz).shape
    tensor = np.zeros(sh + (3, 3))
    tensor[..., 0, 0] = sigma_x
    tensor[..., 1, 1] = sigma_y
    tensor[..., 2, 2] = sigma_z
    tensor[..., 0, 1] = tensor[..., 1, 0] = tau_xy
    tensor[..., 1, 2] = tensor[..., 2, 1] = tau_yz
    tensor[..., 0, 2] = tensor[..., 2, 0] = tau_xz
    
    # eigvalsh returns eigenvalues in ascending order: s3 <= s2 <= s1
    eigenvals = np.linalg.eigvalsh(tensor)
    s1 = eigenvals[..., 2]
    s2 = eigenvals[..., 1]
    s3 = eigenvals[..., 0]
    t_max = (s1 - s3) / 2.0
    return s1, s2, s3, t_max

def _ensure_arrays(*args: _Array) -> tuple[np.ndarray, ...]:
    """Convert scalar/arrays to float ndarrays and broadcast to common shape."""
    arrs = tuple(np.asarray(a, dtype=float) for a in args)
    return np.broadcast_arrays(*arrs)

def _result(stress: _Array, allowable: float) -> _Result:
    """
    Package stress and allowable into the standard output dictionary.
    
    Parameters
    ----------
    stress : array-like
        The computed governing stress (or dimensionless ratio/index).
    allowable : float
        The allowable strength (often 1.0 for dimensionless index criteria).
    """
    strs = np.asarray(stress, dtype=float)
    allow = float(allowable)

    with np.errstate(divide='ignore', invalid='ignore'):
        sf = np.where(strs > 0, allow / strs, np.inf)

    return {
        'stress': strs,
        'SF': sf,
        'MoS': sf - 1.0,
        'passed': sf >= 1.0,
    }

# ===========================================================================
# Abstract base
# ===========================================================================

class FailureCriterion(ABC):
    """Abstract base class for all failure criteria."""

    @property
    @abstractmethod
    def material_type(self) -> _MaterialType:
        """Type of material the criterion targets (Isotropic vs Composite)."""
        pass

    def evaluate(self, **kwargs) -> _Result:
        """
        Evaluate the failure criterion. Validates input keys to prevent
        using composite metrics (sigma_1) for metals and vice versa.

        Returns
        -------
        dict with keys 'stress', 'SF', 'MoS', 'passed'.
        """
        is_comp = any(k in ('sigma_1', 'sigma_2', 'tau_12', 'eps_1', 'eps_2', 'gamma_12') for k in kwargs)
        is_iso  = any(k in ('sigma_x', 'sigma_y', 'tau_xy') for k in kwargs)

        if self.material_type == _MaterialType.ISOTROPIC and is_comp:
            raise TypeError("Isotropic criteria expect global axes stresses (sigma_x, sigma_y, tau_xy).")
        if self.material_type == _MaterialType.COMPOSITE and is_iso:
            raise TypeError("Composite criteria expect material axes stresses/strains (sigma_1, eps_1, etc.).")

        return self._evaluate_impl(**kwargs)

    @abstractmethod
    def _evaluate_impl(self, **kwargs) -> _Result:
        pass


# ===========================================================================
# Isotropic / Metal criteria
# ===========================================================================

class VonMisesCriterion(FailureCriterion):
    """
    Von Mises yield criterion (2D plane stress).

    Parameters
    ----------
    yield_strength : float
        Material yield stress, σ_y (MPa).
    """
    material_type = _MaterialType.ISOTROPIC

    def __init__(self, yield_strength: float):
        if yield_strength <= 0:
            raise ValueError("yield_strength must be positive.")
        self.yield_strength = float(yield_strength)

    def _evaluate_impl(
        self,
        sigma_x: _Array = 0.0,
        sigma_y: _Array = 0.0,
        sigma_z: _Array = 0.0,
        tau_xy: _Array = 0.0,
        tau_yz: _Array = 0.0,
        tau_xz: _Array = 0.0,
        **_,
    ) -> _Result:
        sx, sy, sz, txy, tyz, txz = _ensure_arrays(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz)
        
        # Exact 3D Von Mises invariant
        sigma_vm = np.sqrt(
            0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2) +
            3.0 * (txy**2 + tyz**2 + txz**2)
        )
        return _result(stress=sigma_vm, allowable=self.yield_strength)


class TrescaCriterion(FailureCriterion):
    """
    Tresca (maximum-shear-stress) yield criterion (2D plane stress).

    Parameters
    ----------
    yield_strength : float
        Material yield stress, σ_y (MPa).
    """
    material_type = _MaterialType.ISOTROPIC

    def __init__(self, yield_strength: float):
        if yield_strength <= 0:
            raise ValueError("yield_strength must be positive.")
        self.yield_strength = float(yield_strength)

    def _evaluate_impl(
        self,
        sigma_x: _Array = 0.0,
        sigma_y: _Array = 0.0,
        sigma_z: _Array = 0.0,
        tau_xy: _Array = 0.0,
        tau_yz: _Array = 0.0,
        tau_xz: _Array = 0.0,
        **_,
    ) -> _Result:
        sx, sy, sz, txy, tyz, txz = _ensure_arrays(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz)
        s1, s2, s3, t_max = _principal_stresses_3d(sx, sy, sz, txy, tyz, txz)

        # Tresca equivalent stress is s1 - s2, which is exactly 2 * t_max
        return _result(stress=2.0 * t_max, allowable=self.yield_strength)


class MaxPrincipalStressCriterion(FailureCriterion):
    """
    Maximum Principal Stress criterion (brittle fracture).

    Parameters
    ----------
    Ftu : float
        Tensile ultimate strength (MPa).
    Fcu : float, optional
        Compressive ultimate strength (MPa, positive value).
        Defaults to Ftu if not specified.
    """
    material_type = _MaterialType.ISOTROPIC

    def __init__(self, Ftu: float, Fcu: float | None = None):
        if Ftu <= 0:
            raise ValueError("Ftu must be positive.")
        self.Ftu = float(Ftu)
        self.Fcu = float(Fcu) if Fcu is not None else self.Ftu
        if self.Fcu <= 0:
            raise ValueError("Fcu must be positive.")

    def _evaluate_impl(
        self,
        sigma_x: _Array = 0.0,
        sigma_y: _Array = 0.0,
        sigma_z: _Array = 0.0,
        tau_xy: _Array = 0.0,
        tau_yz: _Array = 0.0,
        tau_xz: _Array = 0.0,
        **_,
    ) -> _Result:
        sx, sy, sz, txy, tyz, txz = _ensure_arrays(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_xz)
        s1, s2, s3, _ = _principal_stresses_3d(sx, sy, sz, txy, tyz, txz)

        # Max Principal considers s1 and s3
        index = np.maximum(s1 / self.Ftu, -s3 / self.Fcu)
        return _result(stress=index, allowable=1.0)


# ===========================================================================
# Composite ply-level criteria (stresses in material / ply axes)
# ===========================================================================

class MaximumStressCriterion(FailureCriterion):
    """
    Maximum Stress criterion for orthotropic plies (material axes).

    Parameters
    ----------
    Xt, Xc : float
        Longitudinal tensile / compressive strength (MPa, both positive).
    Yt, Yc : float
        Transverse tensile / compressive strength (MPa, both positive).
    S : float
        In-plane shear strength (MPa).
    """
    material_type = _MaterialType.COMPOSITE

    def __init__(self, Xt: float, Xc: float, Yt: float, Yc: float, S: float, S13: float | None = None, S23: float | None = None):
        if any(v <= 0 for v in [Xt, Xc, Yt, Yc, S]):
            raise ValueError("All strengths must be positive.")
        self.Xt, self.Xc = float(Xt), float(Xc)
        self.Yt, self.Yc = float(Yt), float(Yc)
        self.S = float(S)
        self.S13 = float(S13) if S13 and float(S13) > 0 else self.S
        self.S23 = float(S23) if S23 and float(S23) > 0 else self.S

    def _evaluate_impl(
        self,
        sigma_1: _Array = 0.0,
        sigma_2: _Array = 0.0,
        tau_12:  _Array = 0.0,
        tau_13:  _Array = 0.0,
        tau_23:  _Array = 0.0,
        **_,
    ) -> _Result:
        s1, s2, t12, t13, t23 = _ensure_arrays(sigma_1, sigma_2, tau_12, tau_13, tau_23)

        terms = np.stack([
            s1 / self.Xt,
            -s1 / self.Xc,
            s2 / self.Yt,
            -s2 / self.Yc,
            np.abs(t12) / self.S,
            np.abs(t13) / self.S13,
            np.abs(t23) / self.S23,
        ], axis=0)
        
        index = np.max(terms, axis=0)
        return _result(stress=index, allowable=1.0)


class TsaiHillCriterion(FailureCriterion):
    """
    Tsai-Hill criterion for orthotropic plies.

    Parameters
    ----------
    Xt, Xc : float  Longitudinal tensile / compressive strength (MPa).
    Yt, Yc : float  Transverse tensile / compressive strength (MPa).
    S : float       In-plane shear strength (MPa).
    """
    material_type = _MaterialType.COMPOSITE

    def __init__(self, Xt: float, Xc: float, Yt: float, Yc: float, S: float, S13: float | None = None, S23: float | None = None):
        if any(v <= 0 for v in [Xt, Xc, Yt, Yc, S]):
            raise ValueError("All strengths must be positive.")
        self.Xt, self.Xc = float(Xt), float(Xc)
        self.Yt, self.Yc = float(Yt), float(Yc)
        self.S = float(S)
        self.S13 = float(S13) if S13 and float(S13) > 0 else self.S
        self.S23 = float(S23) if S23 and float(S23) > 0 else self.S

    def _evaluate_impl(
        self,
        sigma_1: _Array = 0.0,
        sigma_2: _Array = 0.0,
        tau_12:  _Array = 0.0,
        tau_13:  _Array = 0.0,
        tau_23:  _Array = 0.0,
        **_,
    ) -> _Result:
        s1, s2, t12, t13, t23 = _ensure_arrays(sigma_1, sigma_2, tau_12, tau_13, tau_23)

        X = np.where(s1 >= 0, self.Xt, self.Xc)
        Y = np.where(s2 >= 0, self.Yt, self.Yc)

        index = (s1 / X) ** 2 - (s1 * s2 / X ** 2) + (s2 / Y) ** 2 + (t12 / self.S) ** 2 + (t13 / self.S13) ** 2 + (t23 / self.S23) ** 2
        return _result(stress=index, allowable=1.0)


class TsaiWuCriterion(FailureCriterion):
    """
    Tsai-Wu tensor polynomial criterion for orthotropic plies.

    Parameters
    ----------
    Xt, Xc : float  Longitudinal tensile / compressive ultimate strengths (MPa).
    Yt, Yc : float  Transverse tensile / compressive ultimate strengths (MPa).
    S : float       In-plane shear strength (MPa).
    F12 : float, optional
        Interaction coefficient. Default = −0.5/√(Xt·Xc·Yt·Yc).
    """
    material_type = _MaterialType.COMPOSITE

    def __init__(
        self,
        Xt: float, Xc: float,
        Yt: float, Yc: float,
        S: float,
        F12: float | None = None,
        S13: float | None = None,
        S23: float | None = None,
    ):
        if any(v <= 0 for v in [Xt, Xc, Yt, Yc, S]):
            raise ValueError("All strengths must be positive.")
        self.Xt, self.Xc = float(Xt), float(Xc)
        self.Yt, self.Yc = float(Yt), float(Yc)
        self.S = float(S)
        self.S13 = float(S13) if S13 and float(S13) > 0 else self.S
        self.S23 = float(S23) if S23 and float(S23) > 0 else self.S

        self.F1  = 1.0 / Xt - 1.0 / Xc
        self.F2  = 1.0 / Yt - 1.0 / Yc
        self.F11 = 1.0 / (Xt * Xc)
        self.F22 = 1.0 / (Yt * Yc)
        self.F66 = 1.0 / (self.S * self.S)
        self.F55 = 1.0 / (self.S13 * self.S13)
        self.F44 = 1.0 / (self.S23 * self.S23)
        self.F12 = (
            float(F12)
            if F12 is not None
            else -0.5 / np.sqrt(Xt * Xc * Yt * Yc)
        )

    def _evaluate_impl(
        self,
        sigma_1: _Array = 0.0,
        sigma_2: _Array = 0.0,
        tau_12:  _Array = 0.0,
        tau_13:  _Array = 0.0,
        tau_23:  _Array = 0.0,
        **_,
    ) -> _Result:
        s1, s2, t12, t13, t23 = _ensure_arrays(sigma_1, sigma_2, tau_12, tau_13, tau_23)

        index = (
            self.F1  * s1
            + self.F2  * s2
            + self.F11 * s1 ** 2
            + self.F22 * s2 ** 2
            + self.F66 * t12 ** 2
            + self.F55 * t13 ** 2
            + self.F44 * t23 ** 2
            + 2 * self.F12 * s1 * s2
        )
        return _result(stress=index, allowable=1.0)


class MaximumStrainCriterion(FailureCriterion):
    """
    Maximum Strain criterion for orthotropic plies.

    Parameters
    ----------
    eps_Xt, eps_Xc : float  Longitudinal ultimate strains (positive).
    eps_Yt, eps_Yc : float  Transverse ultimate strains (positive).
    gamma_S : float         Ultimate in-plane shear strain.
    """
    material_type = _MaterialType.COMPOSITE

    def __init__(
        self,
        eps_Xt: float, eps_Xc: float,
        eps_Yt: float, eps_Yc: float,
        gamma_S: float,
    ):
        if any(v <= 0 for v in [eps_Xt, eps_Xc, eps_Yt, eps_Yc, gamma_S]):
            raise ValueError("All ultimate strains must be positive.")
        self.eps_Xt, self.eps_Xc = float(eps_Xt), float(eps_Xc)
        self.eps_Yt, self.eps_Yc = float(eps_Yt), float(eps_Yc)
        self.gamma_S = float(gamma_S)

    def _evaluate_impl(
        self,
        eps_1:    _Array = 0.0,
        eps_2:    _Array = 0.0,
        gamma_12: _Array = 0.0,
        **_,
    ) -> _Result:
        e1, e2, g12 = _ensure_arrays(eps_1, eps_2, gamma_12)

        index = np.maximum.reduce([
            e1  / self.eps_Xt,
            -e1 / self.eps_Xc,
            e2  / self.eps_Yt,
            -e2 / self.eps_Yc,
            np.abs(g12) / self.gamma_S,
        ])
        return _result(stress=index, allowable=1.0)
