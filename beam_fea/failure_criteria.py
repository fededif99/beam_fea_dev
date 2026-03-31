"""
failure_criteria.py
===================
Unified Failure Criteria for metals and composite laminates.

All criteria accept scalars or NumPy arrays and return a dict containing:
  - 'FI'     : Failure Index (ndarray). FI < 1 → safe; FI >= 1 → failed.
  - 'MoS'    : Margin of Safety = (1 / FI) - 1. MoS >= 0 → safe.
  - 'passed' : Boolean array. True where FI < 1.

Usage
-----
Isotropic (metal)
>>> from beam_fea.failure_criteria import VonMisesCriterion
>>> crit = VonMisesCriterion(yield_strength=250.0)  # MPa
>>> result = crit.evaluate(sigma_x=180.0, sigma_y=50.0, tau_xy=30.0)
>>> result['FI']   # → 0.73  (safe)

Composite ply (material axes)
>>> from beam_fea.failure_criteria import TsaiWuCriterion
>>> crit = TsaiWuCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
>>> result = crit.evaluate(sigma_1=300.0, sigma_2=10.0, tau_12=5.0)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Union

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_Array = Union[float, np.ndarray]
_Result = Dict[str, np.ndarray]


def _fi_to_mos(FI: np.ndarray) -> np.ndarray:
    """Compute Margin of Safety: MoS = (1/FI) - 1. Clipped at -1 to avoid div/0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mos = np.where(FI > 0, 1.0 / FI - 1.0, np.inf)
    return mos


def _result(FI: _Array) -> _Result:
    """Package a Failure Index into the standard result dict."""
    FI = np.asarray(FI, dtype=float)
    return {
        'FI': FI,
        'MoS': _fi_to_mos(FI),
        'passed': FI < 1.0,
    }


# ===========================================================================
# Abstract base
# ===========================================================================

class FailureCriterion(ABC):
    """Abstract base class for all failure criteria."""

    @abstractmethod
    def evaluate(self, **kwargs) -> _Result:
        """
        Evaluate the failure criterion.

        Returns
        -------
        dict with keys 'FI', 'MoS', 'passed'.
        """


# ===========================================================================
# Isotropic / Metal criteria
# ===========================================================================

class VonMisesCriterion(FailureCriterion):
    """
    Von Mises yield criterion (2D plane stress).

    FI = σ_vm / σ_y       where σ_vm = √(σ₁² − σ₁σ₂ + σ₂²)

    Parameters
    ----------
    yield_strength : float
        Material yield stress, σ_y (MPa).
    """

    def __init__(self, yield_strength: float):
        if yield_strength <= 0:
            raise ValueError("yield_strength must be positive.")
        self.yield_strength = float(yield_strength)

    def evaluate(
        self,
        sigma_x: _Array = 0.0,
        sigma_y: _Array = 0.0,
        tau_xy: _Array = 0.0,
        **_,
    ) -> _Result:
        """
        Parameters
        ----------
        sigma_x, sigma_y : float or ndarray
            Normal stresses in the element x/y axes (MPa).
        tau_xy : float or ndarray
            In-plane shear stress (MPa).
        """
        sigma_x = np.asarray(sigma_x, dtype=float)
        sigma_y = np.asarray(sigma_y, dtype=float)
        tau_xy  = np.asarray(tau_xy,  dtype=float)

        # Principal stresses (2D)
        avg  = (sigma_x + sigma_y) / 2
        R    = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        s1, s2 = avg + R, avg - R

        sigma_vm = np.sqrt(s1 ** 2 - s1 * s2 + s2 ** 2)
        return _result(sigma_vm / self.yield_strength)


class TrescaCriterion(FailureCriterion):
    """
    Tresca (maximum-shear-stress) yield criterion (2D plane stress).

    FI = (σ₁ − σ₂) / σ_y

    Parameters
    ----------
    yield_strength : float
        Material yield stress, σ_y (MPa).
    """

    def __init__(self, yield_strength: float):
        if yield_strength <= 0:
            raise ValueError("yield_strength must be positive.")
        self.yield_strength = float(yield_strength)

    def evaluate(
        self,
        sigma_x: _Array = 0.0,
        sigma_y: _Array = 0.0,
        tau_xy: _Array = 0.0,
        **_,
    ) -> _Result:
        sigma_x = np.asarray(sigma_x, dtype=float)
        sigma_y = np.asarray(sigma_y, dtype=float)
        tau_xy  = np.asarray(tau_xy,  dtype=float)

        avg  = (sigma_x + sigma_y) / 2
        R    = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        s1, s2 = avg + R, avg - R

        return _result((s1 - s2) / self.yield_strength)


class MaxPrincipalStressCriterion(FailureCriterion):
    """
    Maximum Principal Stress criterion (brittle fracture).

    FI = max(σ₁/Ftu, −σ₂/Fcu)  using plane-stress principal stresses.

    Parameters
    ----------
    Ftu : float
        Tensile ultimate strength (MPa).
    Fcu : float, optional
        Compressive ultimate strength (MPa, positive value).
        Defaults to Ftu if not specified.
    """

    def __init__(self, Ftu: float, Fcu: float | None = None):
        if Ftu <= 0:
            raise ValueError("Ftu must be positive.")
        self.Ftu = float(Ftu)
        self.Fcu = float(Fcu) if Fcu is not None else self.Ftu

    def evaluate(
        self,
        sigma_x: _Array = 0.0,
        sigma_y: _Array = 0.0,
        tau_xy: _Array = 0.0,
        **_,
    ) -> _Result:
        sigma_x = np.asarray(sigma_x, dtype=float)
        sigma_y = np.asarray(sigma_y, dtype=float)
        tau_xy  = np.asarray(tau_xy,  dtype=float)

        avg  = (sigma_x + sigma_y) / 2
        R    = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        s1, s2 = avg + R, avg - R

        FI = np.maximum(s1 / self.Ftu, -s2 / self.Fcu)
        return _result(FI)


# ===========================================================================
# Composite ply-level criteria (stresses in material / ply axes)
# ===========================================================================

class MaximumStressCriterion(FailureCriterion):
    """
    Maximum Stress criterion for orthotropic plies (material axes).

    FI = max(σ₁ / Xt, −σ₁ / Xc, σ₂ / Yt, −σ₂ / Yc, |τ₁₂| / S)

    Parameters
    ----------
    Xt, Xc : float
        Longitudinal tensile / compressive strength (MPa, both positive).
    Yt, Yc : float
        Transverse tensile / compressive strength (MPa, both positive).
    S : float
        In-plane shear strength (MPa).
    """

    def __init__(self, Xt: float, Xc: float, Yt: float, Yc: float, S: float):
        self.Xt, self.Xc = float(Xt), float(Xc)
        self.Yt, self.Yc = float(Yt), float(Yc)
        self.S = float(S)

    def evaluate(
        self,
        sigma_1: _Array = 0.0,
        sigma_2: _Array = 0.0,
        tau_12:  _Array = 0.0,
        **_,
    ) -> _Result:
        """
        Parameters
        ----------
        sigma_1, sigma_2 : float or ndarray
            Normal stresses in ply material axes (MPa).
        tau_12 : float or ndarray
            In-plane shear stress in ply material axes (MPa).
        """
        s1 = np.asarray(sigma_1, dtype=float)
        s2 = np.asarray(sigma_2, dtype=float)
        t  = np.asarray(tau_12,  dtype=float)

        # Broadcast all to common shape before stacking
        s1, s2, t = np.broadcast_arrays(s1, s2, t)

        terms = np.stack([
            s1 / self.Xt,
            -s1 / self.Xc,
            s2 / self.Yt,
            -s2 / self.Yc,
            np.abs(t) / self.S,
        ], axis=0)
        FI = np.max(terms, axis=0)
        return _result(FI)


class TsaiHillCriterion(FailureCriterion):
    """
    Tsai-Hill criterion for orthotropic plies.

    FI = (σ₁/X)² − (σ₁·σ₂/X²) + (σ₂/Y)² + (τ₁₂/S)²

    where X = Xt when σ₁ ≥ 0, else Xc; Y = Yt when σ₂ ≥ 0, else Yc.

    Parameters
    ----------
    Xt, Xc : float  Longitudinal tensile / compressive strength (MPa).
    Yt, Yc : float  Transverse tensile / compressive strength (MPa).
    S : float       In-plane shear strength (MPa).
    """

    def __init__(self, Xt: float, Xc: float, Yt: float, Yc: float, S: float):
        self.Xt, self.Xc = float(Xt), float(Xc)
        self.Yt, self.Yc = float(Yt), float(Yc)
        self.S = float(S)

    def evaluate(
        self,
        sigma_1: _Array = 0.0,
        sigma_2: _Array = 0.0,
        tau_12:  _Array = 0.0,
        **_,
    ) -> _Result:
        s1 = np.asarray(sigma_1, dtype=float)
        s2 = np.asarray(sigma_2, dtype=float)
        t  = np.asarray(tau_12,  dtype=float)

        # Use tensile or compressive strength based on sign
        X = np.where(s1 >= 0, self.Xt, self.Xc)
        Y = np.where(s2 >= 0, self.Yt, self.Yc)

        FI = (s1 / X) ** 2 - (s1 * s2 / X ** 2) + (s2 / Y) ** 2 + (t / self.S) ** 2
        return _result(FI)


class TsaiWuCriterion(FailureCriterion):
    """
    Tsai-Wu tensor polynomial criterion for orthotropic plies.

        F₁·σ₁ + F₂·σ₂ + F₁₁·σ₁² + F₂₂·σ₂² + F₆₆·τ₁₂² + 2F₁₂·σ₁·σ₂ = 1

    The left-hand side is the Failure Index (FI).

    Parameters
    ----------
    Xt, Xc : float  Longitudinal tensile / compressive ultimate strengths (MPa).
    Yt, Yc : float  Transverse tensile / compressive ultimate strengths (MPa).
    S : float       In-plane shear strength (MPa).
    F12 : float, optional
        Interaction coefficient. Default = −0.5/√(Xt·Xc·Yt·Yc), the
        Tsai-Hahn estimate (conservative, commonly used industry value).
    """

    def __init__(
        self,
        Xt: float, Xc: float,
        Yt: float, Yc: float,
        S: float,
        F12: float | None = None,
    ):
        self.Xt, self.Xc = float(Xt), float(Xc)
        self.Yt, self.Yc = float(Yt), float(Yc)
        self.S = float(S)

        # Strength tensors
        self.F1  = 1.0 / Xt - 1.0 / Xc
        self.F2  = 1.0 / Yt - 1.0 / Yc
        self.F11 = 1.0 / (Xt * Xc)
        self.F22 = 1.0 / (Yt * Yc)
        self.F66 = 1.0 / (S * S)
        self.F12 = (
            float(F12)
            if F12 is not None
            else -0.5 / np.sqrt(Xt * Xc * Yt * Yc)
        )

    def evaluate(
        self,
        sigma_1: _Array = 0.0,
        sigma_2: _Array = 0.0,
        tau_12:  _Array = 0.0,
        **_,
    ) -> _Result:
        s1 = np.asarray(sigma_1, dtype=float)
        s2 = np.asarray(sigma_2, dtype=float)
        t  = np.asarray(tau_12,  dtype=float)

        FI = (
            self.F1  * s1
            + self.F2  * s2
            + self.F11 * s1 ** 2
            + self.F22 * s2 ** 2
            + self.F66 * t  ** 2
            + 2 * self.F12 * s1 * s2
        )
        return _result(FI)


class MaximumStrainCriterion(FailureCriterion):
    """
    Maximum Strain criterion for orthotropic plies.

    FI = max(ε₁/εXt, −ε₁/εXc, ε₂/εYt, −ε₂/εYc, |γ₁₂|/γS)

    Parameters
    ----------
    eps_Xt, eps_Xc : float  Longitudinal ultimate strains (tensile/compressive, both positive).
    eps_Yt, eps_Yc : float  Transverse ultimate strains (tensile/compressive, both positive).
    gamma_S : float          Ultimate in-plane shear strain.
    """

    def __init__(
        self,
        eps_Xt: float, eps_Xc: float,
        eps_Yt: float, eps_Yc: float,
        gamma_S: float,
    ):
        self.eps_Xt, self.eps_Xc = float(eps_Xt), float(eps_Xc)
        self.eps_Yt, self.eps_Yc = float(eps_Yt), float(eps_Yc)
        self.gamma_S = float(gamma_S)

    def evaluate(
        self,
        eps_1:   _Array = 0.0,
        eps_2:   _Array = 0.0,
        gamma_12: _Array = 0.0,
        **_,
    ) -> _Result:
        e1  = np.asarray(eps_1,    dtype=float)
        e2  = np.asarray(eps_2,    dtype=float)
        g12 = np.asarray(gamma_12, dtype=float)

        FI = np.maximum.reduce([
            e1  / self.eps_Xt,
            -e1 / self.eps_Xc,
            e2  / self.eps_Yt,
            -e2 / self.eps_Yc,
            np.abs(g12) / self.gamma_S,
        ])
        return _result(FI)
