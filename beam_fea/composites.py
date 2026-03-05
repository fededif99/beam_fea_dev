"""
composites.py
=============
Classical Laminate Theory (CLT) for composite materials.

Provides tools to calculate ABD matrices from ply properties and stack-up.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Ply:
    """
    A single orthotropic ply (lamina).

    Attributes:
    -----------
    E1 : float
        Young's modulus in fiber direction (MPa)
    E2 : float
        Young's modulus in transverse direction (MPa)
    nu12 : float
        Major Poisson's ratio
    G12 : float
        In-plane shear modulus (MPa)
    thickness : float
        Ply thickness (mm)
    rho : float
        Density (kg/mm³)
    name : str
        Ply identifier
    """
    E1: float
    E2: float
    nu12: float
    G12: float
    thickness: float
    rho: float = 0.0
    name: str = "Generic Ply"

    def reduced_stiffness_matrix(self) -> np.ndarray:
        """
        Calculate the reduced stiffness matrix [Q] for plane stress.

        Q = [[Q11, Q12, 0],
             [Q12, Q22, 0],
             [0,   0,   Q66]]
        """
        nu21 = self.nu12 * self.E2 / self.E1
        denom = 1 - self.nu12 * nu21

        Q11 = self.E1 / denom
        Q22 = self.E2 / denom
        Q12 = self.nu12 * self.E2 / denom
        Q66 = self.G12

        return np.array([
            [Q11, Q12, 0],
            [Q12, Q22, 0],
            [0,   0,   Q66]
        ])

    def transformed_reduced_stiffness(self, angle_deg: float) -> np.ndarray:
        """
        Calculate the transformed reduced stiffness matrix [Qbar] for a given angle.

        Parameters:
        -----------
        angle_deg : float
            Angle in degrees relative to laminate x-axis.
        """
        theta = np.radians(angle_deg)
        c = np.cos(theta)
        s = np.sin(theta)

        Q = self.reduced_stiffness_matrix()
        Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]

        Qb11 = Q11*c**4 + 2*(Q12 + 2*Q66)*s**2*c**2 + Q22*s**4
        Qb12 = (Q11 + Q22 - 4*Q66)*s**2*c**2 + Q12*(s**4 + c**4)
        Qb22 = Q11*s**4 + 2*(Q12 + 2*Q66)*s**2*c**2 + Q22*c**4
        Qb16 = (Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c
        Qb26 = (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3
        Qb66 = (Q11 + Q22 - 2*Q12 - 2*Q66)*s**2*c**2 + Q66*(s**4 + c**4)

        return np.array([
            [Qb11, Qb12, Qb16],
            [Qb12, Qb22, Qb26],
            [Qb16, Qb26, Qb66]
        ])


class Laminate:
    """
    A laminate stack-up consisting of multiple plies.
    """

    def __init__(self, name: str = "Laminate"):
        self.name = name
        self.plies: List[Tuple[Ply, float]] = [] # (Ply object, angle in degrees)
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.ABD = np.zeros((6, 6))
        self.total_thickness = 0.0

    def add_ply(self, ply: Ply, angle: float):
        """Add a ply to the stack (from bottom up)."""
        self.plies.append((ply, angle))
        self._calculate_properties()

    def add_stack(self, ply: Ply, angles: List[float]):
        """Add multiple plies of the same type."""
        for angle in angles:
            self.plies.append((ply, angle))
        self._calculate_properties()

    def _calculate_properties(self):
        """Calculate ABD matrices using Classical Laminate Theory."""
        if not self.plies:
            return

        n = len(self.plies)
        self.total_thickness = sum(p.thickness for p, a in self.plies)

        # z coordinates from bottom to top, centered at mid-plane
        z = np.zeros(n + 1)
        z[0] = -self.total_thickness / 2.0
        for i in range(n):
            z[i+1] = z[i] + self.plies[i][0].thickness

        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))

        for i in range(n):
            ply, angle = self.plies[i]
            Qbar = ply.transformed_reduced_stiffness(angle)

            # A_ij = sum(Qbar_ij * (z_k - z_k-1))
            self.A += Qbar * (z[i+1] - z[i])
            # B_ij = 1/2 * sum(Qbar_ij * (z_k^2 - z_k-1^2))
            self.B += 0.5 * Qbar * (z[i+1]**2 - z[i]**2)
            # D_ij = 1/3 * sum(Qbar_ij * (z_k^3 - z_k-1^3))
            self.D += (1/3.0) * Qbar * (z[i+1]**3 - z[i]**3)

        # Assemble ABD matrix
        self.ABD = np.block([
            [self.A, self.B],
            [self.B, self.D]
        ])

    def get_effective_properties(self) -> dict:
        """
        Calculate equivalent isotropic properties for 1D beam analysis.

        Ex = (A11*A22 - A12^2) / (A22 * t)
        Eb = 12 / (D_inv[0,0] * t^3)
        """
        if self.total_thickness == 0:
            return {}

        t = self.total_thickness

        # Equivalent Axial Modulus (Ex)
        # Using the A matrix (in-plane)
        Ex = (self.A[0,0]*self.A[1,1] - self.A[0,1]**2) / (self.A[1,1] * t)

        # Equivalent Bending Modulus (Eb)
        # We need the inverse of the ABD matrix to get accurate bending properties
        # for asymmetric laminates, but for symmetric D_inv[0,0] is 1/D[0,0]
        try:
            ABD_inv = np.linalg.inv(self.ABD)
            d11 = ABD_inv[3, 3]
            Eb = 12.0 / (d11 * t**3)
        except np.linalg.LinAlgError:
            # Fallback for simple D-only approximation if inversion fails
            Eb = 12.0 * self.D[0,0] / t**3

        # Equivalent Shear Modulus (Gxy)
        Gxy = self.A[2,2] / t

        # Average Density
        avg_rho = sum(p.rho * p.thickness for p, a in self.plies) / t if t > 0 else 0

        # Effective Poisson's Ratio
        nu_xy = self.A[0,1] / self.A[1,1] if self.A[1,1] != 0 else 0

        return {
            'Ex': Ex,
            'Eb': Eb,
            'Gxy': Gxy,
            'nu_xy': nu_xy,
            'rho': avg_rho,
            'thickness': t
        }

    def to_material(self, preference: str = 'axial') -> 'Material':
        """
        Convert effective properties to a Material object for BeamSolver.

        Parameters:
        -----------
        preference : str
            'axial' (uses Ex) or 'bending' (uses Eb).
            Since 1D beam elements typically use one E, choose based on dominant load.
        """
        from .materials import Material
        props = self.get_effective_properties()

        E_eff = props['Ex'] if preference == 'axial' else props['Eb']

        return Material(
            name=f"Composite_{self.name}_{preference}",
            E=E_eff,
            G=props['Gxy'],
            nu=props['nu_xy'],
            rho=props['rho']
        )

    def __str__(self):
        res = f"Laminate: {self.name}\n"
        res += f"  Total Thickness: {self.total_thickness:.3f} mm\n"
        res += f"  Stack-up: {[a for p, a in self.plies]}\n"
        res += f"  A11: {self.A[0,0]:.2e}, D11: {self.D[0,0]:.2e}"
        return res
