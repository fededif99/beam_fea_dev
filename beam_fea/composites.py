"""
composites.py
=============
Classical Laminate Theory (CLT) for composite materials.

Provides tools to calculate ABD matrices from ply properties and stack-up.
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .materials import Material


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
    G13 : float
        Transverse shear modulus (1-3 plane, MPa)
    G23 : float
        Transverse shear modulus (2-3 plane, MPa)
    thickness : float
        Ply thickness (mm)
    rho : float
        Density (kg/mm³)
    Xt : float
        Longitudinal tensile strength (MPa)
    Xc : float
        Longitudinal compressive strength (MPa)
    Yt : float
        Transverse tensile strength (MPa)
    Yc : float
        Transverse compressive strength (MPa)
    S : float
        In-plane shear strength (MPa)
    name : str
        Ply identifier
    """
    E1: float
    E2: float
    nu12: float
    G12: float
    thickness: float
    G13: float = 0.0
    G23: float = 0.0
    rho: float = 0.0
    Xt: float = 0.0
    Xc: float = 0.0
    Yt: float = 0.0
    Yc: float = 0.0
    S: float = 0.0
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

    def calculate_failure_index(self, sigma_local: np.ndarray, criterion: str = 'max_stress') -> float:
        """
        Calculate the failure index for the ply given local stresses.

        Parameters:
        -----------
        sigma_local : np.ndarray
            Stresses in material coordinates [sigma_1, sigma_2, tau_12].
        criterion : str
            Failure criterion: 'max_stress', 'tsai_hill', or 'tsai_wu'.

        Returns:
        --------
        failure_index : float
            Index > 1.0 indicates predicted failure.
        """
        s1, s2, s12 = sigma_local[0], sigma_local[1], sigma_local[2]

        if criterion == 'max_stress':
            if any(v <= 0 for v in [self.Xt, self.Xc, self.Yt, self.Yc, self.S]): return 0.0
            f1 = s1 / self.Xt if s1 >= 0 else abs(s1) / self.Xc
            f2 = s2 / self.Yt if s2 >= 0 else abs(s2) / self.Yc
            f12 = abs(s12) / self.S
            return max(f1, f2, f12)

        elif criterion == 'tsai_hill':
            X = self.Xt if s1 >= 0 else self.Xc
            Y = self.Yt if s2 >= 0 else self.Yc
            if any(v <= 0 for v in [X, Y, self.S]): return 0.0
            return (s1/X)**2 - (s1*s2)/(X**2) + (s2/Y)**2 + (s12/self.S)**2

        elif criterion == 'tsai_wu':
            if any(v <= 0 for v in [self.Xt, self.Xc, self.Yt, self.Yc, self.S]): return 0.0
            F1 = 1/self.Xt - 1/self.Xc
            F11 = 1/(self.Xt * self.Xc)
            F2 = 1/self.Yt - 1/self.Yc
            F22 = 1/(self.Yt * self.Yc)
            F66 = 1/(self.S**2)
            # F12 is typically assumed as -0.5 * sqrt(F11 * F22)
            F12 = -0.5 * np.sqrt(F11 * F22)

            return F1*s1 + F11*s1**2 + F2*s2 + F22*s2**2 + F66*s12**2 + 2*F12*s1*s2

        return 0.0


class Laminate:
    """
    A laminate stack-up consisting of multiple plies.
    """

    def __init__(self, name: str = "Laminate", beam_type: str = 'narrow'):
        """
        Initialize laminate.

        Parameters:
        -----------
        name : str
            Identifier
        beam_type : str
            'narrow' (assumes sigma_y=0, uses ABD inverse for stiffness)
            'wide' (assumes epsilon_y=0, uses A11/D11 directly)
        """
        self.name = name
        self.beam_type = beam_type
        self.plies: List[Tuple[Ply, float]] = [] # (Ply object, angle in degrees)
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.ABD = np.zeros((6, 6))
        # Transverse shear stiffness (A44, A45, A55)
        self.A_shear = np.zeros((2, 2))
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
        """Calculate ABD and transverse shear matrices using CLT."""
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

        # Transverse shear stiffness A_shear (A44, A45, A55)
        # Integrated over thickness: A_ij = sum(Qbar_trans_ij * thickness)
        self.A_shear = np.zeros((2, 2))
        for i in range(n):
            ply, angle = self.plies[i]
            theta = np.radians(angle)
            c, s = np.cos(theta), np.sin(theta)

            # Transverse shear stiffness in material axes
            Q44, Q55 = ply.G23, ply.G13
            # Transformed to laminate axes
            Qb44 = Q44*c**2 + Q55*s**2
            Qb45 = (Q55 - Q44)*s*c
            Qb55 = Q44*s**2 + Q55*c**2

            Qbar_trans = np.array([[Qb44, Qb45], [Qb45, Qb55]])
            self.A_shear += Qbar_trans * ply.thickness

    @property
    def rho(self) -> float:
        """Average density of the laminate."""
        if not self.plies: return 0.0
        t = sum(p.thickness for p, a in self.plies)
        if t == 0: return 0.0
        return sum(p.rho * p.thickness for p, a in self.plies) / t

    def get_effective_properties(self) -> dict:
        """
        Calculate equivalent engineering properties for 1D beam analysis.

        Choice of formula depends on `beam_type`:
        - 'narrow' (sigma_y=0): Uses the inverse of ABD (compliance).
        - 'wide' (epsilon_y=0): Uses the stiffness terms A11, D11 directly.

        For narrow beams (Reddy, 2003):
          Ex_eff = 1 / (a11 * t)      where [a] = [A]^{-1}
          Eb_eff = 12 / (d11 * t^3)    where [d] = [ABD]^{-1}[3:6, 3:6]
        """
        if self.total_thickness == 0:
            return {}

        t = self.total_thickness

        # Compliance matrix [a] = [A]^{-1}
        # This is the correct general approach for all laminate types.
        try:
            A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(self.A)

        Ex   = 1.0 / (A_inv[0, 0] * t)
        Ey   = 1.0 / (A_inv[1, 1] * t)
        Gxy  = 1.0 / (A_inv[2, 2] * t)
        nu_xy = -A_inv[0, 1] / A_inv[0, 0]

        # Equivalent Bending Modulus (Eb) via full ABD inverse
        # For asymmetric laminates, using D[0,0] only would be wrong because
        # the B matrix couples bending and extension, modifying effective Eb.
        if self.beam_type == 'narrow':
            try:
                ABD_inv = np.linalg.inv(self.ABD)
                a11 = np.linalg.inv(self.A)[0, 0] # for Ex
                d11 = ABD_inv[3, 3] # for Eb
                Ex = 1.0 / (a11 * t)
                Eb = 12.0 / (d11 * t**3)
            except np.linalg.LinAlgError:
                Ex = self.A[0, 0] / t
                Eb = 12.0 * self.D[0, 0] / t**3
        else:
            # Wide beam assumption (Plate theory)
            Ex = self.A[0, 0] / t
            Eb = 12.0 * self.D[0, 0] / t**3

        # Average density (volume-weighted)
        avg_rho = sum(p.rho * p.thickness for p, a in self.plies) / t if t > 0 else 0

        return {
            'Ex': Ex,
            'Ey': Ey,
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

    def get_sectional_stiffness(self, section) -> dict:
        """
        Calculate width-integrated sectional stiffness values for the
        anisotropic 1D beam element (Jones 1999; Reddy 2003).

        Theory
        ------
        **Narrow beam** (free transverse edges → N_y = 0):
        Effective stiffnesses are derived from the full 6×6 ABD compliance
        [a, b; c, d] = [A B; B^T D]^{-1}, NOT from A/B/D directly.
        This correctly captures how the stress-free lateral edge modifies coupling:

          EA_eff = width / a11     where a11 = ABD_inv[0, 0]  (axial compliance)
          EI_eff = width / d11     where d11 = ABD_inv[3, 3]  (bending compliance)
          ES_eff = -a13 * width    where a13 = ABD_inv[0, 3]  (coupling compliance)

        For symmetric laminates ([B] = 0) a13 = 0 → ES_eff = 0 as expected.
        Using B[0,0] directly overestimates coupling for asymmetric narrow beams
        because it ignores the free-edge constraint on N_y (sigma_y = 0).

        **Wide beam** (constrained transverse edges → ε_y = 0, plate-strip):
        Stiffness terms are used directly:

          EA_eff = A11 * width
          ES_eff = B11 * width
          EI_eff = D11 * width

        References
        ----------
        - Jones, R.M. (1999). Mechanics of Composite Materials, 2nd ed. §4.5
        - Reddy, J.N. (2004). Mechanics of Laminated Composite Plates, §4.2
        """
        width = getattr(section, 'width', getattr(section, 'diameter', 1.0))

        if self.beam_type == 'narrow':
            try:
                ABD_inv = np.linalg.inv(self.ABD)
                a11 = ABD_inv[0, 0]   # axial compliance
                d11 = ABD_inv[3, 3]   # bending compliance
                a13 = ABD_inv[0, 3]   # extension-bending coupling compliance
                if abs(a11) < 1e-30 or abs(d11) < 1e-30:
                    raise np.linalg.LinAlgError("Near-singular compliance")
                EA = width / a11
                EI = width / d11
                # a13 < 0 for laminates where tension induces hogging (positive curvature)
                ES = -a13 * width
            except np.linalg.LinAlgError:
                # Fallback for degenerate laminates
                EA = self.A[0, 0] * width
                EI = self.D[0, 0] * width
                ES = self.B[0, 0] * width
        else:
            # Wide beam: use ABD stiffness terms directly
            EA = self.A[0, 0] * width
            ES = self.B[0, 0] * width
            EI = self.D[0, 0] * width

        # Transverse Shear Stiffness (out-of-plane shear, A55 component)
        GA_s = self.A_shear[1, 1] * width

        return {'EA': EA, 'ES': ES, 'EI': EI, 'GA_s': GA_s}

    def get_linear_density(self, section) -> float:
        """Calculate mass per unit length for the laminate."""
        width = getattr(section, 'width', getattr(section, 'diameter', 1.0))
        return self.rho * width * self.total_thickness

    def __str__(self):
        res = f"Laminate: {self.name}\n"
        res += f"  Total Thickness: {self.total_thickness:.3f} mm\n"
        res += f"  Stack-up: {[a for p, a in self.plies]}\n"
        res += f"  A11: {self.A[0,0]:.2e}, D11: {self.D[0,0]:.2e}"
        return res
