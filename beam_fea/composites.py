"""
composites.py
=============
Classical Laminate Theory (CLT) for composite materials.

Provides tools to calculate ABD matrices from ply properties and stack-up.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .materials import Material


@dataclass(frozen=True)
class Ply:
    """
    A single orthotropic ply (lamina).
    ... (Docstring omitted for brevity in replace, but keeping it in real file)
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
    S13: float = 0.0
    S23: float = 0.0
    name: str = "Generic Ply"

    def reduced_stiffness_matrix(self) -> np.ndarray:
        """Calculate the reduced stiffness matrix [Q] for plane stress."""
        nu21 = self.nu12 * self.E2 / self.E1
        denom = 1 - self.nu12 * nu21
        return np.array([
            [self.E1 / denom, self.nu12 * self.E2 / denom, 0],
            [self.nu12 * self.E2 / denom, self.E2 / denom, 0],
            [0, 0, self.G12]
        ])

    def transformed_reduced_stiffness(self, angle_deg: float) -> np.ndarray:
        """Calculate the transformed reduced stiffness matrix [Qbar]."""
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        c2, s2, cs = c**2, s**2, s*c

        Q = self.reduced_stiffness_matrix()
        Q11, Q12, Q22, Q66 = Q[0,0], Q[0,1], Q[1,1], Q[2,2]

        return np.array([
            [Q11*c**4 + 2*(Q12 + 2*Q66)*s2*c2 + Q22*s**4,
             (Q11 + Q22 - 4*Q66)*s2*c2 + Q12*(s**4 + c**4),
             (Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c],
            [(Q11 + Q22 - 4*Q66)*s2*c2 + Q12*(s**4 + c**4),
             Q11*s**4 + 2*(Q12 + 2*Q66)*s2*c2 + Q22*c**4,
             (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3],
            [(Q11 - Q12 - 2*Q66)*s*c**3 + (Q12 - Q22 + 2*Q66)*s**3*c,
             (Q11 - Q12 - 2*Q66)*s**3*c + (Q12 - Q22 + 2*Q66)*s*c**3,
             (Q11 + Q22 - 2*Q12 - 2*Q66)*s2*c2 + Q66*(s**4 + c**4)]
        ])

    def calculate_safety_factor(self, sigma_local: np.ndarray, criterion: str = 'max_stress') -> float:
        """Calculate the safety factor for the ply given local stresses."""
        s1, s2, s12 = sigma_local[0], sigma_local[1], sigma_local[2]

        # Determine normalized failure intensity first
        fi = 0.0
        if criterion == 'max_stress':
            if any(v <= 0 for v in [self.Xt, self.Xc, self.Yt, self.Yc, self.S]): fi = 0.0
            else:
                f1 = s1 / self.Xt if s1 >= 0 else abs(s1) / self.Xc
                f2 = s2 / self.Yt if s2 >= 0 else abs(s2) / self.Yc
                f12 = abs(s12) / self.S
                fi = max(f1, f2, f12)
        elif criterion == 'tsai_hill':
            X = self.Xt if s1 >= 0 else self.Xc
            Y = self.Yt if s2 >= 0 else self.Yc
            if any(v <= 0 for v in [X, Y, self.S]): fi = 0.0
            else:
                fi = (s1/X)**2 - (s1*s2)/(X**2) + (s2/Y)**2 + (s12/self.S)**2
        elif criterion == 'tsai_wu':
            if any(v <= 0 for v in [self.Xt, self.Xc, self.Yt, self.Yc, self.S]): fi = 0.0
            else:
                F1, F11 = 1/self.Xt - 1/self.Xc, 1/(self.Xt * self.Xc)
                F2, F22 = 1/self.Yt - 1/self.Yc, 1/(self.Yt * self.Yc)
                F66, F12 = 1/(self.S**2), -0.5 * np.sqrt(1/(self.Xt * self.Xc * self.Yt * self.Yc))
                fi = F1*s1 + F11*s1**2 + F2*s2 + F22*s2**2 + F66*s12**2 + 2*F12*s1*s2

        # Safety Factor is inversely proportional to failure intensity
        return 1.0 / fi if fi > 0 else np.inf


class Laminate:
    """
    An immutable laminate stack-up consisting of multiple plies.
    """

    def __init__(self, name: str = "Laminate", beam_type: str = 'narrow', 
                 stack: Optional[List[Tuple[Ply, Union[float, List[float]]]]] = None):
        """
        Initialize an immutable laminate.

        Parameters:
        -----------
        name : str
            Identifier
        beam_type : str
            'narrow' (sigma_y=0) or 'wide' (epsilon_y=0)
        stack : list of (Ply, angle(s))
            The stacking sequence definition.
        """
        self.name = name
        self.beam_type = beam_type
        
        # Flatten the stack into a list of (Ply, float)
        self.plies: List[Tuple[Ply, float]] = []
        if stack:
            for ply, angles in stack:
                if isinstance(angles, (int, float)):
                    self.plies.append((ply, float(angles)))
                else:
                    for angle in angles:
                        self.plies.append((ply, float(angle)))

        # Pre-calculated properties
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.ABD = np.zeros((6, 6))
        self.A_shear = np.zeros((2, 2))
        self.total_thickness = 0.0
        self._rho_avg = 0.0
        
        if self.plies:
            self._calculate_properties()

    @classmethod
    def from_single_material(cls, name: str, ply: Ply, angles: List[float], beam_type: str = 'narrow') -> 'Laminate':
        """
        Convenience factory to create a laminate from a single material.

        Parameters:
        -----------
        name : str
            Identifier
        ply : Ply
            The material for all plies
        angles : list of float
            The stacking sequence
        beam_type : str
            'narrow' or 'wide'
        """
        return cls(name=name, beam_type=beam_type, stack=[(ply, angles)])

    def _calculate_properties(self):
        """Vectorized calculation of ABD and transverse shear matrices."""
        n = len(self.plies)
        thicknesses = np.array([p.thickness for p, a in self.plies])
        angles = np.array([a for p, a in self.plies])
        rhos = np.array([p.rho for p, a in self.plies])
        
        self.total_thickness = np.sum(thicknesses)
        self._rho_avg = np.sum(rhos * thicknesses) / self.total_thickness if self.total_thickness > 0 else 0
        
        z = np.zeros(n + 1)
        z[0] = -self.total_thickness / 2.0
        z[1:] = z[0] + np.cumsum(thicknesses)
        
        dz = z[1:] - z[:-1]
        dz2 = z[1:]**2 - z[:-1]**2
        dz3 = z[1:]**3 - z[:-1]**3

        # Vectorized Qbar construction
        theta = np.radians(angles)
        c, s = np.cos(theta), np.sin(theta)
        c2, s2, c4, s4 = c**2, s**2, c**4, s**4
        
        # Reduced stiffness components for all plies
        E1s = np.array([p.E1 for p, a in self.plies])
        E2s = np.array([p.E2 for p, a in self.plies])
        nu12s = np.array([p.nu12 for p, a in self.plies])
        G12s = np.array([p.G12 for p, a in self.plies])
        
        nu21s = nu12s * E2s / E1s
        denoms = 1 - nu12s * nu21s
        Q11s = E1s / denoms
        Q22s = E2s / denoms
        Q12s = nu12s * E2s / denoms
        Q66s = G12s
        
        # Transformed reduced stiffness (Qbar) components
        Qb11 = Q11s*c4 + 2*(Q12s + 2*Q66s)*s2*c2 + Q22s*s4
        Qb12 = (Q11s + Q22s - 4*Q66s)*s2*c2 + Q12s*(s4 + c4)
        Qb22 = Q11s*s4 + 2*(Q12s + 2*Q66s)*s2*c2 + Q22s*c4
        Qb16 = (Q11s - Q12s - 2*Q66s)*s*c**3 + (Q12s - Q22s + 2*Q66s)*s**3*c
        Qb26 = (Q11s - Q12s - 2*Q66s)*s**3*c + (Q12s - Q22s + 2*Q66s)*s*c**3
        Qb66 = (Q11s + Q22s - 2*Q12s - 2*Q66s)*s2*c2 + Q66s*(s4 + c4)
        
        # A, B, D as sums
        self.A = np.array([
            [np.sum(Qb11 * dz), np.sum(Qb12 * dz), np.sum(Qb16 * dz)],
            [np.sum(Qb12 * dz), np.sum(Qb22 * dz), np.sum(Qb26 * dz)],
            [np.sum(Qb16 * dz), np.sum(Qb26 * dz), np.sum(Qb66 * dz)]
        ])
        self.B = 0.5 * np.array([
            [np.sum(Qb11 * dz2), np.sum(Qb12 * dz2), np.sum(Qb16 * dz2)],
            [np.sum(Qb12 * dz2), np.sum(Qb22 * dz2), np.sum(Qb26 * dz2)],
            [np.sum(Qb16 * dz2), np.sum(Qb26 * dz2), np.sum(Qb66 * dz2)]
        ])
        # Note: 1/3 * summation
        val_dz3 = (1/3.0) * dz3
        self.D = np.array([
            [np.sum(Qb11 * val_dz3), np.sum(Qb12 * val_dz3), np.sum(Qb16 * val_dz3)],
            [np.sum(Qb12 * val_dz3), np.sum(Qb22 * val_dz3), np.sum(Qb26 * val_dz3)],
            [np.sum(Qb16 * val_dz3), np.sum(Qb26 * val_dz3), np.sum(Qb66 * val_dz3)]
        ])

        self.ABD = np.block([[self.A, self.B], [self.B, self.D]])

        # Vectorized Transverse Shear
        G13s = np.array([p.G13 for p, a in self.plies])
        G23s = np.array([p.G23 for p, a in self.plies])
        Qb44 = G23s*c2 + G13s*s2
        Qb45 = (G13s - G23s)*s*c
        Qb55 = G23s*s2 + G13s*c2
        
        self.A_shear = np.array([
            [np.sum(Qb44 * dz), np.sum(Qb45 * dz)],
            [np.sum(Qb45 * dz), np.sum(Qb55 * dz)]
        ])

    @property
    def rho(self) -> float:
        """Average density of the laminate."""
        return self._rho_avg

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
        # Standardized robust width extraction using bounding box
        width = section.z_right - section.z_left

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
        width = section.z_right - section.z_left
        return self.rho * width * self.total_thickness

    def __str__(self):
        res = f"Laminate: {self.name}\n"
        res += f"  Total Thickness: {self.total_thickness:.3f} mm\n"
        res += f"  Stack-up: {[a for p, a in self.plies]}\n"
        res += f"  A11: {self.A[0,0]:.2e}, D11: {self.D[0,0]:.2e}"
        return res
