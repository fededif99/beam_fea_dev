"""
element_matrices.py
===================
Local element matrix calculations for different beam theories.

Provides stiffness and mass matrices for:
- Euler-Bernoulli beam elements
- Timoshenko beam elements
"""

import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod



class BeamElementMatrices(ABC):
    """Abstract base class for beam element matrices."""
    
    def __init__(self, E: float, G: float, I: float, A: float, L: float, rho: float):
        """
        Initialize beam element properties.
        
        Parameters:
        -----------
        E : float
            Young's modulus (MPa)
        G : float
            Shear modulus (MPa)
        I : float
            Second moment of area (mm⁴)
        A : float
            Cross-sectional area (mm²)
        L : float
            Element length (mm)
        rho : float
            Material density (kg/mm³)
        """
        self.E = E
        self.G = G
        self.I = I
        self.A = A
        self.L = L
        self.rho = rho
        self.m = rho * A  # Mass per unit length
    
    @abstractmethod
    def stiffness_matrix(self) -> np.ndarray:
        """Calculate local stiffness matrix."""
        pass
    
    @abstractmethod
    def mass_matrix(self) -> np.ndarray:
        """Calculate local mass matrix."""
        pass

    @abstractmethod
    def recover_forces_consistent(self, u_local: np.ndarray, xi: Union[float, np.ndarray],
                                 dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Statically consistent force recovery (Homogeneous + Particular).
        Enforces local element equilibrium by integrating distributed loads explicitly.

        Parameters:
        -----------
        u_local : np.ndarray
            Local element displacement vector [u1, v1, theta1, u2, v2, theta2].
        xi : Union[float, np.ndarray]
            Normalized position(s) along element [0, 1].
        dist_load : Tuple[float, float, float, float]
            Distributed loads: (wy1, wy2, wx1, wx2).

        Returns:
        --------
        (axial_force, shear_force, bending_moment) : Tuple[np.ndarray, np.ndarray, np.ndarray]
            The recovered internal forces.
        """
        pass

    @abstractmethod
    def interpolate_forces_homogeneous(self, u_local: np.ndarray, xi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pure FEA interpolation using shape function derivatives (Homogeneous only).
        Follows the mathematical displacement approximation without local equilibrium corrections.

        Parameters:
        -----------
        u_local : np.ndarray
            Local element displacement vector [u1, v1, theta1, u2, v2, theta2].
        xi : Union[float, np.ndarray]
            Normalized position(s) along element [0, 1].

        Returns:
        --------
        (axial_force, shear_force, bending_moment) : Tuple[np.ndarray, np.ndarray, np.ndarray]
            The interpolated internal forces.
        """
        pass


class UnifiedBeamElement(BeamElementMatrices):
    """
    Unified coupled Timoshenko beam element.
    
    Captures axial-bending coupling (ES) and transverse shear deformation (GA_s).
    By setting GA_s = None or phi=0, it recovers the Euler-Bernoulli formulation.
    By setting ES = 0, it recovers the standard isotropic formulation.

    Constitutive relations::
        N = EA·ε₀ + ES·κ
        M = ES·ε₀ + EI·κ
        V = GA_s·(v' - θ)

    DOFs: [u1, v1, theta1, u2, v2, theta2]
    """

    def __init__(self, EA: float, ES: float, EI: float, L: float, rho_total: float,
                 GA_s: float = None, force_euler: bool = False):
        """
        Initialize unified element.

        Parameters:
        ----------
        EA : float
            Axial stiffness (N)
        ES : float
            Coupling stiffness (N·mm)
        EI : float
            Bending stiffness (N·mm²)
        L : float
            Length (mm)
        rho_total : float
            Mass per unit length (kg/mm)
        GA_s : float, optional
            Transverse shear stiffness (N).
        force_euler : bool
            If True, ignores GA_s and uses Euler-Bernoulli (phi=0).
        """
        super().__init__(E=1.0, G=1.0, I=1.0, A=1.0, L=L, rho=rho_total)
        self.EA = EA
        self.ES = ES
        self.EI = EI
        self.rho_total = rho_total
        self.GA_s = GA_s
        self.force_euler = force_euler

    def _phi(self) -> float:
        """Timoshenko shear-flexibility parameter φ."""
        if self.force_euler or self.GA_s is None or self.GA_s == 0.0:
            return 0.0
        return 12.0 * self.EI / (self.GA_s * self.L ** 2)

    def stiffness_matrix(self) -> np.ndarray:
        """
        Coupled Timoshenko stiffness matrix for anisotropic beam element.

        The full 6×6 matrix is assembled from three contributions:

        1. **Axial sub-matrix** (DOFs u1, u2)::

               EA/L * [[1, -1], [-1, 1]]

        2. **Timoshenko bending sub-matrix** (DOFs v1, θ1, v2, θ2):
           Standard Timoshenko form with shear-flexibility φ = 12EI/(GA_s·L²).
           Setting GA_s = None gives φ = 0 → classical Euler-Bernoulli result.

        3. **Coupling sub-matrix** (u1/u2 coupled to θ1/θ2):
           Derived from the ES energy term via::

               ∫₀ᴸ u′ v″ dx = (u₂−u₁)/L · (θ₂−θ₁)

           giving kernel ES/L on the (u, θ) cross-terms.
        """
        EA, ES, EI, L = self.EA, self.ES, self.EI, self.L
        phi = self._phi()
        fac = 1.0 / (1.0 + phi)

        K = np.zeros((6, 6))

        # --- 1. Axial sub-matrix ---
        k_axial = (EA / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K[np.ix_([0, 3], [0, 3])] += k_axial

        # --- 2. Timoshenko bending sub-matrix ---
        # DOF order within bending block: [v1, θ1, v2, θ2] → indices [1, 2, 4, 5]
        k11 =  12.0 * EI * fac / L**3
        k12 =   6.0 * EI * fac / L**2
        k22 = (4.0 + phi) * EI * fac / L
        k24 = (2.0 - phi) * EI * fac / L

        k_bend = np.array([
            [ k11,  k12, -k11,  k12],
            [ k12,  k22, -k12,  k24],
            [-k11, -k12,  k11, -k12],
            [ k12,  k24, -k12,  k22],
        ])
        K[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] += k_bend

        # --- 3. Coupling sub-matrix (axial ↔ rotation) ---
        # Energy: U_coup = ES ∫ u′ v″ dx = ES · (u₂−u₁)/L · (θ₂−θ₁)
        # K_coup = ∂²U_coup/∂qᵢ∂qⱼ
        # Non-zero entries at (u1,θ1), (u1,θ2), (u2,θ1), (u2,θ2)
        k_coupling = (ES / L) * np.array([
            [ 0,  0,  1,  0,  0, -1],  # u1
            [ 0,  0,  0,  0,  0,  0],  # v1
            [ 1,  0,  0, -1,  0,  0],  # θ1
            [ 0,  0, -1,  0,  0,  1],  # u2
            [ 0,  0,  0,  0,  0,  0],  # v2
            [-1,  0,  0,  1,  0,  0],  # θ2
        ])
        K += k_coupling

        return K

    def mass_matrix(self, consistent: bool = True) -> np.ndarray:
        """
        Consistent or lumped mass matrix for the unified beam element.

        For Euler-Bernoulli (phi=0) the standard consistent mass matrix is used.
        For Timoshenko (phi>0) the exact consistent mass matrix is assembled from
        the phi-dependent Timoshenko shape functions, as derived by Friedman & Kosmatka
        (1993). This includes:

        - Translational inertia terms (m*L integral)
        - **Rotatory inertia terms** (m*r² = EI/EA · m/L integral), crucial for
          thick/short beams (L/h < 10) and high-frequency modal analysis.

        References
        ----------
        Friedman, Z. & Kosmatka, J.B. (1993). "An improved two-node Timoshenko beam
        finite element." Computers & Structures, 47(3), 473-481.
        """
        m, L = self.rho_total, self.L
        phi = self._phi()

        if not consistent:
            # Lumped: translational DOFs only, rotational DOFs zero
            return np.diag([m*L/2, m*L/2, 0.0, m*L/2, m*L/2, 0.0])

        if phi == 0:
            # Standard Euler-Bernoulli consistent mass matrix
            M = (m * L / 420) * np.array([
                [ 140,     0,          0,      70,     0,          0],
                [   0,   156,      22*L,       0,    54,      -13*L],
                [   0,  22*L,   4*L**2,        0,  13*L,   -3*L**2],
                [  70,     0,          0,     140,     0,          0],
                [   0,    54,      13*L,       0,   156,      -22*L],
                [   0, -13*L,  -3*L**2,        0, -22*L,    4*L**2],
            ])
        else:
            # --- Timoshenko consistent mass matrix (Friedman & Kosmatka, 1993) ---
            # Translational inertia scaling factor (same denominator as stiffness)
            fac = 1.0 / (1.0 + phi)
            p = phi  # shorthand

            # Rotatory inertia scaling: r² = EI / (EA·L²) for the section.
            # If EA and EI are available use them; fall back to zero if undefined.
            if self.EA > 0 and self.EI > 0 and L > 0:
                r_sq = self.EI / (self.EA * L**2)   # (radius of gyration / L)²
            else:
                r_sq = 0.0

            m_r = m * r_sq  # rotatory inertia mass per unit length * r²

            # --- Translational sub-matrix coefficients ---
            # (v1, th1, v2, th2) block — indices [1,2,4,5]
            c1 = fac**2 * (13.0/35.0 + 7.0*p/10.0 + p**2/3.0)
            c2 = fac**2 * (11.0/210.0 + 11.0*p/120.0 + p**2/24.0) * L
            c3 = fac**2 * (9.0/70.0 + 3.0*p/10.0 + p**2/6.0)
            c4 = fac**2 * (-13.0/420.0 - 3.0*p/40.0 - p**2/24.0) * L
            c5 = fac**2 * (1.0/105.0 + p/60.0 + p**2/120.0) * L**2
            c6 = fac**2 * (-1.0/140.0 - p/60.0 - p**2/120.0) * L**2

            # --- Rotatory inertia sub-matrix coefficients ---
            r1 = fac**2 * (6.0/5.0)
            r2 = fac**2 * (1.0/10.0 - p/2.0) * L
            r3 = fac**2 * (-6.0/5.0)
            r4 = fac**2 * (1.0/10.0 - p/2.0) * L
            r5 = fac**2 * (2.0/15.0 + p/6.0 + p**2/3.0) * L**2
            r6 = fac**2 * (-1.0/30.0 - p/6.0 + p**2/6.0) * L**2

            # (v1, th1, v2, th2) bending block
            Mb = m * L * np.array([
                [c1 + m_r*r1/m,  c2 + m_r*r2/m,  c3 + m_r*r3/m,  c4 + m_r*r4/m],
                [c2 + m_r*r2/m,  c5 + m_r*r5/m,  -c4 - m_r*r4/m, c6 + m_r*r6/m],
                [c3 + m_r*r3/m, -c4 - m_r*r4/m,  c1 + m_r*r1/m, -c2 - m_r*r2/m],
                [c4 + m_r*r4/m,  c6 + m_r*r6/m, -c2 - m_r*r2/m,  c5 + m_r*r5/m],
            ])

            M = np.zeros((6, 6))
            # Axial DOFs (u1, u2): indices 0, 3 — standard consistent axial mass
            M[0, 0] = m * L / 3.0
            M[0, 3] = m * L / 6.0
            M[3, 0] = m * L / 6.0
            M[3, 3] = m * L / 3.0
            # Bending + rotatory inertia block: indices [1,2,4,5]
            M[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = Mb

        return M

    def interpolate_forces_homogeneous(self, u_local: np.ndarray, xi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure FEA interpolation for Unified element."""
        L, EA, ES, EI = self.L, self.EA, self.ES, self.EI
        u1, v1, theta1, u2, v2, theta2 = u_local
        phi = self._phi()
        fac = 1.0 / (1.0 + phi)

        # eps0 = (u2-u1)/L
        eps0 = (u2 - u1) / L * np.ones_like(xi)

        # Curvature kappa = d_theta/dx
        d_theta = (fac/L) * (
            (6 - 12*xi)*v1/L + (-4 + 6*xi - phi)*theta1 +
            (-6 + 12*xi)*v2/L + (-2 + 6*xi + phi)*theta2
        )

        # Coupled constitutive laws
        axial_force = EA * eps0 + ES * d_theta
        bending_moment = ES * eps0 + EI * d_theta

        # Shear (constant across element for homogeneous solution)
        shear_force = -(12 * EI * fac / L**3) * (v1 - v2 + 0.5 * L * (theta1 + theta2)) * np.ones_like(xi)

        return axial_force, shear_force, bending_moment

    def recover_forces_consistent(self, u_local: np.ndarray, xi: Union[float, np.ndarray],
                                 dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Statically consistent recovery for Unified element."""
        L = self.L
        wy1, wy2, wx1, wx2 = dist_load
        f_eq = np.zeros(6)
        f_eq[1] = (L / 20) * (7*wy1 + 3*wy2)
        f_eq[2] = (L**2 / 60) * (3*wy1 + 2*wy2)
        f_eq[4] = (L / 20) * (3*wy1 + 7*wy2)
        f_eq[5] = -(L**2 / 60) * (2*wy1 + 3*wy2)
        f_eq[0] = (L / 6) * (2*wx1 + wx2)
        f_eq[3] = (L / 6) * (wx1 + 2*wx2)

        f_total = self.stiffness_matrix() @ u_local - f_eq
        N1, V1, M1 = f_total[0], f_total[1], f_total[2]

        int_wx = L * (wx1 * (xi - 0.5*xi**2) + wx2 * (0.5*xi**2))
        axial_force = -N1 - int_wx
        int_wy = L * (wy1 * (xi - 0.5*xi**2) + wy2 * (0.5*xi**2))
        shear_force = V1 + int_wy
        int_wy_dist = L**2 * (wy1 * (0.5*xi**2 - xi**3/6) + wy2 * (xi**3/6))
        bending_moment = -M1 + V1 * (xi * L) + int_wy_dist

        return axial_force, shear_force, bending_moment



class EulerBernoulliElement(UnifiedBeamElement):
    """
    Deprecated: Lightweight wrapper for UnifiedBeamElement.
    Forces Euler-Bernoulli behavior (phi=0).
    """
    def __init__(self, E, G, I, A, L, rho):
        super().__init__(EA=E*A, ES=0, EI=E*I, L=L, rho_total=rho*A, force_euler=True)
        # For legacy test compatibility
        self.E, self.G, self.I, self.A, self.rho = E, G, I, A, rho


class TimoshenkoElement(UnifiedBeamElement):
    """
    Deprecated: Lightweight wrapper for UnifiedBeamElement.
    Uses Timoshenko shear deformation.
    """
    def __init__(self, E, G, I, A, L, rho, kappa=5/6):
        super().__init__(EA=E*A, ES=0, EI=E*I, L=L, rho_total=rho*A, GA_s=G*A*kappa, force_euler=False)
        # For legacy test compatibility
        self.E, self.G, self.I, self.A, self.rho, self.kappa = E, G, I, A, rho, kappa

# Keep for backward compatibility/internal mapping
AnisotropicBeamElement = UnifiedBeamElement


def get_rotation_matrix(angle_rad: float) -> np.ndarray:
    """
    Create a 6x6 rotation matrix for a 2D beam element.

    Transforms local DOFs [u1, v1, th1, u2, v2, th2] to global.
    T = [
        [ c  s  0  0  0  0 ]
        [-s  c  0  0  0  0 ]
        [ 0  0  1  0  0  0 ]
        [ 0  0  0  c  s  0 ]
        [ 0  0  0 -s  c  0 ]
        [ 0  0  0  0  0  1 ]
    ]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    R = np.array([
        [ c, s, 0],
        [-s, c, 0],
        [ 0, 0, 1]
    ])

    T = np.zeros((6, 6))
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    return T


def calculate_shear_correction_factor(section_type: str) -> float:
    """
    Get shear correction factor for common cross-sections.
    
    Parameters:
    -----------
    section_type : str
        Type of cross-section
        
    Returns:
    --------
    kappa : float
        Shear correction factor
    """
    factors = {
        'rectangular': 5/6,
        'circular': 9/10,
        'i_beam': 0.5,
        'box': 0.5,
        'channel': 0.5
    }
    
    return factors.get(section_type.lower(), 5/6)


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("ELEMENT MATRIX EXAMPLES")
    print("="*70)
    
    # Properties
    E = 200000  # MPa
    G = E / (2 * (1 + 0.3))
    I = 6666.67  # mm⁴
    A = 200  # mm²
    L = 100  # mm
    rho = 7.85e-6  # kg/mm³
    
    # Euler-Bernoulli element
    print("\n1. Euler-Bernoulli Element:")
    eb_elem = EulerBernoulliElement(E, G, I, A, L, rho)
    K_eb = eb_elem.stiffness_matrix()
    M_eb = eb_elem.mass_matrix()
    print(f"   Stiffness matrix (6×6):")
    print(f"   K[1,1] = {K_eb[1,1]:.2e} (transverse stiffness)")
    print(f"   K[2,2] = {K_eb[2,2]:.2e} (rotational stiffness)")
    
    # Timoshenko element
    print("\n2. Timoshenko Element:")
    tim_elem = TimoshenkoElement(E, G, I, A, L, rho)
    K_tim = tim_elem.stiffness_matrix()
    print(f"   Stiffness matrix (6×6):")
    print(f"   K[1,1] = {K_tim[1,1]:.2e} (includes shear deformation)")
    print(f"   Difference: {(K_eb[1,1] - K_tim[1,1])/K_eb[1,1]*100:.1f}% softer")
