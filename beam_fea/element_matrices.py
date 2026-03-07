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

# Standard shear correction factor for rectangular sections
SHEAR_CORRECTION_FACTOR = 5/6


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


class EulerBernoulliElement(BeamElementMatrices):
    """
    Euler-Bernoulli beam element (neglects shear deformation).
    
    Valid for slender beams where L/h > 10-15.
    DOFs: [u1, v1, θ1, u2, v2, θ2]
    """
    
    def stiffness_matrix(self) -> np.ndarray:
        """
        Calculate local stiffness matrix for Euler-Bernoulli beam.
        
        Orthonormal local coordinates: x (axial), y (transverse)
        DOFs per node: [u, v, θ]
        
        K = [
            [ EA/L   0          0          -EA/L   0          0        ]
            [ 0      12EI/L^3   6EI/L^2    0       -12EI/L^3  6EI/L^2  ]
            [ 0      6EI/L^2    4EI/L      0       -6EI/L^2   2EI/L    ]
            [ -EA/L  0          0          EA/L    0          0        ]
            [ 0      -12EI/L^3  -6EI/L^2   0       12EI/L^3   -6EI/L^2 ]
            [ 0      6EI/L^2    2EI/L      0       -6EI/L^2   4EI/L    ]
        ]
        
        Returns:
        --------
        K_local : np.ndarray (6×6)
            Local stiffness matrix
        """
        E, I, A, L = self.E, self.I, self.A, self.L
        
        K = np.array([
            [ A*E/L,              0,              0,    -A*E/L,              0,              0 ],
            [     0,  12*E*I/L**3,   6*E*I/L**2,         0, -12*E*I/L**3,   6*E*I/L**2 ],
            [     0,   6*E*I/L**2,      4*E*I/L,         0,  -6*E*I/L**2,      2*E*I/L ],
            [-A*E/L,              0,              0,     A*E/L,              0,              0 ],
            [     0, -12*E*I/L**3,  -6*E*I/L**2,         0,  12*E*I/L**3,  -6*E*I/L**2 ],
            [     0,   6*E*I/L**2,      2*E*I/L,         0,  -6*E*I/L**2,      4*E*I/L ]
        ])
        
        return K
    
    def mass_matrix(self, consistent: bool = True) -> np.ndarray:
        """
        Calculate local mass matrix.
        
        Parameters:
        -----------
        consistent : bool
            If True, use consistent mass matrix
            If False, use lumped mass matrix
            
        Returns:
        --------
        M_local : np.ndarray (6×6)
            Local mass matrix
        """
        if consistent:
            return self._consistent_mass_matrix()
        else:
            return self._lumped_mass_matrix()
    
    def _consistent_mass_matrix(self) -> np.ndarray:
        """Consistent mass matrix for Euler-Bernoulli beam."""
        m, L = self.m, self.L
        
        M = (m * L / 420) * np.array([
            [ 140,      0,         0,   70,      0,         0 ],
            [   0,    156,    22*L,     0,     54,   -13*L ],
            [   0,   22*L,  4*L**2,     0,   13*L, -3*L**2 ],
            [  70,      0,         0,  140,      0,         0 ],
            [   0,     54,    13*L,     0,    156,   -22*L ],
            [   0, -13*L, -3*L**2,     0, -22*L,  4*L**2 ]
        ])
        
        return M
    
    def _lumped_mass_matrix(self) -> np.ndarray:
        """Lumped mass matrix for Euler-Bernoulli beam."""
        m, L = self.m, self.L
        
        # Half mass at each node
        M = np.diag([m*L/2, m*L/2, 0, m*L/2, m*L/2, 0])
        
        return M

    def interpolate_forces_homogeneous(self, u_local: np.ndarray, xi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure FEA interpolation (Hermite derivatives) for Euler-Bernoulli."""
        L, E, I, A = self.L, self.E, self.I, self.A
        u1, v1, theta1, u2, v2, theta2 = u_local

        # Axial (linear)
        axial_force = A * E * (u2 - u1) / L * np.ones_like(xi)

        # Bending (Cubic Hermite second derivatives)
        d2N1 = (6 - 12*xi) / L**2
        d2N2 = (4 - 6*xi) / L
        d2N3 = (-6 + 12*xi) / L**2
        d2N4 = (2 - 6*xi) / L
        bending_moment = -E * I * (d2N1*v1 + d2N2*theta1 + d2N3*v2 + d2N4*theta2)

        # Shear (Cubic Hermite third derivatives)
        d3N1 = -12 / L**3 * np.ones_like(xi)
        d3N2 = -6 / L**2 * np.ones_like(xi)
        d3N3 = 12 / L**3 * np.ones_like(xi)
        d3N4 = -6 / L**2 * np.ones_like(xi)
        shear_force = -E * I * (d3N1*v1 + d3N2*theta1 + d3N3*v2 + d3N4*theta2)

        return axial_force, shear_force, bending_moment

    def recover_forces_consistent(self, u_local: np.ndarray, xi: Union[float, np.ndarray],
                                 dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Statically consistent recovery for Euler-Bernoulli."""
        L = self.L
        wy1, wy2, wx1, wx2 = dist_load

        # Equivalent nodal loads
        f_eq = np.zeros(6)
        f_eq[1] = (L/20) * (7*wy1 + 3*wy2)
        f_eq[2] = (L**2/60) * (3*wy1 + 2*wy2)
        f_eq[4] = (L/20) * (3*wy1 + 7*wy2)
        f_eq[5] = -(L**2/60) * (2*wy1 + 3*wy2)
        f_eq[0] = (L/6) * (2*wx1 + wx2)
        f_eq[3] = (L/6) * (wx1 + 2*wx2)

        f_total = self.stiffness_matrix() @ u_local - f_eq
        N1, V1, M1 = f_total[0], f_total[1], f_total[2]

        # Integration
        int_w = L * (wy1 * (xi - 0.5*xi**2) + wy2 * (0.5*xi**2))
        shear_force = V1 + int_w

        int_w_dist = L**2 * (wy1 * (0.5*xi**2 - xi**3/6) + wy2 * (xi**3/6))
        bending_moment = -M1 + V1 * (xi * L) + int_w_dist

        int_wx = L * (wx1 * (xi - 0.5*xi**2) + wx2 * (0.5*xi**2))
        axial_force = -N1 - int_wx
        
        return axial_force, shear_force, bending_moment


class TimoshenkoElement(BeamElementMatrices):
    """
    Timoshenko beam element (includes shear deformation).
    
    Valid for thick beams where L/h < 10-15.
    DOFs: [u1, v1, θ1, u2, v2, θ2]
    """
    
    def __init__(self, E: float, G: float, I: float, A: float, L: float,
                 rho: float, kappa: float = SHEAR_CORRECTION_FACTOR):
        """
        Initialize Timoshenko beam element.
        
        Parameters:
        -----------
        kappa : float
            Shear correction factor (default 5/6 for rectangular sections)
        """
        super().__init__(E, G, I, A, L, rho)
        self.kappa = kappa  # Shear correction factor
        self.As = kappa * A  # Effective shear area
    
    def stiffness_matrix(self) -> np.ndarray:
        """
        Calculate local stiffness matrix for Timoshenko beam.
        
        Returns:
        --------
        K_local : np.ndarray (6×6)
            Local stiffness matrix
        """
        E, G, I, A, L = self.E, self.G, self.I, self.A, self.L
        As = self.As
        
        # Shear flexibility parameter
        phi = 12 * E * I / (G * As * L**2)
        
        K = np.zeros((6, 6))
        
        # Axial terms
        K[0, 0] = K[3, 3] = A * E / L
        K[0, 3] = K[3, 0] = -A * E / L
        
        # Bending and shear coupling
        K[1, 1] = K[4, 4] = 12 * E * I / (L**3 * (1 + phi))
        K[1, 4] = K[4, 1] = -12 * E * I / (L**3 * (1 + phi))
        
        K[1, 2] = K[2, 1] = 6 * E * I / (L**2 * (1 + phi))
        K[1, 5] = K[5, 1] = 6 * E * I / (L**2 * (1 + phi))
        K[4, 2] = K[2, 4] = -6 * E * I / (L**2 * (1 + phi))
        K[4, 5] = K[5, 4] = -6 * E * I / (L**2 * (1 + phi))
        
        K[2, 2] = K[5, 5] = (4 + phi) * E * I / (L * (1 + phi))
        K[2, 5] = K[5, 2] = (2 - phi) * E * I / (L * (1 + phi))
        
        return K
    
    def mass_matrix(self, consistent: bool = True) -> np.ndarray:
        """
        Calculate local mass matrix for Timoshenko beam.
        
        Parameters:
        -----------
        consistent : bool
            If True, use consistent mass matrix
            
        Returns:
        --------
        M_local : np.ndarray (6×6)
            Local mass matrix
        """
        if consistent:
            return self._consistent_mass_matrix_timoshenko()
        else:
            return self._lumped_mass_matrix()
    
    def _consistent_mass_matrix_timoshenko(self) -> np.ndarray:
        """
        Consistent mass matrix for Timoshenko beam.
        Includes rotational inertia and shear deformation effects.
        """
        E, G, I, A, L, rho = self.E, self.G, self.I, self.A, self.L, self.rho
        As = self.As
        m = rho * A
        
        # Shear flexibility parameter
        phi = 12 * E * I / (G * As * L**2)

        M = np.zeros((6, 6))

        # Axial terms (same as Euler-Bernoulli)
        M[0, 0] = M[3, 3] = rho * A * L / 3
        M[0, 3] = M[3, 0] = rho * A * L / 6

        # Bending and shear terms (including rotational inertia)
        # Coefficients for translational mass
        c1 = m * L / (420 * (1 + phi)**2)
        m1 = 156 + 294*phi + 140*phi**2
        m2 = (22 + 77*phi + 70*phi**2) * L
        m3 = 54 + 126*phi + 70*phi**2
        m4 = -(13 + 63*phi + 70*phi**2) * L
        m5 = (4 + 14*phi + 14*phi**2) * L**2
        m6 = -(3 + 7*phi + 7*phi**2) * L**2

        # Coefficients for rotational mass
        c2 = rho * I / (30 * L * (1 + phi)**2)
        r1 = 36
        r2 = (3 - 15*phi) * L
        r3 = (4 + 5*phi + 10*phi**2) * L**2
        r4 = (-1 + 5*phi - 5*phi**2) * L**2

        # Assemble v and theta DOFs: indices 1, 2, 4, 5
        # Node 1
        M[1, 1] = c1 * m1 + c2 * r1
        M[1, 2] = M[2, 1] = c1 * m2 + c2 * r2
        M[2, 2] = c1 * m5 + c2 * r3

        # Node 2
        M[4, 4] = c1 * m1 + c2 * r1
        M[4, 5] = M[5, 4] = -(c1 * m2 + c2 * r2)
        M[5, 5] = c1 * m5 + c2 * r3

        # Coupling
        M[1, 4] = M[4, 1] = c1 * m3 - c2 * r1
        M[1, 5] = M[5, 1] = c1 * m4 + c2 * r2
        M[2, 4] = M[4, 2] = -(c1 * m4 + c2 * r2)
        M[2, 5] = M[5, 2] = c1 * m6 + c2 * r4
        
        return M
    
    def _lumped_mass_matrix(self) -> np.ndarray:
        """Lumped mass matrix."""
        m, L = self.m, self.L
        return np.diag([m*L/2, m*L/2, 0, m*L/2, m*L/2, 0])

    def interpolate_forces_homogeneous(self, u_local: np.ndarray, xi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure FEA interpolation for Timoshenko."""
        # For Timoshenko, pure interpolation from displacement field (v, theta)
        # N = EA * eps = EA * u'
        # V = kappa*G*A * (v' - theta)
        # M = EI * theta'
        L, E, G, I, A, As = self.L, self.E, self.G, self.I, self.A, self.As
        u1, v1, theta1, u2, v2, theta2 = u_local
        phi = 12 * E * I / (G * As * L**2)
        fac = 1.0 / (1.0 + phi)

        # Axial
        axial_force = A * E * (u2 - u1) / L * np.ones_like(xi)

        # Bending Moment M = EI * d_theta/dx
        # Shape functions for theta: N_theta = [N2', N4', N6', N8'] in some texts
        # For our DOF [v1, theta1, v2, theta2]:
        # theta(xi) = (1/(1+phi)) * [ (6xi/L - 6xi^2/L)v1 + (1 - 4xi + 3xi^2 + phi(1-xi))theta1 + (-6xi/L + 6xi^2/L)v2 + (-2xi + 3xi^2 + phi*xi)theta2 ]
        d_theta = (fac/L) * (
            (6 - 12*xi)*v1/L + (-4 + 6*xi - phi)*theta1 +
            (-6 + 12*xi)*v2/L + (-2 + 6*xi + phi)*theta2
        )
        bending_moment = E * I * d_theta

        # Shear Force V = As*G * (dv/dx - theta)
        # dv/dx = (fac/L) * [ (6xi-6xi^2)v1/L + (L(1-4xi+3xi^2)+phi*L(1-xi))theta1 ... ] - theta
        # Alternatively, for Timoshenko elements, homogeneous shear is constant across element:
        # V_h = -12EI / (L^3(1+phi)) * (v1 - v2 + L/2(theta1 + theta2))
        shear_force = -(12 * E * I * fac / L**3) * (v1 - v2 + 0.5 * L * (theta1 + theta2)) * np.ones_like(xi)

        return axial_force, shear_force, bending_moment

    def recover_forces_consistent(self, u_local: np.ndarray, xi: Union[float, np.ndarray],
                                 dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Statically consistent recovery for Timoshenko."""
        L = self.L
        wy1, wy2, wx1, wx2 = dist_load
        f_eq = np.zeros(6)
        f_eq[1] = (L/20) * (7*wy1 + 3*wy2)
        f_eq[2] = (L**2/60) * (3*wy1 + 2*wy2)
        f_eq[4] = (L/20) * (3*wy1 + 7*wy2)
        f_eq[5] = -(L**2/60) * (2*wy1 + 3*wy2)
        f_eq[0] = (L/6) * (2*wx1 + wx2)
        f_eq[3] = (L/6) * (wx1 + 2*wx2)

        f_total = self.stiffness_matrix() @ u_local - f_eq
        N1, V1, M1 = f_total[0], f_total[1], f_total[2]
        
        int_wx = L * (wx1 * (xi - 0.5*xi**2) + wx2 * (0.5*xi**2))
        axial_force = -N1 - int_wx
        int_wy = L * (wy1 * (xi - 0.5*xi**2) + wy2 * (0.5*xi**2))
        shear_force = V1 + int_wy
        int_wy_dist = L**2 * (wy1 * (0.5*xi**2 - xi**3/6) + wy2 * (xi**3/6))
        bending_moment = -M1 + V1 * (xi * L) + int_wy_dist
        
        return axial_force, shear_force, bending_moment


class AnisotropicBeamElement(BeamElementMatrices):
    """
    Timoshenko-based beam element for anisotropic (composite) materials.

    Captures axial-bending coupling via width-integrated CLT sectional stiffnesses
    (EA = A11·b, ES = B11·b, EI = D11·b). The bending sub-matrix uses the
    Timoshenko formulation (phi-correction) to account for transverse shear
    deformation, which is important for composite sandwich or thick-section beams.

    Constitutive relations (per Reddy 2003 §5.2; Kollár & Springer 2003 §4.1)::

        N = EA·ε₀ + ES·κ
        M = ES·ε₀ + EI·κ

    where ε₀ = u' (axial strain) and κ = v'' (curvature).

    DOFs: [u1, v1, theta1, u2, v2, theta2]
    """

    def __init__(self, EA: float, ES: float, EI: float, L: float, rho_total: float,
                 GA_s: float = None, kappa: float = SHEAR_CORRECTION_FACTOR):
        """
        Initialize with width-integrated stiffnesses.

        Parameters
        ----------
        EA : float
            Axial stiffness (N) — width-integrated A11·b
        ES : float
            Bend-extension coupling stiffness (N·mm) — width-integrated B11·b
        EI : float
            Bending stiffness (N·mm²) — width-integrated D11·b
        L : float
            Element length (mm)
        rho_total : float
            Linear mass density (kg/mm) — mass per unit length
        GA_s : float, optional
            Effective shear stiffness (N) — κ·G·A_s for the cross section.
            If None, the bending sub-matrix degenerates to Euler-Bernoulli
            (phi → 0, i.e. no shear deformation).
        kappa : float
            Shear correction factor (used only if GA_s is None). Default 5/6.
        """
        super().__init__(E=1.0, G=1.0, I=1.0, A=1.0, L=L, rho=rho_total)
        self.EA = EA
        self.ES = ES
        self.EI = EI
        self.rho_total = rho_total
        # GA_s: effective shear stiffness for Timoshenko correction.
        # If not provided, set phi=0 (falls back to Euler-Bernoulli bending).
        self.GA_s = GA_s  # may be None → phi=0

    def _phi(self) -> float:
        """Timoshenko shear-flexibility parameter φ = 12EI / (GA_s · L²)."""
        if self.GA_s is None or self.GA_s == 0.0:
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
        """Consistent or lumped mass matrix based on linear mass density."""
        m, L = self.rho_total, self.L

        if not consistent:
            return np.diag([m*L/2, m*L/2, 0.0, m*L/2, m*L/2, 0.0])

        M = (m * L / 420) * np.array([
            [ 140,     0,       0,     70,     0,       0],
            [   0,   156,   22*L,      0,    54,  -13*L],
            [   0,  22*L, 4*L**2,      0,  13*L, -3*L**2],
            [  70,     0,       0,    140,     0,       0],
            [   0,    54,   13*L,      0,   156,  -22*L],
            [   0, -13*L, -3*L**2,     0, -22*L, 4*L**2],
        ])
        return M

    def interpolate_forces_homogeneous(self, u_local: np.ndarray, xi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure FEA interpolation for Anisotropic element."""
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
        """Statically consistent recovery for Anisotropic element."""
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
