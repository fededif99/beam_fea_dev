"""
element_matrices.py
===================
Local element matrix calculations for different beam theories.

Provides stiffness and mass matrices for:
- Euler-Bernoulli beam elements
- Timoshenko beam elements
"""

import numpy as np
from typing import Tuple
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
    def interpolate_internal_forces(self, u_local: np.ndarray, xi: float,
                                   dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[float, float, float]:
        """
        Interpolate internal forces at a normalized position.

        Uses superposition of homogeneous and particular solutions for static consistency.
        
        Parameters:
        -----------
        u_local : np.ndarray
            Local element displacement vector [u1, v1, theta1, u2, v2, theta2]
        xi : float
            Normalized position along element [0, 1]
        dist_load : Tuple[float, float, float, float]
            Distributed loads: (wy1, wy2, wx1, wx2)
            
        Returns:
        --------
        (axial_force, shear_force, bending_moment) : Tuple[float, float, float]
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

    def interpolate_internal_forces(self, u_local: np.ndarray, xi: float,
                                   dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[float, float, float]:
        """
        Interpolate axial, shear, and moment for Euler-Bernoulli beam.
        """
        L = self.L
        E = self.E
        I = self.I
        A = self.A
        
        u1 = u_local[0]
        v1 = u_local[1]
        theta1 = u_local[2]
        u2 = u_local[3]
        v2 = u_local[4]
        theta2 = u_local[5]
        
        wy1, wy2, wx1, wx2 = dist_load
        
        # 1. Homogeneous Solution (from nodal displacements)
        # Hermite shape functions second derivatives (for moment: M = -EI * d2v/dx2)
        d2N1 = (6 - 12*xi) / L**2
        d2N2 = (4 - 6*xi) / L
        d2N3 = (-6 + 12*xi) / L**2
        d2N4 = (2 - 6*xi) / L
        
        # Hermite shape functions third derivatives (for shear: V = -EI * d3v/dx3)
        d3N1 = -12 / L**3
        d3N2 = -6 / L**2
        d3N3 = 12 / L**3
        d3N4 = -6 / L**2
        
        moment_h = -E * I * (d2N1 * v1 + d2N2 * theta1 + d2N3 * v2 + d2N4 * theta2)
        shear_h = -E * I * (d3N1 * v1 + d3N2 * theta1 + d3N3 * v2 + d3N4 * theta2)

        # For statically consistent internal force recovery:
        # 1. Total force F = K_local * u_local - F_equivalent_distributed
        # 2. V(xi) = V1_total - integral[w(s) L ds]
        # 3. M(xi) = -M1_total + V1_total*(xi*L) - integral[w(s)*(xi*L - s*L) ds]

        # Consistent nodal forces (equivalent loads) for linear distributed load:
        # Fy1 = L/20 * (7*wy1 + 3*wy2)
        # M1  = L^2/60 * (3*wy1 + 2*wy2)
        # Fy2 = L/20 * (3*wy1 + 7*wy2)
        # M2  = -L^2/60 * (2*wy1 + 3*wy2)
        f_eq = np.zeros(6)
        f_eq[1] = (L/20) * (7*wy1 + 3*wy2)
        f_eq[2] = (L**2/60) * (3*wy1 + 2*wy2)
        f_eq[4] = (L/20) * (3*wy1 + 7*wy2)
        f_eq[5] = -(L**2/60) * (2*wy1 + 3*wy2)

        # Axial equivalent loads
        # Fx1 = L/6 * (2*wx1 + wx2)
        # Fx2 = L/6 * (wx1 + 2*wx2)
        f_eq[0] = (L/6) * (2*wx1 + wx2)
        f_eq[3] = (L/6) * (wx1 + 2*wx2)

        # Total nodal reactions at element ends (internal forces)
        # f_total = K*u - f_eq represents the internal forces at the nodes
        # For node 1, f_total[1] is the shear force at the start of the element.
        f_total = self.stiffness_matrix() @ u_local - f_eq

        V1 = f_total[1]
        M1 = f_total[2]
        N1 = f_total[0]

        # In this solver's convention:
        # Positive V is upward on left face.
        # Positive M is bottom-tension (which is -M1 if M1 is CCW acting on the beam end).

        # V(xi) = V1 + integral[w(s) L ds] from 0 to xi (if w is positive upward)
        int_w = L * (wy1 * (xi - 0.5*xi**2) + wy2 * (0.5*xi**2))
        shear_force = V1 + int_w

        # M(xi) = -M1 + V1 * (xi*L) + integral[w(s)*(xi*L - s*L) ds]
        int_w_dist = L**2 * (wy1 * (0.5*xi**2 - xi**3/6) + wy2 * (xi**3/6))
        bending_moment = -M1 + V1 * (xi * L) + int_w_dist

        # Axial: N(xi) = -N1 - integral[wx(s) L ds] (if N is positive tension)
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

    def interpolate_internal_forces(self, u_local: np.ndarray, xi: float,
                                   dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[float, float, float]:
        """
        Interpolate axial, shear, and moment for Timoshenko beam.
        """
        L = self.L
        wy1, wy2, wx1, wx2 = dist_load

        # Consistent force recovery using superposition
        f_eq = np.zeros(6)
        f_eq[1] = (L/20) * (7*wy1 + 3*wy2)
        f_eq[2] = (L**2/60) * (3*wy1 + 2*wy2)
        f_eq[4] = (L/20) * (3*wy1 + 7*wy2)
        f_eq[5] = -(L**2/60) * (2*wy1 + 3*wy2)
        f_eq[0] = (L/6) * (2*wx1 + wx2)
        f_eq[3] = (L/6) * (wx1 + 2*wx2)

        f_total = self.stiffness_matrix() @ u_local - f_eq

        N1, V1, M1 = f_total[0], f_total[1], f_total[2]
        
        # Axial
        int_wx = L * (wx1 * (xi - 0.5*xi**2) + wx2 * (0.5*xi**2))
        axial_force = -N1 - int_wx
        
        # Shear
        int_wy = L * (wy1 * (xi - 0.5*xi**2) + wy2 * (0.5*xi**2))
        shear_force = V1 + int_wy
        
        # Moment
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
