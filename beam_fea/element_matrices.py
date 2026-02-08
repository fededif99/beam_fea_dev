"""
element_matrices.py
===================
Local element matrix calculations for different beam theories.

Provides stiffness and mass matrices for:
- Euler-Bernoulli beam elements
- Timoshenko beam elements
- Geometric stiffness (for buckling analysis)
"""

import numpy as np
from typing import Tuple
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
    def interpolate_internal_forces(self, u_local: np.ndarray, xi: float) -> Tuple[float, float]:
        """
        Interpolate shear force and bending moment at a normalized position.
        
        Parameters:
        -----------
        u_local : np.ndarray
            Local element displacement vector [u1, v1, theta1, u2, v2, theta2]
        xi : float
            Normalized position along element [0, 1]
            
        Returns:
        --------
        shear : float
        moment : float
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

    def interpolate_internal_forces(self, u_local: np.ndarray, xi: float) -> Tuple[float, float]:
        """
        Interpolate shear and moment for Euler-Bernoulli beam using Hermite shape functions.
        """
        L = self.L
        E = self.E
        I = self.I
        
        v1 = u_local[1]
        theta1 = u_local[2]
        v2 = u_local[4]
        theta2 = u_local[5]
        
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
        
        # Results consistent with V=V1, M=-M1 (standard diagram convention for this solver)
        bending_moment = -E * I * (d2N1 * v1 + d2N2 * theta1 + d2N3 * v2 + d2N4 * theta2)
        shear_force = -E * I * (d3N1 * v1 + d3N2 * theta1 + d3N3 * v2 + d3N4 * theta2)
        
        return shear_force, bending_moment


class TimoshenkoElement(BeamElementMatrices):
    """
    Timoshenko beam element (includes shear deformation).
    
    Valid for thick beams where L/h < 10-15.
    DOFs: [u1, v1, θ1, u2, v2, θ2]
    """
    
    def __init__(self, E: float, G: float, I: float, A: float, L: float,
                 rho: float, kappa: float = 5/6):
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
        """Consistent mass matrix for Timoshenko beam (simplified)."""
        # Use Euler-Bernoulli mass matrix as approximation
        # (exact Timoshenko mass matrix is more complex)
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
        """Lumped mass matrix."""
        m, L = self.m, self.L
        return np.diag([m*L/2, m*L/2, 0, m*L/2, m*L/2, 0])

    def interpolate_internal_forces(self, u_local: np.ndarray, xi: float) -> Tuple[float, float]:
        """
        Interpolate shear and moment for Timoshenko beam.
        For simplicity, we use the average shear and linear moment interpolation 
        consistent with Timoshenko beam theory.
        """
        # Get element stiffness for force recovery
        K_local = self.stiffness_matrix()
        forces = K_local @ u_local
        
        # forces = [F1, V1, M1, F2, V2, M2]
        # In Timoshenko beams, shear is constant within the element (averaging V1 and V2)
        # and moment is linear.
        
        V1, M1 = forces[1], forces[2]
        V2, M2 = forces[4], forces[5]
        
        # Consistent with EulerBernoulliElement signs: Shear=V1, Moment=-M1
        shear_force = V1 
        bending_moment = -M1 * (1-xi) - M2 * xi
        
        return shear_force, bending_moment


class GeometricStiffness:
    """Geometric stiffness matrix for stability analysis."""
    
    @staticmethod
    def geometric_stiffness_matrix(L: float, P: float) -> np.ndarray:
        """
        Calculate geometric stiffness matrix for beam element.
        
        Used in buckling and large deflection analysis.
        
        Parameters:
        -----------
        L : float
            Element length (mm)
        P : float
            Axial force (N), positive for tension
            
        Returns:
        --------
        Kg : np.ndarray (6×6)
            Geometric stiffness matrix
        """
        Kg = np.zeros((6, 6))
        
        # Only affects transverse DOFs
        Kg[1, 1] = Kg[4, 4] = 6 / (5 * L)
        Kg[1, 2] = Kg[2, 1] = 1 / 10
        Kg[1, 4] = Kg[4, 1] = -6 / (5 * L)
        Kg[1, 5] = Kg[5, 1] = 1 / 10
        
        Kg[2, 2] = 2 * L / 15
        Kg[2, 4] = Kg[4, 2] = -1 / 10
        Kg[2, 5] = Kg[5, 2] = -L / 30
        
        Kg[4, 5] = Kg[5, 4] = -1 / 10
        Kg[5, 5] = 2 * L / 15
        
        # Scale by axial force
        Kg *= P
        
        return Kg


class DampingMatrix:
    """Damping matrix calculations."""
    
    @staticmethod
    def rayleigh_damping(M: np.ndarray, K: np.ndarray,
                        alpha: float, beta: float) -> np.ndarray:
        """
        Calculate Rayleigh damping matrix: C = α*M + β*K
        
        Parameters:
        -----------
        M : np.ndarray
            Mass matrix
        K : np.ndarray
            Stiffness matrix
        alpha : float
            Mass proportional damping coefficient
        beta : float
            Stiffness proportional damping coefficient
            
        Returns:
        --------
        C : np.ndarray
            Damping matrix
        """
        return alpha * M + beta * K
    
    @staticmethod
    def modal_damping(damping_ratios: np.ndarray, omega: np.ndarray,
                     phi: np.ndarray) -> np.ndarray:
        """
        Calculate damping matrix from modal damping ratios.
        
        Parameters:
        -----------
        damping_ratios : np.ndarray
            Damping ratio for each mode
        omega : np.ndarray
            Natural frequencies (rad/s)
        phi : np.ndarray
            Mode shapes (columns)
            
        Returns:
        --------
        C : np.ndarray
            Damping matrix
        """
        n_modes = len(damping_ratios)
        C_modal = np.diag(2 * damping_ratios * omega)
        
        # Transform to physical coordinates
        C = phi @ C_modal @ phi.T
        
        return C


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
    
    # Geometric stiffness
    print("\n3. Geometric Stiffness (buckling):")
    P = -1000  # N (compression)
    Kg = GeometricStiffness.geometric_stiffness_matrix(L, P)
    print(f"   For P = {P} N compression:")
    print(f"   Kg[1,1] = {Kg[1,1]:.2e}")
