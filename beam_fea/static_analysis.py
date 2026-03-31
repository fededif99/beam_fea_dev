"""
static_analysis.py
==================
Static structural analysis solver.

Solves: K*u = F
Where:
- K: Global stiffness matrix
- u: Displacement vector
- F: Force vector
"""

import numpy as np
import warnings
from typing import Union, Tuple


class StaticAnalysis:
    """Static linear analysis solver."""
    
    def __init__(self, use_sparse: bool = True):
        """
        Initialize static analysis.
        
        Parameters:
        -----------
        use_sparse : bool
            Use sparse matrix solver (default: True)
        """
        self.use_sparse = use_sparse
        self.displacements = None
        self.reactions = None
        self.strain_energy = None
    
    def solve(self, K, F: np.ndarray, bc_set=None) -> np.ndarray:
        """
        Solve static equilibrium: K*u = F
        
        Parameters:
        -----------
        K : np.ndarray or scipy.sparse.matrix
            Global stiffness matrix
        F : np.ndarray
            Global force vector
        bc_set : BoundaryConditionSet, optional
            Boundary conditions (if not already applied)
            
        Returns:
        --------
        u : np.ndarray
            Displacement vector
        """
        # Apply boundary conditions if provided
        if bc_set is not None:
            K, F = bc_set.apply_to_system(K, F)
        
        from scipy.sparse import issparse
        
        if issparse(K) or self.use_sparse:
            from scipy.sparse.linalg import spsolve
            self.displacements = spsolve(K, F)
        else:
            # Dense fallback
            try:
                cond_number = np.linalg.cond(K)
                if cond_number > 1e12:
                    warnings.warn(
                        f"Stiffness matrix is ill-conditioned (cond = {cond_number:.2e}). "
                        "Check boundary conditions."
                    )
            except np.linalg.LinAlgError:
                pass
                
            self.displacements = np.linalg.solve(K, F)
        
        return self.displacements
    
    def calculate_reactions(self, K_original, F_applied: np.ndarray) -> np.ndarray:
        """
        Calculate reaction forces at supports.
        
        Parameters:
        -----------
        K_original : np.ndarray or scipy.sparse.matrix
            Original (unconstrained) stiffness matrix
        F_applied : np.ndarray
            Applied forces (before BC application)
            
        Returns:
        --------
        reactions : np.ndarray
            Reaction force vector
        """
        if self.displacements is None:
            raise ValueError("Must solve system before calculating reactions")
        
        # R = K*u - F
        self.reactions = K_original @ self.displacements - F_applied
        
        return self.reactions
    
    def calculate_strain_energy(self, K: np.ndarray) -> float:
        """
        Calculate total strain energy: U = (1/2) * u^T * K * u
        
        Parameters:
        -----------
        K : np.ndarray
            Stiffness matrix
            
        Returns:
        --------
        U : float
            Strain energy
        """
        if self.displacements is None:
            raise ValueError("Must solve system first")
        
        self.strain_energy = 0.5 * self.displacements @ K @ self.displacements
        
        return self.strain_energy
    



class StressAnalysis:
    """
    Post-processing utilities for stress state calculations.

    All methods accept scalars or NumPy arrays interchangeably via natural
    NumPy broadcasting. When arrays are passed the return type matches the
    input shape element-wise.
    """

    @staticmethod
    def calculate_bending_stress(
        M: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        I: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate bending stress: σ = −M·y / I

        Sign convention (sagging-positive):
        - A **CCW (sagging, M > 0)** moment bends the beam into a "U" shape.
          Top fibre (y > 0) is in **compression** (σ < 0).
          Bottom fibre (y < 0) is in **tension** (σ > 0).
        - A **CW (hogging, M < 0)** moment bends the beam into an "∩" shape.
          Top fibre (y > 0) is in **tension** (σ > 0).
          Bottom fibre (y < 0) is in **compression** (σ < 0).

        Parameters
        ----------
        M : float or ndarray
            Bending moment (N·mm). Positive = sagging (CCW).
        y : float or ndarray
            Distance from neutral axis (mm). Positive = towards top fibre.
        I : float or ndarray
            Second moment of area (mm⁴).

        Returns
        -------
        sigma : float or ndarray
            Bending stress (MPa). Positive = tension.
        """
        return -M * y / I

    @staticmethod
    def calculate_axial_stress(
        N: Union[float, np.ndarray],
        A: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate axial stress: σ = N / A

        Parameters
        ----------
        N : float or ndarray
            Axial force (N). Positive = tension.
        A : float or ndarray
            Cross-sectional area (mm²).

        Returns
        -------
        sigma : float or ndarray
            Axial stress (MPa). Positive = tension.
        """
        return N / A

    @staticmethod
    def calculate_shear_stress(
        V: Union[float, np.ndarray],
        Q: Union[float, np.ndarray],
        I: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate shear stress via the Jourawski formula: τ = V·Q / (I·t)

        Parameters
        ----------
        V : float or ndarray
            Shear force (N).
        Q : float or ndarray
            First moment of area above the point of interest (mm³).
        I : float or ndarray
            Second moment of area (mm⁴).
        t : float or ndarray
            Section width at the point of interest (mm).

        Returns
        -------
        tau : float or ndarray
            Shear stress (MPa).
        """
        return V * Q / (I * t)

    @staticmethod
    def calculate_principal_stresses(
        sigma_x: Union[float, np.ndarray],
        sigma_y: Union[float, np.ndarray],
        tau_xy: Union[float, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Calculate 2D principal stresses via Mohr's circle.

        Parameters
        ----------
        sigma_x, sigma_y : float or ndarray
            Normal stresses (MPa).
        tau_xy : float or ndarray
            Shear stress (MPa).

        Returns
        -------
        (sigma_1, sigma_2) : tuple of float or ndarray
            Principal stresses (MPa), where sigma_1 >= sigma_2.
        """
        avg_sigma = (sigma_x + sigma_y) / 2
        R = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        return (avg_sigma + R, avg_sigma - R)

    @staticmethod
    def calculate_von_mises(
        sigma_x: Union[float, np.ndarray],
        sigma_y: Union[float, np.ndarray],
        tau_xy: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate von Mises equivalent stress (2D plane stress): σ_vm

        σ_vm = sqrt(σ₁² − σ₁·σ₂ + σ₂²)

        Parameters
        ----------
        sigma_x, sigma_y : float or ndarray
            Normal stresses (MPa).
        tau_xy : float or ndarray
            Shear stress (MPa).

        Returns
        -------
        sigma_vm : float or ndarray
            Von Mises equivalent stress (MPa). Always non-negative.
        """
        sigma_1, sigma_2 = StressAnalysis.calculate_principal_stresses(sigma_x, sigma_y, tau_xy)
        # For 2D plane stress: sigma_3 = 0
        return np.sqrt(sigma_1 ** 2 - sigma_1 * sigma_2 + sigma_2 ** 2)


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("STATIC ANALYSIS EXAMPLES")
    print("="*70)
    
    # Example 1: Simple linear system
    print("\n1. Linear Static Analysis:")
    K = np.array([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 1]
    ], dtype=float)
    
    F = np.array([0, 0, 10], dtype=float)
    
    # Apply BC: constrain first DOF
    K[0, :] = 0
    K[:, 0] = 0
    K[0, 0] = 1
    F[0] = 0
    
    analysis = StaticAnalysis()
    u = analysis.solve(K, F)
    
    print(f"   Displacements: {u}")
    
    # Example 2: Stress calculations
    print("\n2. Stress Calculations:")
    M = 1000  # N·mm
    I = 6666.67  # mm⁴
    y = 25  # mm from neutral axis
    
    sigma = StressAnalysis.calculate_bending_stress(M, y, I)
    print(f"   Bending stress at y={y}mm: σ = {sigma:.2f} MPa")
    
    # Example 3: Strain energy
    print("\n3. Strain Energy:")
    U = analysis.calculate_strain_energy(K)
    print(f"   Total strain energy: U = {U:.4f} N·mm")
