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
    """Post-processing for stress analysis."""
    
    @staticmethod
    def calculate_bending_stress(M: float, y: float, I: float) -> float:
        """
        Calculate bending stress: σ = -M*y/I
        
        Parameters:
        -----------
        M : float
            Bending moment (N·mm)
        y : float
            Distance from neutral axis (mm)
        I : float
            Second moment of area (mm⁴)
            
        Returns:
        --------
        sigma : float
            Bending stress (MPa)
        """
        return -M * y / I
    
    @staticmethod
    def calculate_axial_stress(N: float, A: float) -> float:
        """
        Calculate axial stress: σ = N/A
        
        Parameters:
        -----------
        N : float
            Axial force (N)
        A : float
            Cross-sectional area (mm²)
            
        Returns:
        --------
        sigma : float
            Axial stress (MPa)
        """
        return N / A
    
    @staticmethod
    def calculate_shear_stress(V: float, Q: float, I: float, t: float) -> float:
        """
        Calculate shear stress: τ = VQ/(It)
        
        Parameters:
        -----------
        V : float
            Shear force (N)
        Q : float
            First moment of area (mm³)
        I : float
            Second moment of area (mm⁴)
        t : float
            Width at point (mm)
            
        Returns:
        --------
        tau : float
            Shear stress (MPa)
        """
        return V * Q / (I * t)
    
    @staticmethod
    def calculate_von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
        """
        Calculate von Mises equivalent stress.
        
        Parameters:
        -----------
        sigma_x, sigma_y : float
            Normal stresses (MPa)
        tau_xy : float
            Shear stress (MPa)
            
        Returns:
        --------
        sigma_vm : float
            von Mises stress (MPa)
        """
        return np.sqrt(sigma_x**2 + sigma_y**2 - sigma_x*sigma_y + 3*tau_xy**2)


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
