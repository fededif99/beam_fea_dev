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
from typing import Tuple, Optional
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class StaticAnalysis:
    """Static linear analysis solver."""
    
    def __init__(self, use_sparse: bool = False):
        """
        Initialize static analysis.
        
        Parameters:
        -----------
        use_sparse : bool
            Use sparse matrix solver (for large systems)
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
        
        # Check if sparse
        if issparse(K) or self.use_sparse:
            from scipy.sparse.linalg import spsolve
            # Skip condition number check for sparse matrices (too expensive)
            self.displacements = spsolve(K, F)
        else:
            # Check for singularity (dense only)
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
        # Works for both dense and sparse K
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
    
    def calculate_element_forces(self, elem_K: np.ndarray, 
                                 elem_dofs: np.ndarray) -> np.ndarray:
        """
        Calculate internal forces in an element.
        
        Parameters:
        -----------
        elem_K : np.ndarray
            Element stiffness matrix
        elem_dofs : np.ndarray
            Element DOF indices
            
        Returns:
        --------
        forces : np.ndarray
            Internal element forces
        """
        if self.displacements is None:
            raise ValueError("Must solve system first")
        
        # Extract element displacements
        u_elem = self.displacements[elem_dofs]
        
        # F_elem = K_elem * u_elem
        forces = elem_K @ u_elem
        
        return forces


class NonlinearStaticAnalysis:
    """Nonlinear static analysis using Newton-Raphson iteration."""
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        """
        Initialize nonlinear analysis.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.displacements = None
        self.converged = False
        self.num_iterations = 0
    
    def solve_newton_raphson(self, K_func, F: np.ndarray,
                            u0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve nonlinear system using Newton-Raphson.
        
        Parameters:
        -----------
        K_func : callable
            Function that returns tangent stiffness matrix K(u)
        F : np.ndarray
            Applied force vector
        u0 : np.ndarray, optional
            Initial guess (default: zero)
            
        Returns:
        --------
        u : np.ndarray
            Displacement vector
        """
        n = len(F)
        u = u0 if u0 is not None else np.zeros(n)
        
        for iteration in range(self.max_iterations):
            # Get tangent stiffness at current displacement
            K_tangent = K_func(u)
            
            # Residual: R = F - K(u)*u
            R = F - K_tangent @ u
            
            # Check convergence
            norm_R = np.linalg.norm(R)
            if norm_R < self.tolerance:
                self.converged = True
                self.num_iterations = iteration + 1
                break
            
            # Solve for displacement increment: K_tangent * du = R
            du = np.linalg.solve(K_tangent, R)
            
            # Update displacement
            u += du
        
        if not self.converged:
            warnings.warn(
                f"Newton-Raphson did not converge in {self.max_iterations} iterations. "
                f"Final residual norm: {norm_R:.2e}"
            )
        
        self.displacements = u
        return u
    
    def solve_load_increment(self, K_func, F_total: np.ndarray,
                            num_steps: int = 10) -> list:
        """
        Solve using incremental loading.
        
        Parameters:
        -----------
        K_func : callable
            Stiffness matrix function
        F_total : np.ndarray
            Total applied force
        num_steps : int
            Number of load steps
            
        Returns:
        --------
        results : list
            List of displacement vectors at each step
        """
        n = len(F_total)
        u = np.zeros(n)
        results = []
        
        dF = F_total / num_steps
        
        for step in range(num_steps):
            F_step = (step + 1) * dF
            
            # Use previous solution as initial guess
            u = self.solve_newton_raphson(K_func, F_step, u0=u)
            results.append(u.copy())
        
        return results


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


class InfluenceLineAnalysis:
    """Influence line analysis for beams."""
    
    @staticmethod
    def calculate_influence_line(mesh, bc_set, node: int, dof: int,
                                num_positions: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate influence line for a specific DOF.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh
        bc_set : BoundaryConditionSet
            Boundary conditions
        node : int
            Node where response is measured
        dof : int
            DOF index (0=u, 1=v, 2=θ)
        num_positions : int
            Number of load positions
            
        Returns:
        --------
        positions : np.ndarray
            Load positions along beam
        responses : np.ndarray
            Response values at each position
        """
        # This is a simplified example
        # Full implementation would require assembly and solving for each load position
        
        total_length = mesh.nodes[-1].x
        positions = np.linspace(0, total_length, num_positions)
        responses = np.zeros(num_positions)
        
        # Would implement unit load at each position and record response
        
        return positions, responses


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
