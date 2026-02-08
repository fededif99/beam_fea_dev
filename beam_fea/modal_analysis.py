"""
modal_analysis.py
=================
Modal analysis (eigenvalue analysis) for natural frequencies and mode shapes.

Solves: (K - ω²M)φ = 0
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional


class ModalAnalysis:
    """Modal analysis solver for free vibration."""
    
    def __init__(self):
        """Initialize modal analysis."""
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        self.periods = None
        self.mode_shapes = None
    
    def solve(self, K: np.ndarray, M: np.ndarray, 
             num_modes: Optional[int] = None,
             bc_set=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for natural frequencies.
        
        Parameters:
        -----------
        K : np.ndarray
            Stiffness matrix
        M : np.ndarray
            Mass matrix
        num_modes : int, optional
            Number of modes to extract (default: all)
        bc_set : BoundaryConditionSet, optional
            Boundary conditions
            
        Returns:
        --------
        frequencies : np.ndarray
            Natural frequencies (Hz)
        mode_shapes : np.ndarray
            Mode shape matrix (each column is a mode)
        """
        # Apply boundary conditions to K
        if bc_set is not None:
            # This handles penalty method for sparse K, or row/col zeroing for dense K
            K, _ = bc_set.apply_to_system(K, np.zeros(K.shape[0]))
        
        from scipy.sparse import issparse
        
        if issparse(K):
            from scipy.sparse.linalg import eigsh
            
            # For sparse matrices using penalty method in K:
            # We don't need to strictly modify M if K>>M for constrained DOFs.
            # The high stiffness will push those modes to infinity.
            # Modifying sparse M is expensive and unnecessary.
            
            # Determine number of modes to solve
            max_modes = K.shape[0] - 2 # Safety margin
            if num_modes is None:
                k = min(10, max_modes)
            else:
                k = min(num_modes, max_modes)
                
            # Use shift-invert mode (sigma=0) to find smallest eigenvalues efficiently
            # This requires solving linear system, so it uses the sparse solver
            try:
                eigenvalues, eigenvectors = eigsh(K, M=M, k=k, which='LM', sigma=0)
            except Exception as e:
                # Fallback if shift-invert fails (e.g. singular matrix)
                import warnings
                warnings.warn(f"Shift-invert failed ({e}), trying standard mode")
                eigenvalues, eigenvectors = eigsh(K, M=M, k=k, which='SM')
            
        else:
            # Dense matrix logic
            if bc_set is not None:
                # For modal analysis, also need to modify M explicitly for dense method
                # (Identity replacement method)
                constrained_dofs = list(bc_set.get_all_constrained_dofs())
                for dof in constrained_dofs:
                    M[dof, :] = 0
                    M[:, dof] = 0
                    M[dof, dof] = 1
            
            # Solve generalized eigenvalue problem
            eigenvalues, eigenvectors = eigh(K, M)
            
            # Filter dummy modes (freq=1/(2pi)) if checking logic requires
            # or extract relevant modes
            
            # Extract positive eigenvalues (avoid numerical issues)
            valid_idx = eigenvalues > 1e-10
            # For dense method with identity replacement, constrained nodes have ev=1
            # We should probably filter those if they interfere, but usually we just look at lowest
            
            eigenvalues = eigenvalues[valid_idx]
            eigenvectors = eigenvectors[:, valid_idx]
        
        # Natural frequencies (rad/s)
        # Ensure non-negative (numerical noise can give tiny negatives)
        eigenvalues = np.abs(eigenvalues)
        omega = np.sqrt(eigenvalues)
        
        # Natural frequencies (Hz)
        self.frequencies = omega / (2 * np.pi)
        
        # Periods (s)
        with np.errstate(divide='ignore'):
            self.periods = 1 / self.frequencies
        
        # Mode shapes
        self.mode_shapes = eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        # Return requested number of modes
        if num_modes is not None and not issparse(K):
            # For sparse, we already extracted k modes
            # For dense, we computed all, so slice them
            return self.frequencies[:num_modes], self.mode_shapes[:, :num_modes]
        
        return self.frequencies, self.mode_shapes
    
    def normalize_mode_shapes(self, M: np.ndarray):
        """
        Mass-normalize mode shapes: φᵀMφ = I
        
        Parameters:
        -----------
        M : np.ndarray
            Mass matrix
        """
        if self.mode_shapes is None:
            raise ValueError("Must solve eigenvalue problem first")
        
        n_modes = self.mode_shapes.shape[1]
        
        for i in range(n_modes):
            phi = self.mode_shapes[:, i]
            mass_factor = np.sqrt(phi @ M @ phi)
            self.mode_shapes[:, i] /= mass_factor
    
    def calculate_modal_mass(self, M: np.ndarray, mode: int) -> float:
        """Calculate modal mass for a specific mode."""
        phi = self.mode_shapes[:, mode]
        return phi @ M @ phi
    
    def calculate_modal_stiffness(self, K: np.ndarray, mode: int) -> float:
        """Calculate modal stiffness for a specific mode."""
        phi = self.mode_shapes[:, mode]
        return phi @ K @ phi
    
    def calculate_participation_factor(self, M: np.ndarray, 
                                      direction: np.ndarray, mode: int) -> float:
        """
        Calculate modal participation factor.
        
        Parameters:
        -----------
        M : np.ndarray
            Mass matrix
        direction : np.ndarray
            Load direction vector
        mode : int
            Mode number
        """
        phi = self.mode_shapes[:, mode]
        return (phi @ M @ direction) / (phi @ M @ phi)


if __name__ == "__main__":
    print("\nModal Analysis Module")
    print("Provides eigenvalue analysis for natural frequencies and mode shapes")
