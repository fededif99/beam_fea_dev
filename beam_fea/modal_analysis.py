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
        from scipy.sparse import issparse
        num_dofs = K.shape[0]
        
        # Decide between sparse and dense solver
        # Dense is often more robust for small systems and handles singular M better
        use_sparse = issparse(K) and num_dofs > 1000
        
        if use_sparse:
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
            # Dense matrix logic with partitioning for robustness
            from scipy.sparse import issparse
            if issparse(K): K = K.toarray()
            if issparse(M): M = M.toarray()
            
            if bc_set is not None:
                # 1. Identify active DOFs
                all_dofs = np.arange(num_dofs)
                constrained_dofs = bc_set.get_all_constrained_dofs()
                active_dofs = [d for d in all_dofs if d not in constrained_dofs]
                
                if not active_dofs:
                    raise ValueError("All DOFs are constrained. No modal analysis possible.")
                
                # 2. Extract active submatrices
                K_act = K[np.ix_(active_dofs, active_dofs)]
                M_act = M[np.ix_(active_dofs, active_dofs)]
                
                # 3. Solve generalized eigenvalue problem for active system
                # Using 'eigh' with partitioned matrices is most stable
                eigenvalues, eigenvectors_act = eigh(K_act, M_act)
                
                # 4. Map mode shapes back to full space (zero for constrained DOFs)
                eigenvectors = np.zeros((num_dofs, len(eigenvalues)))
                for i, dof in enumerate(active_dofs):
                    eigenvectors[dof, :] = eigenvectors_act[i, :]
                    
            else:
                # No BCs - solve full system (e.g. free-free)
                eigenvalues, eigenvectors = eigh(K, M)
            
            # Extract positive eigenvalues (avoid numerical issues with precision)
            valid_idx = eigenvalues > 1e-12
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
        if num_modes is not None:
            self.frequencies = self.frequencies[:num_modes]
            self.mode_shapes = self.mode_shapes[:, :num_modes]

        # Automatic mass-normalization: phi.T @ M @ phi = I
        self.normalize_mode_shapes(M)
        
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
                                      direction: np.ndarray, mode_idx: int) -> float:
        """
        Calculate modal participation factor Γ_i.
        
        Γ_i = (φ_iᵀ M L) / (φ_iᵀ M φ_i)

        If shapes are mass-normalized, denominator is 1.0.
        """
        phi = self.mode_shapes[:, mode_idx]
        # Ensure direction is a vector of same size as M
        L = direction
        denom = phi @ M @ phi
        if abs(denom) < 1e-12: return 0.0
        return (phi @ M @ L) / denom

    def calculate_effective_modal_mass(self, M: np.ndarray,
                                      direction: np.ndarray, mode_idx: int) -> float:
        """
        Calculate effective modal mass m_eff_i.

        m_eff_i = (φ_iᵀ M L)² / (φ_iᵀ M φ_i)
        """
        phi = self.mode_shapes[:, mode_idx]
        L = direction
        num = (phi @ M @ L)**2
        den = (phi @ M @ phi)
        if abs(den) < 1e-12: return 0.0
        return num / den

    def get_modal_participation_summary(self, M: np.ndarray) -> dict:
        """
        Calculate participation factors and effective masses for all solved modes.
        Considers both X (axial) and Y (transverse) base excitation.
        """
        num_dofs = M.shape[0]
        num_modes = len(self.frequencies)

        # Influence vectors for base excitation (unit displacement at all nodes)
        L_x = np.zeros(num_dofs); L_x[0::3] = 1.0
        L_y = np.zeros(num_dofs); L_y[1::3] = 1.0

        total_mass_x = L_x @ M @ L_x
        total_mass_y = L_y @ M @ L_y

        results = {
            'total_mass_x': total_mass_x,
            'total_mass_y': total_mass_y,
            'modes': []
        }

        for i in range(num_modes):
            mx_eff = self.calculate_effective_modal_mass(M, L_x, i)
            my_eff = self.calculate_effective_modal_mass(M, L_y, i)

            results['modes'].append({
                'mode': i + 1,
                'frequency': self.frequencies[i],
                'mass_x': mx_eff,
                'mass_y': my_eff,
                'percent_x': mx_eff / total_mass_x if total_mass_x > 0 else 0,
                'percent_y': my_eff / total_mass_y if total_mass_y > 0 else 0
            })

        return results


if __name__ == "__main__":
    print("\nModal Analysis Module")
    print("Provides eigenvalue analysis for natural frequencies and mode shapes")
