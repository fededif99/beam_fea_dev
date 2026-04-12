import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, Dict, Any


class ModalAnalysis:
    """Modal analysis solver for free vibration."""
    
    # Threshold for switching to sparse solver
    SPARSE_THRESHOLD = 500  # Based on 1D beam bandwidth efficiency

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
        Solve generalized eigenvalue problem for natural frequencies.
        
        Solves: (K - ω²M)φ = 0
        """
        from scipy.sparse import issparse
        num_dofs = K.shape[0]
        
        # Default to 10 modes if not specified
        if num_modes is None:
            num_modes = 10
            
        # Determine solver type based on matrix type and size
        use_sparse = issparse(K) and num_dofs > self.SPARSE_THRESHOLD
        
        # 1. Boundary Condition Handling via Partitioning (Elimination Method)
        active_indices = None
        if bc_set is not None:
            constrained_dofs = bc_set.get_all_constrained_dofs()
            active_mask = np.ones(num_dofs, dtype=bool)
            active_mask[list(constrained_dofs)] = False
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) == 0:
                raise ValueError("All DOFs constrained. No modal analysis possible.")
                
            K_act = K[np.ix_(active_indices, active_indices)]
            M_act = M[np.ix_(active_indices, active_indices)]
        else:
            K_act = K
            M_act = M
            
        num_act_dofs = K_act.shape[0]

        # 2. Eigenvalue Solver Execution
        if use_sparse:
            # Determine number of modes to solve
            max_modes = num_act_dofs - 2
            k = min(num_modes, max_modes) if max_modes > 0 else num_act_dofs
            k = max(1, k) # safety fallback
                
            # Use shift-invert mode (sigma=0) for efficient low-frequency extraction
            try:
                eigenvalues, eigenvectors_act = eigsh(K_act, M=M_act, k=k, which='LM', sigma=0)
            except Exception as e:
                import warnings
                warnings.warn(f"Sparse shift-invert failed ({e}), trying standard SM mode")
                eigenvalues, eigenvectors_act = eigsh(K_act, M=M_act, k=k, which='SM')
            
        else:
            # Dense solver path
            if issparse(K_act): K_act = K_act.toarray()
            if issparse(M_act): M_act = M_act.toarray()
            
            # Use subset_by_index for dense solver if num_modes is requested
            subset = [0, min(num_modes, num_act_dofs) - 1] if num_modes else None
            try:
                eigenvalues, eigenvectors_act = eigh(K_act, M_act, subset_by_index=subset)
            except Exception as e:
                # Fallback to full decomposition if subset fails
                eigenvalues, eigenvectors_act = eigh(K_act, M_act)
                if num_modes:
                    eigenvalues = eigenvalues[:num_modes]
                    eigenvectors_act = eigenvectors_act[:, :num_modes]
            
        # 3. Back-map eigenvectors to full model space
        if active_indices is not None:
            eigenvectors = np.zeros((num_dofs, len(eigenvalues)))
            eigenvectors[active_indices, :] = eigenvectors_act
        else:
            eigenvectors = eigenvectors_act
            
        # Post-process results
        # Ensure positive eigenvalues (handle numerical noise/rigid body modes)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(eigenvalues)
        
        self.frequencies = omega / (2 * np.pi)
        with np.errstate(divide='ignore'):
            self.periods = 1 / self.frequencies
        
        self.mode_shapes = eigenvectors
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        # Mass-normalize all extracted modes at once
        self.normalize_mode_shapes(M)
        
        return self.frequencies, self.mode_shapes
    
    def normalize_mode_shapes(self, M: np.ndarray):
        """Mass-normalize mode shapes: φᵀMφ = I (Vectorized)"""
        if self.mode_shapes is None:
            raise ValueError("Must solve eigenvalue problem first")
        
        # Calculate modal masses: diag(Phi.T @ M @ Phi) (Vectorized)
        # Avoid explicit large dense matrix multiplication if M is sparse
        modal_masses = np.sum(self.mode_shapes * (M @ self.mode_shapes), axis=0)
        mass_factors = np.sqrt(np.maximum(modal_masses, 0.0))
        
        # Avoid division by zero for rigid body modes with zero mass
        mass_factors[mass_factors < 1e-15] = 1.0
        self.mode_shapes /= mass_factors
    
    def calculate_modal_mass(self, M: np.ndarray, mode: int) -> float:
        """Calculate modal mass for a specific mode."""
        phi = self.mode_shapes[:, mode]
        return float(phi @ M @ phi)
    
    def calculate_modal_stiffness(self, K: np.ndarray, mode: int) -> float:
        """Calculate modal stiffness for a specific mode."""
        phi = self.mode_shapes[:, mode]
        return float(phi @ K @ phi)
    
    def calculate_participation_factor(self, M: np.ndarray, 
                                      direction: np.ndarray, mode_idx: int) -> float:
        """Calculate modal participation factor Γ_i = (φ_iᵀ M L) / (φ_iᵀ M φ_i)"""
        phi = self.mode_shapes[:, mode_idx]
        denom = phi @ M @ phi
        if abs(denom) < 1e-12: return 0.0
        return float((phi @ M @ direction) / denom)

    def get_modal_participation_summary(self, M: np.ndarray) -> Dict[str, Any]:
        """
        Calculate participation factors and effective masses (Vectorized).
        """
        if self.frequencies is None:
            raise ValueError("Must solve eigenvalue problem first")
            
        num_dofs = M.shape[0]
        num_modes = len(self.frequencies)

        # Influence vectors for base excitation
        L_x = np.zeros(num_dofs); L_x[0::3] = 1.0
        L_y = np.zeros(num_dofs); L_y[1::3] = 1.0

        total_mass_x = float(L_x @ M @ L_x)
        total_mass_y = float(L_y @ M @ L_y)

        # Vectorized Modal Mass (Denominator)
        m_modes = np.sum(self.mode_shapes * (M @ self.mode_shapes), axis=0)
        
        # Vectorized Numerators: (Phi.T @ (M @ L))**2
        numer_x = (self.mode_shapes.T @ (M @ L_x))**2
        numer_y = (self.mode_shapes.T @ (M @ L_y))**2
        
        m_eff_x = numer_x / m_modes
        m_eff_y = numer_y / m_modes

        results = {
            'total_mass_x': total_mass_x,
            'total_mass_y': total_mass_y,
            'modes': []
        }

        for i in range(num_modes):
            results['modes'].append({
                'mode': i + 1,
                'frequency': float(self.frequencies[i]),
                'mass_x': float(m_eff_x[i]),
                'mass_y': float(m_eff_y[i]),
                'percent_x': float(m_eff_x[i] / total_mass_x) if total_mass_x > 0 else 0.0,
                'percent_y': float(m_eff_y[i] / total_mass_y) if total_mass_y > 0 else 0.0
            })

        return results


if __name__ == "__main__":
    print("\nModal Analysis Module (Optimized)")
    print("Provides high-performance eigenvalue analysis")
