import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from typing import Tuple

# Add current dir to path
sys.path.append(os.getcwd())

# Import New
from beam_fea.modal_analysis import ModalAnalysis as ModalAnalysisNew
# Import Old (copy we made)
try:
    from beam_fea.modal_analysis_old import ModalAnalysis as ModalAnalysisOld
except ImportError:
    print("Error: modal_analysis_old.py not found. Please ensure backup exists.")
    sys.exit(1)

def create_test_system(n_elements: int) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Create a simple large system for benchmarking."""
    n_nodes = n_elements + 1
    num_dofs = n_nodes * 3
    
    # Simple diagonal-heavy matrices to simulate a beam
    # This is just for timing, doesn't need to be physically perfect
    K_diag = np.random.rand(num_dofs) * 1e6 + 1e5
    M_diag = np.random.rand(num_dofs) * 10 + 1
    
    K = sp.diags([K_diag, -0.1*K_diag[:-1], -0.1*K_diag[:-1]], [0, 1, -1], format='csr')
    M = sp.diags([M_diag], [0], format='csr')
    
    return K, M

def run_benchmark(n_dofs: int, num_modes: int):
    print(f"\n--- Benchmarking System with {n_dofs} DOFs, {num_modes} modes ---")
    K, M = create_test_system(n_dofs // 3)
    
    # Dense comparison (since that's where subset_by_index helps)
    K_dense = K.toarray()
    M_dense = M.toarray()
    
    # 1. Benchmark Solve (Dense)
    print("Timing Solve (Dense)...")
    
    start = time.perf_counter()
    ma_old = ModalAnalysisOld()
    ma_old.solve(K_dense, M_dense, num_modes=num_modes)
    t_old_solve = time.perf_counter() - start
    
    start = time.perf_counter()
    ma_new = ModalAnalysisNew()
    ma_new.solve(K_dense, M_dense, num_modes=num_modes)
    t_new_solve = time.perf_counter() - start
    
    print(f"  Old Solve: {t_old_solve:.4f}s")
    print(f"  New Solve: {t_new_solve:.4f}s (Speedup: {t_old_solve/t_new_solve:.2f}x)")
    
    # 2. Benchmark Participation Summary
    print("Timing Participation Summary...")
    
    start = time.perf_counter()
    ma_old.get_modal_participation_summary(M_dense)
    t_old_part = time.perf_counter() - start
    
    start = time.perf_counter()
    ma_new.get_modal_participation_summary(M_dense)
    t_new_part = time.perf_counter() - start
    
    print(f"  Old Participation: {t_old_part:.4f}s")
    print(f"  New Participation: {t_new_part:.4f}s (Speedup: {t_old_part/t_new_part:.2f}x)")

if __name__ == "__main__":
    # Test cases
    # 1. Medium system (Dense path)
    run_benchmark(n_dofs=300, num_modes=10)
    
    # 2. Larger system (Dense path, subset extraction)
    run_benchmark(n_dofs=900, num_modes=20)
    
    # 3. Large system (Sparse path)
    # The new sparse path has better vectorized post-processing
    run_benchmark(n_dofs=3000, num_modes=50)
