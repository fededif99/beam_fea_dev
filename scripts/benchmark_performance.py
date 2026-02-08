"""
Benchmark Performance Script
============================

Measures execution time for:
1. Mesh generation
2. Matrix assembly
3. Static solution
4. Modal solution

Compares performance across different mesh sizes.
"""

import time
import numpy as np
import sys
import os

# Ensure we import from the local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beam_fea import BeamSolver, MeshGenerator, get_material, rectangular, LoadCase, BoundaryConditionSet

def run_benchmark(num_elements):
    print(f"\nBenchmark: {num_elements} elements")
    print("-" * 30)
    
    # 1. Setup
    start_time = time.perf_counter()
    mesh = MeshGenerator.beam_mesh_1d(length=1000, num_elements=num_elements)
    material = get_material('steel')
    section = rectangular(50, 100)
    solver = BeamSolver(mesh, material, section)
    bc_set = BoundaryConditionSet("Benchmark Supports")
    bc_set.pinned_support(0)
    bc_set.pinned_support(num_elements)
    
    load_case = LoadCase("Benchmark Load")
    load_case.point_load(node=num_elements//2, fy=-1000)
    setup_time = time.perf_counter() - start_time
    print(f"Setup:        {setup_time:.4f} s")
    
    # 2. Assembly
    start_time = time.perf_counter()
    solver.assemble_global_matrices()
    assembly_time = time.perf_counter() - start_time
    print(f"Assembly:     {assembly_time:.4f} s")
    
    # 3. Static Solve
    start_time = time.perf_counter()
    solver.solve_static(load_case, bc_set)
    static_time = time.perf_counter() - start_time
    print(f"Static Solve: {static_time:.4f} s")
    
    # 4. Modal Solve (First 5 modes)
    start_time = time.perf_counter()
    solver.solve_modal(bc_set, num_modes=5)
    modal_time = time.perf_counter() - start_time
    print(f"Modal Solve:  {modal_time:.4f} s")
        
    total_time = setup_time + assembly_time + static_time + modal_time
    return assembly_time, static_time, modal_time

if __name__ == "__main__":
    print("Running Performance Benchmark (Optimized)...")
    print("============================================")
    
    sizes = [100, 1000, 5000, 10000]
    results = []
    
    for size in sizes:
        try:
            res = run_benchmark(size)
            results.append((size, *res))
        except Exception as e:
            print(f"Failed at N={size}: {e}")
            break
            
    print("\nSummary Results")
    print("="*75)
    print(f"{'Elements':<10} | {'Assembly (s)':<15} | {'Static (s)':<15} | {'Modal (s)':<15}")
    print("-" * 75)
    for size, t_asm, t_static, t_modal in results:
        print(f"{size:<10} | {t_asm:<15.4f} | {t_static:<15.4f} | {t_modal:<15.4f}")
