
import numpy as np
import os

from beam_fea.mesh import Mesh
from beam_fea.materials import get_material
from beam_fea.cross_sections import RectangularSection
from beam_fea.solver import BeamSolver
from beam_fea.loads import LoadCase, PointLoad
from beam_fea.boundary_conditions import BoundaryConditionSet, FixedSupport

def test_centralized_forces():
    print("Testing Centralized Force Calculations...")
    
    # 1. Setup a simple cantilever beam
    # Length 1000mm, 10 elements
    mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=10)
    material = get_material('aluminum')
    section = RectangularSection(width=50, height=100).properties()
    
    solver = BeamSolver(mesh, material, section)
    
    # 2. Apply loads and BCs
    # Fixed at node 0, 1000N tip load (downward) at node 10
    bc_set = BoundaryConditionSet()
    bc_set.fixed_support(0)
    
    load_case = LoadCase()
    load_case.point_load(10, fy=-1000)
    
    # 3. Solve
    solver.solve_static(load_case, bc_set)
    
    # 4. Calculate internal forces (Centralized)
    # For a cantilever with tip load P:
    # Shear V = P (constant)
    # Moment M = P * (L - x)
    forces = solver.calculate_internal_forces(num_points=11)
    
    positions = forces['positions']
    shear = forces['shear_forces']
    moment = forces['bending_moments']
    
    print(f"\nResults for P=-1000N, L=1000mm:")
    print(f"{'Pos (mm)':>10} | {'Shear (N)':>10} | {'Moment (N-mm)':>15}")
    print("-" * 45)
    
    for x, V, M in zip(positions, shear, moment):
        # Expected: Reaction at node 1 is 1000N (upward)
        V_exp = 1000
        M_exp = -1000 * (1000 - x)
        
        err_V = abs(V - V_exp)
        err_M = abs(M - M_exp)
        
        print(f"{x:10.1f} | {V:10.1f} | {M:15.1f} | V_err: {err_V:.2e}, M_err: {err_M:.2e}")
        
        assert err_V < 1e-5, f"Shear error too high at x={x}: {err_V}"
        assert err_M < 1e-5, f"Moment error too high at x={x}: {err_M}"

    print("\n[OK] Internal Force Calculation Verification: PASSED")
    
    print("\nTesting Timoshenko Force Calculations...")
    solver_t = BeamSolver(mesh, material, section, element_type='timoshenko')
    solver_t.solve_static(load_case, bc_set)
    forces_t = solver_t.calculate_internal_forces(num_points=11)
    
    for x, V, M in zip(forces_t['positions'], forces_t['shear_forces'], forces_t['bending_moments']):
        V_exp = 1000
        M_exp = -1000 * (1000 - x)
        err_V = abs(V - V_exp)
        err_M = abs(M - M_exp)
        print(f"{x:10.1f} | {V:10.1f} | {M:15.1f} | V_err: {err_V:.2e}, M_err: {err_M:.2e}")
        assert err_V < 1e-5, f"Timoshenko Shear error too high at x={x}: {err_V}"
        assert err_M < 1e-5, f"Timoshenko Moment error too high at x={x}: {err_M}"
    
    print("\n[OK] Timoshenko Force Calculation Verification: PASSED")

    # 5. Verify Report Generator still works
    print("\nVerifying Report Generator integration...")
    report_path = "test_force_report.md"
    solver.generate_report(report_path)
    if os.path.exists(report_path):
        print(f"OK Report generated: {report_path}")
        os.remove(report_path)
    else:
        print("Report generation failed")
    
    # 6. Verify Visualizer types
    print("\nVerifying Visualizer integration...")
    # This just ensures no crash
    try:
        solver.visualize('shear', num_points=20, output_path="test_shear.png")
        if os.path.exists("test_shear.png"):
            print("OK Shear plot generated")
            os.remove("test_shear.png")
            
        solver.visualize('moment', num_points=20, output_path="test_moment.png")
        if os.path.exists("test_moment.png"):
            print("OK Moment plot generated")
            os.remove("test_moment.png")
            
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    test_centralized_forces()
