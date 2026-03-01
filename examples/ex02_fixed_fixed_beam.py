"""
Example 2: Fixed-Fixed Beam with Uniform Load
===============================================

Objective:
----------
Analyze a beam fixed at both ends under a uniformly distributed load.

Theory:
-------
For a fixed-fixed beam of length L with uniform load w (N/mm):
    δ_max = (w*L⁴) / (384*E*I)  
    M_max = (w*L²) / 12  (at supports)

This example demonstrates:
- Fixed boundary conditions at both ends
- Uniformly distributed load application
- Verification against classical beam theory
"""

import os

from beam_fea import (
    BeamSolver, MeshGenerator, get_material, 
    LoadCase, BoundaryConditionSet, hollow_circular
)

def run_example():
    print(f"[*] Running {os.path.basename(__file__)}...")
    
    # 1. Parameters
    L = 3000.0  # mm (3 m)
    w = -5.0    # N/mm (5 N/mm downward)
    
    # Material: Titanium Ti-6Al-4V (aerospace tubing)
    titanium = get_material('ti_6al_4v')
    
    # Section: Hollow circular tube (D=80mm, t=5mm)
    # Common for aerospace structures
    props = hollow_circular(outer_diameter=80, thickness=5)
    
    # 2. Create Model
    num_elems = 30
    mesh = MeshGenerator.beam_mesh_1d(L, num_elems)
    solver = BeamSolver(mesh, titanium, props)
    
    # 3. Boundary Conditions: Fixed at both ends
    bc = BoundaryConditionSet("Fixed-Fixed")
    bc.fixed_support(0)
    bc.fixed_support(num_elems)
    
    # 4. Load: Uniform distributed load
    lc = LoadCase("Uniform Load")
    lc.uniform_load(list(range(num_elems)), wy=w)
    
    # 5. Solve
    print("  Solving...")
    solver.solve_static(lc, bc)
    
    # 6. Verification (Results in report)
    solver.get_max_deflection()
    
    # 7. Extract 3D Stress Field
    # Analyzing stresses at 31 points along length (including nodes and center)
    solver.calculate_stresses(num_x_points=31, num_y_points=20, num_z_points=20)
    
    # 8. Generate Report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "ex02_report.md")
    solver.generate_report(report_path)
    print(f"[SUCCESS] Report generated: {report_path}")
    
    return None


if __name__ == "__main__":
    run_example()
