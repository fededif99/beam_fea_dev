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
    BeamSolver, Mesh, get_material, 
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
    # Coarse mesh (5 elements) demonstrating high-accuracy force recovery
    num_elems = 5
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=num_elems)
    solver = BeamSolver(mesh, titanium, props)
    
    # 3. Boundary Conditions: Fixed at both ends
    bc = BoundaryConditionSet("Fixed-Fixed")
    bc.fixed_support(0)
    bc.fixed_support(num_elems)
    
    # 4. Load: Uniform distributed load
    lc = LoadCase("Uniform Load")
    lc.distributed_load(element=list(range(num_elems)), distribution='uniform', wy=w)
    
    # 5. Solve
    print("  Solving...")
    solver.solve_static(lc, bc)
    
    # 6. Verification (Results in report)
    solver.get_max_deflection()
    
    # 7. Extract 3D Stress Field
    # High-resolution stress recovery (50 points) from 5-element mesh
    solver.calculate_stresses(num_x_points=50, num_y_points=20, num_z_points=20)
    
    # 8. Generate Report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "ex02_report.md")
    solver.generate_report(report_path)
    print(f"[SUCCESS] Report generated: {report_path}")
    
    return None


if __name__ == "__main__":
    run_example()
