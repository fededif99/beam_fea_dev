"""
Example 1: Cantilever Beam Analysis
=====================================

Objective:
----------
Verify the static solver against analytical beam theory using a cantilever
beam under a point load.

Theory:
-------
For a cantilever beam of length L with tip load P:
    δ_max = (P*L³) / (3*E*I)

This example demonstrates:
- Setting up a simple FEA model
- Applying boundary conditions (fixed support)
- Applying loads (point load)
- Comparing FEA results with analytical solution
"""

import os

from beam_fea import (
    BeamSolver, Mesh, get_material, 
    LoadCase, BoundaryConditionSet, i_beam
)

def run_example():
    
    # 1. Define Parameters
    L = 2000.0  # mm (2 m)
    P = -10000.0  # N (10 kN, downward)
    
    # Material: Aluminum 7075-T6 (aerospace grade)
    aluminum = get_material('aluminum_7075')
    
    # Section: I-Beam (wing spar mockup)
    # IPE 200: h=200mm, b=100mm, tw=5.6mm, tf=8.5mm
    props = i_beam(
        flange_width=100,
        total_height=200,
        web_thickness=5.6,
        flange_thickness=8.5
    )
    
    # 2. Create Model
    # 20 elements provides good accuracy
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=20)
    
    solver = BeamSolver(mesh, aluminum, props)
    
    # 3. Apply Boundary Conditions & Loads
    # Fixed at x=0 (root)
    bc = BoundaryConditionSet("Cantilever Support")
    bc.fixed_support(0)
    
    # Point load at tip (node 20)
    lc = LoadCase("Tip Load 10kN")
    lc.point_load(node=20, fy=P)
    
    # 4. Solve
    solver.solve_static(lc, bc)
    
    # 5. Verification (Calculation only, results in report)
    disp_fem, node_fem = solver.get_max_deflection()
    
    # Analytical solution
    E = aluminum.E
    I = props.Iy
    disp_analytical = (P * L**3) / (3 * E * I)
    
    error = abs((disp_fem - disp_analytical) / disp_analytical) * 100
    
    # 6. Extract 3D Stress Field
    solver.calculate_stresses(num_x_points=21, num_y_points=20, num_z_points=10)
    
    # 7. Generate Report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "ex01_report.md")
    solver.generate_report(report_path)
    
    return error


if __name__ == "__main__":
    run_example()
