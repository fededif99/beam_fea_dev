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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beam_fea import (
    BeamSolver, MeshGenerator, Material, 
    LoadCase, BoundaryConditionSet
)
from beam_fea.cross_sections import i_beam

def run_example():
    print("Running Example 1: Cantilever Beam Verification")
    print("-----------------------------------------------")
    
    # 1. Define Parameters
    L = 2000.0  # mm (2 m)
    P = -10000.0  # N (10 kN, downward)
    
    # Material: Aluminum 7075-T6 (aerospace grade)
    aluminum = Material("Aluminum 7075-T6", E=71700, nu=0.33, rho=2.81e-6, yield_strength=503)
    
    # Section: I-Beam (wing spar mockup)
    # IPE 200: h=200mm, b=100mm, tw=5.6mm, tf=8.5mm
    props = i_beam(
        flange_width=100,
        total_height=200,
        web_thickness=5.6,
        flange_thickness=8.5
    )
    print(f"Section Properties:\n  I_y = {props.Iy:.2f} mm^4\n  Area = {props.A:.2f} mm^2")
    
    # 2. Create Model
    # 20 elements provides good accuracy
    mesh = MeshGenerator.beam_mesh_1d(L, num_elements=20)
    
    solver = BeamSolver(mesh, aluminum, props)
    
    # 3. Apply Boundary Conditions & Loads
    # Fixed at x=0 (root)
    bc = BoundaryConditionSet("Cantilever Support")
    bc.fixed_support(0)
    
    # Point load at tip (node 20)
    lc = LoadCase("Tip Load 10kN")
    lc.point_load(node=20, fy=P)
    
    # 4. Solve
    print("Solving...")
    solver.solve_static(lc, bc)
    
    # 5. Verification
    disp_fem, node_fem = solver.get_max_deflection()
    
    # Analytical solution
    E = aluminum.E
    I = props.Iy
    disp_analytical = (P * L**3) / (3 * E * I)
    
    error = abs((disp_fem - disp_analytical) / disp_analytical) * 100
    
    print(f"Max Deflection (FEA): {disp_fem:.4f} mm at node {node_fem}")
    print(f"Max Deflection (Analytical): {disp_analytical:.4f} mm")
    print(f"Error: {error:.4f}%")
    
    # 6. Generate Report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "ex01_report.md")
    solver.generate_report(report_path)
    print(f"Report generated: {report_path}")
    
    return error


if __name__ == "__main__":
    run_example()
