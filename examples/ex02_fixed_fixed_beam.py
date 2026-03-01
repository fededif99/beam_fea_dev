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
    BeamSolver, MeshGenerator, Material, 
    LoadCase, BoundaryConditionSet
)
from beam_fea.cross_sections import hollow_circular

def run_example():
    print("Running Example 2: Fixed-Fixed Beam with UDL")
    print("--------------------------------------------")
    
    # 1. Parameters
    L = 3000.0  # mm (3 m)
    w = -5.0    # N/mm (5 N/mm downward)
    
    # Material: Titanium Ti-6Al-4V (aerospace tubing)
    titanium = Material(
        "Titanium Ti-6Al-4V",
        E=113800,
        nu=0.34,
        rho=4.43e-6,
        yield_strength=880
    )
    
    # Section: Hollow circular tube (D=80mm, t=5mm)
    # Common for aerospace structures
    props = hollow_circular(outer_diameter=80, thickness=5)
    print(f"Section Properties:")
    print(f"  I_y = {props.Iy:.2f} mm^4")
    print(f"  Area = {props.A:.2f} mm^2")
    
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
    print("Solving...")
    solver.solve_static(lc, bc)
    
    # 6. Verification
    disp_fem, node_fem = solver.get_max_deflection()
    
    # Analytical solution (center deflection)
    E = titanium.E
    I = props.Iy
    disp_analytical = (w * L**4) / (384 * E * I)
    
    error = abs((disp_fem - disp_analytical) / disp_analytical) * 100
    
    print(f"\nResults:")
    print(f"Max Deflection (FEA): {disp_fem:.4f} mm at node {node_fem}")
    print(f"Max Deflection (Analytical): {disp_analytical:.4f} mm")
    print(f"Error: {error:.4f}%")
    
    # Calculate maximum moment
    moment_analytical = abs(w * L**2 / 12)
    print(f"\nExpected Max Moment: {moment_analytical/1000:.2f} kN·m (at supports)")
    
    # 7. Extract 3D Stress Field
    print("\nCalculating 3D Stress Field...")
    # Analyzing stresses at 31 points along length (including nodes and center)
    stresses = solver.calculate_stresses(num_x_points=31, num_y_points=20, num_z_points=20)
    
    import numpy as np
    max_bend = np.max(np.abs(stresses['bending']))
    max_shear = np.max(np.abs(stresses['shear']))
    max_vm = np.max(stresses['von_mises'])
    
    print(f"  Maximum Bending Stress: {max_bend:.2f} MPa")
    print(f"  Maximum Shear Stress:   {max_shear:.2f} MPa")
    print(f"  Peak von Mises Stress:  {max_vm:.2f} MPa")
    
    # 8. Generate Report
    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(base_dir, "ex02_report.md")
    solver.generate_report(report_path)
    print(f"\nReport generated: {report_path}")
    
    return error


if __name__ == "__main__":
    run_example()
