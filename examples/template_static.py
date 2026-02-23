"""
Template: Linear Static Analysis
================================
This template demonstrates a standard workflow for solving a beam problem
using linear static analysis (displacements, forces, and stresses).

Workflow: 
1. Define Material & Cross-Section
2. Generate Mesh
3. Apply Loads & Boundary Conditions
4. Solve Analysis
5. Retrieve Results & Generate Report
"""

import os
import sys

# Ensure beam_fea is in the path (if running from repository root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beam_fea import (
    BeamSolver, MeshGenerator, get_material, 
    LoadCase, BoundaryConditionSet
)
from beam_fea.cross_sections import rectangular

def main():
    # 1. MATERIAL & SECTION
    # Using a predefined library material (e.g., 'steel', 'aluminium')
    material = get_material("aluminium_7075")
    
    # Defining a rectangular section: width=100mm, height=200mm
    section = rectangular(width=100.0, height=200.0)

    # 2. MESH GENERATION
    # Create a 2000mm beam with 20 elements
    length = 2000.0
    num_elements = 20
    mesh = MeshGenerator.beam_mesh_1d(length, num_elements)
    print(f"[*] Mesh created: {mesh.num_nodes} nodes, {mesh.num_elements} elements")

    # 3. LOADS
    # Create a load case and add a point load at the tip (last node)
    lc = LoadCase("Tip Load Case")
    lc.point_load(node=mesh.num_nodes - 1, fy=-5000.0)  # 5 kN downwards
    
    # Add a uniform load of 2 N/mm across all elements
    lc.uniform_load(element=list(range(mesh.num_elements)), wy=-2.0)

    # 4. BOUNDARY CONDITIONS
    # Create a BC set and fix the left end (node 0)
    bc = BoundaryConditionSet("Fixed Support")
    bc.fixed_support(node=0)

    # 5. SOLVE
    # Initialize solver and run static analysis
    solver = BeamSolver(mesh, material, section, element_type='euler')
    solver.solve_static(lc, bc)
    
    # 6. RESULTS
    disp_max, node_idx = solver.get_max_deflection()
    print(f"[SUCCESS] Analysis Complete.")
    print(f"  - Max Deflection: {disp_max:.4f} mm at x = {mesh.nodes[node_idx].x:.1f} mm")

    # 7. (Optional) STRESS & REPORT
    # Calculate detailed 3D stresses
    stresses = solver.calculate_stresses(num_x_points=50)
    print(f"  - Peak von Mises Stress: {stresses['von_mises'].max():.2f} MPa")
    
    # Generate an automated Markdown report
    report_path = os.path.join(os.path.dirname(__file__), "static_template_results.md")
    solver.generate_report(report_path)
    print(f"[*] Report saved to: {report_path}")

if __name__ == "__main__":
    main()
