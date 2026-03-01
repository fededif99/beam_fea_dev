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
import numpy as np


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
    length = 750
    num_elements = 50
    mesh = MeshGenerator.beam_mesh_1d(length, num_elements)
    print(f"[*] Mesh created: {mesh.num_nodes} nodes, {mesh.num_elements} elements")

    # 3. LOADS
    # Create a load case and add a point load at the tip (last node)
    lc = LoadCase("Center Load")
    lc.point_load(x=length/2, fy=-1500000.0)  # 1500 kN downwards 
    

    # 4. BOUNDARY CONDITIONS
    bc = BoundaryConditionSet("Pin - Pin")
    bc.pinned_support(node=0)
    bc.pinned_support(node=mesh.num_nodes -1 ) 


    # 5. SOLVE
    # Initialize solver and run static analysis
    solver = BeamSolver(mesh, material, section, element_type='euler')
    solver.solve_static(lc, bc)
    
    # 6. RESULTS
    disp_max, node_idx = solver.get_max_deflection()
    print(f"\n" + "="*50)
    print(f"ANALYSIS RESULTS: {lc.name}")
    print(f"="*50)
    print(f"Moment of inertia: {section.Iy:.2e} mm^4")
    
    # 7. PEAK INTERNAL FORCES
    # Uses default resolution (num_nodes)
    internal = solver.get_max_internal_forces()
    max_moment = internal['moment']['value']
    max_moment_pos = internal['moment']['x']
    print(f"Max Bending Moment: {max_moment/1000:.2f} kN·mm at x = {max_moment_pos:.1f} mm")
    print(f"Max Deflection: {abs(disp_max):.4f} mm at x = {mesh.nodes[node_idx].x:.1f} mm")

    # 7. REACTION FORCES
    print(f"\nREACTION FORCES:")
    print(f"{'Node':<6} | {'Fx (N)':<12} | {'Fy (N)':<12} | {'Mz (N·mm)':<12}")
    print("-" * 50)
    
    # Track which nodes have reactions to avoid duplicates
    support_nodes = sorted(list(set(bc.node for bc in bc.conditions if hasattr(bc, 'node'))))
    for node_id in support_nodes:
        # Reactions are stored as K*u - F for all DOFs
        rx = solver.reactions[3*node_id]
        ry = solver.reactions[3*node_id + 1]
        rm = solver.reactions[3*node_id + 2]
        print(f"{node_id:<6} | {rx:>12.2f} | {ry:>12.2f} | {rm:>12.2f}")
    
    # 8. STRESS ANALYSIS
    # Calculate detailed 3D stresses using default resolution (num_nodes)
    stresses = solver.calculate_stresses()
    
    print(f"\nPEAK STRESSES (MPa):")
    print(f"  - Peak Bending:   {np.max(np.abs(stresses['bending'])):.2f}")
    print(f"  - Peak Shear:     {np.max(np.abs(stresses['shear'])):.2f}")
    print(f"  - Max Principal:  {np.max(stresses['sigma_1']):.2f}")
    print(f"  - Min Principal:  {np.min(stresses['sigma_2']):.2f}")
    print(f"  - Peak von Mises: {stresses['von_mises'].max():.2f}")
    
    # 9. (Optional) REPORT
    # Generate an automated Markdown report
    report_path = os.path.join(os.path.dirname(__file__), "static_template_results.md")
    solver.generate_report(report_path)
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
