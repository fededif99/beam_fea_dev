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
from beam_fea import (
    BeamSolver, Mesh, get_material, 
    LoadCase, BoundaryConditionSet, rectangular
)

def main():
    # 1. MATERIAL & SECTION
    material = get_material('steel_a36')
    section = rectangular(width=100.0, height=200.0)

    # 2. MESH GENERATION
    length = 750
    num_elements = 50
    mesh = Mesh.from_path([(0, 0), (length, 0)], elements_per_segment=num_elements)

    print(f"[*] Running {os.path.basename(__file__)}...")

    # 3. LOADS
    # Create a load case and add a point load at the tip (last node)
    lc = LoadCase("Center Load")
    lc.point_load(x=length/2, fy=-1500000.0)  # 1500 kN downwards 
    

    # 4. BOUNDARY CONDITIONS
    bc = BoundaryConditionSet("Pin - Pin")
    bc.pinned_support(node=0)
    bc.pinned_support(node=mesh.num_nodes -1 ) 


    # 5. SOLVE
    print("  Solving...")
    # element_type='euler' or 'timoshenko'
    solver = BeamSolver(mesh, material, section, element_type='euler')
    solver.solve_static(lc, bc)
    
    # 6. RESULTS (Processed for report)
    # The unified engine recovers displacements, internal forces,
    # and 3D stress fields (including axial/bending decomposition and shear)
    solver.get_max_deflection()
    solver.get_max_internal_forces()
    solver.calculate_stresses()
    
    # 7. REPORT
    report_path = os.path.join(os.path.dirname(__file__), "static_template_results.md")
    # For isotropic materials, failure criteria defaults to Von Mises in plots,
    # but we can also specify it for the reporting tables.
    solver.generate_report(report_path, failure_criterion='tsai_wu')
    print(f"[SUCCESS] Template results saved to: {report_path}")

if __name__ == "__main__":
    main()
