"""
Template: Modal Analysis (Vibration)
=====================================
This template demonstrates how to perform modal analysis to find the 
natural frequencies and mode shapes of a structure.

Workflow: 
1. Define Material & Cross-Section
2. Generate Mesh
3. Apply Boundary Conditions
4. Solve Modal Analysis
5. Retrieve Frequencies
"""


from beam_fea import (
    BeamSolver, Mesh, get_material, 
    LoadCase, BoundaryConditionSet, circular
)

def main():
    # 1. MATERIAL & SECTION
    material = get_material('aluminum_6061')
    section = circular(diameter=50.0)

    # 2. MESH GENERATION
    length = 1000.0
    mesh = Mesh.from_path([(0, 0), (length, 0)], elements_per_segment=50)

    import os
    print(f"[*] Running {os.path.basename(__file__)}...")

    # 3. BOUNDARY CONDITIONS
    # Define a simply supported beam (Pinned at both ends)
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(node=0)
    bc.pinned_support(node=mesh.num_nodes - 1)

    # 4. SOLVE MODAL
    print("  Solving Modal Analysis...")
    # Unified model handles modal analysis with consistent mass matrices
    solver = BeamSolver(mesh, material, section, element_type='timoshenko')
    num_modes = 5
    frequencies, mode_shapes = solver.solve_modal(bc, num_modes=num_modes)
    
    # 5. RESULTS
    print(f"[SUCCESS] Calculated {len(frequencies)} natural frequencies.")
    
    # 6. REPORT GENERATION
    import os
    report_path = os.path.join(os.path.dirname(__file__), "template_modal_report.md")
    solver.generate_report(report_path)
    print(f"[SUCCESS] Report generated: {report_path}")

if __name__ == "__main__":
    main()
