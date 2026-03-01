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
    BeamSolver, MeshGenerator, get_material, 
    LoadCase, BoundaryConditionSet, circular
)

def main():
    # Material (Aluminum from database)
    material = get_material('aluminum_6061')
    
    # Defining a circular section: diameter=50mm
    section = circular(diameter=50.0)

    # 2. MESH GENERATION
    # Create a 1000mm beam with 50 elements (Higher density for better modal accuracy)
    length = 1000.0
    mesh = MeshGenerator.beam_mesh_1d(length, num_elements=50)
    import os
    print(f"[*] Running {os.path.basename(__file__)}...")

    # 3. BOUNDARY CONDITIONS
    # Define a simply supported beam (Pinned at both ends)
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(node=0)
    bc.pinned_support(node=mesh.num_nodes - 1)

    # 4. SOLVE MODAL
    print("  Solving Modal Analysis...")
    solver = BeamSolver(mesh, material, section)
    num_modes = 5
    frequencies, mode_shapes = solver.solve_modal(bc, num_modes=num_modes)
    
    # 5. RESULTS
    print(f"[SUCCESS] Calculated {len(frequencies)} natural frequencies.")
    
    # 6. REPORT GENERATION
    import os
    report_path = os.path.join(os.getcwd(), "template_modal_report.md")
    solver.generate_report(report_path)
    print(f"[SUCCESS] Report generated: {report_path}")

if __name__ == "__main__":
    main()
