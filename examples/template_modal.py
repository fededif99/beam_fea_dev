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
    LoadCase, BoundaryConditionSet
)
from beam_fea.cross_sections import circular

def main():
    # 1. MATERIAL & SECTION
    material = get_material('aluminium_7075')
    
    # Defining a circular section: diameter=50mm
    section = circular(diameter=50.0)

    # 2. MESH GENERATION
    # Create a 1000mm beam with 50 elements (Higher density for better modal accuracy)
    length = 1000.0
    mesh = MeshGenerator.beam_mesh_1d(length, num_elements=50)
    print(f"[*] Mesh created: {mesh.num_nodes} nodes, {mesh.num_elements} elements")

    # 3. BOUNDARY CONDITIONS
    # Define a simply supported beam (Pinned at both ends)
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(node=0)
    bc.pinned_support(node=mesh.num_nodes - 1)

    # 4. SOLVE MODAL
    # Initialize solver and find the first 5 natural frequencies
    # Note: Modal analysis does NOT require a LoadCase.
    solver = BeamSolver(mesh, material, section)
    num_modes = 5
    frequencies, mode_shapes = solver.solve_modal(bc, num_modes=num_modes)
    
    # 5. RESULTS
    print(f"[SUCCESS] Modal Analysis Complete.")
    print(f"\nFirst {num_modes} Natural Frequencies:")
    print("-" * 30)
    for i, f in enumerate(frequencies, 1):
        print(f"  Mode {i}: {f:8.2f} Hz")
    print("-" * 30)

    # 6. (Optional) VISUALIZATION
    # Note: Report generation for modal analysis is primarily numeric, 
    # but mode shapes can be plotted using the visualizer.
    print("[TIP] Use 'solver.visualize('modal', mode=1)' in an interactive environment to see mode shapes.")

if __name__ == "__main__":
    main()
