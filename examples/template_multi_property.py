"""
Template: Multi-Property Analysis & Force Recovery
==================================================
This template demonstrates advanced features of the Beam FEA library:
1. Per-element material and cross-section overrides (Stepped/Composite beams).
2. High-accuracy intra-element force recovery for distributed loads on coarse meshes.

Workflow:
1. Define multiple Materials & Cross-Sections
2. Generate Mesh
3. Assign element-specific properties
4. Apply distributed loads
5. Solve and observe accurate force diagrams
"""

import os
import numpy as np
from beam_fea import (
    BeamSolver, MeshGenerator, get_material,
    LoadCase, BoundaryConditionSet, rectangular, circular
)

def main():
    print(f"[*] Running {os.path.basename(__file__)}...")

    # 1. DEFINE PROPERTIES
    # Materials
    steel = get_material('steel_a36')
    aluminum = get_material('aluminum_7075')

    # Sections
    sec_large = rectangular(width=100.0, height=200.0)
    sec_small = rectangular(width=100.0, height=100.0)

    # 2. MESH GENERATION
    # We use a very coarse mesh (only 2 elements) to demonstrate intra-element recovery
    length = 2000.0
    mesh = MeshGenerator.beam_mesh_1d(length, num_elements=2)

    # 3. ASSIGN PER-ELEMENT OVERRIDES
    # Element 0 (0-1000mm): Steel + Large Section
    mesh.elements[0].material = steel
    mesh.elements[0].section = sec_large

    # Element 1 (1000-2000mm): Aluminum + Small Section
    mesh.elements[1].material = aluminum
    mesh.elements[1].section = sec_small

    # 4. LOADS & BOUNDARY CONDITIONS
    # Simply supported beam
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(node=0)
    bc.roller_support(node=2)

    # Uniformly Distributed Load over the entire beam
    lc = LoadCase("Uniform Load")
    lc.uniform_load(x_start=0, x_end=2000, wy=-10.0) # 10 N/mm downward

    # 5. SOLVE
    # Solver requires default properties, though elements are overridden here
    solver = BeamSolver(mesh, steel, sec_large)
    solver.solve_static(lc, bc)

    # 6. RESULTS
    # Force recovery at high resolution (100 points) to see smooth diagrams
    # despite having only 2 finite elements.
    solver.calculate_internal_forces(num_points=100)
    solver.calculate_stresses(num_x_points=50)

    # 7. REPORT
    report_path = os.path.join(os.path.dirname(__file__), "multi_property_report.md")
    solver.generate_report(report_path)

    print(f"[SUCCESS] Multi-property analysis complete.")
    print(f"  Max Deflection: {solver.get_max_deflection()[0]:.4f} mm")
    print(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    main()
