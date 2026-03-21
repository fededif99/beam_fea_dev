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
    BeamSolver, Mesh, get_material,
    LoadCase, BoundaryConditionSet, rectangular, circular,
    PropertySet
)

def main():

    # 1. DEFINE PROPERTIES
    # Materials
    steel = get_material('steel_a36')
    aluminum = get_material('aluminum_7075')

    # Sections
    sec_large = rectangular(width=100.0, height=200.0)
    sec_small = rectangular(width=100.0, height=100.0)

    # Use a single PropertySet collector
    props = PropertySet(name="Main Properties")
    
    # 1. Base properties (default for the whole beam)
    props.add(material=steel, section=sec_large)
    
    # 2. Specific override for Element 1
    props.add(material=aluminum, section=sec_small, elements=1)

    # 2. MESH GENERATION
    # We use a very coarse mesh (only 2 elements) to demonstrate intra-element recovery
    length = 2000.0
    mesh = Mesh.from_path([(0, 0), (length, 0)], elements_per_segment=2)

    # 3. LOADS & BOUNDARY CONDITIONS
    # Simply supported beam
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(node=0)
    bc.roller_support(node=2)

    # Uniformly Distributed Load over the entire beam
    lc = LoadCase("Uniform Load")
    lc.distributed_load(x_start=0, x_end=2000, distribution='uniform', wy=-10.0) # 10 N/mm downward

    # 4. SOLVE
    # Solver accepts PropertySet for unified management.
    # Note: element_type='timoshenko' handles both slender and thick beams.
    solver = BeamSolver(mesh, props, element_type='timoshenko')
    solver.solve_static(lc, bc)

    # 6. RESULTS & RECOVERY STRATEGIES
    print("\nComparing internal force strategies (on a very coarse 2-element mesh):")

    # 6.1. Standard Interpolation (FEA Approximation)
    # This will show a step-wise shear and linear moment (approximating the curve)
    res_std = solver.calculate_internal_forces(num_points=100, strategy='standard')
    M_mid_std = res_std['bending_moments'][50]
    print(f"  [Standard Interpolation] Mid-span Moment: {M_mid_std:.1f} N-mm")

    # 6.2. Consistent Recovery (Analytical Fidelity)
    # This will show a perfect parabolic moment even with just 2 elements
    res_con = solver.calculate_internal_forces(num_points=100, strategy='consistent')
    M_mid_con = res_con['bending_moments'][50]
    print(f"  [Consistent Recovery   ] Mid-span Moment: {M_mid_con:.1f} N-mm (Exact)")

    # Stresses automatically use the consistent recovery strategy for report accuracy
    solver.calculate_stresses(num_x_points=50)

    # 7. REPORT
    report_path = os.path.join(os.path.dirname(__file__), "multi_property_report.md")
    solver.generate_report(report_path)


if __name__ == "__main__":
    main()
