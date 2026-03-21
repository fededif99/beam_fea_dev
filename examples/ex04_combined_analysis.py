"""
ex04_combined_analysis.py
========================
Example demonstrating a combined static and modal report.
"""

from beam_fea import BeamSolver, Mesh, LoadCase, BoundaryConditionSet, get_material

def run_combined_example():
    # 1. Setup Model
    # 2000mm beam, 40 elements
    mesh = Mesh.from_path([(0, 0), (2000, 0)], elements_per_segment=40)
    material = get_material("Steel")
    
    # Rectangular Section 50x100 mm using convenience helper
    from beam_fea.cross_sections import rectangular
    section = rectangular(width=50, height=100)
    
    solver = BeamSolver(mesh, material, section)
    
    # 2. Define Analysis Cases
    # Boundary Conditions: Fixed-Pinned
    bc = BoundaryConditionSet()
    bc.fixed_support(node=0)
    bc.pinned_support(node=mesh.num_nodes - 1)
    
    # Static Loads: 5000N downward point load + 5 N/mm UDL
    lc = LoadCase("Operational Loads")
    lc.point_load(node=mesh.num_nodes // 2, fy=-5000)
    lc.distributed_load(distribution='uniform', wy=-5)
    
    # 3. Run Both Analyses
    print("Running static analysis...")
    solver.solve_static(lc, bc)
    
    print("Running modal analysis...")
    solver.solve_modal(bc, num_modes=5)
    
    # 4. Generate Combined Report
    print("Generating combined report...")
    import os
    report_path = os.path.join(os.path.dirname(__file__), "ex04_combined_report.md")
    solver.generate_report(report_path)
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    run_combined_example()
