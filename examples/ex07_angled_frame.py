
from beam_fea import BeamSolver, Mesh, get_material, rectangular, BoundaryConditionSet, LoadCase
import os
import numpy as np

def run_angled_frame_demo():
    print("\n" + "="*60)
    print("EXAMPLE 7: ANGLED FRAME (CRANKED BEAM) ANALYSIS")
    print("="*60)

    # 1. Mesh Generation
    # Define a frame: (0,0) -> (1000, 0) -> (1000, 1000)
    points = [(0, 0), (1000, 0), (1000, 1000)]
    elems = [20, 20]
    mesh = Mesh.from_path(points, elems)

    print(f"Mesh created with {mesh.num_nodes} nodes and {mesh.num_elements} elements.")
    print(f"Waypoints: {mesh.waypoint_nodes}")

    # 2. Properties
    material = get_material('aluminium')
    section = rectangular(height=100, width=200)
    solver = BeamSolver(mesh, material, section)

    # 3. Boundary Conditions & Loads
    # Fixed at the base (Node 0)
    bc = BoundaryConditionSet("L-Frame Fixed-Free")
    bc.fixed_support(mesh.waypoint_nodes[0])

    # Tip load at the end of the vertical segment (Last node)
    load = LoadCase("Tip Horizontal Load")
    # Applying a horizontal force at the top of the 'column'
    load.point_load(node=mesh.waypoint_nodes[2], fx=10000) # 10 kN rightward

    # Let's also add a vertical point load at the elbow for variety
    load.point_load(node=mesh.waypoint_nodes[1], fy=-5000) # 5 kN downward

    # 4. Solve
    solver.solve_static(load, bc)

    # 5. Results
    max_rec = solver.get_max_deflection()
    print(f"\nSolve complete.")
    print(f"Max displacement magnitude: {max_rec['res']:.4f} mm at Node {int(max_rec['node_id'])}")

    # 6. Report
    report_path = os.path.join(os.path.dirname(__file__), "angled_frame_report.md")
    solver.generate_report(report_path)
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    run_angled_frame_demo()
