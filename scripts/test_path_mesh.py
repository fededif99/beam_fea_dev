
from beam_fea import MeshGenerator, BeamSolver, get_material, rectangular, BoundaryConditionSet, LoadCase
import os

def test_cranked_beam():
    print("\nTesting Cranked Beam (Angled multi-segment mesh)...")

    # Define a cranked beam path
    # (0,0) -> (1000, 0) [Horizontal] -> (1500, 500) [Angled]
    points = [(0, 0), (1000, 0), (1500, 500)]
    elems = [20, 10]

    mesh = MeshGenerator.path_mesh(points, elems)
    print(f"Mesh generated with {mesh.num_nodes} nodes.")
    print(f"Waypoints: {mesh.waypoint_nodes}")

    # Verify waypoints
    assert len(mesh.waypoint_nodes) == 3
    assert mesh.waypoint_nodes[0] == 0
    assert mesh.waypoint_nodes[1] == 20
    assert mesh.waypoint_nodes[2] == 30

    # Try a simple solve
    mat = get_material('steel')
    sec = rectangular(100, 200)
    solver = BeamSolver(mesh, mat, sec)

    bc = BoundaryConditionSet("Fixed-Free Cranked")
    bc.fixed_support(mesh.waypoint_nodes[0])

    load = LoadCase("Tip Load")
    load.point_load(node=mesh.waypoint_nodes[2], fy=-1000)

    solver.solve_static(load, bc)
    max_def, _ = solver.get_max_deflection()
    print(f"Solve complete. Max deflection: {abs(max_def):.4f} mm")

if __name__ == "__main__":
    test_cranked_beam()
