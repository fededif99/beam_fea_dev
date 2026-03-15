
from beam_fea import Mesh, BeamSolver, get_material, rectangular, BoundaryConditionSet
import numpy as np

def test_modal_mass_sum():
    print("\nTesting Modal Mass Sum (Mass Participation Verification)...")

    # Simple cantilever beam
    L = 1000.0
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=50)
    mat = get_material('steel')
    sec = rectangular(50, 50)

    solver = BeamSolver(mesh, mat, sec, element_type='euler')

    bc = BoundaryConditionSet("Cantilever")
    bc.fixed_support(0)

    # Solve for many modes to check mass convergence
    num_modes = 20
    frequencies, _ = solver.solve_modal(bc, num_modes=num_modes)

    part = solver.last_modal_participation

    cum_y = sum(m['percent_y'] for m in part['modes'])
    print(f"Total Structural Mass (Y): {part['total_mass_y']:.4f} kg")
    print(f"Cumulative Mass Participation (Y) for {num_modes} modes: {cum_y*100:.2f}%")

    # For a cantilever, the first few modes should capture most of the mass (>80%)
    assert cum_y > 0.80
    print("SUCCESS: Modal mass participation is correctly calculated and converging.")

if __name__ == "__main__":
    test_modal_mass_sum()
