"""
examples/ex05_composite_beam.py
==============================
Example of using Classical Laminate Theory (CLT) to model a composite beam.
"""

from beam_fea import BeamSolver, MeshGenerator, BoundaryConditionSet, LoadCase, rectangular
from beam_fea.composites import Ply, Laminate
import numpy as np

def run_composite_analysis():
    # 1. Define Ply Properties (Carbon/Epoxy UD)
    carbon_ply = Ply(
        name="T300_Epoxy",
        E1=135000,   # Longitudinal Modulus (MPa)
        E2=10000,    # Transverse Modulus (MPa)
        nu12=0.3,    # Poisson's ratio
        G12=5000,    # Shear Modulus (MPa)
        thickness=0.125, # mm
        rho=1.6e-6   # kg/mm^3
    )

    # 2. Create Laminate Stack-up [0/45/-45/90]s
    lam = Laminate("Wing_Spar_Flange")
    # Add plies from bottom to top
    lam.add_stack(carbon_ply, [0, 45, -45, 90, 90, -45, 45, 0])

    print("\n" + "="*50)
    print(f"Laminate Analysis: {lam.name}")
    print("="*50)
    print(lam)

    props = lam.get_effective_properties()
    print(f"\nEffective Properties:")
    print(f"  Ex (Axial):   {props['Ex']:.1f} MPa")
    print(f"  Eb (Bending): {props['Eb']:.1f} MPa")
    print(f"  Gxy (Shear):  {props['Gxy']:.1f} MPa")
    print(f"  Thickness:    {props['thickness']:.3f} mm")

    # 3. Create FEA Model
    # We'll model a 1-meter composite strip (width 25mm)
    L = 1000
    w = 25
    h = props['thickness']

    mesh = MeshGenerator.beam_mesh_1d(length=L, num_elements=50)

    # Convert Laminate to a Material for the solver
    # We prefer 'bending' modulus for this transverse load case
    composite_mat = lam.to_material(preference='bending')
    section = rectangular(width=w, height=h)

    solver = BeamSolver(mesh, composite_mat, section)

    # 4. Boundary Conditions & Loads (Simply Supported with UDL)
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(0)
    bc.roller_support(50) # node at x=L

    load = LoadCase("UDL")
    load.uniform_load(x_start=0, x_end=L, wy=-1.0) # 1 N/mm

    # 5. Solve
    solver.solve_static(load, bc)

    # 6. Results
    max_def, _ = solver.get_max_deflection()
    print(f"\nFEA Results:")
    print(f"  Max Deflection: {abs(max_def):.4f} mm")

    # Validation against analytical: v_max = 5wL^4 / (384EI)
    I = (w * h**3) / 12
    E = composite_mat.E
    v_analytical = (5 * 1.0 * L**4) / (384 * E * I)
    print(f"  Analytical:     {v_analytical:.4f} mm")
    print(f"  Error:          {abs(abs(max_def) - v_analytical)/v_analytical*100:.4e}%")

    # 7. Generate Report
    solver.generate_report("composite_beam_report.md")
    print("\nReport generated: composite_beam_report.md")

if __name__ == "__main__":
    run_composite_analysis()
