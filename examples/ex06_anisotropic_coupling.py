"""
examples/ex06_anisotropic_coupling.py
=====================================
Demonstrates bend-extension coupling in asymmetric composite beams.
"""

from beam_fea import BeamSolver, MeshGenerator, BoundaryConditionSet, LoadCase, rectangular
from beam_fea.composites import Ply, Laminate
import numpy as np

def run_coupling_demo():
    # 1. Define Carbon Ply
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=1.0, rho=1.6e-6)

    # 2. Setup TWO Laminates: Symmetric vs Asymmetric
    # Symmetric [0/90]s
    lam_sym = Laminate("Symmetric [0/90]s")
    lam_sym.add_stack(ply, [0, 90, 90, 0])

    # Asymmetric [0/90]
    lam_asym = Laminate("Asymmetric [0/90]")
    lam_asym.add_stack(ply, [0, 0, 90, 90]) # Stacked as 2 plies of 0 then 2 of 90 for same total thickness

    print("\n" + "="*60)
    print("DEMONSTRATING BEND-EXTENSION COUPLING")
    print("="*60)
    print(f"Asymmetric B11: {lam_asym.B[0,0]:.2e}")
    print(f"Symmetric B11:  {lam_sym.B[0,0]:.2e}")

    # 3. FEA Model
    L = 500
    w = 10
    mesh = MeshGenerator.beam_mesh_1d(length=L, num_elements=20)
    section = rectangular(width=w, height=4.0)

    # Solve Symmetric Case
    solver_sym = BeamSolver(mesh, lam_sym, section)
    bc = BoundaryConditionSet("Cantilever")
    bc.fixed_support(0)
    load = LoadCase("Tip Load")
    load.point_load(node=20, fy=-100)

    u_sym = solver_sym.solve_static(load, bc)
    tip_u_sym = u_sym[3*20]

    # Solve Asymmetric Case
    solver_asym = BeamSolver(mesh, lam_asym, section)
    u_asym = solver_asym.solve_static(load, bc)
    tip_u_asym = u_asym[3*20]

    print(f"\nResults for Transverse Tip Load (Fy=-100 N):")
    print(f"  Symmetric Beam Tip Axial Displacement (u):  {tip_u_sym:.4e} mm")
    print(f"  Asymmetric Beam Tip Axial Displacement (u): {tip_u_asym:.4e} mm")

    if abs(tip_u_asym) > 10 * abs(tip_u_sym):
        print("\nSUCCESS: Coupling behavior captured!")
        print("The asymmetric laminate stretched/contracted due to bending.")

    # Internal Force Recovery
    forces = solver_asym.calculate_internal_forces(num_points=10)
    print(f"\nAsymmetric Beam Internal Axial Force at root: {forces['axial_forces'][0]:.2f} N")

    # Generate Report to verify composite material formatting
    import os
    report_path = os.path.join(os.path.dirname(__file__), "anisotropic_coupling_report.md")
    solver_asym.generate_report(report_path)
    print(f"\nReport generated: {report_path}")

if __name__ == "__main__":
    run_coupling_demo()
