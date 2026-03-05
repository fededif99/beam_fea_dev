import pytest
import numpy as np
from beam_fea import BeamSolver, MeshGenerator, BoundaryConditionSet, LoadCase, rectangular
from beam_fea.composites import Ply, Laminate

def test_bend_extension_coupling():
    """
    Test that an asymmetric laminate produces axial displacement under transverse load.
    Laminate: [0/90] (highly asymmetric)
    """
    # 1. Setup asymmetric laminate
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=1.0)
    lam = Laminate("Asymmetric")
    lam.add_ply(ply, 0)
    lam.add_ply(ply, 90)

    # B11 should be non-zero
    assert abs(lam.B[0,0]) > 1e-3

    # 2. Model: Cantilever beam
    L = 1000
    w = 20
    mesh = MeshGenerator.beam_mesh_1d(length=L, num_elements=10)
    section = rectangular(width=w, height=2.0)

    solver = BeamSolver(mesh, lam, section)

    # 3. BCs: Fixed at start
    # We must NOT constrain node 0 fx for this test if we want to see global coupling,
    # OR we check axial force at the fixed end.
    # Actually, if we fix node 0 (u, v, theta), and apply Fy at tip,
    # the tip should show non-zero axial displacement 'u'.
    bc = BoundaryConditionSet()
    bc.fixed_support(0)

    load = LoadCase()
    load.point_load(node=10, fy=-100) # Transverse tip load

    # 4. Solve
    displ = solver.solve_static(load, bc)

    # 5. Verify coupling
    # For a standard cantilever, tip u is 0 for Fy load.
    # For asymmetric composite, tip u should be non-zero.
    tip_u = displ[3*10]
    tip_v = displ[3*10 + 1]

    print(f"Tip deflection: v={tip_v:.4f} mm, u={tip_u:.4e} mm")

    assert abs(tip_u) > 1e-6
    # Note: sign of u depends on stack-up and load direction

def test_symmetric_no_coupling():
    """
    Test that a symmetric laminate produces ZERO axial displacement under transverse load.
    Laminate: [0/90]s
    """
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=1.0)
    lam = Laminate("Symmetric")
    lam.add_stack(ply, [0, 90, 90, 0])

    assert abs(lam.B[0,0]) < 1e-10

    L = 1000
    mesh = MeshGenerator.beam_mesh_1d(length=L, num_elements=10)
    section = rectangular(width=20, height=4.0)
    solver = BeamSolver(mesh, lam, section)

    bc = BoundaryConditionSet()
    bc.fixed_support(0)
    load = LoadCase()
    load.point_load(node=10, fy=-100)

    displ = solver.solve_static(load, bc)
    tip_u = displ[3*10]

    assert abs(tip_u) < 1e-12
