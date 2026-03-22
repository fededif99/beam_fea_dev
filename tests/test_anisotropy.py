import pytest
import numpy as np
from beam_fea import BeamSolver, Mesh, BoundaryConditionSet, LoadCase, rectangular
from beam_fea.composites import Ply, Laminate
from beam_fea.element_matrices import AnisotropicBeamElement

def test_bend_extension_coupling():
    """
    Test that an asymmetric laminate produces axial displacement under transverse load.
    Laminate: [0/90] (highly asymmetric)
    """
    # 1. Setup asymmetric laminate
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=1.0)
    lam = Laminate("Asymmetric", stack=[(ply, 0), (ply, 90)])

    # B11 should be non-zero
    assert abs(lam.B[0,0]) > 1e-3

    # 2. Model: Cantilever beam
    L = 1000
    w = 20
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=10)
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

    assert abs(tip_u) > 1e-10  # Non-zero coupling (compliance-based ES is correctly smaller than B[0,0])
    # Note: sign of u depends on stack-up and load direction

def test_symmetric_no_coupling():
    """
    Test that a symmetric laminate produces ZERO axial displacement under transverse load.
    Laminate: [0/90]s
    """
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=1.0)
    lam = Laminate("Symmetric", stack=[(ply, [0, 90, 90, 0])])

    assert abs(lam.B[0,0]) < 1e-10

    L = 1000
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=10)
    section = rectangular(width=20, height=4.0)
    solver = BeamSolver(mesh, lam, section)

    bc = BoundaryConditionSet()
    bc.fixed_support(0)
    load = LoadCase()
    load.point_load(node=10, fy=-100)

    displ = solver.solve_static(load, bc)
    tip_u = displ[3*10]

    assert abs(tip_u) < 1e-12


def test_anisotropic_element_timoshenko_softening():
    """
    Verify that providing GA_s (Timoshenko shear) makes AnisotropicBeamElement
    softer (higher transverse flexibility) than the EB-only case (GA_s=None).
    For a short beam this should produce a meaningful difference.
    """
    EA = 5e6   # N
    ES = 0.0   # no coupling, isolate bending
    EI = 1e8   # N*mm^2
    L  = 50.0  # mm (short: L/h small => shear matters)
    GA_s = 2e4  # N — shear stiffness
    rho_total = 1.6e-3

    elem_eb = AnisotropicBeamElement(EA=EA, ES=ES, EI=EI, L=L, rho_total=rho_total, GA_s=None)
    elem_ts = AnisotropicBeamElement(EA=EA, ES=ES, EI=EI, L=L, rho_total=rho_total, GA_s=GA_s)

    K_eb = elem_eb.stiffness_matrix()
    K_ts = elem_ts.stiffness_matrix()

    # Timoshenko transverse stiffness K[1,1] < EB (softer)
    assert K_ts[1, 1] < K_eb[1, 1]

    # Both must be symmetric
    assert np.allclose(K_eb, K_eb.T, atol=1e-10)
    assert np.allclose(K_ts, K_ts.T, atol=1e-10)


def test_anisotropic_udl_force_recovery():
    """
    For a simply-supported symmetric composite beam under UDL, the
    internal force recovery must give:
      - Mid-span moment  ≈ q*L^2/8  (analytical)
      - End shear        ≈ q*L/2
    This test specifically validates the distributed load particular solution
    that was missing in the original implementation.
    """
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.5)
    lam = Laminate("SS_UDL", stack=[(ply, [0, 90, 90, 0])])  # symmetric => B=0

    L = 1000.0
    w = 20.0
    q = -2.0  # N/mm (downward UDL)

    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=20)
    section = rectangular(width=w, height=lam.total_thickness)
    solver = BeamSolver(mesh, lam, section)

    bc = BoundaryConditionSet()
    bc.pinned_support(0)
    bc.roller_support(20)

    load = LoadCase()
    load.distributed_load(x_start=0, x_end=L, distribution='uniform', wy=q)

    solver.solve_static(load, bc)

    forces = solver.calculate_internal_forces(num_points=201)

    # Analytical values — bottom-tension positive convention:
    # Downward UDL (q<0) produces sagging at midspan => bottom in tension => M_mid > 0
    # M_mid = q * L^2 / 8  =>  (-2) * 1000000 / 8 = -250000 => but bottom-tension =>  +250000
    M_mid_analytical = -q * L**2 / 8   # positive (bottom-tension)
    # Left-end reaction force = +q_magnitude * L/2 = -q * L / 2
    V_end_analytical  = -q * L / 2     # positive (upward reaction = upward shear at left face)

    # Mid-span moment
    idx_mid = len(forces['positions']) // 2
    M_mid = forces['bending_moments'][idx_mid]
    assert M_mid == pytest.approx(M_mid_analytical, rel=0.02)

    # Left-end shear
    V_end = forces['shear_forces'][0]
    assert V_end == pytest.approx(V_end_analytical, rel=0.02)
