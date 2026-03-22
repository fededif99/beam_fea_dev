import pytest
import numpy as np
from beam_fea.composites import Ply, Laminate

def test_ply_reduced_stiffness():
    # Example properties (Carbon/Epoxy)
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.125)
    Q = ply.reduced_stiffness_matrix()

    # Q11 = E1 / (1 - nu12*nu21)
    # nu21 = nu12 * E2 / E1 = 0.3 * 10000 / 150000 = 0.02
    # Q11 = 150000 / (1 - 0.3*0.02) = 150000 / 0.994 = 150905.4
    assert Q[0,0] == pytest.approx(150905.4, rel=1e-4)
    assert Q[1,1] == pytest.approx(10060.36, rel=1e-4)
    assert Q[0,1] == pytest.approx(3018.10, rel=1e-4)
    assert Q[2,2] == 5000

def test_quasi_isotropic_laminate():
    # A quasi-isotropic laminate [0/45/90/-45]s should have A11 approx A22 and A16 approx 0
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.125)
    lam = Laminate("QI", stack=[(ply, [0, 45, 90, -45, -45, 90, 45, 0])])

    A = lam.A
    assert A[0,0] == pytest.approx(A[1,1], rel=1e-2)
    assert A[0,2] == pytest.approx(0, abs=1e-10) # Symmetric and balanced
    assert A[1,2] == pytest.approx(0, abs=1e-10)

def test_symmetric_laminate_coupling():
    # Symmetric laminate should have B = 0
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.125)
    lam = Laminate("Symmetric", stack=[(ply, [0, 90, 90, 0])])

    assert np.all(np.abs(lam.B) < 1e-10)

def test_effective_properties_export():
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.125, rho=1.6e-6)
    lam = Laminate("Test", stack=[(ply, [0] * 4)])

    props = lam.get_effective_properties()
    # For all 0 deg, Ex should be close to E1
    assert props['Ex'] == pytest.approx(150000, rel=1e-2)
    assert props['Eb'] == pytest.approx(150000, rel=1e-2)
    # Ey should be close to E2 for all-0 laminate
    assert props['Ey'] == pytest.approx(10060, rel=1e-2)

    mat = lam.to_material()
    assert mat.E == pytest.approx(150000, rel=1e-2)
    assert mat.rho == 1.6e-6


def test_effective_ex_compliance_vs_naive():
    """
    Verify that Ex uses the compliance-matrix formula 1/(a11*t).
    For off-axis laminates, this differs from the naive (A11*A22-A12^2)/(A22*t).
    A [+15/-15] balanced-but-off-axis laminate has A16=A26=0 so both should match;
    an unbalanced [15/15] laminate has A16 != 0 and they should differ.
    """
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.25)

    # Balanced [+15/-15]: both methods agree
    lam_bal = Laminate("balanced", stack=[(ply, [15, -15])])
    props_bal = lam_bal.get_effective_properties()
    # Naive formula for balanced
    A = lam_bal.A
    naive_Ex = (A[0,0]*A[1,1] - A[0,1]**2) / (A[1,1] * lam_bal.total_thickness)
    assert props_bal['Ex'] == pytest.approx(naive_Ex, rel=1e-3)

    # Unbalanced [15/15]: compliance approach is correct; naive is wrong
    lam_unbal = Laminate("unbalanced", stack=[(ply, [15, 15])])
    props_unbal = lam_unbal.get_effective_properties()
    A_u = lam_unbal.A
    naive_Ex_u = (A_u[0,0]*A_u[1,1] - A_u[0,1]**2) / (A_u[1,1] * lam_unbal.total_thickness)
    # Compliance-based Ex and naive Ex should differ for unbalanced laminate
    assert abs(props_unbal['Ex'] - naive_Ex_u) / props_unbal['Ex'] > 0.001
    # But compliance-based Ex must equal 1/(a11*t)
    A_inv = np.linalg.inv(A_u)
    correct_Ex = 1.0 / (A_inv[0, 0] * lam_unbal.total_thickness)
    assert props_unbal['Ex'] == pytest.approx(correct_Ex, rel=1e-9)

if __name__ == "__main__":
    # Quick manual run
    test_ply_reduced_stiffness()
    print("Ply Q matrix test passed")
    test_quasi_isotropic_laminate()
    print("QI Laminate A matrix test passed")
