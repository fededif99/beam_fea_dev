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
    lam = Laminate("QI")
    lam.add_stack(ply, [0, 45, 90, -45, -45, 90, 45, 0])

    A = lam.A
    assert A[0,0] == pytest.approx(A[1,1], rel=1e-2)
    assert A[0,2] == pytest.approx(0, abs=1e-10) # Symmetric and balanced
    assert A[1,2] == pytest.approx(0, abs=1e-10)

def test_symmetric_laminate_coupling():
    # Symmetric laminate should have B = 0
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.125)
    lam = Laminate("Symmetric")
    lam.add_stack(ply, [0, 90, 90, 0])

    assert np.all(np.abs(lam.B) < 1e-10)

def test_effective_properties_export():
    ply = Ply(E1=150000, E2=10000, nu12=0.3, G12=5000, thickness=0.125, rho=1.6e-6)
    lam = Laminate("Test")
    lam.add_stack(ply, [0, 0, 0, 0]) # Pure 0 degree

    props = lam.get_effective_properties()
    # For all 0 deg, Ex should be close to E1
    assert props['Ex'] == pytest.approx(150000, rel=1e-2)
    assert props['Eb'] == pytest.approx(150000, rel=1e-2)

    mat = lam.to_material()
    assert mat.E == pytest.approx(150000, rel=1e-2)
    assert mat.rho == 1.6e-6

if __name__ == "__main__":
    # Quick manual run
    test_ply_reduced_stiffness()
    print("Ply Q matrix test passed")
    test_quasi_isotropic_laminate()
    print("QI Laminate A matrix test passed")
