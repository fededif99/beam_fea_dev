"""
Unit tests for beam_fea.element_matrices module
"""

import pytest
import numpy as np
from beam_fea.element_matrices import EulerBernoulliElement, TimoshenkoElement


class TestEulerBernoulliElement:
    """Test EulerBernoulliElement class"""
    
    def test_element_creation(self):
        """Test creating Euler-Bernoulli element"""
        elem = EulerBernoulliElement(
            E=200000,  # MPa
            G=80000,   # MPa
            I=1e6,     # mm^4
            A=1000,    # mm^2
            L=1000,    # mm
            rho=7.85e-6  # kg/mm^3
        )
        assert elem.E == 200000
        assert elem.L == 1000
    
    def test_stiffness_matrix_shape(self):
        """Test that stiffness matrix has correct shape"""
        elem = EulerBernoulliElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        K = elem.stiffness_matrix()
        assert K.shape == (6, 6)
    
    def test_stiffness_matrix_symmetry(self):
        """Test that stiffness matrix is symmetric"""
        elem = EulerBernoulliElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        K = elem.stiffness_matrix()
        assert np.allclose(K, K.T)
    
    def test_mass_matrix_consistent(self):
        """Test consistent mass matrix"""
        elem = EulerBernoulliElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        M = elem.mass_matrix(consistent=True)
        assert M.shape == (6, 6)
        assert np.allclose(M, M.T)  # Should be symmetric
    
    def test_mass_matrix_lumped(self):
        """Test lumped mass matrix"""
        elem = EulerBernoulliElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        M = elem.mass_matrix(consistent=False)
        assert M.shape == (6, 6)
        # Lumped mass should be diagonal
        off_diagonal = M - np.diag(np.diag(M))
        assert np.allclose(off_diagonal, 0)


class TestTimoshenkoElement:
    """Test TimoshenkoElement class"""
    
    def test_element_creation(self):
        """Test creating Timoshenko element"""
        elem = TimoshenkoElement(
            E=200000,
            G=80000,
            I=1e6,
            A=1000,
            L=1000,
            rho=7.85e-6,
            kappa=5/6
        )
        assert elem.kappa == 5/6
    
    def test_stiffness_matrix_shape(self):
        """Test that stiffness matrix has correct shape"""
        elem = TimoshenkoElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        K = elem.stiffness_matrix()
        assert K.shape == (6, 6)
    
    def test_stiffness_matrix_symmetry(self):
        """Test that stiffness matrix is symmetric"""
        elem = TimoshenkoElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        K = elem.stiffness_matrix()
        assert np.allclose(K, K.T)
    
    def test_mass_matrix(self):
        """Test mass matrix"""
        elem = TimoshenkoElement(
            E=200000, G=80000, I=1e6, A=1000, L=1000, rho=7.85e-6
        )
        M = elem.mass_matrix()
        assert M.shape == (6, 6)
        assert np.allclose(M, M.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
