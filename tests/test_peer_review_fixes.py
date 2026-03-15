import pytest
import numpy as np
from beam_fea.cross_sections import rectangular, c_channel, t_beam, TBeamSection
from beam_fea.modal_analysis import ModalAnalysis
from beam_fea.mesh import Mesh
from beam_fea.materials import get_material

def test_rectangular_h_cubed_fix():
    """Verify that RectangularSection uses h**3, not h**h."""
    b, h = 10, 2
    # If it used h**h, it would be (10 * 2**2) / 12 = 40/12 = 3.33
    # With h**3, it is (10 * 2**3) / 12 = 80/12 = 6.66
    rect = rectangular(b, h)
    assert pytest.approx(rect.Iy) == (b * h**3) / 12

def test_c_channel_name_error_fix():
    """Verify that CChannelSection properties can be calculated without NameError."""
    # This would have failed before with NameError: Sy
    try:
        chan = c_channel(height=100, flange_width=50, web_thickness=5, flange_thickness=8)
        assert chan.Sy > 0
    except NameError as e:
        pytest.fail(f"CChannelSection raised NameError: {e}")

def test_t_beam_stress_profile_signs():
    """Verify that TBeamSection stresses (Q) remain positive."""
    t_sect = TBeamSection(flange_width=100, flange_thickness=10, web_height=100, web_thickness=10)
    t = t_sect.properties()
    y_coords = np.linspace(t.y_bottom, t.y_top, 20)
    z_coords = np.zeros_like(y_coords)
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')
    
    mask, t_yz, Q = t_sect.get_stress_profile(Y, Z)
    
    # Q should be non-negative
    assert np.all(Q >= -1e-10)
    # Peak should be positive
    assert np.max(Q) > 0

def test_modal_analysis_dense_slicing():
    """Verify that ModalAnalysis handles num_modes correctly for dense matrices."""
    # Create a small dense problem
    K = np.eye(10) * 1000
    M = np.eye(10)
    
    ma = ModalAnalysis()
    # Solve for 3 modes
    freqs, shapes = ma.solve(K, M, num_modes=3)
    
    assert len(freqs) == 3
    assert shapes.shape[1] == 3

def test_element_lookup_binary_search():
    """Verify solver binary search optimization for internal forces."""
    from beam_fea.solver import BeamSolver
    
    mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=100)
    mat = get_material('steel')
    sect = rectangular(10, 10)
    solver = BeamSolver(mesh, mat, sect)
    
    # Mock displacements (required for internal force calc)
    solver.displacements = np.zeros(mesh.num_dofs)
    
    # This triggers the searchsorted optimization
    forces = solver.calculate_internal_forces(num_points=10)
    assert len(forces['positions']) == 10
    assert forces['positions'][0] == 0.0
    assert forces['positions'][-1] == 1000.0

if __name__ == "__main__":
    pytest.main([__file__])
