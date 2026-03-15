import numpy as np

from beam_fea.mesh import Mesh
from beam_fea.materials import Material
from beam_fea.cross_sections import IBeamSection
from beam_fea.boundary_conditions import BoundaryConditionSet
from beam_fea.loads import LoadCase
from beam_fea.solver import BeamSolver

def test_3d_stress_field_generation():
    print("Testing 3D Stress Field Generation...")
    
    # Create mesh
    mesh = Mesh.from_path([(0, 0), (1000.0, 0)], elements_per_segment=10)
    
    # Create material and section
    mat = Material(name='steel', E=200000, nu=0.3, rho=7850)
    sec = IBeamSection(flange_width=100.0, total_height=200.0, web_thickness=8.0, flange_thickness=12.0).properties()
    
    # Create BCs and loads
    bc_set = BoundaryConditionSet()
    bc_set.fixed_support(0)
    
    load_case = LoadCase()
    load_case.point_load(10, fy=-5000.0) # 5 kN load at the tip
    
    # Create solver
    solver = BeamSolver(mesh, mat, sec)
    solver.solve_static(load_case, bc_set)
    
    # Calculate stresses
    stresses = solver.calculate_stresses(num_x_points=11, num_y_points=20, num_z_points=10)
    
    # Assertions
    assert 'axial' in stresses
    assert 'bending' in stresses
    assert 'shear' in stresses
    assert 'von_mises' in stresses
    assert 'mask' in stresses
    
    assert stresses['axial'].shape == (11, 20, 10)
    assert stresses['bending'].shape == (11, 20, 10)
    
    print(f"Stress dictionary keys: {list(stresses.keys())}")
    print(f"Axial matrix shape: {stresses['axial'].shape}")
    print(f"Max Axial: {np.max(np.abs(stresses['axial'])):.4f} MPa")
    print(f"Max Bending: {np.max(np.abs(stresses['bending'])):.4f} MPa")
    print(f"Max Shear: {np.max(np.abs(stresses['shear'])):.4f} MPa")
    print(f"Max von Mises: {np.max(stresses['von_mises']):.4f} MPa")
    print("SUCCESS!")

if __name__ == "__main__":
    test_3d_stress_field_generation()
