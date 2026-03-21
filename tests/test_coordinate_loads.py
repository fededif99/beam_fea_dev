
import numpy as np
import pytest
from beam_fea.mesh import Mesh
from beam_fea.loads import LoadCase
from beam_fea.materials import Material
from beam_fea.cross_sections import rectangular
from beam_fea.solver import BeamSolver
from beam_fea.boundary_conditions import BoundaryConditionSet

def test_point_load_coordinate():
    """Verify PointLoad(x=...) is consistent with shape functions."""
    mesh = Mesh.from_path([(0, 0), (1000.0, 0)], elements_per_segment=1) # Single element [0, 1000]
    lc = LoadCase()
    
    # Apply load at mid-span (x=500.0)
    # For xi=0.5: 
    # N1 = 0.5, N2 = L*0.125 = 125, N3 = 0.5, N4 = -L*0.125 = -125
    fy = -1000.0
    lc.point_load(x=500.0, fy=fy)
    
    F = lc.create_force_vector(mesh.num_dofs, mesh)
    
    # Node 0 (start)
    assert np.isclose(F[1], fy * 0.5)
    assert np.isclose(F[2], fy * 125.0)
    
    # Node 1 (end)
    assert np.isclose(F[4], fy * 0.5)
    assert np.isclose(F[5], fy * -125.0)

def test_concentrated_moment_coordinate():
    """Verify concentrated_moment(x=...) matches shape function derivatives."""
    mesh = Mesh.from_path([(0, 0), (1000.0, 0)], elements_per_segment=1)
    lc = LoadCase()
    
    # Moment at x=500.0 (xi=0.5)
    # dN1/dx = -1.5/L, dN2/dx = -0.25, dN3/dx = 1.5/L, dN4/dx = -0.25
    mz = 10000.0
    lc.moment(x=500.0, mz=mz)
    F = lc.create_force_vector(mesh.num_dofs, mesh)
    
    # Node 0
    assert np.isclose(F[1], mz * -1.5 / 1000.0)
    assert np.isclose(F[2], mz * -0.25)
    
    # Node 1
    assert np.isclose(F[4], mz * 1.5 / 1000.0)
    assert np.isclose(F[5], mz * -0.25)

def test_udl_coordinate_range():
    """Verify UDL over a partial coordinate range."""
    mesh = Mesh.from_path([(0, 0), (1000.0, 0)], elements_per_segment=1)
    lc = LoadCase()
    
    # UDL from x=0 to x=500.0
    wy = -10.0
    lc.distributed_load(x_start=0, x_end=500, distribution='uniform', wy=wy)
    F = lc.create_force_vector(mesh.num_dofs, mesh)
    
    # Analytical integrals of shape functions from 0 to 0.5
    # L=1000
    # Int(N1) = xi - xi^3 + 0.5xi^4 = 0.5 - 0.125 + 0.03125 = 0.40625
    # Int(N2) = L * (0.5xi^2 - 2/3xi^3 + 0.25*xi^4) = 1000 * (0.125 - 0.08333 + 0.015625) = 57.29...
    
    # FY1 = wy * L * 0.40625 = -10 * 1000 * 0.40625 = -4062.5
    assert np.isclose(F[1], -4062.5)

def test_coordinate_parametric_consistency():
    """Verify result stays same when refining mesh."""
    material = Material("Steel", 200000, 0.3, 7.85e-9)
    section = rectangular(50, 100)
    
    # Cantilever with load at x=750
    L = 1000.0
    P = -1000.0
    
    # Case 1: 2 elements (load at node 1.5... wait)
    mesh2 = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=2)
    lc2 = LoadCase()
    lc2.point_load(x=750.0, fy=P)
    bc2 = BoundaryConditionSet()
    bc2.fixed_support(0)
    solver2 = BeamSolver(mesh2, material, section)
    solver2.solve_static(lc2, bc2)
    max_rec2 = solver2.get_max_deflection()
    disp2 = max_rec2['res']
    
    # Case 2: 10 elements (load at node 7.5)
    mesh10 = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=10)
    lc10 = LoadCase()
    lc10.point_load(x=750.0, fy=P)
    bc10 = BoundaryConditionSet()
    bc10.fixed_support(0)
    solver10 = BeamSolver(mesh10, material, section)
    solver10.solve_static(lc10, bc10)
    max_rec10 = solver10.get_max_deflection()
    disp10 = max_rec10['res']
    
    assert np.isclose(disp2, disp10, rtol=1e-5)

def test_coordinate_consistency_signs():
    """Verify sign convention: +Y load -> +Y deflection, +Mz moment -> CCW rotation."""
    mesh = Mesh.from_path([(0, 0), (1000.0, 0)], elements_per_segment=10)
    material = Material("Steel", 200000, 0.3, 7.85e-9)
    section = rectangular(50, 100)
    solver = BeamSolver(mesh, material, section)
    
    # Case A: Positive FY at tip
    bc = BoundaryConditionSet()
    bc.fixed_support(0)
    lc_y = LoadCase()
    lc_y.point_load(x=1000.0, fy=1000.0)
    solver.solve_static(lc_y, bc)
    assert solver.displacements[31] > 0 # Y displacement at end (DOF 3*10 + 1)
    
    # Case B: Positive MZ at tip
    lc_m = LoadCase()
    lc_m.moment(x=1000.0, mz=100000.0)
    solver.solve_static(lc_m, bc)
    assert solver.displacements[32] > 0 # Theta at end (CCW) (DOF 3*10 + 2)

def test_triangular_load_coordinate():
    """Verify TriangularDistributedLoad(x=...) maps correctly."""
    mesh = Mesh.from_path([(0, 0), (1000.0, 0)], elements_per_segment=1)
    lc = LoadCase()
    
    # Triangular load from 0 to 1000, peak at start (xi=0)
    w_peak = -10.0
    lc.distributed_load(x_start=0, x_end=1000, distribution='triangular', w_peak=w_peak, peak_loc='start')
    F = lc.create_force_vector(mesh.num_dofs, mesh)
    
    # Analytical: Fy1 = (7*w1+3*w2)*L/20 = (7*-10+0)*1000/20 = -3500
    # Fy2 = (3*w1+7*w2)*L/20 = (3*-10+0)*1000/20 = -1500
    assert np.isclose(F[1], -3500.0)
    assert np.isclose(F[4], -1500.0)
