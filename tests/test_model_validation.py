import pytest
import warnings
from beam_fea import BeamSolver, Mesh, Material, LoadCase, BoundaryConditionSet
from beam_fea.cross_sections import RectangularSection

def test_missing_mesh():
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 10).properties()
    with pytest.raises(ValueError, match="No mesh assigned"):
        solver = BeamSolver(None, material, section)
        solver.solve_static(LoadCase(), BoundaryConditionSet())

def test_empty_mesh():
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 10).properties()
    mesh = Mesh()
    solver = BeamSolver(mesh, material, section)
    with pytest.raises(ValueError, match="Mesh has no nodes"):
        solver.solve_static(LoadCase(), BoundaryConditionSet())

def test_missing_properties():
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=1)
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 10).properties()
    
    with pytest.raises(ValueError, match="Property Assignment Incomplete"):
        solver = BeamSolver(mesh, None, section)
        
    with pytest.raises(ValueError, match="Property Assignment Incomplete"):
        solver = BeamSolver(mesh, material, None)

def test_no_bc():
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=1)
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 10).properties()
    solver = BeamSolver(mesh, material, section)
    
    with pytest.raises(ValueError, match="No boundary conditions or spring supports defined"):
        solver.solve_static(LoadCase(), BoundaryConditionSet())

def test_instability_y():
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=1)
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 10).properties()
    solver = BeamSolver(mesh, material, section)
    
    bc = BoundaryConditionSet()
    bc.roller_support(0, direction='x') # No Y constraint
    
    with pytest.raises(ValueError, match="unstable in the transverse direction"):
        solver.solve_static(LoadCase(), bc)

def test_slenderness_warning():
    # Stout beam: L=50, H=20 -> L/H = 2.5
    mesh = Mesh.from_path([(0, 0), (50, 0)], elements_per_segment=5)
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 20).properties()
    
    # Use Euler for a stout beam -> should warn
    solver = BeamSolver(mesh, material, section, element_type='euler')
    
    bc = BoundaryConditionSet()
    bc.fixed_support(0)
    
    with pytest.warns(UserWarning, match=r"element\(s\) have low slenderness ratio"):
        solver.solve_static(LoadCase(), bc)

def test_axial_stability_warning():
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=1)
    material = Material('Steel', E=210000, nu=0.3, rho=7.85e-6)
    section = RectangularSection(10, 10).properties()
    solver = BeamSolver(mesh, material, section)
    
    bc = BoundaryConditionSet()
    bc.roller_support(0, direction='y') # Only Y constraint
    
    with pytest.warns(UserWarning, match="No X-direction constraints found"):
        solver.solve_static(LoadCase(), bc)
