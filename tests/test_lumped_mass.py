import pytest
import numpy as np
from beam_fea.mesh import Mesh
from beam_fea.materials import Material, get_material
from beam_fea.cross_sections import RectangularSection
from beam_fea.properties import PropertySet
from beam_fea.loads import LoadCase, LumpedMass, PointLoad
from beam_fea.boundary_conditions import BoundaryConditionSet
from beam_fea.solver import BeamSolver

def create_model():
    mesh = Mesh()
    mesh.add_node(0.0, 0.0)
    mesh.add_node(500.0, 0.0)
    mesh.add_node(1000.0, 0.0)
    mesh.add_element(0, 1)
    mesh.add_element(1, 2)

    import copy
    mat = copy.copy(get_material('Steel'))
    # Change density to 0 to isolate the lumped mass effect
    mat.rho = 0.0
    sec = RectangularSection(width=50, height=100).properties()
    bc_set = BoundaryConditionSet()
    bc_set.fixed_support(0)

    props = PropertySet(material=mat, section=sec)
    solver = BeamSolver(mesh, props)
    return solver, bc_set, mesh

def test_lumped_mass_gravity_deflection():
    """Verify that a LumpedMass with apply_gravity=True acts like a PointLoad."""
    solver1, bc_set1, _ = create_model()
    # Test with standard PointLoad
    lc_point = LoadCase()
    mass_kg = 50.0  # 50 kg mass
    weight_n = -mass_kg * 9.80665
    lc_point.point_load(node=2, fy=weight_n)
    
    solver1.solve_static(lc_point, bc_set1)
    defl1 = solver1.get_max_deflection()['res']

    solver2, bc_set2, _ = create_model()
    # Test with LumpedMass
    lc_mass = LoadCase()
    lc_mass.lumped_mass(node=2, m=mass_kg, apply_gravity=True)
    
    solver2.solve_static(lc_mass, bc_set2)
    defl2 = solver2.get_max_deflection()['res']

    assert deflate_is_close(defl1, defl2), f"Expected {defl1}, got {defl2}"

def deflate_is_close(a, b, rtol=1e-5):
    return abs(a - b) <= rtol * max(abs(a), abs(b))

def test_lumped_mass_modal_shift():
    """Verify that adding a LumpedMass lowers the natural frequency."""
    # Since mat.rho = 0, standard mass matrix will be zeros.
    # To have a valid initial frequency, we need non-zero density.
    mesh = Mesh()
    mesh.add_node(0.0, 0.0)
    mesh.add_node(500.0, 0.0)
    mesh.add_node(1000.0, 0.0)
    mesh.add_element(0, 1)
    mesh.add_element(1, 2)

    mat = get_material('Steel') # Default rho is > 0
    sec = RectangularSection(width=50, height=100).properties()
    bc_set = BoundaryConditionSet()
    bc_set.fixed_support(0)

    props = PropertySet(material=mat, section=sec)
    
    solver1 = BeamSolver(mesh, props)
    freqs1, _ = solver1.solve_modal(bc_set, num_modes=1)
    f1 = freqs1[0]

    # Add Lumped Mass
    lc_mass = LoadCase()
    lc_mass.lumped_mass(node=2, m=10.0, Izz=1.0) # Apply 10 kg at the tip
    
    solver2 = BeamSolver(mesh, props)
    freqs2, _ = solver2.solve_modal(bc_set, num_modes=1, load_case=lc_mass)
    f2 = freqs2[0]

    # Added mass should decrease natural frequency
    assert f2 < f1, f"Expected frequency to drop. Original: {f1}, With Mass: {f2}"

def test_lumped_mass_no_mass_error():
    """Verify that solve_modal raises an error if an empty load case is given."""
    solver, bc_set, _ = create_model()
    empty_lc = LoadCase()
    empty_lc.point_load(node=2, fy=-100) # Contains no LumpedMass
    
    with pytest.raises(ValueError, match="does not contain any LumpedMass loads"):
        solver.solve_modal(bc_set, load_case=empty_lc)
