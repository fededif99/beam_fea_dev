import pytest
import numpy as np
from beam_fea import Mesh, get_material, rectangular, BeamSolver, PropertySet

def test_single_propertyset_global():
    # Test that a single PropertySet with no elements applies to all
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=5)
    steel = get_material('steel_a36')
    sec = rectangular(10, 20)
    
    ps = PropertySet(material=steel, section=sec)
    solver = BeamSolver(mesh, ps)
    
    for i in range(mesh.num_elements):
        assert solver.properties.get_material(i) == steel
        assert solver.properties.get_section(i) == sec

def test_multiple_propertysets_precedence():
    # Test that later assignments override earlier ones
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=10)
    steel = get_material('steel_a36')
    alum = get_material('aluminum_7075')
    sec1 = rectangular(10, 20)
    sec2 = rectangular(20, 40)
    
    ps = PropertySet(name="Multi Set")
    # Assignment 1: All are Steel/Sec1
    ps.add(material=steel, section=sec1)
    # Assignment 2: Elements 5-9 are Alum/Sec2
    ps.add(material=alum, section=sec2, elements=range(5, 10))
    
    solver = BeamSolver(mesh, ps)
    
    for i in range(5):
        assert solver.properties.get_material(i) == steel
        assert solver.properties.get_section(i) == sec1
    for i in range(5, 10):
        assert solver.properties.get_material(i) == alum
        assert solver.properties.get_section(i) == sec2

def test_list_merging_logic():
    # Test that passing a list of sets to the solver works (merging)
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=10)
    steel = get_material('steel_a36')
    sec1 = rectangular(10, 20)
    sec2 = rectangular(20, 40)
    
    ps1 = PropertySet(material=steel, section=sec1)
    ps2 = PropertySet(section=sec2, elements=range(5, 10))
    
    solver = BeamSolver(mesh, [ps1, ps2])
    
    assert solver.properties.get_material(9) == steel
    assert solver.properties.get_section(9) == sec2

def test_validation_incomplete_coverage():
    # Test that missing assignments raise a ValueError
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=10)
    steel = get_material('steel_a36')
    sec = rectangular(10, 20)
    
    # Assign only elements 0-4
    ps = PropertySet(material=steel, section=sec, elements=range(5))
    with pytest.raises(ValueError, match="Property Assignment Incomplete"):
        BeamSolver(mesh, ps)

def test_legacy_compatibility():
    # Test that passing material/section directly still works
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=5)
    steel = get_material('steel_a36')
    sec = rectangular(10, 20)
    
    solver = BeamSolver(mesh, material=steel, section=sec)
    
    for i in range(5):
        assert solver.properties.get_material(i) == steel
        assert solver.properties.get_section(i) == sec

def test_partial_overrides_mismatch():
    # Test a set that only provides material for some elements, 
    # relying on a global set for section
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=5)
    steel = get_material('steel_a36')
    alum = get_material('aluminum_7075')
    sec = rectangular(10, 20)
    
    ps = PropertySet(material=steel, section=sec)
    ps.add(material=alum, elements=[2])
    
    solver = BeamSolver(mesh, ps)
    
    assert solver.properties.get_material(2) == alum
    assert solver.properties.get_section(2) == sec # Still sec from base assignment
