"""
Test configuration and fixtures for beam_fea tests
"""


import pytest
from pathlib import Path


@pytest.fixture
def simple_mesh():
    """Fixture providing a simple beam mesh"""
    from beam_fea.mesh import Mesh
    return Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=10)


@pytest.fixture
def steel_material():
    """Fixture providing steel material"""
    from beam_fea.materials import get_material
    return get_material('steel')


@pytest.fixture
def rectangular_section():
    """Fixture providing rectangular cross-section"""
    from beam_fea.cross_sections import rectangular
    return rectangular(width=50, height=100)
