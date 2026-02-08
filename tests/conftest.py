"""
Test configuration and fixtures for beam_fea tests
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import beam_fea
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def simple_mesh():
    """Fixture providing a simple beam mesh"""
    from beam_fea.mesh import MeshGenerator
    return MeshGenerator.beam_mesh_1d(length=1000, num_elements=10)


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
