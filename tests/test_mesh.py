"""
Unit tests for beam_fea.mesh module
"""

import pytest
import numpy as np
import warnings
from beam_fea.mesh import Mesh, MeshGenerator, MeshRefinement


class TestMesh:
    """Test Mesh class"""
    
    def test_empty_mesh(self):
        """Test creating empty mesh"""
        mesh = Mesh()
        assert mesh.num_nodes == 0
        assert mesh.num_elements == 0
        assert mesh.num_dofs == 0
    
    def test_add_node(self):
        """Test adding nodes to mesh"""
        mesh = Mesh()
        node_id = mesh.add_node(10.0, 20.0)
        assert node_id == 0
        assert mesh.num_nodes == 1
        
        node_id2 = mesh.add_node(30.0, 40.0)
        assert node_id2 == 1
        assert mesh.num_nodes == 2
    
    def test_add_element(self):
        """Test adding elements to mesh"""
        mesh = Mesh()
        mesh.add_node(0.0, 0.0)
        mesh.add_node(10.0, 0.0)
        
        elem_id = mesh.add_element(0, 1)
        assert elem_id == 0
        assert mesh.num_elements == 1
    
    def test_add_element_invalid_node(self):
        """Test that adding element with invalid node raises error"""
        mesh = Mesh()
        mesh.add_node(0.0, 0.0)
        
        with pytest.raises(ValueError, match="Invalid node1 ID"):
            mesh.add_element(5, 0)  # Node 5 doesn't exist
    
    def test_add_element_same_nodes(self):
        """Test that adding element with same nodes raises error"""
        mesh = Mesh()
        mesh.add_node(0.0, 0.0)
        
        with pytest.raises(ValueError, match="Cannot create element with same node"):
            mesh.add_element(0, 0)
    
    def test_num_dofs(self):
        """Test DOF calculation"""
        mesh = Mesh()
        mesh.add_node(0.0, 0.0)
        mesh.add_node(10.0, 0.0)
        assert mesh.num_dofs == 6  # 2 nodes * 3 DOFs per node


class TestMeshGenerator:
    """Test MeshGenerator class"""
    
    def test_line_mesh(self):
        """Test generating line mesh"""
        mesh = MeshGenerator.line_mesh((0, 0), (100, 0), num_elements=10)
        assert mesh.num_nodes == 11
        assert mesh.num_elements == 10
        
        # Check first and last node positions
        assert mesh.nodes[0, 0] == 0
        assert mesh.nodes[-1, 0] == 100
    
    def test_beam_mesh_1d_horizontal(self):
        """Test generating horizontal beam mesh"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = MeshGenerator.beam_mesh_1d(length=1000, num_elements=20)
        assert mesh.num_nodes == 21
        assert mesh.num_elements == 20
        assert mesh.nodes[-1, 0] == 1000
        assert mesh.nodes[-1, 1] == 0
    
    def test_beam_mesh_1d_vertical(self):
        """Test generating vertical beam mesh"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = MeshGenerator.beam_mesh_1d(
                length=500, 
                num_elements=10, 
                orientation='vertical'
            )
        assert mesh.num_nodes == 11
        assert mesh.nodes[-1, 0] == 0
        assert mesh.nodes[-1, 1] == 500
    
    def test_beam_mesh_negative_length(self):
        """Test that negative length raises error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="Length must be positive"):
                MeshGenerator.beam_mesh_1d(length=-100, num_elements=10)
    
    def test_beam_mesh_zero_elements(self):
        """Test that zero elements raises error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="Number of elements must be at least 1"):
                MeshGenerator.beam_mesh_1d(length=100, num_elements=0)
    
    def test_beam_mesh_invalid_orientation(self):
        """Test that invalid orientation raises error"""
        with pytest.raises(ValueError, match="Orientation must be"):
            MeshGenerator.beam_mesh_1d(
                length=100, 
                num_elements=10, 
                orientation='diagonal'
            )
    
    def test_curved_beam(self):
        """Test generating curved beam mesh"""
        mesh = MeshGenerator.curved_beam(
            radius=100,
            start_angle=0,
            end_angle=90,
            num_elements=10
        )
        assert mesh.num_nodes == 11
        assert mesh.num_elements == 10
    
    def test_curved_beam_negative_radius(self):
        """Test that negative radius raises error"""
        with pytest.raises(ValueError, match="Radius must be positive"):
            MeshGenerator.curved_beam(
                radius=-100,
                start_angle=0,
                end_angle=90,
                num_elements=10
            )
    
    def test_multi_span_beam(self):
        """Test generating multi-span beam"""
        mesh = MeshGenerator.multi_span_beam(
            span_lengths=[500, 800, 500],
            elements_per_span=[5, 8, 5]
        )
        assert mesh.num_nodes == 19  # 5+1 + 8 + 5
        assert mesh.num_elements == 18  # 5 + 8 + 5


class TestMeshRefinement:
    """Test MeshRefinement class"""
    
    def test_uniform_refinement(self):
        """Test uniform mesh refinement"""
        mesh = MeshGenerator.beam_mesh_1d(length=100, num_elements=5)
        refined = MeshRefinement.refine_uniform(mesh, refinement_level=1)
        
        assert refined.num_elements == 10  # Each element split in 2
        assert refined.num_nodes == 11
    
    def test_multiple_refinements(self):
        """Test multiple refinement levels"""
        mesh = MeshGenerator.beam_mesh_1d(length=100, num_elements=5)
        refined = MeshRefinement.refine_uniform(mesh, refinement_level=2)
        
        assert refined.num_elements == 20  # 5 * 2^2
    
    def test_get_element_sizes(self):
        """Test getting element sizes"""
        mesh = MeshGenerator.beam_mesh_1d(length=100, num_elements=10)
        sizes = MeshRefinement.get_element_sizes(mesh)
        
        assert len(sizes) == 10
        # All elements should be approximately the same size
        assert np.allclose(sizes, 10.0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
