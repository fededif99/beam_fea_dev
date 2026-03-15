"""
Unit tests for beam_fea.mesh module
"""

import pytest
import numpy as np
import warnings
from beam_fea.mesh import Mesh, MeshRefinement


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


class TestMeshConstructors:
    """Test Mesh static constructors (from_path, from_arc)"""
    
    def test_from_path_single_line(self):
        """Test generating line mesh via from_path"""
        mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=10)
        assert mesh.num_nodes == 11
        assert mesh.num_elements == 10
        
        # Check first and last node positions
        assert mesh.nodes[0, 0] == 0
        assert mesh.nodes[-1, 0] == 100
    
    def test_from_path_multi_segment(self):
        """Test multi-segment path mesh"""
        points = [(0, 0), (100, 0), (100, 100)]
        mesh = Mesh.from_path(points, elements_per_segment=[10, 10])
        assert mesh.num_nodes == 21
        assert mesh.num_elements == 20
        assert mesh.waypoint_nodes == [0, 10, 20]
    
    def test_from_path_invalid_points(self):
        """Test error for too few points"""
        with pytest.raises(ValueError, match="Path must have at least 2 points"):
            Mesh.from_path([(0,0)], elements_per_segment=1)
    
    def test_from_path_grading(self):
        """Test graded mesh via from_path"""
        mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=10, grading_ratios=2.0)
        sizes = MeshRefinement.get_element_sizes(mesh)
        assert sizes[0] < sizes[-1]
    
    def test_from_arc(self):
        """Test generating curved beam mesh"""
        mesh = Mesh.from_arc(
            radius=100,
            start_angle=0,
            end_angle=90,
            num_elements=10
        )
        assert mesh.num_nodes == 11
        assert mesh.num_elements == 10
    
    def test_from_arc_negative_radius(self):
        """Test that negative radius raises error"""
        with pytest.raises(ValueError, match="Radius must be positive"):
            Mesh.from_arc(
                radius=-100,
                start_angle=0,
                end_angle=90,
                num_elements=10
            )


class TestMeshRefinement:
    """Test MeshRefinement class"""
    
    def test_uniform_refinement(self):
        """Test uniform mesh refinement"""
        mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=5)
        refined = MeshRefinement.refine_uniform(mesh, refinement_level=1)
        
        assert refined.num_elements == 10  # Each element split in 2
        assert refined.num_nodes == 11
    
    def test_multiple_refinements(self):
        """Test multiple refinement levels"""
        mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=5)
        refined = MeshRefinement.refine_uniform(mesh, refinement_level=2)
        
        assert refined.num_elements == 20  # 5 * 2^2
    
    def test_get_element_sizes(self):
        """Test getting element sizes"""
        mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=10)
        sizes = MeshRefinement.get_element_sizes(mesh)
        
        assert len(sizes) == 10
        # All elements should be approximately the same size
        assert np.allclose(sizes, 10.0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
