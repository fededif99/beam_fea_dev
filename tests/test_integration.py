"""
Integration tests for beam_fea package
"""

import pytest
import numpy as np
from beam_fea import (
    BeamSolver, Mesh, get_material,
    rectangular, LoadCase, BoundaryConditionSet
)


class TestSimpleBeamAnalysis:
    """Integration test for simple beam analysis"""
    
    def test_simply_supported_beam_point_load(self):
        """Test simply supported beam with central point load"""
        # Create mesh
        mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=20)
        
        # Define material and section
        material = get_material('steel')
        section = rectangular(width=50, height=100)
        
        # Create solver
        solver = BeamSolver(mesh, material, section)
        
        # Setup boundary conditions (pinned at both ends)
        bc_set = BoundaryConditionSet()
        bc_set.pinned_support(0)
        bc_set.pinned_support(20)
        
        # Apply central point load
        load_case = LoadCase("Central Point Load")
        load_case.point_load(node=10, fy=-1000)  # 1000 N downward
        
        # Solve
        displacements = solver.solve_static(load_case, bc_set)
        
        # Check that we got results
        assert displacements is not None
        assert len(displacements) == mesh.num_dofs
        
        # Check that supports have zero displacement
        assert abs(displacements[1]) < 1e-6  # y-displacement at node 0
        assert abs(displacements[3*20 + 1]) < 1e-6  # y-displacement at node 20
        
        # Check that maximum deflection is at center (negative = downward)
        v_displacements = displacements[1::3]
        max_deflection_idx = np.argmin(v_displacements)
        assert max_deflection_idx == 10  # Should be at center
    
    def test_cantilever_beam_end_load(self):
        """Test cantilever beam with end load"""
        # Create mesh
        mesh = Mesh.from_path([(0, 0), (500, 0)], elements_per_segment=10)
        
        # Define material and section
        material = get_material('aluminum')
        section = rectangular(width=30, height=60)
        
        # Create solver
        solver = BeamSolver(mesh, material, section)
        
        # Fixed support at left end
        bc_set = BoundaryConditionSet()
        bc_set.fixed_support(0)
        
        # Apply end load
        load_case = LoadCase("End Load")
        load_case.point_load(node=10, fy=-500)
        
        # Solve
        displacements = solver.solve_static(load_case, bc_set)
        
        # Check that fixed end has zero displacement and rotation
        assert abs(displacements[0]) < 1e-6  # x-displacement
        assert abs(displacements[1]) < 1e-6  # y-displacement
        assert abs(displacements[2]) < 1e-6  # rotation
        
        # Check that free end has maximum deflection
        v_displacements = displacements[1::3]
        assert np.argmin(v_displacements) == 10
    
    def test_modal_analysis(self):
        """Test modal analysis for natural frequencies"""
        # Create mesh
        mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=20)
        
        # Define material and section
        material = get_material('steel')
        section = rectangular(width=50, height=100)
        
        # Create solver
        solver = BeamSolver(mesh, material, section)
        
        # Simply supported
        bc_set = BoundaryConditionSet()
        bc_set.pinned_support(0)
        bc_set.pinned_support(20)
        
        # Solve for first 5 modes
        frequencies, mode_shapes = solver.solve_modal(bc_set, num_modes=5)
        
        # Check that we got 5 frequencies
        assert len(frequencies) == 5
        
        # Check that frequencies are positive
        assert np.all(frequencies > 0)
        
        # Check that frequencies are non-decreasing (may have duplicates)
        assert np.all(np.diff(frequencies) >= 0)
        
        # Check mode shapes shape
        assert mode_shapes.shape[0] == mesh.num_dofs
        assert mode_shapes.shape[1] == 5


class TestDistributedLoad:
    """Test distributed loads"""
    
    def test_uniformly_distributed_load(self):
        """Test beam with uniformly distributed load"""
        # Create mesh
        mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=10)
        
        # Define material and section
        material = get_material('steel')
        section = rectangular(width=50, height=100)
        
        # Create solver
        solver = BeamSolver(mesh, material, section)
        
        # Simply supported
        bc_set = BoundaryConditionSet()
        bc_set.pinned_support(0)
        bc_set.pinned_support(10)
        
        # Apply distributed load to all elements
        load_case = LoadCase("UDL")
        for i in range(10):
            load_case.distributed_load(element=i, distribution='uniform', wy=-10)  # 10 N/mm
        
        # Solve
        displacements = solver.solve_static(load_case, bc_set)
        
        # Check that we got results
        assert displacements is not None
        
        # Maximum deflection should be near center
        v_displacements = displacements[1::3]
        max_deflection_idx = np.argmin(v_displacements)
        assert 4 <= max_deflection_idx <= 6  # Should be near center


class TestMultiSpanBeam:
    """Test multi-span continuous beam"""
    
    def test_two_span_beam(self):
        """Test two-span continuous beam"""
        # Create mesh
        mesh = Mesh.from_path([(0, 0), (500, 0), (1000, 0)], elements_per_segment=[10, 10])
        
        # Define material and section
        material = get_material('steel')
        section = rectangular(width=50, height=100)
        
        # Create solver
        solver = BeamSolver(mesh, material, section)
        
        # Pinned supports at nodes 0, 10, and 20
        bc_set = BoundaryConditionSet()
        bc_set.pinned_support(0)
        bc_set.pinned_support(10)
        bc_set.pinned_support(20)
        
        # Apply loads
        load_case = LoadCase("Point Loads")
        load_case.point_load(node=5, fy=-1000)
        load_case.point_load(node=15, fy=-1000)
        
        # Solve
        displacements = solver.solve_static(load_case, bc_set)
        
        # Check that supports have zero y-displacement
        assert abs(displacements[1]) < 1e-6
        assert abs(displacements[3*10 + 1]) < 1e-6
        assert abs(displacements[3*20 + 1]) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
