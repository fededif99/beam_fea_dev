"""
Beam FEA Package
================
A comprehensive finite element analysis package for beam structures.

Module Organization:
-------------------
- materials.py: Material property definitions
- cross_sections.py: Cross-section property calculations
- mesh.py: Mesh generation and management
- element_matrices.py: Local element matrix calculations
- loads.py: Load definitions and application
- boundary_conditions.py: Support and constraint definitions
- static_analysis.py: Static structural analysis
- modal_analysis.py: Modal/eigenvalue analysis
- visualizer.py: Result visualization
- solver.py: Main coordinating solver

Usage Example:
--------------
>>> from beam_fea import BeamSolver, MeshGenerator, get_material
>>> from beam_fea.cross_sections import rectangular
>>> from beam_fea.loads import LoadCase
>>> 
>>> # Create mesh
>>> mesh = MeshGenerator.beam_mesh_1d(length=1000, num_elements=20)
>>> 
>>> # Define material and section
>>> material = get_material('steel')
>>> section = rectangular(width=50, height=100)
>>> 
>>> # Create solver
>>> solver = BeamSolver(mesh, material, section)
>>> 
>>> # Setup and solve
>>> bc_set = BoundaryConditionSet()
>>> bc_set.pinned_support(0)
>>> bc_set.pinned_support(20)
>>> load_case = LoadCase("Point Load")
>>> load_case.point_load(node=10, fy=-100)
>>> 
>>> displacements = solver.solve_static(load_case, bc_set)
>>> solver.visualize()
"""

# Set headless matplotlib backend before any other imports
import os
import matplotlib
if os.environ.get('GITHUB_ACTIONS') == 'true' or os.environ.get('DISPLAY') is None:
    try:
        matplotlib.use('Agg')
    except Exception:
        pass


# Core modules
# Core modules
from .materials import Material, get_material, MATERIAL_LIBRARY
from .cross_sections import (
    SectionProperties,
    RectangularSection, CircularSection, IBeamSection, 
    BoxSection, TBeamSection, CChannelSection, HollowCircularSection, LSection,
    rectangular, circular, i_beam, box, t_beam, c_channel, hollow_circular, l_section
)
from .mesh import Mesh, Node, Element, MeshGenerator, MeshRefinement
from .element_matrices import EulerBernoulliElement, TimoshenkoElement
from .boundary_conditions import (
    BoundaryConditionSet,
    FixedSupport, PinnedSupport, RollerSupport
)
from .loads import (
    LoadCase, LoadCombination, PointLoad, 
    UniformDistributedLoad, TrapezoidalDistributedLoad
)
from .static_analysis import StaticAnalysis, StressAnalysis
from .modal_analysis import ModalAnalysis
from .visualizer import BeamVisualizer
from .solver import BeamSolver
from .report_generator import BeamReportGenerator
from .plot_style import PlotStyle, smart_units, DEFAULT_STYLE

from ._version import __version__
__author__ = 'Beam FEA Development Team'

__all__ = [
    # Materials
    'Material', 'get_material', 'MATERIAL_LIBRARY',
    
    # Cross-sections
    'SectionProperties',
    'RectangularSection', 'CircularSection', 'IBeamSection',
    'BoxSection', 'TBeamSection', 'CChannelSection', 'HollowCircularSection', 'LSection',
    'rectangular', 'circular', 'i_beam', 'box', 't_beam', 'c_channel', 'hollow_circular', 'l_section',
    
    # Mesh
    'Mesh', 'Node', 'Element', 'MeshGenerator', 'MeshRefinement',
    
    # Elements
    'EulerBernoulliElement', 'TimoshenkoElement',
    
    # Boundary Conditions
    'BoundaryConditionSet',
    'FixedSupport', 'PinnedSupport', 'RollerSupport',
    
    # Loads
    'LoadCase', 'LoadCombination',
    'PointLoad', 'UniformDistributedLoad', 'TrapezoidalDistributedLoad',
    
    # Analysis
    'StaticAnalysis', 'ModalAnalysis', 'StressAnalysis',
    
    # Visualization
    'BeamVisualizer',
    
    # Plot Style
    'PlotStyle', 'smart_units', 'DEFAULT_STYLE',
    
    # Report Generation
    'BeamReportGenerator',
    
    # Main Solver
    'BeamSolver',
]
