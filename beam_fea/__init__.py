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
Usage Example:
--------------
>>> from beam_fea import BeamSolver, Mesh, get_material
>>> from beam_fea.cross_sections import rectangular
>>> from beam_fea.loads import LoadCase
>>> 
>>> # Create mesh
>>> mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=20)
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
from .materials import Material, get_material, MATERIAL_LIBRARY
from .cross_sections import (
    SectionProperties,
    RectangularSection, CircularSection, IBeamSection, 
    BoxSection, TBeamSection, CChannelSection, HollowCircularSection, LSection,
    rectangular, circular, i_beam, box, t_beam, c_channel, hollow_circular, l_section
)
from .mesh import Mesh, MeshRefinement
from .batch import BatchProcessor
from .properties import PropertySet
from .composites import Ply, Laminate
from .element_matrices import EulerBernoulliElement, TimoshenkoElement
from .boundary_conditions import (
    BoundaryConditionSet,
    FixedSupport, PinnedSupport, RollerSupport
)
from .loads import (
    LoadCase, LoadCombination, PointLoad, ConcentratedMoment, DistributedLoad
)
from .static_analysis import StaticAnalysis, StressAnalysis
from .failure_criteria import (
    FailureCriterion,
    VonMisesCriterion,
    TrescaCriterion,
    MaxPrincipalStressCriterion,
    MaximumStressCriterion,
    TsaiHillCriterion,
    TsaiWuCriterion,
    MaximumStrainCriterion,
)
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
    'Mesh', 'MeshRefinement',
    
    # Batch
    'BatchProcessor',

    # Properties
    'PropertySet',

    # Composites
    'Ply', 'Laminate',

    # Elements
    'EulerBernoulliElement', 'TimoshenkoElement',
    
    # Boundary Conditions
    'BoundaryConditionSet',
    'FixedSupport', 'PinnedSupport', 'RollerSupport',
    
    # Loads
    'LoadCase', 'LoadCombination',
    'PointLoad', 'ConcentratedMoment', 'DistributedLoad',
    
    # Analysis
    'StaticAnalysis', 'ModalAnalysis', 'StressAnalysis',

    # Failure Criteria
    'FailureCriterion',
    'VonMisesCriterion', 'TrescaCriterion', 'MaxPrincipalStressCriterion',
    'MaximumStressCriterion', 'TsaiHillCriterion', 'TsaiWuCriterion',
    'MaximumStrainCriterion',
    
    # Visualization
    'BeamVisualizer',
    
    # Plot Style
    'PlotStyle', 'smart_units', 'DEFAULT_STYLE',
    
    # Report Generation
    'BeamReportGenerator',
    
    # Main Solver
    'BeamSolver',
]
