"""
solver.py
=========
Main FEA orchestrator that coordinates specialized engines.
"""

import numpy as np
import warnings
import pandas as pd
from typing import Optional, Union, List

from .mesh import Mesh
from .materials import Material
from .cross_sections import SectionProperties
from .properties import PropertySet
from .boundary_conditions import BoundaryConditionSet
from .loads import LoadCase, LoadCombination
from .static_analysis import StaticAnalysis
from .modal_analysis import ModalAnalysis
from .visualizer import BeamVisualizer
from .assembly import AssemblyEngine
from .analysis import AnalysisEngine, AnalysisType

class BeamSolver:
    """
    Coordinator for beam structural analysis.
    
    Delegates heavy lifting to specialized engines:
    - AssemblyEngine: Matrix construction.
    - AnalysisEngine: Solution execution.
    - ResultsEngine: Data extraction and formatting.
    """
    
    def __init__(self, mesh: Mesh, material: Union[Material, PropertySet, List[PropertySet]],
                 section: Optional[SectionProperties] = None, element_type: str = 'euler'):
        """
        Initialize beam solver.
        """
        self.mesh = mesh
        self.element_type = element_type.lower()
        
        # 1. Standardize properties into a unified PropertySet collector
        if isinstance(material, list):
            self.properties = PropertySet(name="Merged Properties")
            for ps in material:
                if isinstance(ps, PropertySet):
                    self.properties._assignments.extend(ps._assignments)
                else:
                    raise TypeError("List of properties must contain only PropertySet objects")
        elif isinstance(material, PropertySet):
            self.properties = material
        else:
            self.properties = PropertySet(material=material, section=section)

        # 2. Resolve and pre-calculate all element-wise properties
        if self.mesh and self.mesh.num_elements > 0:
            self.properties.resolve(self.mesh.num_elements)

        # System matrices
        self.K_global = None
        self.M_global = None
        
        # Result State
        self.displacements = None
        self.reactions = None
        self.last_frequencies = None
        self.last_mode_shapes = None
        self.last_modal_participation = None
        
        # Analysis objects
        self.static_solver = StaticAnalysis(use_sparse=True)
        self.modal_solver = ModalAnalysis()
        self.analysis_engine = AnalysisEngine(self)
        self.visualizer = BeamVisualizer(mesh)
        
        # State for reporting
        self.last_load_case = None
        self.last_bc_set = None
        self.recovery_strategy = 'consistent'
        
        # Result Caches
        self._cached_forces = None
        self._cached_forces_params = None
        self._cached_stresses = None
        self._cached_stresses_params = None

    def _validate_model(self, bc_set: BoundaryConditionSet = None):
        """
        Validate model state and cross-module consistency.
        """
        if self.mesh is None:
            raise ValueError("No mesh assigned to BeamSolver.")
        if self.mesh.num_nodes == 0:
            raise ValueError("Mesh has no nodes.")
        if self.mesh.num_elements == 0:
            raise ValueError("Mesh has no elements.")

        # 1. Property Coverage Validation
        # BeamSolver already calls self.properties.resolve() in __init__ which handles this
        if not self.properties._is_resolved:
             self.properties.resolve(self.mesh.num_elements)

        if bc_set is None:
            raise ValueError("No boundary conditions mapping (bc_set) provided for analysis.")

        # 2. Stability Checks
        constrained_dofs = bc_set.get_all_constrained_dofs()
        if not constrained_dofs and not bc_set.spring_supports:
            raise ValueError("No boundary conditions or spring supports defined. The structure is unstable.")

        # Check for basic rigid body stability
        has_x_const = any(dof % 3 == 0 for dof in constrained_dofs)
        has_y_const = any(dof % 3 == 1 for dof in constrained_dofs)
        
        if not has_x_const:
            warnings.warn("No X-direction constraints found. The beam may be free to slide axially.")
        if not has_y_const:
            raise ValueError("No Y-direction constraints found. The beam is unstable in the transverse direction.")

        # 3. Slenderness Ratio Check (L/h) - Per Element Validation
        # Collect all flagged elements first, then emit a single batched warning to avoid log noise.
        coords = self.mesh.nodes
        elements = self.mesh.elements
        low_slenderness = []  # list of (element_id, slenderness)

        for i in range(self.mesh.num_elements):
            node1, node2 = elements[i]
            L_el = np.linalg.norm(coords[node2] - coords[node1])
            sec = self.properties.get_section(i)

            # Section height (depth)
            if hasattr(sec, 'y_top') and sec.y_top is not None:
                h = sec.y_top - sec.y_bottom
            else:
                h = np.sqrt(sec.A)

            if h > 0:
                slenderness = L_el / h
                if slenderness < 10 and self.element_type == 'euler':
                    low_slenderness.append((i, slenderness))

        if low_slenderness:
            elem_summary = ", ".join(f"el[{i}] L/h={s:.1f}" for i, s in low_slenderness)
            warnings.warn(
                f"{len(low_slenderness)} element(s) have low slenderness ratio (L/h < 10). "
                f"Euler-Bernoulli elements may under-predict deflections for short/deep beams. "
                f"Affected: [{elem_summary}]. Consider using element_type='timoshenko'.",
                stacklevel=2
            )

        # Coordinates Bounds Check
        max_x = np.max(self.mesh.nodes[:, 0])
        for load in (self.last_load_case.loads if self.last_load_case else []):
            if hasattr(load, 'x') and load.x is not None:
                if load.x > max_x + 1e-6:
                    warnings.warn(f"Load at x={load.x} is outside beam length ({max_x})")

    def _assemble_stiffness_matrix(self):
        """Internal helper to trigger assembly."""
        self.K_global, _ = AssemblyEngine.assemble(
            self.mesh, self.properties, self.element_type, 'stiffness'
        )

    def _assemble_mass_matrix(self):
        """Internal helper to trigger assembly."""
        _, self.M_global = AssemblyEngine.assemble(
            self.mesh, self.properties, self.element_type, 'mass'
        )

    def assemble_global_matrices(self):
        """Force assembly of both K and M."""
        self.K_global, self.M_global = AssemblyEngine.assemble(
            self.mesh, self.properties, self.element_type, 'both'
        )

    def solve_static(self, load_case: LoadCase, bc_set: BoundaryConditionSet):
        """Public wrapper for static analysis."""
        self._validate_model(bc_set)
        return self.analysis_engine.run(AnalysisType.STATIC, load_case=load_case, bc_set=bc_set)

    def solve_modal(self, bc_set: BoundaryConditionSet, num_modes: Optional[int] = None, 
                   load_case: Optional[Union[LoadCase, 'LoadCombination']] = None):
        """Public wrapper for modal analysis."""
        self._validate_model(bc_set)
        return self.analysis_engine.run(AnalysisType.MODAL, bc_set=bc_set, 
                                       num_modes=num_modes, load_case=load_case)

    def solve_batch(self, load_cases: List[LoadCase], bc_set: BoundaryConditionSet,
                    mode: str = 'light', failure_criterion: str = 'tsai_wu'):
        """Public wrapper for batch static analysis."""
        if not isinstance(load_cases, list) or len(load_cases) == 0:
            raise ValueError("load_cases must be a non-empty list of LoadCase objects.")
        if not all(isinstance(lc, LoadCase) for lc in load_cases):
            raise TypeError("All items in load_cases must be LoadCase objects.")
        return self.analysis_engine.run(AnalysisType.BATCH, load_cases=load_cases, 
                                       bc_set=bc_set, mode=mode,
                                       failure_criterion=failure_criterion)

    # --- Result Extraction Wrappers ---
    
    def verify_equilibrium(self) -> dict:
        """
        Verify analytical equilibrium of the last static solution.
        Returns residuals for forces and moments.
        """
        from .post_processing import ResultsEngine
        return ResultsEngine.verify_equilibrium(self)

    def calculate_internal_forces(self, num_points: int = None, strategy: str = None) -> dict:
        from .post_processing import InternalForceEngine
        strat = strategy or self.recovery_strategy
        key = (num_points, strat)
        if self._cached_forces is not None and self._cached_forces_params == key:
            return self._cached_forces
        self._cached_forces = InternalForceEngine.calculate(self, num_points, strat)
        self._cached_forces_params = key
        return self._cached_forces

    def calculate_stresses(self, num_x_points: int = None, num_y_points: int = 20, num_z_points: int = 20) -> dict:
        from .post_processing import StressEngine
        params = (num_x_points, num_y_points, num_z_points, self.recovery_strategy)
        if self._cached_stresses is not None and self._cached_stresses_params == params:
            return self._cached_stresses
        self._cached_stresses = StressEngine.calculate(self, num_x_points, num_y_points, num_z_points)
        self._cached_stresses_params = params
        return self._cached_stresses

    def get_max_deflection(self) -> 'pd.Series':
        from .post_processing import ResultsEngine
        df = ResultsEngine.get_nodal_displacements(self)
        return df.iloc[df['res'].idxmax()]

    def get_max_internal_forces(self, num_points: int = 100) -> dict:
        res = self.calculate_internal_forces(num_points)
        v_idx, m_idx = np.argmax(np.abs(res['shear_forces'])), np.argmax(np.abs(res['bending_moments']))
        return {
            'shear': {'value': res['shear_forces'][v_idx], 'x': res['path_positions'][v_idx]},
            'moment': {'value': res['bending_moments'][m_idx], 'x': res['path_positions'][m_idx]}
        }

    def export_results(self, filepath: str):
        """Export batch results to CSV."""
        if not hasattr(self, 'batch_summary'):
            raise ValueError("No batch summary available. Run solve_batch() first.")
        self.batch_summary.to_csv(filepath, index=False)
        return filepath

    def generate_batch_report(self, output_path: str):
        from .report_generator import BatchReportGenerator
        if not hasattr(self, 'batch_summary'):
            raise ValueError("No batch results to report. Run solve_batch() first.")
        return BatchReportGenerator(self).generate_report(output_path)

    def generate_report(self, output_path: str, deformation_scale: Union[float, str] = 'auto', 
                        failure_criterion: str = 'tsai_wu'):
        from .report_generator import BeamReportGenerator
        if self.displacements is None and self.last_frequencies is None:
            raise ValueError("Must run analysis before generating report.")
        
        gen = BeamReportGenerator(
            solver=self, mesh=self.mesh,
            load_case=self.last_load_case, bc_set=self.last_bc_set,
            displacements=self.displacements, reactions=self.reactions,
            failure_criterion=failure_criterion
        )
        if self.last_frequencies is not None:
            gen.frequencies, gen.mode_shapes = self.last_frequencies, self.last_mode_shapes
        return gen.generate_report(output_path, deformation_scale)

    def visualize(self, analysis_type: str = 'static', **kwargs):
        if analysis_type == 'static':
            return self.visualizer.plot_deformed_shape(self.displacements, **kwargs)
        elif analysis_type in ['shear', 'moment']:
            num_points = kwargs.pop('num_points', max(50, 4 * self.mesh.num_elements))
            forces = self.calculate_internal_forces(num_points)
            if analysis_type == 'shear':
                return self.visualizer.plot_shear_force(forces['shear_forces'], forces['path_positions'], **kwargs)
            return self.visualizer.plot_bending_moment(forces['bending_moments'], forces['path_positions'], **kwargs)

if __name__ == "__main__":
    print("Beam Solver Orchestrator Ready")
