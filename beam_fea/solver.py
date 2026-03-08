"""
solver.py
=========
Main FEA solver that coordinates all modules.
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple, List
from .mesh import Mesh, MeshGenerator
from .materials import Material, get_material
from .cross_sections import CrossSection, SectionProperties
from .properties import PropertySet
from .element_matrices import EulerBernoulliElement, TimoshenkoElement
from .boundary_conditions import BoundaryConditionSet
from .loads import LoadCase
from .static_analysis import StaticAnalysis
from .modal_analysis import ModalAnalysis
from .visualizer import BeamVisualizer


class BeamSolver:
    """
    Main FEA solver for beam structures.
    
    Coordinates meshing, assembly, solution, and post-processing.
    """
    
    def __init__(self, mesh: Mesh, material: Union[Material, PropertySet, List[PropertySet]],
                 section: Optional[SectionProperties] = None, element_type: str = 'euler'):
        """
        Initialize beam solver.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh
        material : Material, PropertySet, or list of PropertySet
            Material properties or declarative PropertySet(s).
        section : SectionProperties, optional
            Cross-section properties. Required if material is a Material object.
        element_type : str
            'euler' or 'timoshenko' (default: 'euler')
        """
        self.mesh = mesh
        self.element_type = element_type.lower()
        
        # 1. Standardize properties into a unified PropertySet collector
        if isinstance(material, list):
            # Merge multiple sets into a new master set
            self.properties = PropertySet(name="Merged Properties")
            for ps in material:
                if isinstance(ps, PropertySet):
                    self.properties._assignments.extend(ps._assignments)
                else:
                    raise TypeError(f"List of properties must contain only PropertySet objects, got {type(ps)}")
        elif isinstance(material, PropertySet):
            self.properties = material
        else:
            # Legacy/Simple path: Material + Section -> single PropertySet
            self.properties = PropertySet(material=material, section=section)

        # 2. Resolve and Validate
        if self.mesh and self.mesh.num_elements > 0:
            self.properties.resolve(self.mesh.num_elements)

        # Legacy pointers for reporting/internal compatibility
        self.material = self.properties.default_material
        self.section = self.properties.default_section

        # System matrices
        self.K_global = None
        self.M_global = None
        
        # Results
        self.displacements = None
        self.reactions = None
        
        # Performance & Caching
        self._cached_forces = None
        self._cached_forces_params = None
        self._cached_stresses = None
        self._cached_stresses_params = None
        
        # Modal Results
        self.last_frequencies = None
        self.last_mode_shapes = None
        self.last_modal_participation = None
        
        # Analysis objects
        self.static_solver = StaticAnalysis(use_sparse=True)
        self.modal_solver = ModalAnalysis()
        self.visualizer = BeamVisualizer(mesh)
        
        # State for reporting
        self.last_load_case = None
        self.last_bc_set = None

        # Post-processing settings
        self.recovery_strategy = 'consistent'
        self.results = None
    
    def _validate_model(self, bc_set: BoundaryConditionSet = None):
        """
        Validate the model state before analysis to catch common errors.
        """
        # 1. Mesh Validation
        if self.mesh is None:
            raise ValueError("No mesh assigned to BeamSolver.")
        if self.mesh.num_nodes == 0:
            raise ValueError("Mesh has no nodes.")
        if self.mesh.num_elements == 0:
            raise ValueError("Mesh has no elements.")
            
        # 2. Property Validation
        if self.material is None:
            raise ValueError("No material assigned to BeamSolver.")
        if self.section is None:
            raise ValueError("No cross-section properties assigned to BeamSolver.")
            
        # 3. Boundary Condition Validation
        if bc_set is None:
            raise ValueError("No boundary conditions mapping (bc_set) provided for analysis.")
            
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

        # 4. Slenderness Ratio Check (L/h)
        coords = self.mesh.get_node_coords()
        if len(coords) > 0:
            L_total = np.max(coords[:, 0]) - np.min(coords[:, 0])
            
            # Section height (depth)
            if hasattr(self.section, 'y_top') and self.section.y_top is not None:
                h = self.section.y_top - self.section.y_bottom
            else:
                # Fallback for simple properties
                h = np.sqrt(self.section.A) 
                
            if h > 0:
                slenderness = L_total / h
                
                if slenderness < 10 and self.element_type == 'euler':
                    warnings.warn(
                        f"Slenderness ratio L/h = {slenderness:.1f} is low (< 10). "
                        f"Euler-Bernoulli elements may under-predict deflections by ignoring shear. "
                        f"Consider using element_type='timoshenko' for more accurate results."
                    )
                elif slenderness > 30 and self.element_type == 'timoshenko':
                    # Timoshenko is fine but overkill for very slender beams
                    pass 
    
    def _assemble_system_matrices(self, assembly_type: str = 'both'):
        """
        Unified global matrix assembly for stiffness and mass.

        Parameters:
        -----------
        assembly_type : str
            'stiffness', 'mass', or 'both'
        """
        from scipy.sparse import coo_matrix
        from .element_matrices import UnifiedBeamElement, get_rotation_matrix

        num_dofs = self.mesh.num_dofs
        K_rows, K_cols, K_data = [], [], []
        M_rows, M_cols, M_data = [], [], []

        coords = self.mesh.get_node_coords()
        is_euler = (self.element_type == 'euler')
        do_k = assembly_type in ['stiffness', 'both']
        do_m = assembly_type in ['mass', 'both']

        for elem in self.mesh.elements:
            p1, p2 = coords[elem.node1], coords[elem.node2]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            L = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            T = get_rotation_matrix(angle)

            # Unified property retrieval
            mat = self.properties.get_material(elem.id)
            sec = self.properties.get_section(elem.id)
            stiff = mat.get_sectional_stiffness(sec)
            rho_lin = mat.get_linear_density(sec)

            element = UnifiedBeamElement(
                EA=stiff['EA'], ES=stiff['ES'], EI=stiff['EI'],
                L=L, rho_total=rho_lin, GA_s=stiff['GA_s'] * sec.shear_factor,
                force_euler=is_euler
            )

            dof_indices = np.array([
                3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
            ])
            ii, jj = np.meshgrid(dof_indices, dof_indices, indexing='ij')

            if do_k:
                k_local = element.stiffness_matrix()
                # Transform to global: K_global = T.T @ K_local @ T
                k_global = T.T @ k_local @ T
                K_rows.append(ii.ravel()); K_cols.append(jj.ravel()); K_data.append(k_global.ravel())
            if do_m:
                m_local = element.mass_matrix()
                # Transform to global: M_global = T.T @ M_local @ T
                m_global = T.T @ m_local @ T
                M_rows.append(ii.ravel()); M_cols.append(jj.ravel()); M_data.append(m_global.ravel())

        if do_k:
            self.K_global = coo_matrix((np.concatenate(K_data), (np.concatenate(K_rows), np.concatenate(K_cols))),
                                       shape=(num_dofs, num_dofs)).tocsr()
        if do_m:
            self.M_global = coo_matrix((np.concatenate(M_data), (np.concatenate(M_rows), np.concatenate(M_cols))),
                                       shape=(num_dofs, num_dofs)).tocsr()

    def _assemble_stiffness_matrix(self):
        """Assemble global stiffness matrix."""
        self._assemble_system_matrices(assembly_type='stiffness')

    def _assemble_mass_matrix(self):
        """Assemble global mass matrix."""
        self._assemble_system_matrices(assembly_type='mass')

    def assemble_global_matrices(self):
        """Assemble both global stiffness and mass matrices upfront."""
        self._assemble_system_matrices(assembly_type='both')
    
    def solve_static(self, load_case: LoadCase, bc_set: BoundaryConditionSet):
        """
        Solve static analysis.
        
        Only the stiffness matrix is required; the mass matrix is not assembled.
        """
        # Pre-analysis validation
        self._validate_model(bc_set)
        
        if self.K_global is None:
            self._assemble_stiffness_matrix()

        # Store for reporting
        self.last_load_case = load_case
        self.last_bc_set = bc_set

        F = load_case.create_force_vector(self.mesh.num_dofs, self.mesh)

        K_bc, F_bc = bc_set.apply_to_system(self.K_global, F)

        self.displacements = self.static_solver.solve(K_bc, F_bc)

        self.reactions = self.static_solver.calculate_reactions(self.K_global, F)

        # Reset cache
        self._cached_forces = None
        self._cached_forces_params = None
        self._cached_stresses = None
        self._cached_stresses_params = None

        return self.displacements

    def solve_batch(self, load_cases: List[LoadCase], bc_set: BoundaryConditionSet, mode: str = 'light'):
        """
        Perform static analysis for multiple load cases.

        Parameters:
        -----------
        load_cases : list of LoadCase
            Load cases to analyze
        bc_set : BoundaryConditionSet
            Boundary conditions (shared across all cases)
        mode : str
            'light' (default) - stores only peak results and summary
            'full' - stores complete nodal results for every case
        """
        if not isinstance(load_cases, list) or len(load_cases) == 0:
            raise ValueError("load_cases must be a non-empty list of LoadCase objects.")

        if not all(isinstance(lc, LoadCase) for lc in load_cases):
            raise TypeError("All items in load_cases must be LoadCase objects.")

        import pandas as pd
        import copy
        from .post_processing import StressEngine

        self.batch_results = []
        summary_rows = []

        # Ensure K is assembled once
        if self.K_global is None:
            self._assemble_stiffness_matrix()

        for lc in load_cases:
            # Solve current case
            self.solve_static(lc, bc_set)

            # Extract internal forces and stresses
            forces = self.calculate_internal_forces()
            stresses = self.calculate_stresses()

            # Find peaks
            max_v, max_v_idx = self.get_max_deflection()
            max_forces = self.get_max_internal_forces()

            # Stress peaks
            vm_flat = stresses['von_mises'].flatten()
            max_vm_idx_flat = np.argmax(vm_flat)
            max_vm_idx = np.unravel_index(max_vm_idx_flat, stresses['von_mises'].shape)
            max_vm = vm_flat[max_vm_idx_flat]

            peak_x = stresses['x'][max_vm_idx[0]]
            peak_y = stresses['y'][max_vm_idx[1], max_vm_idx[2]]
            peak_z = stresses['z'][max_vm_idx[1], max_vm_idx[2]]

            # Factor of Safety
            yield_strength = getattr(self.material, 'yield_strength', None) or 0.0
            fos = yield_strength / max_vm if max_vm > 0 and yield_strength > 0 else float('nan')

            summary_row = {
                'case_name': lc.name,
                'max_deflection': max_v,
                'max_deflection_x': self.mesh.nodes[max_v_idx].x,
                'max_shear': max_forces['shear']['value'],
                'max_shear_x': max_forces['shear']['x'],
                'max_moment': max_forces['moment']['value'],
                'max_moment_x': max_forces['moment']['x'],
                'max_von_mises': max_vm,
                'max_vm_x': peak_x,
                'max_vm_y': peak_y,
                'max_vm_z': peak_z,
                'fos': fos
            }

            # Add ply info if laminate
            if hasattr(self, 'laminate_results'):
                # Find which ply has the peak VM
                # max_vm_idx is (x_idx, y_idx, z_idx)
                # StationEvaluator etc map x_idx back to element
                from .post_processing import StationEvaluator
                plan = StationEvaluator.get_evaluation_plan(self, len(stresses['x']))
                # Find which element the peak x is in
                eid_peak = -1
                for eid, indices in plan['points_by_element'].items():
                    if max_vm_idx[0] in indices:
                        eid_peak = eid
                        break

                if eid_peak != -1 and eid_peak in self.laminate_results:
                    ply_data = self.laminate_results[eid_peak]['ply_data']
                    # y coord of peak is peak_y. Find which ply contains it.
                    for ply in ply_data:
                        z_b, z_t = ply['z_range']
                        if z_b - 1e-6 <= peak_y <= z_t + 1e-6:
                            summary_row['peak_ply'] = ply['name']
                            break

            summary_rows.append(summary_row)

            if mode == 'full':
                self.batch_results.append({
                    'name': lc.name,
                    'displacements': self.displacements.copy(),
                    'reactions': self.reactions.copy(),
                    'forces': copy.deepcopy(forces),
                    'stresses': copy.deepcopy(stresses)
                })

        self.batch_summary = pd.DataFrame(summary_rows)
        return self.batch_summary

    def export_results(self, filepath: str):
        """
        Export batch analysis results to CSV.
        """
        if not hasattr(self, 'batch_summary'):
            raise ValueError("No batch results to export. Run solve_batch() first.")

        self.batch_summary.to_csv(filepath, index=False)
        return filepath

    def generate_batch_report(self, output_path: str):
        """
        Generate a summary report for batch analysis.
        """
        if not hasattr(self, 'batch_summary'):
            raise ValueError("No batch results to report. Run solve_batch() first.")

        from .report_generator import BatchReportGenerator
        report_gen = BatchReportGenerator(self)
        return report_gen.generate_report(output_path)

    def solve_modal(self, bc_set: BoundaryConditionSet, num_modes: int = 10):
        """
        Solve modal analysis.
        
        Assembles the stiffness matrix if not already done, then assembles the
        mass matrix lazily (only when this method is first called).
        """
        # Pre-analysis validation
        self._validate_model(bc_set)
        
        # Ensure Stiffness is assembled
        if self.K_global is None:
            self._assemble_stiffness_matrix()
        if self.M_global is None:
            self._assemble_mass_matrix()

        frequencies, mode_shapes = self.modal_solver.solve(
            self.K_global, self.M_global, num_modes, bc_set
        )
    
        self.last_bc_set = bc_set
        self.last_frequencies = frequencies
        self.last_mode_shapes = mode_shapes

        # Calculate participation factors
        self.last_modal_participation = self.modal_solver.get_modal_participation_summary(self.M_global)

        return frequencies, mode_shapes
    
    def get_max_deflection(self):
        """Get maximum deflection and location."""
        if self.displacements is None:
            raise ValueError("Must solve first")
        
        v = self.displacements[1::3]
        max_idx = np.argmax(np.abs(v))
        return v[max_idx], max_idx
    
    def get_max_internal_forces(self, num_points: int = 100) -> dict:
        """
        Find absolute maximum values of internal forces and their locations.
        
        Returns:
        --------
        max_values : dict
            Peak values for 'shear' and 'moment' with 'value' and 'x' keys.
        """
        res = self.calculate_internal_forces(num_points)
        pos = res['positions']
        
        v_idx = np.argmax(np.abs(res['shear_forces']))
        m_idx = np.argmax(np.abs(res['bending_moments']))
        
        return {
            'shear': {'value': res['shear_forces'][v_idx], 'x': pos[v_idx]},
            'moment': {'value': res['bending_moments'][m_idx], 'x': pos[m_idx]}
        }

    def calculate_internal_forces(self, num_points: int = None, strategy: str = None) -> dict:
        """
        Calculate internal force distributions using the specified strategy.
        """
        from .post_processing import InternalForceEngine
        
        strat = strategy or self.recovery_strategy
        
        # Check cache
        cache_key = (num_points, strat)
        if self._cached_forces is not None and self._cached_forces_params == cache_key:
            return self._cached_forces

        res = InternalForceEngine.calculate(self, num_points, strat)
        
        # Update cache
        self._cached_forces = res
        self._cached_forces_params = cache_key
        
        return res
        
    def calculate_stresses(self, num_x_points: int = None, num_y_points: int = 20, num_z_points: int = 20) -> dict:
        """
        Calculate detailed 3D stress field over the beam's length and cross-section.
        """
        from .post_processing import StressEngine

        params = (num_x_points, num_y_points, num_z_points, self.recovery_strategy)
        if self._cached_stresses is not None and self._cached_stresses_params == params:
            return self._cached_stresses

        res = StressEngine.calculate(self, num_x_points, num_y_points, num_z_points)
        
        # Update cache
        self._cached_stresses = res
        self._cached_stresses_params = params
        
        return res

    
    def generate_report(self, output_path: str, deformation_scale: Union[float, str] = 'auto',
                        failure_criterion: str = 'tsai_wu'):
        """
        Generate a professional markdown report of the analysis.
        Images are saved to a ``<report_name>_images/`` folder alongside the report.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report (e.g., 'report.md')
        deformation_scale : float or 'auto', optional
            Scale factor for deformed shape plots. Default is 'auto'.
        failure_criterion : str, optional
            Failure criterion for composite plies: 'max_stress', 'tsai_hill', or 'tsai_wu'.
            Default is 'tsai_wu'.
        """
        if self.displacements is None and self.last_frequencies is None:
            raise ValueError("Must run solve_static() or solve_modal() before generating a report")
            
        from .report_generator import BeamReportGenerator
        
        report_gen = BeamReportGenerator(
            solver=self,
            mesh=self.mesh,
            material=self.material,
            section=self.section,
            load_case=self.last_load_case,
            bc_set=self.last_bc_set,
            displacements=self.displacements,
            reactions=self.reactions,
            failure_criterion=failure_criterion
        )
        
        # Add modal results if available
        if self.last_frequencies is not None:
            report_gen.frequencies = self.last_frequencies
            report_gen.mode_shapes = self.last_mode_shapes
        
        return report_gen.generate_report(output_path, deformation_scale)
    
    def visualize(self, analysis_type: str = 'static', **kwargs):
        """
        Visualize results.
        
        Parameters:
        -----------
        analysis_type : str
            'static' (deformed shape), 'shear' (SFD), 'moment' (BMD), or 'modal'
        **kwargs : 
            Passed to the specific plotting method
        """
        if analysis_type == 'static':
            return self.visualizer.plot_deformed_shape(self.displacements, **kwargs)
        elif analysis_type == 'shear':
            default_pts = max(50, 4 * self.mesh.num_elements)
            num_points = kwargs.pop('num_points', default_pts)
            forces = self.calculate_internal_forces(num_points)
            return self.visualizer.plot_shear_force(forces['shear_forces'], forces['positions'], **kwargs)
        elif analysis_type == 'moment':
            default_pts = max(50, 4 * self.mesh.num_elements)
            num_points = kwargs.pop('num_points', default_pts)
            forces = self.calculate_internal_forces(num_points)
            return self.visualizer.plot_bending_moment(forces['bending_moments'], forces['positions'], **kwargs)
        elif analysis_type == 'modal':
            # Would plot mode shapes (implementation in BeamVisualizer)
            pass


if __name__ == "__main__":
    print("\nMain Beam Solver Module")
    print("Coordinates all analysis modules")
