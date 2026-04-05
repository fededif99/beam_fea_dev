"""
solver.py
=========
Main FEA solver that coordinates all modules.
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .mesh import Mesh
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
        if self.mesh.num_nodes > 0:
            coords = self.mesh.nodes
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
        High-performance vectorized global matrix assembly.
        """
        from scipy.sparse import coo_matrix

        num_elements = self.mesh.num_elements
        num_dofs = self.mesh.num_dofs
        coords = self.mesh.nodes
        elements = self.mesh.elements
        is_euler = (self.element_type == 'euler')
        do_k = assembly_type in ['stiffness', 'both']
        do_m = assembly_type in ['mass', 'both']

        # 1. Vectorized Geometry
        p1 = coords[elements[:, 0]]
        p2 = coords[elements[:, 1]]
        diff = p2 - p1
        L = np.sqrt(np.sum(diff**2, axis=1))
        angle = np.arctan2(diff[:, 1], diff[:, 0])

        # 2. Properties (Gathered)
        # Fast path for uniform properties
        if self.properties.is_uniform(num_elements):
            mat0 = self.properties.get_material(0)
            sec0 = self.properties.get_section(0)
            stiff0 = mat0.get_sectional_stiffness(sec0)
            EA = np.full(num_elements, stiff0['EA'])
            ES = np.full(num_elements, stiff0['ES'])
            EI = np.full(num_elements, stiff0['EI'])
            GA_s = np.full(num_elements, stiff0['GA_s'] * sec0.shear_factor)
            rho_lin = np.full(num_elements, mat0.get_linear_density(sec0))
        else:
            EA, ES, EI, GA_s, rho_lin = [np.zeros(num_elements) for _ in range(5)]
            for i in range(num_elements):
                mat = self.properties.get_material(i)
                sec = self.properties.get_section(i)
                stiff = mat.get_sectional_stiffness(sec)
                EA[i], ES[i], EI[i], GA_s[i] = stiff['EA'], stiff['ES'], stiff['EI'], stiff['GA_s'] * sec.shear_factor
                rho_lin[i] = mat.get_linear_density(sec)

        # 3. Vectorized Rotation Transformations
        # T defined as u_local = T @ u_global
        c = np.cos(angle)
        s = np.sin(angle)
        T = np.zeros((num_elements, 6, 6))
        for i in [0, 3]:
            T[:, i, i] = c; T[:, i, i+1] = s
            T[:, i+1, i] = -s; T[:, i+1, i+1] = c
            T[:, i+2, i+2] = 1.0

        # 4. Global Row/Col Indices (Vectorized meshgrid replacement)
        dof_indices = np.zeros((num_elements, 6), dtype=int)
        dof_indices[:, 0:3] = 3 * elements[:, 0:1] + np.array([0, 1, 2])
        dof_indices[:, 3:6] = 3 * elements[:, 1:2] + np.array([0, 1, 2])
        K_rows = np.repeat(dof_indices, 6, axis=1).flatten()
        K_cols = np.tile(dof_indices, (1, 6)).flatten()

        # 5. Stiffness Matrix (Vectorized Generation & Transformation)
        if do_k:
            phi = np.zeros(num_elements)
            if not is_euler:
                mask = GA_s > 0
                phi[mask] = 12.0 * EI[mask] / (GA_s[mask] * L[mask]**2)
            fac = 1.0 / (1.0 + phi)

            K_local = np.zeros((num_elements, 6, 6))
            # Axial
            ka = EA / L
            K_local[:, 0, 0] = ka; K_local[:, 0, 3] = -ka
            K_local[:, 3, 0] = -ka; K_local[:, 3, 3] = ka
            # Bending
            k11 = 12.0 * EI * fac / L**3
            k12 = 6.0 * EI * fac / L**2
            k22 = (4.0 + phi) * EI * fac / L
            k24 = (2.0 - phi) * EI * fac / L
            K_local[:, 1, 1] = k11;  K_local[:, 1, 2] = k12; K_local[:, 1, 4] = -k11; K_local[:, 1, 5] = k12
            K_local[:, 2, 1] = k12;  K_local[:, 2, 2] = k22; K_local[:, 2, 4] = -k12; K_local[:, 2, 5] = k24
            K_local[:, 4, 1] = -k11; K_local[:, 4, 2] = -k12; K_local[:, 4, 4] = k11; K_local[:, 4, 5] = -k12
            K_local[:, 5, 1] = k12;  K_local[:, 5, 2] = k24; K_local[:, 5, 4] = -k12; K_local[:, 5, 5] = k22
            # Coupling
            kc = ES / L
            K_local[:, 0, 2] += kc; K_local[:, 0, 5] -= kc
            K_local[:, 2, 0] += kc; K_local[:, 2, 3] -= kc
            K_local[:, 3, 2] -= kc; K_local[:, 3, 5] += kc
            K_local[:, 5, 0] -= kc; K_local[:, 5, 3] += kc

            K_global_el = np.matmul(T.transpose(0, 2, 1), np.matmul(K_local, T))
            self.K_global = coo_matrix((K_global_el.flatten(), (K_rows, K_cols)), 
                                       shape=(num_dofs, num_dofs)).tocsr()

        # 6. Mass Matrix (Vectorized Generation & Transformation)
        if do_m:
            M_local = np.zeros((num_elements, 6, 6))
            m = rho_lin
            phi = np.zeros(num_elements)
            if not is_euler:
                mask = GA_s > 0
                phi[mask] = 12.0 * EI[mask] / (GA_s[mask] * L[mask]**2)
            
            # Common parts
            M_local[:, 0, 0] = m * L / 3.0; M_local[:, 0, 3] = m * L / 6.0
            M_local[:, 3, 0] = m * L / 6.0; M_local[:, 3, 3] = m * L / 3.0

            if np.all(phi == 0):
                # Euler-Bernoulli Case
                fac = m * L / 420.0
                M_local[:, 1, 1] = 156 * fac;   M_local[:, 1, 2] = 22*L * fac;  M_local[:, 1, 4] = 54 * fac;   M_local[:, 1, 5] = -13*L * fac
                M_local[:, 2, 1] = 22*L * fac;  M_local[:, 2, 2] = 4*L**2 * fac; M_local[:, 2, 4] = 13*L * fac; M_local[:, 2, 5] = -3*L**2 * fac
                M_local[:, 4, 1] = 54 * fac;    M_local[:, 4, 2] = 13*L * fac;  M_local[:, 4, 4] = 156 * fac;  M_local[:, 4, 5] = -22*L * fac
                M_local[:, 5, 1] = -13*L * fac; M_local[:, 5, 2] = -3*L**2 * fac; M_local[:, 5, 4] = -22*L * fac; M_local[:, 5, 5] = 4*L**2 * fac
            else:
                # Timoshenko Case (Friedman & Kosmatka, 1993)
                fac = 1.0 / (1.0 + phi)
                p = phi
                r_sq = np.zeros(num_elements)
                mask = (EA > 0)
                r_sq[mask] = EI[mask] / (EA[mask] * L[mask]**2)
                m_r = m * r_sq

                # Translational
                c1 = fac**2 * (13/35 + 7*p/10 + p**2/3)
                c2 = fac**2 * (11/210 + 11*p/120 + p**2/24) * L
                c3 = fac**2 * (9/70 + 3*p/10 + p**2/6)
                c4 = fac**2 * (-13/420 - 3*p/40 - p**2/24) * L
                c5 = fac**2 * (1/105 + p/60 + p**2/120) * L**2
                c6 = fac**2 * (-1/140 - p/60 - p**2/120) * L**2
                # Rotatory
                r1 = fac**2 * (6/5)
                r2 = fac**2 * (1/10 - p/2) * L
                r3 = fac**2 * (-6/5)
                r4 = fac**2 * (1/10 - p/2) * L
                r5 = fac**2 * (2/15 + p/6 + p**2/3) * L**2
                r6 = fac**2 * (-1/30 - p/6 + p**2/6) * L**2

                Mb_data = m * L
                M_local[:, 1, 1] = Mb_data * (c1 + m_r*r1/m);  M_local[:, 1, 2] = Mb_data * (c2 + m_r*r2/m)
                M_local[:, 1, 4] = Mb_data * (c3 + m_r*r3/m);  M_local[:, 1, 5] = Mb_data * (c4 + m_r*r4/m)
                M_local[:, 2, 1] = M_local[:, 1, 2];          M_local[:, 2, 2] = Mb_data * (c5 + m_r*r5/m)
                M_local[:, 2, 4] = Mb_data * (-c4 - m_r*r4/m); M_local[:, 2, 5] = Mb_data * (c6 + m_r*r6/m)
                M_local[:, 4, 1] = M_local[:, 1, 4];          M_local[:, 4, 2] = M_local[:, 2, 4]
                M_local[:, 4, 4] = Mb_data * (c1 + m_r*r1/m);  M_local[:, 4, 5] = Mb_data * (-c2 - m_r*r2/m)
                M_local[:, 5, 1] = M_local[:, 1, 5];          M_local[:, 5, 2] = M_local[:, 2, 5]
                M_local[:, 5, 4] = M_local[:, 4, 5];          M_local[:, 5, 5] = Mb_data * (c5 + m_r*r5/m)

            M_global_el = np.matmul(T.transpose(0, 2, 1), np.matmul(M_local, T))
            self.M_global = coo_matrix((M_global_el.flatten(), (K_rows, K_cols)), 
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
            max_defl = self.get_max_deflection()
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
                'max_deflection': max_defl['res'],
                'max_deflection_x': max_defl['x'],
                'max_deflection_node': max_defl['node_id'],
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

    def solve_modal(self, bc_set: BoundaryConditionSet, num_modes: Optional[int] = None, load_case: Optional[Union[LoadCase, 'LoadCombination']] = None):
        """
        Solve modal analysis.
        
        Assembles the stiffness matrix if not already done, then assembles the
        mass matrix lazily (only when this method is first called).
        If a `load_case` containing `LumpedMass` loads is provided, those masses
        are applied to a copy of the mass matrix for the modal solve.
        """
        # Pre-analysis validation
        self._validate_model(bc_set)
        
        # Ensure Stiffness and Mass are assembled
        if self.K_global is None:
            self._assemble_stiffness_matrix()
        if self.M_global is None:
            self._assemble_mass_matrix()

        # Handle Lumped Masses via LoadCase
        M_eff = self.M_global
        if load_case is not None:
            if not load_case.has_mass_loads():
                raise ValueError("load_case provided to solve_modal does not contain any LumpedMass loads.")
            M_eff = self.M_global.copy()
            M_eff = load_case.apply_to_mass_matrix(M_eff, self.mesh)

        frequencies, mode_shapes = self.modal_solver.solve(
            self.K_global, M_eff, num_modes, bc_set
        )
    
        self.last_bc_set = bc_set
        self.last_frequencies = frequencies
        self.last_mode_shapes = mode_shapes

        # Calculate participation factors using the effective mass matrix
        self.last_modal_participation = self.modal_solver.get_modal_participation_summary(M_eff)

        return frequencies, mode_shapes
    
    def get_max_deflection(self) -> 'pd.Series':
        """
        Calculate resultant displacement at all nodes and find the maximum.
        
        Returns:
        --------
        max_record : pd.Series
            Series containing node_id, x, y, u, v, and res (resultant)
        """
        import pandas as pd
        if self.displacements is None:
            raise ValueError("Must solve first")
        
        if self.results is None:
            self.results = {}

        # 1. Extract components
        u = self.displacements[0::3]
        v = self.displacements[1::3]
        
        # 2. Compute resultant (2D for now, easily extendable to 3D)
        res = np.sqrt(u**2 + v**2)
        
        # 3. Build DataFrame
        coords = self.mesh.nodes
        df = pd.DataFrame({
            'node_id': np.arange(self.mesh.num_nodes),
            'x': coords[:, 0],
            'y': coords[:, 1],
            'u': u,
            'v': v,
            'res': res
        })
        
        # 4. Store in results
        self.results['displacements'] = df
        
        # 5. Return the record with max resultant
        return df.iloc[df['res'].idxmax()]
    
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
