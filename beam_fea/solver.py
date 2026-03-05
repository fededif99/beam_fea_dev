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
from .element_matrices import EulerBernoulliElement, TimoshenkoElement, SHEAR_CORRECTION_FACTOR
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
            'euler' or 'timoshenko'
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
    
    def _assemble_stiffness_matrix(self):
        """Assemble global stiffness matrix only."""
        from scipy.sparse import coo_matrix

        num_dofs = self.mesh.num_dofs
        K_rows, K_cols, K_data = [], [], []

        from .element_matrices import AnisotropicBeamElement
        from .composites import Laminate

        ElementClass = EulerBernoulliElement if self.element_type == 'euler' else TimoshenkoElement
        coords = self.mesh.get_node_coords()

        for elem in self.mesh.elements:
            p1, p2 = coords[elem.node1], coords[elem.node2]
            L = np.sqrt(np.sum((p2 - p1)**2))

            # Determine properties for this element via collector
            mat = self.properties.get_material(elem.id)
            sec = self.properties.get_section(elem.id)

            # Check for anisotropy (Laminate override)
            if hasattr(mat, '_is_laminate') or isinstance(mat, Laminate):
                # Detect width from section (standardized sections have 'width')
                width = getattr(sec, 'width', 1.0)
                if hasattr(sec, 'diameter'): width = sec.diameter

                EA, ES, EI = mat.get_sectional_stiffness(width)
                rho_total = mat.rho * width * mat.total_thickness
                # Timoshenko shear stiffness: κ · Gxy · (width × thickness)
                props = mat.get_effective_properties()
                GA_s = SHEAR_CORRECTION_FACTOR * props['Gxy'] * (width * mat.total_thickness)
                element = AnisotropicBeamElement(EA=EA, ES=ES, EI=EI, L=L, rho_total=rho_total, GA_s=GA_s)
            else:
                element = ElementClass(
                    E=mat.E,
                    G=mat.G,
                    I=sec.Iy,
                    A=sec.A,
                    L=L,
                    rho=mat.rho
                )

            k_local = element.stiffness_matrix()

            dof_indices = np.array([
                3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
            ])
            ii, jj = np.meshgrid(dof_indices, dof_indices, indexing='ij')

            K_rows.append(ii.ravel())
            K_cols.append(jj.ravel())
            K_data.append(k_local.ravel())

        self.K_global = coo_matrix(
            (np.concatenate(K_data), (np.concatenate(K_rows), np.concatenate(K_cols))),
            shape=(num_dofs, num_dofs)
        ).tocsr()

    def _assemble_mass_matrix(self):
        """Assemble global mass matrix only (called lazily for modal analysis)."""
        from scipy.sparse import coo_matrix

        num_dofs = self.mesh.num_dofs
        M_rows, M_cols, M_data = [], [], []

        from .element_matrices import AnisotropicBeamElement
        from .composites import Laminate

        ElementClass = EulerBernoulliElement if self.element_type == 'euler' else TimoshenkoElement
        coords = self.mesh.get_node_coords()

        for elem in self.mesh.elements:
            p1, p2 = coords[elem.node1], coords[elem.node2]
            L = np.sqrt(np.sum((p2 - p1)**2))

            # Determine properties for this element via collector
            mat = self.properties.get_material(elem.id)
            sec = self.properties.get_section(elem.id)

            # Check for anisotropy
            if hasattr(mat, '_is_laminate') or isinstance(mat, Laminate):
                width = getattr(sec, 'width', 1.0)
                if hasattr(sec, 'diameter'): width = sec.diameter

                EA, ES, EI = mat.get_sectional_stiffness(width)
                rho_total = mat.rho * width * mat.total_thickness
                # Timoshenko shear stiffness: κ · Gxy · (width × thickness)
                props = mat.get_effective_properties()
                GA_s = SHEAR_CORRECTION_FACTOR * props['Gxy'] * (width * mat.total_thickness)
                element = AnisotropicBeamElement(EA=EA, ES=ES, EI=EI, L=L, rho_total=rho_total, GA_s=GA_s)
            else:
                element = ElementClass(
                    E=mat.E,
                    G=mat.G,
                    I=sec.Iy,
                    A=sec.A,
                    L=L,
                    rho=mat.rho
                )

            m_local = element.mass_matrix()

            dof_indices = np.array([
                3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
            ])
            ii, jj = np.meshgrid(dof_indices, dof_indices, indexing='ij')

            M_rows.append(ii.ravel())
            M_cols.append(jj.ravel())
            M_data.append(m_local.ravel())

        self.M_global = coo_matrix(
            (np.concatenate(M_data), (np.concatenate(M_rows), np.concatenate(M_cols))),
            shape=(num_dofs, num_dofs)
        ).tocsr()

    def assemble_global_matrices(self):
        """Assemble both global stiffness and mass matrices.

        Calling this directly assembles both matrices upfront. In normal
        workflows the solver assembles them lazily: K is built on the first
        static or modal solve, M is built only when a modal solve is requested.
        """
        self._assemble_stiffness_matrix()
        self._assemble_mass_matrix()
    
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
        self._cached_stresses = None

        return self.displacements

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

    
    def generate_report(self, output_path: str, deformation_scale: Union[float, str] = 'auto'):
        """
        Generate a professional markdown report of the analysis.
        Images are saved to a ``<report_name>_images/`` folder alongside the report.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report (e.g., 'report.md')
        deformation_scale : float or 'auto', optional
            Scale factor for deformed shape plots. Default is 'auto'.
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
            reactions=self.reactions
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
