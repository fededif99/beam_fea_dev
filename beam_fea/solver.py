"""
solver.py
=========
Main FEA solver that coordinates all modules.
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple
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
    
    def __init__(self, mesh: Mesh, material: Union[Material, PropertySet],
                 section: Optional[SectionProperties] = None, element_type: str = 'euler'):
        """
        Initialize beam solver.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh
        material : Material or PropertySet
            Material properties or a unified PropertySet collector.
        section : SectionProperties, optional
            Cross-section properties. Required if material is a Material object.
        element_type : str
            'euler' or 'timoshenko'
        """
        self.mesh = mesh
        self.element_type = element_type.lower()
        
        # Unified Property Management
        if isinstance(material, PropertySet):
            self.properties = material
        else:
            self.properties = PropertySet(default_material=material, default_section=section)

        # Legacy pointers for reporting compatibility
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

        ElementClass = EulerBernoulliElement if self.element_type == 'euler' else TimoshenkoElement
        coords = self.mesh.get_node_coords()

        for elem in self.mesh.elements:
            p1, p2 = coords[elem.node1], coords[elem.node2]
            L = np.sqrt(np.sum((p2 - p1)**2))

            # Determine properties for this element via collector
            mat = self.properties.get_material(elem.id)
            sec = self.properties.get_section(elem.id)

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

        ElementClass = EulerBernoulliElement if self.element_type == 'euler' else TimoshenkoElement
        coords = self.mesh.get_node_coords()

        for elem in self.mesh.elements:
            p1, p2 = coords[elem.node1], coords[elem.node2]
            L = np.sqrt(np.sum((p2 - p1)**2))

            # Determine properties for this element via collector
            mat = self.properties.get_material(elem.id)
            sec = self.properties.get_section(elem.id)

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

    def calculate_internal_forces(self, num_points: int = None) -> dict:
        """
        Calculate shear force and bending moment distributions along the beam.
        
        Parameters:
        -----------
        num_points : int, optional
            Number of points for interpolation along beam length. 
            Defaults to self.mesh.num_nodes if None.
            
        Returns:
        --------
        results : dict
            Dictionary containing 'positions', 'shear_forces', and 'bending_moments'.
        """
        if self.displacements is None:
            raise ValueError("Must run solve_static() first")
            
        # Check cache
        if self._cached_forces is not None and self._cached_forces_params == num_points:
            return self._cached_forces

        coords = self.mesh.get_node_coords()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        
        # Create evaluation points
        if num_points is None:
            # Physics-aligned: use the exact nodal coordinates by default
            positions = np.sort(coords[:, 0])
            eval_count = len(positions)
        else:
            # Interpolated: use a uniform grid for smooth plotting
            positions = np.linspace(x_min, x_max, num_points)
            eval_count = num_points
            
        axial_forces = np.zeros(eval_count)
        shear_forces = np.zeros(eval_count)
        bending_moments = np.zeros(eval_count)
        
        # Determine evaluation points per element to avoid redundant element instantiation
        from .element_matrices import EulerBernoulliElement, TimoshenkoElement
        
        # Pre-calculate distributed loads per element
        element_dist_loads = {}
        if self.last_load_case:
            from .loads import (UniformDistributedLoad, TrapezoidalDistributedLoad,
                               TriangularDistributedLoad)
            for load in self.last_load_case.loads:
                # Transverse (wy) and Axial (wx)
                wy1 = wy2 = wx1 = wx2 = 0.0

                if isinstance(load, UniformDistributedLoad):
                    wy1 = wy2 = load.wy
                    wx1 = wx2 = load.wx
                elif isinstance(load, TrapezoidalDistributedLoad):
                    wy1, wy2 = load.wy1, load.wy2
                    wx1, wx2 = load.wx1, load.wx2
                elif isinstance(load, TriangularDistributedLoad):
                    if load.peak_loc == 'start':
                        wy1, wy2 = load.w_peak, 0.0
                    else:
                        wy1, wy2 = 0.0, load.w_peak
                else:
                    continue # Not a distributed load

                # Identify elements affected
                if load.element is not None:
                    elems = [load.element] if isinstance(load.element, int) else load.element
                    for eid in elems:
                        curr = element_dist_loads.get(eid, (0, 0, 0, 0))
                        element_dist_loads[eid] = (curr[0] + wy1, curr[1] + wy2, curr[2] + wx1, curr[3] + wx2)
                elif load.x_start is not None and load.x_end is not None:
                    x_s, x_e = min(load.x_start, load.x_end), max(load.x_start, load.x_end)
                    for eid, elem in enumerate(self.mesh.elements):
                        p1, p2 = coords[elem.node1], coords[elem.node2]
                        e_min, e_max = min(p1[0], p2[0]), max(p1[0], p2[0])

                        # Intersection
                        i_min, i_max = max(e_min, x_s), min(e_max, x_e)
                        if i_max > i_min:
                            # For simplicity, if coordinate-based distributed load is used,
                            # we interpolate the intensities at element ends
                            def get_w(x, w1, w2, xs, xe):
                                if abs(xe - xs) < 1e-9: return w1
                                return w1 + (w2 - w1) * (x - xs) / (xe - xs)

                            wya = get_w(p1[0], wy1, wy2, load.x_start, load.x_end)
                            wyb = get_w(p2[0], wy1, wy2, load.x_start, load.x_end)
                            wxa = get_w(p1[0], wx1, wx2, load.x_start, load.x_end)
                            wxb = get_w(p2[0], wx1, wx2, load.x_start, load.x_end)

                            curr = element_dist_loads.get(eid, (0, 0, 0, 0))
                            element_dist_loads[eid] = (curr[0] + wya, curr[1] + wyb, curr[2] + wxa, curr[3] + wxb)

        # Re-instantiate expert and cache element properties only if element changes
        current_elem_idx = -1
        x1, L, u_local = 0.0, 1.0, None
        element_expert = None

        for i, x in enumerate(positions):
            # Find element containing point x
            found_idx = -1
            
            # 1. Try current element (spatial coherence) - Most efficient for evaluation loops
            if current_elem_idx != -1:
                if L > 0 and x1 <= x <= x1 + L:
                    found_idx = current_elem_idx
            
            # 2. Delegate to Mesh for optimized/indexed search
            if found_idx == -1:
                found_idx = self.mesh.find_element_at_x(x)
            
            # Fallback for numerical precision at ends
            if found_idx == -1:
                if x <= x_min:
                    found_idx = 0
                elif x >= x_max:
                    found_idx = self.mesh.num_elements - 1
            
            if found_idx == -1:
                continue
            
            # Update cache and expert only if element truly changes
            if found_idx != current_elem_idx:
                current_elem_idx = found_idx
                elem = self.mesh.elements[current_elem_idx]
                
                # Cache boundary and length
                p1, p2 = coords[elem.node1], coords[elem.node2]
                x1 = p1[0]
                L = np.sqrt(np.sum((p2 - p1)**2))
                
                # Local element displacements
                dof_indices = [
                    3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                    3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
                ]
                u_local = self.displacements[dof_indices]
                
                # Instantiate the correct element type expert
                # Support multiple materials/sections via collector
                mat = self.properties.get_material(elem.id)
                sec = self.properties.get_section(elem.id)
                E, G, I, A, rho = mat.E, mat.G, sec.Iy, sec.A, mat.rho
                
                if self.element_type == 'euler':
                    element_expert = EulerBernoulliElement(E=E, G=G, I=I, A=A, L=L, rho=rho)
                else:
                    element_expert = TimoshenkoElement(E=E, G=G, I=I, A=A, L=L, rho=rho)
            
            # Local position xi [0, 1] - Uses cached L and x1
            xi = np.clip((x - x1) / L, 0, 1) if L > 0 else 0
            
            # Pass distributed load for this element
            d_load = element_dist_loads.get(current_elem_idx, (0.0, 0.0, 0.0, 0.0))
            N, V, M = element_expert.interpolate_internal_forces(u_local, xi, dist_load=d_load)
            axial_forces[i] = N
            shear_forces[i] = V
            bending_moments[i] = M
            
        res = {
            'positions': positions,
            'axial_forces': axial_forces,
            'shear_forces': shear_forces,
            'bending_moments': bending_moments
        }
        
        # Update cache
        self._cached_forces = res
        self._cached_forces_params = num_points
        
        return res
        
    def calculate_stresses(self, num_x_points: int = None, num_y_points: int = 20, num_z_points: int = 20) -> dict:
        """
        Calculate detailed 3D stress field over the beam's length and cross-section.
        
        Parameters:
        -----------
        num_x_points : int, optional
            Number of points along the beam length. Defaults to self.mesh.num_nodes if None.
        num_y_points : int
            Number of grid points in the cross-section y-direction (depth).
        num_z_points : int
            Number of grid points in the cross-section z-direction (width).
            
        Returns:
        --------
        dict
            Dictionary containing evaluation grids and 3D stress arrays:
            'x': 1D array of positions
            'y': 2D grid of y coordinates
            'z': 2D grid of z coordinates
            'mask': 2D boolean mask of the cross section shape
            'axial': 3D array of axial stresses
            'bending': 3D array of bending stresses
            'shear': 3D array of transverse shear stresses
            'von_mises': 3D array of von Mises equivalent stresses
        """
        # 1. Check cache
        params = (num_x_points, num_y_points, num_z_points)
        if self._cached_stresses is not None and self._cached_stresses_params == params:
            return self._cached_stresses

        from .static_analysis import StressAnalysis
        
        # 2. Get internal forces along the beam
        forces = self.calculate_internal_forces(num_x_points)
        x_positions = forces['positions']
        N_x = forces['axial_forces']
        V_x = forces['shear_forces']
        M_x = forces['bending_moments']
        
        # 2. Build 2D grid for the cross-section
        # Use union of all bounding boxes to accommodate multiple sections
        y_min_global, y_max_global = self.properties.default_section.y_bottom, self.properties.default_section.y_top
        z_min_global, z_max_global = self.properties.default_section.z_left, self.properties.default_section.z_right
        
        # Check overrides for bounding box expansions
        for sec in self.properties.section_overrides.values():
            y_min_global = min(y_min_global, sec.y_bottom)
            y_max_global = max(y_max_global, sec.y_top)
            z_min_global = min(z_min_global, sec.z_left)
            z_max_global = max(z_max_global, sec.z_right)

        # Fallback for simple properties
        if y_min_global is None:
            r = np.sqrt(self.section.Iy / self.section.A)
            y_min_global, y_max_global = -2*r, 2*r
        if z_min_global is None:
            r = np.sqrt(self.section.Iz / self.section.A) if self.section.Iz else np.sqrt(self.section.Iy / self.section.A)
            z_min_global, z_max_global = -2*r, 2*r
            
        y_coords = np.linspace(y_min_global, y_max_global, num_y_points)
        z_coords = np.linspace(z_min_global, z_max_global, num_z_points)
        Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')
        
        # 3. Calculate stresses (Vectorized if single section, else loop)
        eval_x_count = len(x_positions)
        shape_3d = (eval_x_count, num_y_points, num_z_points)

        has_multiple_sections = self.properties.has_multiple_sections(self.mesh.num_elements)

        if not has_multiple_sections:
            # Full Vectorization for single-section beams
            sec = self.properties.default_section
            try:
                mask, t_yz, Q_yz = sec.get_stress_profile(Y, Z)
            except AttributeError:
                mask = np.ones_like(Y, dtype=bool)
                t_yz = np.full_like(Y, (sec.z_right or 10) - (sec.z_left or -10))
                Q_yz = (t_yz) / 2.0 * ((sec.y_top or 10)**2 - Y**2)

            try:
                mask, t_yz, Q_yz = sec.get_stress_profile(Y, Z)
            except AttributeError:
                # Fallback for simple properties: use bounding box mask
                y_top = sec.y_top if sec.y_top is not None else np.sqrt(sec.A)/2
                y_bot = sec.y_bottom if sec.y_bottom is not None else -np.sqrt(sec.A)/2
                z_r = sec.z_right if sec.z_right is not None else (sec.Iz/sec.A)**0.5 if sec.Iz else np.sqrt(sec.A)/2
                z_l = sec.z_left if sec.z_left is not None else -(sec.Iz/sec.A)**0.5 if sec.Iz else -np.sqrt(sec.A)/2
                
                mask = (Y <= y_top) & (Y >= y_bot) & (Z <= z_r) & (Z >= z_l)
                t_yz = np.full_like(Y, z_r - z_l)
                Q_yz = (t_yz) / 2.0 * (y_top**2 - Y**2)
                Q_yz[~mask] = 0.0
                t_yz[~mask] = 0.0
            A, Iy = sec.A, sec.Iy

            # Expand forces to 3D
            N_3d = N_x[:, np.newaxis, np.newaxis]
            V_3d = V_x[:, np.newaxis, np.newaxis]
            M_3d = M_x[:, np.newaxis, np.newaxis]

            sigma_a = np.broadcast_to(N_3d / A, shape_3d).copy()
            sigma_b = -M_3d * Y[np.newaxis, :, :] / Iy

            valid_t = t_yz > 1e-9
            shear_base = np.zeros_like(Y)
            shear_base[valid_t] = Q_yz[valid_t] / (Iy * t_yz[valid_t])
            tau_s = V_3d * shear_base[np.newaxis, :, :]

            sigma_x = sigma_a + sigma_b
            avg_sig = sigma_x / 2.0
            R = np.sqrt(avg_sig**2 + tau_s**2)
            sigma_1, sigma_2 = avg_sig + R, avg_sig - R
            sigma_vm = np.sqrt(sigma_x**2 + 3 * tau_s**2)

            # Global Masking
            inv_mask = ~np.broadcast_to(mask[np.newaxis, :, :], shape_3d)
            sigma_a[inv_mask] = sigma_b[inv_mask] = tau_s[inv_mask] = 0.0
            sigma_1[inv_mask] = sigma_2[inv_mask] = sigma_vm[inv_mask] = 0.0

        else:
            # Multi-section beams
            sigma_a = np.zeros(shape_3d)
            sigma_b = np.zeros(shape_3d)
            tau_s = np.zeros(shape_3d)
            sigma_vm = np.zeros(shape_3d)
            sigma_1 = np.zeros(shape_3d)
            sigma_2 = np.zeros(shape_3d)

            profile_cache = {}

            for i, x in enumerate(x_positions):
                elem_idx = self.mesh.find_element_at_x(x)
                if elem_idx == -1:
                    elem_idx = 0 if x <= x_positions[0] else self.mesh.num_elements - 1

                elem = self.mesh.elements[elem_idx]
                sec = self.properties.get_section(elem.id)

                sec_id = id(sec)
                if sec_id not in profile_cache:
                    try:
                        mask, t_yz, Q_yz = sec.get_stress_profile(Y, Z)
                    except AttributeError:
                        # Fallback for simple properties: use bounding box mask
                        y_top = sec.y_top if sec.y_top is not None else np.sqrt(sec.A)/2
                        y_bot = sec.y_bottom if sec.y_bottom is not None else -np.sqrt(sec.A)/2
                        z_r = sec.z_right if sec.z_right is not None else (sec.Iz/sec.A)**0.5 if sec.Iz else np.sqrt(sec.A)/2
                        z_l = sec.z_left if sec.z_left is not None else -(sec.Iz/sec.A)**0.5 if sec.Iz else -np.sqrt(sec.A)/2
                        
                        mask = (Y <= y_top) & (Y >= y_bot) & (Z <= z_r) & (Z >= z_l)
                        t_yz = np.full_like(Y, z_r - z_l)
                        Q_yz = (t_yz) / 2.0 * (y_top**2 - Y**2)
                        Q_yz[~mask] = 0.0
                        t_yz[~mask] = 0.0
                    profile_cache[sec_id] = (mask, t_yz, Q_yz)

                mask, t_yz, Q_yz = profile_cache[sec_id]
                A, Iy = sec.A, sec.Iy

                sigma_a_x = np.full_like(Y, N_x[i] / A)
                sigma_b_x = -M_x[i] * Y / Iy

                tau_s_x = np.zeros_like(Y)
                valid_t = t_yz > 1e-9
                tau_s_x[valid_t] = V_x[i] * Q_yz[valid_t] / (Iy * t_yz[valid_t])

                sig_x = sigma_a_x + sigma_b_x
                avg_sig = sig_x / 2.0
                R = np.sqrt(avg_sig**2 + tau_s_x**2)

                sigma_a[i], sigma_b[i], tau_s[i] = sigma_a_x, sigma_b_x, tau_s_x
                sigma_1[i], sigma_2[i] = avg_sig + R, avg_sig - R
                sigma_vm[i] = np.sqrt(sig_x**2 + 3 * tau_s_x**2)

                # Element Masking
                m_3d = mask # (ny, nz)
                sigma_a[i][~m_3d] = sigma_b[i][~m_3d] = tau_s[i][~m_3d] = 0.0
                sigma_1[i][~m_3d] = sigma_2[i][~m_3d] = sigma_vm[i][~m_3d] = 0.0

        res = {
            'x': x_positions,
            'y': Y,
            'z': Z,
            'mask': mask,
            'axial': sigma_a,
            'bending': sigma_b,
            'shear': tau_s,
            'sigma_1': sigma_1,
            'sigma_2': sigma_2,
            'von_mises': sigma_vm
        }
        
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
