"""
solver.py
=========
Main FEA solver that coordinates all modules.
"""

import numpy as np
from .mesh import Mesh, MeshGenerator
from .materials import Material, get_material
from .cross_sections import CrossSection, SectionProperties
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
    
    def __init__(self, mesh: Mesh, material: Material, 
                 section: SectionProperties, element_type: str = 'euler'):
        """
        Initialize beam solver.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh
        material : Material
            Material properties
        section : SectionProperties
            Cross-section properties
        element_type : str
            'euler' or 'timoshenko'
        """
        self.mesh = mesh
        self.material = material
        self.section = section
        self.element_type = element_type.lower()
        
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
        
        # Analysis objects
        self.static_solver = StaticAnalysis(use_sparse=True)
        self.modal_solver = ModalAnalysis()
        self.visualizer = BeamVisualizer(mesh)
        
        # State for reporting
        self.last_load_case = None
        self.last_bc_set = None
    
    def assemble_global_matrices(self):
        """Assemble global stiffness and mass matrices."""
        from scipy.sparse import coo_matrix
        
        num_dofs = self.mesh.num_dofs
        
        # Lists to store sparse matrix data
        # Stiffness
        K_rows = []
        K_cols = []
        K_data = []
        
        # Mass
        M_rows = []
        M_cols = []
        M_data = []
        
        # Prepare element data
        ElementClass = EulerBernoulliElement if self.element_type == 'euler' else TimoshenkoElement
        
        # Precompute coordinates and lengths for all elements
        coords = self.mesh.get_node_coords()
        
        for elem in self.mesh.elements:
            # Calculate length directly from cached coords
            p1, p2 = coords[elem.node1], coords[elem.node2]
            L = np.sqrt(np.sum((p2 - p1)**2))
            
            # Create element matrices
            element = ElementClass(
                E=self.material.E,
                G=self.material.G,
                I=self.section.Iy,
                A=self.section.A,
                L=L,
                rho=self.material.rho
            )
            
            k_local = element.stiffness_matrix()
            m_local = element.mass_matrix()
            
            # Get DOF indices
            dof_indices = np.array([
                3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
            ])
            
            # Vectorized scatter using meshgrid
            ii, jj = np.meshgrid(dof_indices, dof_indices, indexing='ij')
            
            K_rows.append(ii.ravel())
            K_cols.append(jj.ravel())
            K_data.append(k_local.ravel())
            
            M_rows.append(ii.ravel())
            M_cols.append(jj.ravel())
            M_data.append(m_local.ravel())
        
        # Create sparse matrices (concatenate all arrays once)
        self.K_global = coo_matrix(
            (np.concatenate(K_data), (np.concatenate(K_rows), np.concatenate(K_cols))),
            shape=(num_dofs, num_dofs)
        ).tocsr()
        
        self.M_global = coo_matrix(
            (np.concatenate(M_data), (np.concatenate(M_rows), np.concatenate(M_cols))),
            shape=(num_dofs, num_dofs)
        ).tocsr()
    
    def solve_static(self, load_case: LoadCase, bc_set: BoundaryConditionSet):
        """Solve static analysis."""
        if self.K_global is None:
            self.assemble_global_matrices()
        
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
        """Solve modal analysis."""
        if self.K_global is None:
            self.assemble_global_matrices()
        
        frequencies, mode_shapes = self.modal_solver.solve(
            self.K_global, self.M_global, num_modes, bc_set
        )
        
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

    def calculate_internal_forces(self, num_points: int = 100) -> dict:
        """
        Calculate shear force and bending moment distributions along the beam.
        
        Parameters:
        -----------
        num_points : int
            Number of points for interpolation along beam length.
            
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
        beam_length = x_max - x_min
        
        # Create evaluation points
        positions = np.linspace(x_min, x_max, num_points)
        axial_forces = np.zeros(num_points)
        shear_forces = np.zeros(num_points)
        bending_moments = np.zeros(num_points)
        
        # Determine evaluation points per element to avoid redundant element instantiation
        from .element_matrices import EulerBernoulliElement, TimoshenkoElement
        
        # Re-instantiate expert and cache element properties only if element changes
        current_elem_idx = -1
        x1, L, u_local = 0.0, 1.0, None

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
                # [Optimization]: Pre-extract all properties once per element change
                E, G, I, A, rho = self.material.E, self.material.G, self.section.Iy, self.section.A, self.material.rho
                
                if self.element_type == 'euler':
                    element_expert = EulerBernoulliElement(E=E, G=G, I=I, A=A, L=L, rho=rho)
                else:
                    element_expert = TimoshenkoElement(E=E, G=G, I=I, A=A, L=L, rho=rho)
            
            # Local position xi [0, 1] - Uses cached L and x1
            xi = np.clip((x - x1) / L, 0, 1) if L > 0 else 0
            
            N, V, M = element_expert.interpolate_internal_forces(u_local, xi)
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
        
    def calculate_stresses(self, num_x_points: int = 100, num_y_points: int = 20, num_z_points: int = 20) -> dict:
        """
        Calculate detailed 3D stress field over the beam's length and cross-section.
        
        Parameters:
        -----------
        num_x_points : int
            Number of points along the beam length.
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
        y_min, y_max = self.section.y_bottom, self.section.y_top
        z_min, z_max = self.section.z_left, self.section.z_right
        
        if y_min is None or y_max is None:
            r = np.sqrt(self.section.Iy / self.section.A)
            y_min, y_max = -2*r, 2*r
            
        if z_min is None or z_max is None:
            r = np.sqrt(self.section.Iz / self.section.A) if self.section.Iz else np.sqrt(self.section.Iy / self.section.A)
            z_min, z_max = -2*r, 2*r
            
        y_coords = np.linspace(y_min, y_max, num_y_points)
        z_coords = np.linspace(z_min, z_max, num_z_points)
        Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')
        
        # 3. Query section profile
        try:
            mask, t_yz, Q_yz = self.section.get_stress_profile(Y, Z)
        except AttributeError:
            mask = np.ones_like(Y, dtype=bool)
            t_yz = np.full_like(Y, z_max - z_min)
            Q_yz = (z_max - z_min) / 2.0 * ((y_max)**2 - Y**2)
            
        # 5. Calculate stresses (Full Vectorization)
        A, Iy = self.section.A, self.section.Iy
        shape_3d = (num_x_points, num_y_points, num_z_points)
        
        # Expand forces to 3D: (nx, 1, 1) to allow broadcasting with (ny, nz) grids
        N_3d = N_x[:, np.newaxis, np.newaxis]
        V_3d = V_x[:, np.newaxis, np.newaxis]
        M_3d = M_x[:, np.newaxis, np.newaxis]
        
        # 1. Axial stress (Broadcasting nx -> nx, ny, nz)
        sigma_a = np.broadcast_to(N_3d / A, shape_3d).copy()
        
        # 2. Bending stress (Broadcasting (nx, 1, 1) * (1, ny, nz) -> (nx, ny, nz))
        sigma_b = -M_3d * Y[np.newaxis, :, :] / Iy
        
        # 3. Shear stress
        valid_t = t_yz > 1e-9
        shear_base = np.zeros_like(Y)
        shear_base[valid_t] = Q_yz[valid_t] / (Iy * t_yz[valid_t])
        tau_s = V_3d * shear_base[np.newaxis, :, :]
        
        # 4. von Mises and Principal Stresses
        sigma_x = sigma_a + sigma_b
        # sigma_y = 0 for 1D beam elements in local coordinates
        
        # Principal stresses (plane stress): (sigma_x/2) +/- sqrt((sigma_x/2)^2 + tau_xy^2)
        avg_sigma = sigma_x / 2.0
        R = np.sqrt(avg_sigma**2 + tau_s**2)
        sigma_1 = avg_sigma + R
        sigma_2 = avg_sigma - R
        
        sigma_vm = np.sqrt(sigma_x**2 + 3 * tau_s**2)
        
        # 5. Apply cross-section mask to all 3D arrays at once
        inv_mask = ~np.broadcast_to(mask[np.newaxis, :, :], shape_3d)
        sigma_a[inv_mask] = 0.0
        sigma_b[inv_mask] = 0.0
        tau_s[inv_mask] = 0.0
        sigma_1[inv_mask] = 0.0
        sigma_2[inv_mask] = 0.0
        sigma_vm[inv_mask] = 0.0

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

    
    def generate_report(self, output_path: str, deformation_scale: float = 1.0):
        """
        Generate a professional markdown report of the analysis.
        Images are saved to a ``<report_name>_images/`` folder alongside the report.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report (e.g., 'report.md')
        deformation_scale : float or 'auto', optional
            Scale factor for deformed shape plots. Default is 1.0.
        """
        if self.displacements is None or self.last_load_case is None:
            raise ValueError("Must run solve_static() before generating a report")
            
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
            num_points = kwargs.pop('num_points', 100)
            forces = self.calculate_internal_forces(num_points)
            return self.visualizer.plot_shear_force(forces['shear_forces'], forces['positions'], **kwargs)
        elif analysis_type == 'moment':
            num_points = kwargs.pop('num_points', 100)
            forces = self.calculate_internal_forces(num_points)
            return self.visualizer.plot_bending_moment(forces['bending_moments'], forces['positions'], **kwargs)
        elif analysis_type == 'modal':
            # Would plot mode shapes (implementation in BeamVisualizer)
            pass


if __name__ == "__main__":
    print("\nMain Beam Solver Module")
    print("Coordinates all analysis modules")
