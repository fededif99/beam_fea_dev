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
        
        for elem in self.mesh.elements:
            L = elem.length(self.mesh.nodes)
            
            # Create element matrices
            if self.element_type == 'euler':
                element = EulerBernoulliElement(
                    E=self.material.E,
                    G=self.material.G,
                    I=self.section.Iy,
                    A=self.section.A,
                    L=L,
                    rho=self.material.rho
                )
            elif self.element_type == 'timoshenko':
                element = TimoshenkoElement(
                    E=self.material.E,
                    G=self.material.G,
                    I=self.section.Iy,
                    A=self.section.A,
                    L=L,
                    rho=self.material.rho
                )
            else:
                raise ValueError("element_type must be 'euler' or 'timoshenko'")
            
            k_local = element.stiffness_matrix()
            m_local = element.mass_matrix()
            
            # Get DOF indices
            dof_indices = np.array([
                3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
            ])
            
            # Efficiently append to lists
            for i, gi in enumerate(dof_indices):
                for j, gj in enumerate(dof_indices):
                    # Stiffness
                    val_k = k_local[i, j]
                    if val_k != 0:
                        K_rows.append(gi)
                        K_cols.append(gj)
                        K_data.append(val_k)
                    
                    # Mass
                    val_m = m_local[i, j]
                    if val_m != 0:
                        M_rows.append(gi)
                        M_cols.append(gj)
                        M_data.append(val_m)
        
        # Create sparse matrices
        self.K_global = coo_matrix((K_data, (K_rows, K_cols)), shape=(num_dofs, num_dofs)).tocsr()
        self.M_global = coo_matrix((M_data, (M_rows, M_cols)), shape=(num_dofs, num_dofs)).tocsr()
    
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
    
    def calculate_internal_forces(self, num_points: int = 100):
        """
        Calculate shear force and bending moment along the beam on-demand.
        
        Parameters:
        -----------
        num_points : int
            Number of evaluation points along beam length.
            
        Returns:
        --------
        results : dict
            Dictionary containing 'positions', 'shear_forces', and 'bending_moments'.
        """
        if self.displacements is None:
            raise ValueError("Must run solve_static() first")
            
        coords = self.mesh.get_node_coords()
        beam_length = np.max(coords[:, 0]) - np.min(coords[:, 0])
        
        # Create evaluation points
        positions = np.linspace(0, beam_length, num_points)
        shear_forces = np.zeros(num_points)
        bending_moments = np.zeros(num_points)
        
        # Determine evaluation points per element to avoid redundant element instantiation
        for i, x in enumerate(positions):
            # Find element (simple linear mapping for now, assuming nodes are sorted by x)
            elem_idx = int(x / beam_length * self.mesh.num_elements)
            if elem_idx >= self.mesh.num_elements:
                elem_idx = self.mesh.num_elements - 1
            
            elem = self.mesh.elements[elem_idx]
            L = elem.length(self.mesh.nodes)
            
            # Local element displacements
            dof_indices = np.array([
                3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2
            ])
            u_local = self.displacements[dof_indices]
            
            # Local position xi [0, 1]
            x_local = x - coords[elem.node1, 0]
            xi = np.clip(x_local / L, 0, 1)
            
            # Instantiate the correct element type expert
            if self.element_type == 'euler':
                from .element_matrices import EulerBernoulliElement
                element = EulerBernoulliElement(
                    E=self.material.E, G=self.material.G, I=self.section.Iy,
                    A=self.section.A, L=L, rho=self.material.rho
                )
            else:
                from .element_matrices import TimoshenkoElement
                element = TimoshenkoElement(
                    E=self.material.E, G=self.material.G, I=self.section.Iy,
                    A=self.section.A, L=L, rho=self.material.rho
                )
            
            V, M = element.interpolate_internal_forces(u_local, xi)
            shear_forces[i] = V
            bending_moments[i] = M
            
        return {
            'positions': positions,
            'shear_forces': shear_forces,
            'bending_moments': bending_moments
        }
    
    def generate_report(self, output_path: str, embed_images: bool = True, deformation_scale: float = 1.0):
        """
        Generate a professional markdown report of the analysis.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report (e.g., 'report.md')
        embed_images : bool, optional
            If True, embed images as Base64 strings. If False, save as external PNGs. Default is True.
        deformation_scale : float, optional
            Scale factor for deformed shape plots. Default is 1.0.
        """
        if self.displacements is None or self.last_load_case is None:
            raise ValueError("Must run solve_static() before generating a report")
            
        # Lazy import to avoid circular dependencies if any
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
        
        return report_gen.generate_report(output_path, deformation_scale, embed_images)
    
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
