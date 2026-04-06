"""
assembly.py
===========
Vectorized matrix assembly engine for global stiffness and mass matrices.
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Optional

class AssemblyEngine:
    """
    Dedicated engine for high-performance vectorized global matrix assembly.
    Decouples the assembly logic from the solver.
    """

    @staticmethod
    def assemble(mesh, properties, element_type: str = 'euler', 
                assembly_type: str = 'both') -> Tuple[Optional[csr_matrix], Optional[csr_matrix]]:
        """
        Perform vectorized assembly of global stiffness (K) and mass (M) matrices.
        
        Parameters:
        -----------
        mesh : Mesh
            The finite element mesh.
        properties : PropertySet
            Resolved property set containing cached element arrays.
        element_type : str
            'euler' or 'timoshenko'.
        assembly_type : str
            'stiffness', 'mass', or 'both'.
            
        Returns:
        --------
        (K_global, M_global) : Tuple
            Global matrices in CSR format. None if not requested.
        """
        num_elements = mesh.num_elements
        num_dofs = mesh.num_dofs
        coords = mesh.nodes
        elements = mesh.elements
        is_euler = (element_type.lower() == 'euler')
        
        do_k = assembly_type in ['stiffness', 'both']
        do_m = assembly_type in ['mass', 'both']
        
        # 1. Vectorized Geometry
        n1 = coords[elements[:, 0]]
        n2 = coords[elements[:, 1]]
        diff = n2 - n1
        L = np.sqrt(np.sum(diff**2, axis=1))
        angle = np.arctan2(diff[:, 1], diff[:, 0])

        # 2. Properties (Directly from pre-calculated PropertySet arrays)
        EA = properties.EA
        EI = properties.EI
        ES = properties.ES
        GA_s = properties.GA_s
        rho_lin = properties.rho_lin

        # 3. Vectorized Rotation Transformations
        # T defined as u_local = T @ u_global
        c = np.cos(angle)
        s = np.sin(angle)
        T = np.zeros((num_elements, 6, 6))
        for i in [0, 3]:
            T[:, i, i] = c; T[:, i, i+1] = s
            T[:, i+1, i] = -s; T[:, i+1, i+1] = c
            T[:, i+2, i+2] = 1.0

        # 4. Global Index Mapping
        dof_indices = np.zeros((num_elements, 6), dtype=int)
        dof_indices[:, 0:3] = 3 * elements[:, 0:1] + np.array([0, 1, 2])
        dof_indices[:, 3:6] = 3 * elements[:, 1:2] + np.array([0, 1, 2])
        rows = np.repeat(dof_indices, 6, axis=1).flatten()
        cols = np.tile(dof_indices, (1, 6)).flatten()

        K_global = None
        M_global = None

        # 5. Stiffness Matrix Assembly
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
            K_global = coo_matrix((K_global_el.flatten(), (rows, cols)), 
                                 shape=(num_dofs, num_dofs)).tocsr()

        # 6. Mass Matrix Assembly
        if do_m:
            M_local = np.zeros((num_elements, 6, 6))
            m = rho_lin
            phi = np.zeros(num_elements)
            if not is_euler:
                mask = GA_s > 0
                phi[mask] = 12.0 * EI[mask] / (GA_s[mask] * L[mask]**2)
            
            # Common Axial components
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
            M_global = coo_matrix((M_global_el.flatten(), (rows, cols)), 
                                 shape=(num_dofs, num_dofs)).tocsr()
                                 
        return K_global, M_global
