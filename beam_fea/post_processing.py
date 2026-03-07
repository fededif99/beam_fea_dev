"""
post_processing.py
==================
Internal force and stress recovery engines.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Union

class ForceRecoveryStrategy(ABC):
    """Abstract base class for internal force recovery strategies."""

    @abstractmethod
    def recover(self, element, u_local: np.ndarray, xi: Union[float, np.ndarray],
                dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover internal forces at normalized position(s) xi.

        Parameters:
        -----------
        element : BeamElementMatrices
            The beam element formulation expert (Euler, Timoshenko, etc.).
        u_local : np.ndarray
            Local element displacement vector [u1, v1, theta1, u2, v2, theta2].
        xi : Union[float, np.ndarray]
            Normalized position(s) along element [0, 1].
        dist_load : Tuple[float, float, float, float], optional
            Distributed loads: (wy1, wy2, wx1, wx2).

        Returns:
        --------
        (axial, shear, moment) : Tuple[np.ndarray, np.ndarray, np.ndarray]
            The extracted internal forces at the requested domain points.
        """
        pass

class NodalInterpolationStrategy(ForceRecoveryStrategy):
    """
    Standard FEA interpolation using shape function derivatives.
    Mathematically consistent with the displacement approximation.
    """

    def recover(self, element, u_local: np.ndarray, xi: Union[float, np.ndarray],
                dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Delegate to element's homogeneous interpolation logic
        return element.interpolate_forces_homogeneous(u_local, xi)

class ConsistentRecoveryStrategy(ForceRecoveryStrategy):
    """
    Statically consistent force recovery (Homogeneous + Particular).
    Enforces local equilibrium by integrating distributed loads.
    """

    def recover(self, element, u_local: np.ndarray, xi: Union[float, np.ndarray],
                dist_load: Tuple[float, float, float, float] = (0, 0, 0, 0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Delegate to element's full recovery logic
        return element.recover_forces_consistent(u_local, xi, dist_load)

class ResultsContainer:
    """
    Unified container for analysis results to avoid redundant calculations.
    """
    def __init__(self):
        self.positions = None
        self.axial_forces = None
        self.shear_forces = None
        self.bending_moments = None
        self.stresses = None
        self.displacements = None
        self.reactions = None
        self.recovery_strategy = "consistent"
        self.num_points = 100

class InternalForceEngine:
    """Engine to coordinate internal force calculation across the mesh."""

    @staticmethod
    def calculate(solver, num_points: int = None, strategy_name: str = "consistent") -> dict:
        if solver.displacements is None:
            raise ValueError("Must run solve_static() first")

        mesh = solver.mesh
        coords = mesh.get_node_coords()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])

        if num_points is None:
            positions = np.sort(coords[:, 0])
        else:
            positions = np.linspace(x_min, x_max, num_points)

        eval_count = len(positions)
        axial_forces = np.zeros(eval_count)
        shear_forces = np.zeros(eval_count)
        bending_moments = np.zeros(eval_count)

        # Select strategy
        if strategy_name == "consistent":
            strategy = ConsistentRecoveryStrategy()
        else:
            strategy = NodalInterpolationStrategy()

        # Pre-calculate distributed loads per element
        element_dist_loads = InternalForceEngine._get_element_dist_loads(solver)

        # Vectorization optimization: Group points by element
        # This is faster than the previous station-by-station loop
        points_by_element = {}
        for i, x in enumerate(positions):
            eid = mesh.find_element_at_x(x)
            if eid == -1:
                eid = 0 if x <= x_min else mesh.num_elements - 1
            if eid not in points_by_element:
                points_by_element[eid] = []
            points_by_element[eid].append(i)

        from .element_matrices import EulerBernoulliElement, TimoshenkoElement, AnisotropicBeamElement, SHEAR_CORRECTION_FACTOR
        from .composites import Laminate

        for eid, indices in points_by_element.items():
            elem = mesh.elements[eid]
            p1, p2 = coords[elem.node1], coords[elem.node2]
            x1 = p1[0]
            L = np.sqrt(np.sum((p2 - p1)**2))

            dof_indices = [3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                           3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2]
            u_local = solver.displacements[dof_indices]

            # Local normalized coordinates for this element
            xi = np.array([(positions[idx] - x1) / L for idx in indices])
            xi = np.clip(xi, 0, 1) if L > 0 else np.zeros_like(xi)

            # Element expert
            mat = solver.properties.get_material(elem.id)
            sec = solver.properties.get_section(elem.id)

            if hasattr(mat, '_is_laminate') or isinstance(mat, Laminate):
                width = getattr(sec, 'width', getattr(sec, 'diameter', 1.0))
                EA, ES, EI = mat.get_sectional_stiffness(width)
                rho_total = mat.rho * width * mat.total_thickness
                props = mat.get_effective_properties()
                GA_s = SHEAR_CORRECTION_FACTOR * props['Gxy'] * (width * mat.total_thickness)
                element_expert = AnisotropicBeamElement(EA=EA, ES=ES, EI=EI, L=L, rho_total=rho_total, GA_s=GA_s)
            else:
                E, G, I, A, rho = mat.E, mat.G, sec.Iy, sec.A, mat.rho
                if solver.element_type == 'euler':
                    element_expert = EulerBernoulliElement(E=E, G=G, I=I, A=A, L=L, rho=rho)
                else:
                    element_expert = TimoshenkoElement(E=E, G=G, I=I, A=A, L=L, rho=rho)

            d_load = element_dist_loads.get(eid, (0.0, 0.0, 0.0, 0.0))
            N, V, M = strategy.recover(element_expert, u_local, xi, dist_load=d_load)

            axial_forces[indices] = N
            shear_forces[indices] = V
            bending_moments[indices] = M

        return {
            'positions': positions,
            'axial_forces': axial_forces,
            'shear_forces': shear_forces,
            'bending_moments': bending_moments
        }

    @staticmethod
    def _get_element_dist_loads(solver):
        """Helper to consolidate distributed loads per element."""
        element_dist_loads = {}
        if not solver.last_load_case:
            return element_dist_loads

        from .loads import UniformDistributedLoad, TrapezoidalDistributedLoad, TriangularDistributedLoad
        coords = solver.mesh.get_node_coords()

        for load in solver.last_load_case.loads:
            wy1 = wy2 = wx1 = wx2 = 0.0
            if isinstance(load, UniformDistributedLoad):
                wy1 = wy2 = load.wy; wx1 = wx2 = load.wx
            elif isinstance(load, TrapezoidalDistributedLoad):
                wy1, wy2 = load.wy1, load.wy2; wx1, wx2 = load.wx1, load.wx2
            elif isinstance(load, TriangularDistributedLoad):
                wy1, wy2 = (load.w_peak, 0.0) if load.peak_loc == 'start' else (0.0, load.w_peak)
            else: continue

            if load.element is not None:
                elems = [load.element] if isinstance(load.element, int) else load.element
                for eid in elems:
                    curr = element_dist_loads.get(eid, (0, 0, 0, 0))
                    element_dist_loads[eid] = (curr[0] + wy1, curr[1] + wy2, curr[2] + wx1, curr[3] + wx2)
            elif load.x_start is not None:
                x_s, x_e = min(load.x_start, load.x_end), max(load.x_start, load.x_end)
                for eid, elem in enumerate(solver.mesh.elements):
                    p1, p2 = coords[elem.node1, 0], coords[elem.node2, 0]
                    e_min, e_max = min(p1, p2), max(p1, p2)
                    i_min, i_max = max(e_min, x_s), min(e_max, x_e)
                    if i_max > i_min:
                        def get_w(x, w1, w2, xs, xe):
                            if abs(xe - xs) < 1e-9: return w1
                            return w1 + (w2 - w1) * (x - xs) / (xe - xs)
                        wya = get_w(p1, wy1, wy2, load.x_start, load.x_end)
                        wyb = get_w(p2, wy1, wy2, load.x_start, load.x_end)
                        wxa = get_w(p1, wx1, wx2, load.x_start, load.x_end)
                        wxb = get_w(p2, wx1, wx2, load.x_start, load.x_end)
                        curr = element_dist_loads.get(eid, (0, 0, 0, 0))
                        element_dist_loads[eid] = (curr[0] + wya, curr[1] + wyb, curr[2] + wxa, curr[3] + wxb)
        return element_dist_loads

class StressEngine:
    """Engine to coordinate 3D stress field calculation."""

    @staticmethod
    def calculate(solver, num_x_points: int = None, num_y_points: int = 20, num_z_points: int = 20) -> dict:
        forces = solver.calculate_internal_forces(num_x_points)
        x_positions = forces['positions']
        N_x = forces['axial_forces']
        V_x = forces['shear_forces']
        M_x = forces['bending_moments']

        # Build 2D grid for the cross-section
        # (Logic moved from solver.py and optimized)
        y_min, y_max, z_min, z_max = StressEngine._get_global_bounding_box(solver)

        y_coords = np.linspace(y_min, y_max, num_y_points)
        z_coords = np.linspace(z_min, z_max, num_z_points)
        Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

        eval_x_count = len(x_positions)
        shape_3d = (eval_x_count, num_y_points, num_z_points)

        sigma_a = np.zeros(shape_3d)
        sigma_b = np.zeros(shape_3d)
        tau_s = np.zeros(shape_3d)
        sigma_vm = np.zeros(shape_3d)
        sigma_1 = np.zeros(shape_3d)
        sigma_2 = np.zeros(shape_3d)

        # Group stations by element for vectorization
        points_by_element = {}
        for i, x in enumerate(x_positions):
            eid = solver.mesh.find_element_at_x(x)
            if eid == -1: eid = 0 if x <= x_positions[0] else solver.mesh.num_elements - 1
            if eid not in points_by_element: points_by_element[eid] = []
            points_by_element[eid].append(i)

        profile_cache = {}
        from .composites import Laminate

        for eid, indices in points_by_element.items():
            mat = solver.properties.get_material(eid)
            sec = solver.properties.get_section(eid)
            sec_id = id(sec)

            if sec_id not in profile_cache:
                profile_cache[sec_id] = StressEngine._get_section_profile(sec, Y, Z)

            mask, t_yz, Q_yz = profile_cache[sec_id]

            # --- Special Handling for Laminates (Ply-by-Ply) ---
            if hasattr(mat, '_is_laminate') or isinstance(mat, Laminate):
                lam = mat
                t_lam = lam.total_thickness

                # Laminate mid-plane strains and curvatures for these stations
                # For 1D beam: eps_x = eps_x_0 + z * kappa_x
                # u_local = [u1, v1, th1, u2, v2, th2]
                # Actually we have global forces N, V, M.
                # Midplane resultants per unit width: n = N/width, m = M/width
                width = getattr(sec, 'width', getattr(sec, 'diameter', 1.0))

                N_sub = N_x[indices] / width
                M_sub = M_x[indices] / width

                # Solve [ABD][eps0; kappa] = [n; m]
                # [n_x, n_y, n_xy, m_x, m_y, m_xy]^T
                # In 1D beam, we assume n_y=n_xy=m_y=m_xy = 0
                load_vectors = np.zeros((len(indices), 6))
                load_vectors[:, 0] = N_sub
                load_vectors[:, 3] = M_sub

                try:
                    ABD_inv = np.linalg.inv(lam.ABD)
                    strains_mid = (ABD_inv @ load_vectors.T).T # (n_stations, 6)
                except np.linalg.LinAlgError:
                    strains_mid = np.zeros((len(indices), 6))

                # For each ply, calculate stress at its top, bottom and mid
                ply_stresses = []
                z_ply = -t_lam / 2.0
                for i, (ply, angle) in enumerate(lam.plies):
                    Qbar = ply.transformed_reduced_stiffness(angle)

                    # Heights within the ply (top and bottom relative to midplane)
                    z_bot = z_ply
                    z_top = z_ply + ply.thickness
                    z_mid = z_ply + ply.thickness / 2.0

                    # Strains at these heights: eps = eps0 + z * kappa
                    # eps0 = strains_mid[:, 0:3], kappa = strains_mid[:, 3:6]
                    eps_bot = strains_mid[:, 0:3] + z_bot * strains_mid[:, 3:6]
                    eps_top = strains_mid[:, 0:3] + z_top * strains_mid[:, 3:6]

                    # Stresses at these heights: sig = Qbar * eps
                    sig_bot = (Qbar @ eps_bot.T).T # (n_stations, 3)
                    sig_top = (Qbar @ eps_top.T).T # (n_stations, 3)

                    # Principal and Von Mises Calculation for each station
                    # sigma_x = sig[:, 0], sigma_y = sig[:, 1], tau_xy = sig[:, 2]
                    def calc_vm_principal(sig):
                        avg = (sig[:, 0] + sig[:, 1]) / 2.0
                        r = np.sqrt(((sig[:, 0] - sig[:, 1]) / 2.0)**2 + sig[:, 2]**2)
                        s1 = avg + r
                        s2 = avg - r
                        # 2D plane stress VM: sqrt(s1^2 + s2^2 - s1*s2) or sqrt(sx^2 + sy^2 - sx*sy + 3*txy^2)
                        vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 - sig[:, 0]*sig[:, 1] + 3*sig[:, 2]**2)
                        return vm, s1, s2

                    vm_bot, s1_bot, s2_bot = calc_vm_principal(sig_bot)
                    vm_top, s1_top, s2_top = calc_vm_principal(sig_top)

                    ply_stresses.append({
                        'index': i,
                        'name': ply.name,
                        'angle': angle,
                        'z_range': (z_bot, z_top),
                        'sigma_x': (sig_bot[:, 0] + sig_top[:, 0]) / 2.0, # Average for the station's 3D field
                        'peak_sigma_x': np.maximum(np.abs(sig_bot[:, 0]), np.abs(sig_top[:, 0])),
                        'peak_tau_xy': np.maximum(np.abs(sig_bot[:, 2]), np.abs(sig_top[:, 2])),
                        'peak_von_mises': np.maximum(vm_bot, vm_top),
                        'peak_sigma_1': np.maximum(s1_bot, s1_top),
                        'sigma_bot': sig_bot,
                        'sigma_top': sig_top
                    })
                    z_ply += ply.thickness

                # Map ply stresses back to Y, Z grid for standard results
                # In our 1D beam model, Y is the thickness direction for laminates.
                for idx_in_sub, global_idx in enumerate(indices):
                    for ply_info in ply_stresses:
                        z_b, z_t = ply_info['z_range']
                        # mask_ply identifies points in the grid that belong to this ply
                        mask_ply = mask & (Y >= z_b) & (Y <= z_t)
                        sigma_a[global_idx, mask_ply] = ply_info['sigma_x'][idx_in_sub]
                        # We use von_mises slot for peak ply stress for now
                        sigma_vm[global_idx, mask_ply] = ply_info['peak_sigma_x'][idx_in_sub]

                # Store ply-by-ply data in a separate attribute for querying
                if not hasattr(solver, 'laminate_results'):
                    solver.laminate_results = {}
                solver.laminate_results[eid] = {
                    'ply_data': ply_stresses,
                    'strains_mid': strains_mid,
                    'x_positions': x_positions[indices]
                }
                continue # Skip standard isotropic calculation
            A, Iy = sec.A, sec.Iy

            # Forces for these stations
            N_sub = N_x[indices, np.newaxis, np.newaxis]
            V_sub = V_x[indices, np.newaxis, np.newaxis]
            M_sub = M_x[indices, np.newaxis, np.newaxis]

            # Stress calculation (vectorized across these stations)
            sig_a_sub = N_sub / A
            sig_b_sub = -M_sub * Y[np.newaxis, :, :] / Iy

            valid_t = t_yz > 1e-9
            shear_base = np.zeros_like(Y)
            shear_base[valid_t] = Q_yz[valid_t] / (Iy * t_yz[valid_t])
            tau_s_sub = V_sub * shear_base[np.newaxis, :, :]

            sig_x_sub = sig_a_sub + sig_b_sub
            vm_sub = np.sqrt(sig_x_sub**2 + 3 * tau_s_sub**2)

            # Principal stresses
            avg_sig = sig_x_sub / 2.0
            R = np.sqrt(avg_sig**2 + tau_s_sub**2)

            # Masking
            # mask is (ny, nz), sig_a_sub is (n_stations, ny, nz)
            # Apply mask using broadcasting for all stations
            m_3d = mask[np.newaxis, :, :]
            sig_a_sub = sig_a_sub * m_3d
            sig_b_sub = sig_b_sub * m_3d
            tau_s_sub = tau_s_sub * m_3d
            vm_sub = vm_sub * m_3d

            sigma_a[indices] = sig_a_sub
            sigma_b[indices] = sig_b_sub
            tau_s[indices] = tau_s_sub
            sigma_vm[indices] = vm_sub
            sigma_1[indices] = avg_sig + R
            sigma_2[indices] = avg_sig - R

        return {
            'x': x_positions, 'y': Y, 'z': Z, 'mask': mask,
            'axial': sigma_a, 'bending': sigma_b, 'shear': tau_s,
            'sigma_1': sigma_1, 'sigma_2': sigma_2, 'von_mises': sigma_vm
        }

    @staticmethod
    def _get_global_bounding_box(solver):
        # Union of all sections
        y_min, y_max = solver.properties.default_section.y_bottom, solver.properties.default_section.y_top
        z_min, z_max = solver.properties.default_section.z_left, solver.properties.default_section.z_right

        processed_ids = set()
        unique_secs = []
        for i in range(solver.mesh.num_elements):
            sec = solver.properties.get_section(i)
            if id(sec) not in processed_ids:
                processed_ids.add(id(sec))
                unique_secs.append(sec)

        for sec in unique_secs:
            y_min = min(y_min, sec.y_bottom); y_max = max(y_max, sec.y_top)
            z_min = min(z_min, sec.z_left); z_max = max(z_max, sec.z_right)

        return y_min, y_max, z_min, z_max

    @staticmethod
    def get_ply_stresses(solver, element_id: int, x_station_idx: int = 0):
        """
        Query detailed ply-by-ply stresses for a specific element and station.
        """
        if not hasattr(solver, 'laminate_results') or element_id not in solver.laminate_results:
            return None

        res = solver.laminate_results[element_id]
        ply_data = res['ply_data']

        station_stresses = []
        for ply in ply_data:
            station_stresses.append({
                'ply_index': ply['index'],
                'ply_name': ply['name'],
                'angle': ply['angle'],
                'sigma_x_bot': ply['sigma_bot'][x_station_idx, 0],
                'sigma_x_top': ply['sigma_top'][x_station_idx, 0],
                'sigma_y_bot': ply['sigma_bot'][x_station_idx, 1],
                'sigma_y_top': ply['sigma_top'][x_station_idx, 1],
                'tau_xy_bot': ply['sigma_bot'][x_station_idx, 2],
                'tau_xy_top': ply['sigma_top'][x_station_idx, 2],
            })
        return station_stresses

    @staticmethod
    def _get_section_profile(sec, Y, Z):
        try:
            return sec.get_stress_profile(Y, Z)
        except AttributeError:
            y_top = sec.y_top if sec.y_top is not None else np.sqrt(sec.A)/2
            y_bot = sec.y_bottom if sec.y_bottom is not None else -np.sqrt(sec.A)/2
            z_r = sec.z_right if sec.z_right is not None else (sec.Iz/sec.A)**0.5 if sec.Iz else np.sqrt(sec.A)/2
            z_l = sec.z_left if sec.z_left is not None else -(sec.Iz/sec.A)**0.5 if sec.Iz else -np.sqrt(sec.A)/2
            mask = (Y <= y_top) & (Y >= y_bot) & (Z <= z_r) & (Z >= z_l)
            t_yz = np.full_like(Y, z_r - z_l)
            Q_yz = (t_yz) / 2.0 * (y_top**2 - Y**2)
            Q_yz[~mask] = 0.0; t_yz[~mask] = 0.0
            return mask, t_yz, Q_yz
