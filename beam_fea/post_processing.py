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

class StationEvaluator:
    """Helper to group evaluation stations by element and calculate local kinematics."""

    @staticmethod
    def get_evaluation_plan(solver, num_points: int = None) -> dict:
        """
        Groups global evaluation points into element-local stations.

        Returns:
        --------
        plan : dict
            {positions, points_by_element}
        """
        mesh = solver.mesh
        coords = mesh.get_node_coords()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])

        if num_points is None:
            positions = np.sort(coords[:, 0])
        else:
            positions = np.linspace(x_min, x_max, num_points)

        points_by_element = {}
        for i, x in enumerate(positions):
            eid = mesh.find_element_at_x(x)
            if eid == -1:
                eid = 0 if x <= x_min else mesh.num_elements - 1
            if eid not in points_by_element:
                points_by_element[eid] = []
            points_by_element[eid].append(i)

        return {'positions': positions, 'points_by_element': points_by_element}


class InternalForceEngine:
    """Engine to coordinate internal force calculation across the mesh."""

    @staticmethod
    def calculate(solver, num_points: int = None, strategy_name: str = "consistent") -> dict:
        if solver.displacements is None:
            raise ValueError("Must run solve_static() first")

        eval_plan = StationEvaluator.get_evaluation_plan(solver, num_points)
        positions = eval_plan['positions']
        points_by_element = eval_plan['points_by_element']

        eval_count = len(positions)
        axial_forces = np.zeros(eval_count)
        shear_forces = np.zeros(eval_count)
        bending_moments = np.zeros(eval_count)

        if strategy_name == "consistent":
            strategy = ConsistentRecoveryStrategy()
        else:
            strategy = NodalInterpolationStrategy()

        element_dist_loads = InternalForceEngine._get_element_dist_loads(solver)

        from .element_matrices import UnifiedBeamElement
        coords = solver.mesh.get_node_coords()
        is_euler = (solver.element_type == 'euler')

        for eid, indices in points_by_element.items():
            elem = solver.mesh.elements[eid]
            p1, p2 = coords[elem.node1], coords[elem.node2]
            x1, L = p1[0], np.sqrt(np.sum((p2 - p1)**2))

            dof_indices = [3*elem.node1, 3*elem.node1 + 1, 3*elem.node1 + 2,
                           3*elem.node2, 3*elem.node2 + 1, 3*elem.node2 + 2]
            u_local = solver.displacements[dof_indices]

            xi = np.clip((positions[indices] - x1) / L, 0, 1) if L > 0 else np.zeros(len(indices))

            mat = solver.properties.get_material(elem.id)
            sec = solver.properties.get_section(elem.id)
            stiff = mat.get_sectional_stiffness(sec)
            rho_lin = mat.get_linear_density(sec)

            element_expert = UnifiedBeamElement(
                EA=stiff['EA'], ES=stiff['ES'], EI=stiff['EI'],
                L=L, rho_total=rho_lin, GA_s=stiff['GA_s'] * sec.shear_factor,
                force_euler=is_euler
            )

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
        V_x = forces['shear_forces'] # Transverse shear force V_y
        M_x = forces['bending_moments']

        # Build 2D grid for the cross-section
        y_min, y_max, z_min, z_max = StressEngine._get_global_bounding_box(solver)
        y_coords = np.linspace(y_min, y_max, num_y_points)
        z_coords = np.linspace(z_min, z_max, num_z_points)
        Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

        eval_plan = StationEvaluator.get_evaluation_plan(solver, num_x_points)
        points_by_element = eval_plan['points_by_element']

        eval_x_count = len(x_positions)
        shape_3d = (eval_x_count, num_y_points, num_z_points)

        sigma_a, sigma_b, tau_s, sigma_vm = [np.zeros(shape_3d) for _ in range(4)]
        sigma_1, sigma_2 = np.zeros(shape_3d), np.zeros(shape_3d)

        profile_cache = {}

        for eid, indices in points_by_element.items():
            mat = solver.properties.get_material(eid)
            sec = solver.properties.get_section(eid)
            sec_id = id(sec)

            if sec_id not in profile_cache:
                profile_cache[sec_id] = StressEngine._get_section_profile(sec, Y, Z)

            mask, t_yz, Q_yz = profile_cache[sec_id]

            # --- Unified Through-Thickness Recovery ---
            # Treat Isotropic as a 1-ply laminate for code unification
            from .composites import Laminate, Ply
            if not hasattr(mat, 'plies'):
                # Convert isotropic Material to a single-ply Laminate
                lam = Laminate(name=mat.name)
                # G13/G23 are equal to G for isotropic
                p = Ply(name=mat.name, E1=mat.E, E2=mat.E, nu12=mat.nu, G12=mat.G,
                        G13=mat.G, G23=mat.G, thickness=sec.y_top - sec.y_bottom,
                        rho=mat.rho, Xt=mat.yield_strength or 0.0,
                        Xc=mat.yield_strength or 0.0, Yt=mat.yield_strength or 0.0,
                        Yc=mat.yield_strength or 0.0, S=mat.yield_strength / 1.732 if mat.yield_strength else 0.0)
                lam.add_ply(p, 0.0)
            else:
                lam = mat

            if True: # Always use the advanced recovery logic now
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

                # Transverse Shear Stress Recovery (tau_xz)
                # Recovered by integrating equilibrium: d_sigma_x/dx + d_tau_xz/dz = 0
                # tau_xz(z) = - integral_{-t/2}^{z} (d_sigma_x/dx) dz
                # Since sigma_x = Qbar_11 * eps_x + ...
                # d_sigma_x/dx is related to dN/dx (axial load) and dM/dx (shear force V)
                # In 1D beam: dM/dx = V. dN/dx = -wx (axial distributed load)

                # Recover d_strains_mid/dx: [ABD_inv] * [dN/dx, 0, 0, dM/dx, 0, 0]^T
                # Load rates: dN/dx = -wx_sub, dM/dx = V_sub
                element_dist_loads = InternalForceEngine._get_element_dist_loads(solver)
                d_load = element_dist_loads.get(eid, (0.0, 0.0, 0.0, 0.0))
                # Approximate wx as average across element stations for this estação
                wx_avg = (d_load[2] + d_load[3]) / 2.0

                load_rates = np.zeros((len(indices), 6))
                load_rates[:, 0] = -wx_avg / width
                load_rates[:, 3] = V_x[indices] / width

                try:
                    dstrains_dx = (ABD_inv @ load_rates.T).T # (n_stations, 6)
                except np.linalg.LinAlgError:
                    dstrains_dx = np.zeros((len(indices), 6))

                # For each ply, calculate stress at its top, bottom and mid
                ply_stresses = []
                z_ply = -t_lam / 2.0
                for i, (ply, angle) in enumerate(lam.plies):
                    Qbar = ply.transformed_reduced_stiffness(angle)

                    # Heights within the ply (top, mid, bottom relative to midplane)
                    z_bot = z_ply
                    z_top = z_ply + ply.thickness
                    z_mid = z_ply + ply.thickness / 2.0

                    # Strains at these heights: eps = eps0 + z * kappa
                    # eps0 = strains_mid[:, 0:3], kappa = strains_mid[:, 3:6]
                    eps_bot = strains_mid[:, 0:3] + z_bot * strains_mid[:, 3:6]
                    eps_mid = strains_mid[:, 0:3] + z_mid * strains_mid[:, 3:6]
                    eps_top = strains_mid[:, 0:3] + z_top * strains_mid[:, 3:6]

                    # Stresses at these heights: sig = Qbar * eps (Global laminate coordinates x, y, xy)
                    sig_bot = (Qbar @ eps_bot.T).T # (n_stations, 3)
                    sig_mid = (Qbar @ eps_mid.T).T # (n_stations, 3)
                    sig_top = (Qbar @ eps_top.T).T # (n_stations, 3)

                    # Axial vs Bending decomposition for sigma_x
                    sig_a = (Qbar @ strains_mid[:, 0:3].T).T[:, 0]
                    sig_b_bot = sig_bot[:, 0] - sig_a
                    sig_b_top = sig_top[:, 0] - sig_a

                    # Transform to local material coordinates (1, 2, 12)
                    def to_local(sig, ang):
                        theta = np.radians(ang)
                        c, s = np.cos(theta), np.sin(theta)
                        # sigma_1 = sx*c^2 + sy*s^2 + 2*txy*s*c
                        s1 = sig[:, 0]*c**2 + sig[:, 1]*s**2 + 2*sig[:, 2]*s*c
                        # sigma_2 = sx*s^2 + sy*c^2 - 2*txy*s*c
                        s2 = sig[:, 0]*s**2 + sig[:, 1]*c**2 - 2*sig[:, 2]*s*c
                        # tau_12 = (sy-sx)*s*c + txy*(c^2-s^2)
                        s12 = (sig[:, 1] - sig[:, 0])*s*c + sig[:, 2]*(c**2 - s**2)
                        return np.column_stack([s1, s2, s12])

                    loc_bot = to_local(sig_bot, angle)
                    loc_mid = to_local(sig_mid, angle)
                    loc_top = to_local(sig_top, angle)

                    # Principal and Von Mises Calculation for each station
                    def calc_vm_principal(sig):
                        avg = (sig[:, 0] + sig[:, 1]) / 2.0
                        r = np.sqrt(((sig[:, 0] - sig[:, 1]) / 2.0)**2 + sig[:, 2]**2)
                        s1p = avg + r
                        s2p = avg - r
                        vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 - sig[:, 0]*sig[:, 1] + 3*sig[:, 2]**2)
                        return vm, s1p, s2p

                    vm_bot, s1p_bot, s2p_bot = calc_vm_principal(sig_bot)
                    vm_mid, s1p_mid, s2p_mid = calc_vm_principal(sig_mid)
                    vm_top, s1p_top, s2p_top = calc_vm_principal(sig_top)

                    # Failure Indices
                    fi_max_bot = np.array([ply.calculate_failure_index(s, 'max_stress') for s in loc_bot])
                    fi_th_bot = np.array([ply.calculate_failure_index(s, 'tsai_hill') for s in loc_bot])
                    fi_tw_bot = np.array([ply.calculate_failure_index(s, 'tsai_wu') for s in loc_bot])

                    fi_max_top = np.array([ply.calculate_failure_index(s, 'max_stress') for s in loc_top])
                    fi_th_top = np.array([ply.calculate_failure_index(s, 'tsai_hill') for s in loc_top])
                    fi_tw_top = np.array([ply.calculate_failure_index(s, 'tsai_wu') for s in loc_top])

                    # Integrated transverse shear stress at ply boundaries
                    # tau_xz = - integral (Qbar_11 * d_eps_x/dx + Qbar_12 * d_eps_y/dx + Qbar_16 * d_gamma_xy/dx) dz
                    # d_eps(z)/dx = d_eps0/dx + z * d_kappa/dx
                    def calc_tau_xz(z_start, z_end, tau_start):
                        # integral_{z_start}^{z_end} (dstrains_dx_0 + z * d_kappa_dx) dz
                        # = [dstrains_dx_0 * z + 0.5 * d_kappa_dx * z^2]_{z_start}^{z_end}
                        term0 = dstrains_dx[:, 0:3] * (z_end - z_start)
                        term1 = 0.5 * dstrains_dx[:, 3:6] * (z_end**2 - z_start**2)
                        d_eps_int = term0 + term1
                        d_sig_x_int = (Qbar @ d_eps_int.T).T[:, 0]
                        return tau_start - d_sig_x_int

                    # Initialize tau at bottom of laminate (must be zero)
                    if i == 0:
                        tau_xz_bot = np.zeros(len(indices))
                    else:
                        tau_xz_bot = ply_stresses[-1]['tau_xz_top']

                    tau_xz_mid = calc_tau_xz(z_bot, z_mid, tau_xz_bot)
                    tau_xz_top = calc_tau_xz(z_bot, z_top, tau_xz_bot)

                    ply_stresses.append({
                        'index': i,
                        'name': ply.name,
                        'angle': angle,
                        'z_range': (z_bot, z_top),
                        'tau_xz_bot': tau_xz_bot,
                        'tau_xz_mid': tau_xz_mid,
                        'tau_xz_top': tau_xz_top,
                        'sigma_x': (sig_bot[:, 0] + sig_top[:, 0]) / 2.0,
                        'peak_sigma_x': np.maximum(np.abs(sig_bot[:, 0]), np.abs(sig_top[:, 0])),
                        'peak_sigma_1': np.maximum(np.abs(loc_bot[:, 0]), np.abs(loc_top[:, 0])),
                        'peak_sigma_2': np.maximum(np.abs(loc_bot[:, 1]), np.abs(loc_top[:, 1])),
                        'peak_tau_12': np.maximum(np.abs(loc_bot[:, 2]), np.abs(loc_top[:, 2])),
                        'peak_tau_xy': np.maximum(np.abs(sig_bot[:, 2]), np.abs(sig_top[:, 2])),
                        'peak_tau_xz': np.max(np.abs([tau_xz_bot, tau_xz_mid, tau_xz_top]), axis=0),
                        'peak_von_mises': np.maximum(np.abs(vm_bot), np.abs(vm_top)),
                        'peak_fi_max': np.maximum(fi_max_bot, fi_max_top),
                        'peak_fi_tsai_hill': np.maximum(fi_th_bot, fi_th_top),
                        'peak_fi_tsai_wu': np.maximum(fi_tw_bot, fi_tw_top),
                        'sigma_bot': sig_bot, 'sigma_mid': sig_mid, 'sigma_top': sig_top,
                        'local_bot': loc_bot, 'local_mid': loc_mid, 'local_top': loc_top,
                        'axial_sigma_x': sig_a,
                        'bending_sigma_x_bot': sig_b_bot,
                        'bending_sigma_x_top': sig_b_top
                    })
                    z_ply += ply.thickness

                # Map ply stresses back to Y, Z grid for standard results
                # In our 1D beam model, Y is the thickness direction for laminates.
                for idx_in_sub, global_idx in enumerate(indices):
                    # Transverse beam shear (isotropic sections) fallback
                    # If this is isotropic, tau_s should match VQ/It
                    # Our integrated tau_xz should naturally converge to parabolic for isotropic

                    for ply_info in ply_stresses:
                        z_b, z_t = ply_info['z_range']
                        # mask_ply identifies points in the grid that belong to this ply
                        mask_ply = mask & (Y >= z_b) & (Y <= z_t)

                        # Store in global 3D matrices
                        sigma_a[global_idx, mask_ply] = ply_info['axial_sigma_x'][idx_in_sub]

                        # For bending, we use the top/bottom averaged bending component
                        # (Ideally we'd interpolate linearly through ply, but this is for plotting)
                        y_mid = (z_b + z_t) / 2.0
                        sig_b_avg = (ply_info['bending_sigma_x_bot'][idx_in_sub] + ply_info['bending_sigma_x_top'][idx_in_sub]) / 2.0
                        sigma_b[global_idx, mask_ply] = sig_b_avg

                        tau_s[global_idx, mask_ply] = ply_info['tau_xz_mid'][idx_in_sub]
                        sigma_vm[global_idx, mask_ply] = ply_info['peak_von_mises'][idx_in_sub]

                # Store ply-by-ply data in a separate attribute for querying
                if not hasattr(solver, 'laminate_results'):
                    solver.laminate_results = {}
                solver.laminate_results[eid] = {
                    'ply_data': ply_stresses,
                    'strains_mid': strains_mid,
                    'x_positions': x_positions[indices]
                }

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
                # Global stresses
                'sigma_x_bot': ply['sigma_bot'][x_station_idx, 0],
                'sigma_x_mid': ply['sigma_mid'][x_station_idx, 0],
                'sigma_x_top': ply['sigma_top'][x_station_idx, 0],
                'sigma_y_bot': ply['sigma_bot'][x_station_idx, 1],
                'sigma_y_top': ply['sigma_top'][x_station_idx, 1],
                'tau_xy_bot': ply['sigma_bot'][x_station_idx, 2],
                'tau_xy_top': ply['sigma_top'][x_station_idx, 2],
                # Local material stresses
                'sigma_1_bot': ply['local_bot'][x_station_idx, 0],
                'sigma_1_mid': ply['local_mid'][x_station_idx, 0],
                'sigma_1_top': ply['local_top'][x_station_idx, 0],
                'sigma_2_bot': ply['local_bot'][x_station_idx, 1],
                'sigma_2_mid': ply['local_mid'][x_station_idx, 1],
                'sigma_2_top': ply['local_top'][x_station_idx, 1],
                'tau_12_bot': ply['local_bot'][x_station_idx, 2],
                'tau_12_mid': ply['local_mid'][x_station_idx, 2],
                'tau_12_top': ply['local_top'][x_station_idx, 2],
                # Components
                'axial_x': ply['axial_sigma_x'][x_station_idx],
                'bending_x_bot': ply['bending_sigma_x_bot'][x_station_idx],
                'bending_x_top': ply['bending_sigma_x_top'][x_station_idx],
                # Failure indices
                'fi_max': ply['peak_fi_max'][x_station_idx],
                'fi_tsai_hill': ply['peak_fi_tsai_hill'][x_station_idx],
                'fi_tsai_wu': ply['peak_fi_tsai_wu'][x_station_idx],
                'tau_xz_bot': ply['tau_xz_bot'][x_station_idx],
                'tau_xz_mid': ply['tau_xz_mid'][x_station_idx],
                'tau_xz_top': ply['tau_xz_top'][x_station_idx],
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
