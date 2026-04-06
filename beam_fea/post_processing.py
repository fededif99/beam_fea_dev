"""
post_processing.py
==================
Internal force and stress recovery engines.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict, Optional

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
        Now supports path-length evaluation for angled beams.
        
        Returns:
        --------
        plan : dict
            {positions, points_by_element, path_lengths}
        """
        mesh = solver.mesh
        coords = mesh.nodes
        elements = mesh.elements
        
        # Calculate cumulative path length along the beam using vectorized operations
        n1_coords = coords[elements[:, 0]]
        n2_coords = coords[elements[:, 1]]
        element_lengths = np.linalg.norm(n2_coords - n1_coords, axis=1)
        
        path_coords = np.zeros(mesh.num_elements + 1)
        path_coords[1:] = np.cumsum(element_lengths)
        
        total_length = path_coords[-1]
        
        if num_points is None:
            # Use node locations
            path_positions = path_coords
        else:
            path_positions = np.linspace(0, total_length, num_points)
            
        # Map path positions back to element IDs and local coordinates
        points_by_element = {}
        # Keep 3D for backward compatibility in results, but build it safely
        positions_xyz = np.zeros((len(path_positions), 3))
        
        for eid in range(mesh.num_elements):
            s_start = path_coords[eid]
            s_end = path_coords[eid+1]
            L = element_lengths[eid]
            
            # Find points that fall within this segment
            if eid == mesh.num_elements - 1:
                mask = (path_positions >= s_start) & (path_positions <= s_end)
            else:
                mask = (path_positions >= s_start) & (path_positions < s_end)
                
            indices = np.where(mask)[0]
            if len(indices) > 0:
                points_by_element[eid] = indices.tolist()
                
                # Interpolate global coordinates for these points
                p1 = coords[elements[eid, 0]]
                p2 = coords[elements[eid, 1]]
                xi = (path_positions[indices] - s_start) / L if L > 0 else np.zeros(len(indices))
                
                # Vectorized coordinate interpolation
                pos_2d = p1[None, :] + xi[:, None] * (p2 - p1)[None, :]
                positions_xyz[indices, :2] = pos_2d
                
        return {
            'positions': positions_xyz[:, 0], # X for backward compatibility
            'positions_xyz': positions_xyz,
            'path_positions': path_positions,
            'points_by_element': points_by_element
        }


class InternalForceEngine:
    """Engine to coordinate internal force calculation across the mesh."""

    @staticmethod
    def calculate(solver, num_points: int = None, strategy_name: str = "consistent") -> dict:
        if solver.displacements is None:
            raise ValueError("Must run solve_static() first")

        eval_plan = StationEvaluator.get_evaluation_plan(solver, num_points)
        positions = eval_plan['positions']
        path_positions = eval_plan['path_positions']
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

        from .element_matrices import UnifiedBeamElement, get_rotation_matrix
        coords = solver.mesh.nodes
        elements = solver.mesh.elements
        is_euler = (solver.element_type == 'euler')

        for eid, indices in points_by_element.items():
            node1, node2 = elements[eid]
            p1, p2 = coords[node1], coords[node2]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            x1, L = p1[0], np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            T = get_rotation_matrix(angle)

            dof_indices = [3*node1, 3*node1 + 1, 3*node1 + 2,
                           3*node2, 3*node2 + 1, 3*node2 + 2]
            # Transform global displacements to local: u_local = T @ u_global
            u_local = T @ solver.displacements[dof_indices]

            # Use path coordinates for xi instead of purely X to support angled beams
            s_start = eval_plan['path_positions'][indices[0]] # Approximate start of stations
            # Actually better to use the mapped global coords from xyz
            xi = np.clip((path_positions[indices] - path_positions[indices[0]] + (coords[node1, 0] - coords[node1, 0])) / L, 0, 1)
            # Re-derive xi correctly from xyz positions
            stations_xyz = eval_plan['positions_xyz'][indices]
            # dist from node 1
            dist_from_n1 = np.sqrt(np.sum((stations_xyz[:, :2] - coords[node1])**2, axis=1))
            xi = np.clip(dist_from_n1 / L, 0, 1) if L > 0 else np.zeros(len(indices))

            # Use Pre-calculated Properties
            element_expert = UnifiedBeamElement(
                EA=solver.properties.EA[eid], 
                ES=solver.properties.ES[eid], 
                EI=solver.properties.EI[eid],
                L=L, 
                rho_total=solver.properties.rho_lin[eid], 
                GA_s=solver.properties.GA_s[eid],
                force_euler=is_euler
            )

            d_load = element_dist_loads.get(eid, (0.0, 0.0, 0.0, 0.0))
            N, V, M = strategy.recover(element_expert, u_local, xi, dist_load=d_load)

            axial_forces[indices] = N
            shear_forces[indices] = V
            bending_moments[indices] = M

        return {
            'positions': positions,
            'path_positions': path_positions,
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

        from .loads import DistributedLoad
        coords = solver.mesh.nodes
        elements = solver.mesh.elements

        for load in solver.last_load_case.loads:
            if not isinstance(load, DistributedLoad):
                continue
                
            wy1 = wy2 = wx1 = wx2 = 0.0
            dist = load.distribution.lower()
            if dist == 'uniform':
                wy1 = wy2 = float(load.wy)
                wx1 = wx2 = float(load.wx)
            elif dist == 'linear':
                wy1, wy2 = float(load.wy_start), float(load.wy_end)
                wx1, wx2 = float(load.wx_start), float(load.wx_end)
            elif dist == 'triangular':
                if str(load.peak_loc).lower() == 'start':
                    wy1, wy2 = float(load.w_peak), 0.0
                else:
                    wy1, wy2 = 0.0, float(load.w_peak)
            else:
                continue

            if load.element is not None:
                elems = [load.element] if isinstance(load.element, int) else load.element
                for eid in elems:
                    curr = element_dist_loads.get(eid, (0, 0, 0, 0))
                    element_dist_loads[eid] = (curr[0] + wy1, curr[1] + wy2, curr[2] + wx1, curr[3] + wx2)
            elif load.x_start is not None:
                x_s, x_e = min(load.x_start, load.x_end), max(load.x_start, load.x_end)
                for eid in range(solver.mesh.num_elements):
                    node1, node2 = elements[eid]
                    p1, p2 = coords[node1, 0], coords[node2, 0]
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
        x_positions = eval_plan['path_positions'] # Use path length for X in stresses too
        shape_3d = (eval_x_count, num_y_points, num_z_points)

        sigma_a, sigma_b, tau_s, sigma_vm = [np.zeros(shape_3d) for _ in range(4)]
        sigma_1, sigma_2 = np.zeros(shape_3d), np.zeros(shape_3d)

        profile_cache = {}

        from .element_matrices import get_rotation_matrix
        coords = solver.mesh.nodes
        elements = solver.mesh.elements

        for eid, indices in points_by_element.items():
            node1, node2 = elements[eid]
            p1, p2 = coords[node1], coords[node2]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            angle = np.arctan2(dy, dx)
            T = get_rotation_matrix(angle)

            mat = solver.properties.get_material(eid)
            sec = solver.properties.get_section(eid)
            sec_id = id(sec)

            if sec_id not in profile_cache:
                profile_cache[sec_id] = StressEngine._get_section_profile(sec, Y, Z)

            mask, t_yz, Q_yz = profile_cache[sec_id]

            # --- Rotate Load Vectors for Angled Beams ---
            # Midplane resultants (N, V, M) are in global coords.
            # We need them in local element axes for CLT stress recovery.
            # Local Force Vector [n_local; m_local] = T @ [n_global; m_global]
            # Actually, standard beam internal force vectors are already local in our strategy.
            # But the 'midplane resultants' logic here builds vectors from N_x, V_x, M_x.
            # Are N_x, V_x local or global?
            # In InternalForceEngine, they are LOCAL (axial, shear, moment).
            # So no extra rotation needed here if N_x, V_x are already local.
            # Treat Isotropic as a 1-ply laminate for code unification
            from .composites import Laminate, Ply
            if not hasattr(mat, 'plies'):
                # Convert isotropic Material to a single-ply Laminate
                p = Ply(name=mat.name, E1=mat.E, E2=mat.E, nu12=mat.nu, G12=mat.G,
                        G13=mat.G, G23=mat.G, thickness=sec.y_top - sec.y_bottom,
                        rho=mat.rho, Xt=mat.yield_strength or 0.0,
                        Xc=mat.yield_strength or 0.0, Yt=mat.yield_strength or 0.0,
                        Yc=mat.yield_strength or 0.0, S=mat.yield_strength / 1.732 if mat.yield_strength else 0.0)
                lam = Laminate(name=mat.name, stack=[(p, 0.0)])
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

                    # Calculate Transverse Shear first so it can be evaluated
                    def calc_tau_xz(z_start, z_end, tau_start):
                        term0 = dstrains_dx[:, 0:3] * (z_end - z_start)
                        term1 = 0.5 * dstrains_dx[:, 3:6] * (z_end**2 - z_start**2)
                        d_eps_int = term0 + term1
                        d_sig_x_int = (Qbar @ d_eps_int.T).T[:, 0]
                        return tau_start - d_sig_x_int

                    if i == 0:
                        tau_xz_bot = np.zeros(len(indices))
                    else:
                        tau_xz_bot = ply_stresses[-1]['tau_xz_top']

                    tau_xz_mid = calc_tau_xz(z_bot, z_mid, tau_xz_bot)
                    tau_xz_top = calc_tau_xz(z_bot, z_top, tau_xz_bot)

                    # Transform to local material coordinates (1, 2, 12, 13, 23)
                    def to_local(sig, txz, ang):
                        theta = np.radians(ang)
                        c, s = np.cos(theta), np.sin(theta)
                        s1 = sig[:, 0]*c**2 + sig[:, 1]*s**2 + 2*sig[:, 2]*s*c
                        s2 = sig[:, 0]*s**2 + sig[:, 1]*c**2 - 2*sig[:, 2]*s*c
                        s12 = (sig[:, 1] - sig[:, 0])*s*c + sig[:, 2]*(c**2 - s**2)
                        t13 = txz * c
                        t23 = -txz * s
                        return np.column_stack([s1, s2, s12, t13, t23])

                    loc_bot = to_local(sig_bot, tau_xz_bot, angle)
                    loc_mid = to_local(sig_mid, tau_xz_mid, angle)
                    loc_top = to_local(sig_top, tau_xz_top, angle)

                    # Principal and Von Mises Calculation for each station
                    def calc_vm_principal(sig, txz):
                        avg = (sig[:, 0] + sig[:, 1]) / 2.0
                        r = np.sqrt(((sig[:, 0] - sig[:, 1]) / 2.0)**2 + sig[:, 2]**2)
                        s1p = avg + r
                        s2p = avg - r
                        vm = np.sqrt(sig[:, 0]**2 + sig[:, 1]**2 - sig[:, 0]*sig[:, 1] + 3*(sig[:, 2]**2 + txz**2))
                        return vm, s1p, s2p

                    vm_bot, s1p_bot, s2p_bot = calc_vm_principal(sig_bot, tau_xz_bot)
                    vm_mid, s1p_mid, s2p_mid = calc_vm_principal(sig_mid, tau_xz_mid)
                    vm_top, s1p_top, s2p_top = calc_vm_principal(sig_top, tau_xz_top)

                    # Minimum Safety Factors
                    from .failure_criteria import MaximumStressCriterion, TsaiHillCriterion, TsaiWuCriterion
                    
                    try:
                        c_max = MaximumStressCriterion(Xt=ply.Xt, Xc=ply.Xc, Yt=ply.Yt, Yc=ply.Yc, S=ply.S, S13=ply.S13, S23=ply.S23)
                        sf_max_bot = c_max.evaluate(sigma_1=loc_bot[:,0], sigma_2=loc_bot[:,1], tau_12=loc_bot[:,2], tau_13=loc_bot[:,3], tau_23=loc_bot[:,4])['SF']
                        sf_max_top = c_max.evaluate(sigma_1=loc_top[:,0], sigma_2=loc_top[:,1], tau_12=loc_top[:,2], tau_13=loc_top[:,3], tau_23=loc_top[:,4])['SF']
                        
                        c_th = TsaiHillCriterion(Xt=ply.Xt, Xc=ply.Xc, Yt=ply.Yt, Yc=ply.Yc, S=ply.S, S13=ply.S13, S23=ply.S23)
                        sf_th_bot = c_th.evaluate(sigma_1=loc_bot[:,0], sigma_2=loc_bot[:,1], tau_12=loc_bot[:,2], tau_13=loc_bot[:,3], tau_23=loc_bot[:,4])['SF']
                        sf_th_top = c_th.evaluate(sigma_1=loc_top[:,0], sigma_2=loc_top[:,1], tau_12=loc_top[:,2], tau_13=loc_top[:,3], tau_23=loc_top[:,4])['SF']
                        
                        c_tw = TsaiWuCriterion(Xt=ply.Xt, Xc=ply.Xc, Yt=ply.Yt, Yc=ply.Yc, S=ply.S, S13=ply.S13, S23=ply.S23)
                        sf_tw_bot = c_tw.evaluate(sigma_1=loc_bot[:,0], sigma_2=loc_bot[:,1], tau_12=loc_bot[:,2], tau_13=loc_bot[:,3], tau_23=loc_bot[:,4])['SF']
                        sf_tw_top = c_tw.evaluate(sigma_1=loc_top[:,0], sigma_2=loc_top[:,1], tau_12=loc_top[:,2], tau_13=loc_top[:,3], tau_23=loc_top[:,4])['SF']
                    except ValueError:
                        # If ply has 0s for strengths
                        n_stat = len(indices)
                        sf_max_bot = sf_max_top = sf_th_bot = sf_th_top = sf_tw_bot = sf_tw_top = np.full(n_stat, np.inf)

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
                        'min_sf_max': np.minimum(sf_max_bot, sf_max_top),
                        'min_sf_tsai_hill': np.minimum(sf_th_bot, sf_th_top),
                        'min_sf_tsai_wu': np.minimum(sf_tw_bot, sf_tw_top),
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
                # Failure SFs (Note: using SFs now, lower is more critical)
                'sf_max': ply['min_sf_max'][x_station_idx],
                'sf_tsai_hill': ply['min_sf_tsai_hill'][x_station_idx],
                'sf_tsai_wu': ply['min_sf_tsai_wu'][x_station_idx],
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


class ResultsEngine:
    """
    Standardized engine for results extraction, peak finding, and Pandas export.
    Unifies data handling for Static, Modal, and Batch analysis.
    """
    
    @staticmethod
    def get_nodal_displacements(solver) -> pd.DataFrame:
        if solver.displacements is None:
            return pd.DataFrame()
            
        u = solver.displacements[0::3]
        v = solver.displacements[1::3]
        
        return pd.DataFrame({
            'node_id': np.arange(solver.mesh.num_nodes),
            'x': solver.mesh.nodes[:, 0],
            'y': solver.mesh.nodes[:, 1],
            'u': u,
            'v': v,
            'res': np.sqrt(u**2 + v**2)
        })

    @staticmethod
    def get_peak_summary(solver) -> Dict:
        """Standardized peak result extractor for a single load case."""
        # 1. Deflections
        disp_df = ResultsEngine.get_nodal_displacements(solver)
        max_res = disp_df.iloc[disp_df['res'].idxmax()] if not disp_df.empty else None
        
        # 2. Internal Forces
        forces = InternalForceEngine.calculate(solver)
        v_idx = np.argmax(np.abs(forces['shear_forces']))
        m_idx = np.argmax(np.abs(forces['bending_moments']))
        
        # 3. Peak Stresses & FOS
        stresses = StressEngine.calculate(solver)
        vm_flat = stresses['von_mises'].flatten()
        max_vm_idx_flat = np.argmax(vm_flat)
        vm_peak = vm_flat[max_vm_idx_flat]
        
        # Find spatial coordinates of peak stress
        max_vm_idx = np.unravel_index(max_vm_idx_flat, stresses['von_mises'].shape)
        peak_x = stresses['x'][max_vm_idx[0]]
        peak_y = stresses['y'][max_vm_idx[1], max_vm_idx[2]]
        peak_z = stresses['z'][max_vm_idx[1], max_vm_idx[2]]

        # Use failure criteria module for proper FOS
        from .failure_criteria import VonMisesCriterion
        # Fallback to 250 as default yield if material is missing properties
        yield_s = getattr(solver.material, 'yield_strength', 250.0) or 250.0
        crit = VonMisesCriterion(yield_strength=yield_s)
        # Evaluate peak stress
        fos_result = crit.evaluate(sigma_x=vm_peak)
        
        result = {
            'max_deflection': max_res['res'] if max_res is not None else 0.0,
            'max_deflection_x': max_res['x'] if max_res is not None else 0.0,
            'max_deflection_node': int(max_res['node_id']) if max_res is not None else -1,
            'max_shear': forces['shear_forces'][v_idx],
            'max_shear_x': forces['path_positions'][v_idx],
            'max_moment': forces['bending_moments'][m_idx],
            'max_moment_x': forces['path_positions'][m_idx],
            'max_von_mises': vm_peak,
            'max_vm_x': peak_x,
            'max_vm_y': peak_y,
            'max_vm_z': peak_z,
            'fos': fos_result['SF'],
            'mos': fos_result['MoS']
        }
        
        # Add laminate info if available
        if hasattr(solver, 'laminate_results'):
            # Find element and ply for peak VM
            for eid, data in solver.laminate_results.items():
                for ply in data['ply_data']:
                    z_b, z_t = ply['z_range']
                    if z_b - 1e-6 <= peak_y <= z_t + 1e-6:
                        # Check if this station matches peak_x (using path coord)
                        if any(np.isclose(data['x_positions'], peak_x, rtol=1e-5)):
                             result['peak_ply'] = ply['name']
                             break
        
        return result

    @staticmethod
    def create_batch_summary(solver, load_cases: List, peak_results: List[Dict]) -> pd.DataFrame:
        """Create a consolidated Pandas DataFrame for batch results."""
        rows = []
        for lc, res in zip(load_cases, peak_results):
            row = {'case_name': lc.name}
            row.update(res)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def export_results(df: pd.DataFrame, filepath: str):
        """Export results to CSV."""
        df.to_csv(filepath, index=False)
        return filepath
