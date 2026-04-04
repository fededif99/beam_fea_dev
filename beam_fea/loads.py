"""
loads.py
========
Load definitions and application for finite element analysis.

Supports:
- Point loads (forces)
- Concentrated moments
- Distributed loads (uniform, linear, triangular, custom)
- Load cases and combinations
"""

import numpy as np
import warnings
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Load(ABC):
    """Abstract base class for loads."""
    
    @abstractmethod
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """
        Apply load to global force vector.
        
        Parameters:
        -----------
        F : np.ndarray
            Global force vector
        mesh : Mesh
            Finite element mesh
            
        Returns:
        --------
        F : np.ndarray
            Modified force vector
        """
        pass

    def apply_to_mass_matrix(self, M, mesh):
        """
        Apply load properties that affect the global mass matrix.
        
        Parameters:
        -----------
        M : scipy.sparse matrix
            Global mass matrix
        mesh : Mesh
            Finite element mesh
            
        Returns:
        --------
        M : modified mass matrix
        """
        return M


@dataclass
class PointLoad(Load):
    """
    Point load (force) at a specific node or global x-coordinate.
    
    For concentrated moments, use the ConcentratedMoment class instead.
    
    Attributes:
    -----------
    node : int, optional
        Node ID where load is applied
    x : float, optional
        Global x-coordinate (mm). Used if node is None.
    fx : float or str
        Force in x-direction (N)
    fy : float or str
        Force in y-direction (N)
    """
    
    node: Optional[int] = None
    x: Optional[float] = None
    fx: Union[float, str] = 0.0
    fy: Union[float, str] = 0.0
    
    def __post_init__(self):
        if self.node is not None and self.x is not None:
            warnings.warn("Both 'node' and 'x' specified for PointLoad. 'node' will take precedence.", UserWarning, stacklevel=2)
        if self.node is None and self.x is None:
            raise ValueError("Must specify either 'node' or 'x' for PointLoad.")

    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply point load to global force vector."""
        if self.node is not None:
            F[3 * self.node] += self.fx
            F[3 * self.node + 1] += self.fy
        elif self.x is not None:
            elem_idx = mesh.find_element_at_x(self.x)
            if elem_idx == -1:
                warnings.warn(f"PointLoad at x={self.x} is outside mesh range. Ignored.", UserWarning, stacklevel=2)
                return F
            
            coords = mesh.nodes
            node1, node2 = mesh.elements[elem_idx]
            p1, p2 = coords[node1], coords[node2]
            
            dx = p2 - p1
            length_sq = float(np.dot(dx, dx))
            L = np.sqrt(length_sq)
            if length_sq > 0:
                vec = np.zeros_like(dx)
                vec[0] = self.x - p1[0]
                t = np.dot(vec, dx) / length_sq
                xi = np.clip(t, 0.0, 1.0)
            else:
                xi = 0.0
            
            # Axial (linear shape functions)
            F[3 * node1] += self.fx * (1 - xi)
            F[3 * node2] += self.fx * xi
            
            # Transverse (Hermite shape functions)
            N = np.array([
                1 - 3*xi**2 + 2*xi**3,
                L * (xi - 2*xi**2 + xi**3),
                3*xi**2 - 2*xi**3,
                L * (-xi**2 + xi**3)
            ])
            F[3 * node1 + 1] += self.fy * N[0]
            F[3 * node1 + 2] += self.fy * N[1]
            F[3 * node2 + 1] += self.fy * N[2]
            F[3 * node2 + 2] += self.fy * N[3]
            
        return F
    
    def __str__(self):
        loc = f"node {self.node}" if self.node is not None else f"x={self.x:.1f}mm"
        components = []
        if abs(self.fx) > 1e-10:
            components.append(f"Fx={self.fx:.2f}N")
        if abs(self.fy) > 1e-10:
            components.append(f"Fy={self.fy:.2f}N")
        return f"Point load at {loc}: {', '.join(components)}"


@dataclass
class ConcentratedMoment(Load):
    """
    Concentrated moment at a specific node or global x-coordinate.
    
    Applied via work-equivalent nodal forces using Hermite shape function
    derivatives when specified by coordinate.
    
    Attributes:
    -----------
    node : int, optional
        Node ID where moment is applied
    x : float, optional
        Global x-coordinate (mm). Used if node is None.
    mz : float or str
        Moment about z-axis (N·mm), positive counter-clockwise
    """
    
    node: Optional[int] = None
    x: Optional[float] = None
    mz: Union[float, str] = 0.0
    
    def __post_init__(self):
        if self.node is not None and self.x is not None:
            warnings.warn("Both 'node' and 'x' specified for ConcentratedMoment. 'node' will take precedence.", UserWarning, stacklevel=2)
        if self.node is None and self.x is None:
            raise ValueError("Must specify either 'node' or 'x' for ConcentratedMoment.")

    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply concentrated moment to global force vector."""
        if self.node is not None:
            F[3 * self.node + 2] += self.mz
        elif self.x is not None:
            elem_idx = mesh.find_element_at_x(self.x)
            if elem_idx == -1:
                warnings.warn(f"ConcentratedMoment at x={self.x} is outside mesh range. Ignored.", UserWarning, stacklevel=2)
                return F
            
            coords = mesh.nodes
            node1, node2 = mesh.elements[elem_idx]
            p1, p2 = coords[node1], coords[node2]
            
            dx = p2 - p1
            length_sq = float(np.dot(dx, dx))
            L = np.sqrt(length_sq)
            if length_sq > 0:
                vec = np.zeros_like(dx)
                vec[0] = self.x - p1[0]
                t = np.dot(vec, dx) / length_sq
                xi = np.clip(t, 0.0, 1.0)
            else:
                xi = 0.0
            
            # Shape function derivatives: dNi/dx = dNi/dxi * (1/L)
            dN = np.array([
                (-6*xi + 6*xi**2) / L,
                1 - 4*xi + 3*xi**2,
                (6*xi - 6*xi**2) / L,
                -2*xi + 3*xi**2
            ])
            F[3 * node1 + 1] += self.mz * dN[0]
            F[3 * node1 + 2] += self.mz * dN[1]
            F[3 * node2 + 1] += self.mz * dN[2]
            F[3 * node2 + 2] += self.mz * dN[3]
            
        return F
    
    def __str__(self):
        loc = f"node {self.node}" if self.node is not None else f"x={self.x:.1f}mm"
        return f"Moment at {loc}: Mz={self.mz:.2f}N·mm"


@dataclass
class DistributedLoad(Load):
    """
    Unified distributed load along a coordinate range.
    
    All distributions are internally resolved to piecewise-linear waypoint
    profiles and integrated using vectorized Hermite shape functions.
    
    Attributes:
    -----------
    x_start : float
        Start of load range (mm)
    x_end : float
        End of load range (mm)
    distribution : str
        'uniform', 'linear', 'triangular', or 'custom'
    wy, wx : float
        Intensities for 'uniform' distribution (N/mm)
    wy_start, wy_end : float
        Start/end intensities for 'linear' distribution (N/mm)
    wx_start, wx_end : float
        Start/end axial intensities for 'linear' distribution (N/mm)
    w_peak : float
        Peak intensity for 'triangular' distribution (N/mm)
    peak_loc : str or float
        'start', 'end', or float (x-coordinate) for isosceles triangle
    load_fn : callable
        f(x) -> wy (scalar) or (wx, wy) for 'custom' distribution
    n_points : int
        Gauss-Legendre quadrature order for 'custom' (default 10)
    """
    
    x_start: float = 0.0
    x_end: float = 0.0
    distribution: str = 'uniform'
    
    # Uniform
    wy: Union[float, str] = 0.0
    wx: Union[float, str] = 0.0
    
    # Linear
    wy_start: Union[float, str] = 0.0
    wy_end: Union[float, str] = 0.0
    wx_start: Union[float, str] = 0.0
    wx_end: Union[float, str] = 0.0
    
    # Triangular
    w_peak: Union[float, str] = 0.0
    peak_loc: Union[str, float] = 'end'
    
    # Custom
    load_fn: Optional[Callable] = field(default=None, repr=False)
    n_points: int = 10
    
    # Element-based (legacy compat)
    element: Optional[Union[int, List[int]]] = None

    def __post_init__(self):
        if self.distribution not in ['uniform', 'linear', 'triangular', 'custom']:
            raise ValueError(f"Unknown distribution: '{self.distribution}'.")
        if self.element is None and self.x_end < self.x_start:
            raise ValueError("x_end must be >= x_start for DistributedLoad.")

    def _build_waypoints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build piecewise-linear waypoint arrays (x, wy, wx).
        
        Returns sorted arrays ready for interpolation across elements.
        """
        dist = self.distribution.lower()
        
        if dist == 'uniform':
            xs = np.array([self.x_start, self.x_end])
            wys = np.array([self.wy, self.wy], dtype=float)
            wxs = np.array([self.wx, self.wx], dtype=float)
            
        elif dist == 'linear':
            xs = np.array([self.x_start, self.x_end])
            wys = np.array([self.wy_start, self.wy_end], dtype=float)
            wxs = np.array([self.wx_start, self.wx_end], dtype=float)
            
        elif dist == 'triangular':
            if isinstance(self.peak_loc, str):
                if self.peak_loc.lower() == 'start':
                    xs = np.array([self.x_start, self.x_end])
                    wys = np.array([self.w_peak, 0.0], dtype=float)
                else:  # 'end'
                    xs = np.array([self.x_start, self.x_end])
                    wys = np.array([0.0, self.w_peak], dtype=float)
            else:
                # Isosceles: peak at a specific x-coordinate
                x_peak = float(self.peak_loc)
                x_peak = np.clip(x_peak, self.x_start, self.x_end)
                xs = np.array([self.x_start, x_peak, self.x_end])
                wys = np.array([0.0, self.w_peak, 0.0], dtype=float)
            wxs = np.zeros_like(wys)
            
        else:
            raise ValueError(f"Unknown distribution: '{dist}'. "
                             f"Use 'uniform', 'linear', 'triangular', or 'custom'.")
        
        # Sort by x
        order = np.argsort(xs)
        return xs[order], wys[order], wxs[order]

    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply distributed load to global force vector."""
        if self.distribution.lower() == 'custom':
            return self._apply_custom(F, mesh)
        
        # Handle legacy element-based specification
        if self.element is not None:
            return self._apply_element_based(F, mesh)
        
        # Waypoint-based vectorized path
        wp_x, wp_wy, wp_wx = self._build_waypoints()
        
        coords = mesh.nodes
        elements = mesh.elements
        num_elem = mesh.num_elements
        
        # Vectorized element geometry
        p1_all = coords[elements[:, 0]]
        p2_all = coords[elements[:, 1]]
        e_x1 = p1_all[:, 0]
        e_x2 = p2_all[:, 0]
        e_min = np.minimum(e_x1, e_x2)
        e_max = np.maximum(e_x1, e_x2)
        L_all = np.linalg.norm(p2_all - p1_all, axis=1)
        
        x_min_load = wp_x[0]
        x_max_load = wp_x[-1]
        
        # Process each waypoint segment
        for seg in range(len(wp_x) - 1):
            seg_x0, seg_x1 = wp_x[seg], wp_x[seg + 1]
            seg_wy0, seg_wy1 = wp_wy[seg], wp_wy[seg + 1]
            seg_wx0, seg_wx1 = wp_wx[seg], wp_wx[seg + 1]
            seg_len = seg_x1 - seg_x0
            if seg_len <= 0:
                continue
            
            # Vectorized overlap detection
            inter_min = np.maximum(e_min, seg_x0)
            inter_max = np.minimum(e_max, seg_x1)
            mask = inter_max > inter_min
            
            if not np.any(mask):
                continue
            
            # Filtered arrays
            idx = np.where(mask)[0]
            n1 = elements[idx, 0]
            n2 = elements[idx, 1]
            L = L_all[idx]
            imin = inter_min[idx]
            imax = inter_max[idx]
            emin = e_min[idx]
            
            # Local coordinates
            xi1 = (imin - emin) / L
            xi2 = (imax - emin) / L
            
            # Interpolated intensities at intersection boundaries
            t1 = (imin - seg_x0) / seg_len
            t2 = (imax - seg_x0) / seg_len
            wya = seg_wy0 + (seg_wy1 - seg_wy0) * t1
            wyb = seg_wy0 + (seg_wy1 - seg_wy0) * t2
            wxa = seg_wx0 + (seg_wx1 - seg_wx0) * t1
            wxb = seg_wx0 + (seg_wx1 - seg_wx0) * t2
            
            # Apply trapezoidal integration
            self._apply_linear_vectorized(F, n1, n2, L, xi1, xi2, wya, wyb, wxa, wxb)
        
        return F

    def _apply_element_based(self, F: np.ndarray, mesh) -> np.ndarray:
        """Legacy element-based application for backward compatibility."""
        coords = mesh.nodes
        elements = mesh.elements
        elem_list = [self.element] if isinstance(self.element, int) else self.element
        
        for eid in elem_list:
            node1, node2 = elements[eid]
            L = np.linalg.norm(coords[node2] - coords[node1])
            n1 = np.array([node1])
            n2 = np.array([node2])
            L_arr = np.array([L])
            xi1 = np.array([0.0])
            xi2 = np.array([1.0])
            
            dist = self.distribution.lower()
            if dist == 'uniform':
                wya = np.array([self.wy], dtype=float)
                wyb = np.array([self.wy], dtype=float)
                wxa = np.array([self.wx], dtype=float)
                wxb = np.array([self.wx], dtype=float)
            elif dist == 'linear':
                wya = np.array([self.wy_start], dtype=float)
                wyb = np.array([self.wy_end], dtype=float)
                wxa = np.array([self.wx_start], dtype=float)
                wxb = np.array([self.wx_end], dtype=float)
            elif dist == 'triangular':
                if isinstance(self.peak_loc, str) and self.peak_loc.lower() == 'start':
                    wya = np.array([self.w_peak], dtype=float)
                    wyb = np.array([0.0])
                else:
                    wya = np.array([0.0])
                    wyb = np.array([self.w_peak], dtype=float)
                wxa = np.array([0.0])
                wxb = np.array([0.0])
            else:
                continue
            
            self._apply_linear_vectorized(F, n1, n2, L_arr, xi1, xi2, wya, wyb, wxa, wxb)
        
        return F

    @staticmethod
    def _apply_linear_vectorized(F, n1, n2, L, xi1, xi2, wya, wyb, wxa, wxb):
        """
        Vectorized work-equivalent nodal forces for linearly varying load.
        
        All inputs are 1-D arrays of length N (number of elements to process).
        """
        # Guard against zero-length segments
        valid = np.abs(xi2 - xi1) > 1e-12
        if not np.any(valid):
            return
        
        n1, n2 = n1[valid], n2[valid]
        L, xi1, xi2 = L[valid], xi1[valid], xi2[valid]
        wya, wyb, wxa, wxb = wya[valid], wyb[valid], wxa[valid], wxb[valid]
        
        # w(xi) = base + slope * xi
        wy_slope = (wyb - wya) / (xi2 - xi1)
        wy_base = wya - wy_slope * xi1
        wx_slope = (wxb - wxa) / (xi2 - xi1)
        wx_base = wxa - wx_slope * xi1
        
        # Vectorized shape function integrals
        # I1[i] = integral(Ni, xi1..xi2), I2[i] = integral(xi*Ni, xi1..xi2)
        def hermite_integrals(xi):
            """Antiderivatives of Hermite shape functions and xi*Hermite."""
            xi2_ = xi**2
            xi3_ = xi**3
            xi4_ = xi**4
            xi5_ = xi**5
            
            I1 = np.array([
                xi - xi3_ + 0.5*xi4_,
                L * (0.5*xi2_ - (2/3)*xi3_ + 0.25*xi4_),
                xi3_ - 0.5*xi4_,
                L * (-(1/3)*xi3_ + 0.25*xi4_)
            ])
            I2 = np.array([
                0.5*xi2_ - 0.75*xi4_ + 0.4*xi5_,
                L * ((1/3)*xi3_ - 0.5*xi4_ + 0.2*xi5_),
                0.75*xi4_ - 0.4*xi5_,
                L * (-0.25*xi4_ + 0.2*xi5_)
            ])
            # Axial (linear shape functions)
            Ia = np.array([
                xi - 0.5*xi2_,
                0.5*xi2_ - (1/3)*xi3_,
                0.5*xi2_,
                (1/3)*xi3_
            ])
            return I1, I2, Ia
        
        I1_b, I2_b, Ia_b = hermite_integrals(xi2)
        I1_a, I2_a, Ia_a = hermite_integrals(xi1)
        dI1 = I1_b - I1_a  # shape (4, N)
        dI2 = I2_b - I2_a
        dIa = Ia_b - Ia_a
        
        # Transverse contributions: L * (wy_base * dI1 + wy_slope * dI2)
        fy_v1  = L * (wy_base * dI1[0] + wy_slope * dI2[0])
        fy_th1 = L * (wy_base * dI1[1] + wy_slope * dI2[1])
        fy_v2  = L * (wy_base * dI1[2] + wy_slope * dI2[2])
        fy_th2 = L * (wy_base * dI1[3] + wy_slope * dI2[3])
        
        # Axial contributions: L * (wx_base * dIa[0:2] + wx_slope * dIa[1:3])
        fx_1 = L * (wx_base * dIa[0] + wx_slope * dIa[1])
        fx_2 = L * (wx_base * dIa[2] + wx_slope * dIa[3])
        
        # Scatter into global force vector
        np.add.at(F, 3 * n1,     fx_1)
        np.add.at(F, 3 * n1 + 1, fy_v1)
        np.add.at(F, 3 * n1 + 2, fy_th1)
        np.add.at(F, 3 * n2,     fx_2)
        np.add.at(F, 3 * n2 + 1, fy_v2)
        np.add.at(F, 3 * n2 + 2, fy_th2)

    def _apply_custom(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply custom load function using Gauss-Legendre quadrature per element."""
        from numpy.polynomial.legendre import leggauss
        
        coords = mesh.nodes
        elements = mesh.elements
        gp, gw = leggauss(self.n_points)
        
        x_lo = min(self.x_start, self.x_end)
        x_hi = max(self.x_start, self.x_end)
        
        for eid in range(mesh.num_elements):
            node1, node2 = elements[eid]
            x1, x2 = coords[node1, 0], coords[node2, 0]
            e_min, e_max = min(x1, x2), max(x1, x2)
            
            inter_min = max(e_min, x_lo)
            inter_max = min(e_max, x_hi)
            if inter_max <= inter_min:
                continue
            
            L = np.linalg.norm(coords[node2] - coords[node1])
            
            # Map Gauss points from [-1, 1] to [inter_min, inter_max]
            mid = 0.5 * (inter_min + inter_max)
            half = 0.5 * (inter_max - inter_min)
            x_gauss = mid + half * gp
            
            for i, (xg, w) in enumerate(zip(x_gauss, gw)):
                result = self.load_fn(xg)
                if isinstance(result, (tuple, list, np.ndarray)):
                    wxg, wyg = float(result[0]), float(result[1])
                else:
                    wxg, wyg = 0.0, float(result)
                
                # Local coordinate
                xi = (xg - e_min) / L
                xi = np.clip(xi, 0, 1)
                
                # Hermite shape functions
                N = np.array([
                    1 - 3*xi**2 + 2*xi**3,
                    L * (xi - 2*xi**2 + xi**3),
                    3*xi**2 - 2*xi**3,
                    L * (-xi**2 + xi**3)
                ])
                
                # Jacobian for this Gauss point: half (from [-1,1] mapping)
                jac = half * w
                
                F[3 * node1 + 1] += wyg * N[0] * jac
                F[3 * node1 + 2] += wyg * N[1] * jac
                F[3 * node2 + 1] += wyg * N[2] * jac
                F[3 * node2 + 2] += wyg * N[3] * jac
                
                # Axial (linear shape functions)
                F[3 * node1] += wxg * (1 - xi) * jac
                F[3 * node2] += wxg * xi * jac
        
        return F

    def __str__(self):
        dist = self.distribution.lower()
        loc = f"x=[{self.x_start}, {self.x_end}]"
        if self.element is not None:
            loc = f"element {self.element}"
        
        if dist == 'uniform':
            return f"Uniform load on {loc}: wy={self.wy:.4f} N/mm"
        elif dist == 'linear':
            return f"Linear load on {loc}: wy={self.wy_start:.4f} to {self.wy_end:.4f} N/mm"
        elif dist == 'triangular':
            return f"Triangular load on {loc}: peak={self.w_peak:.4f} at {self.peak_loc}"
        elif dist == 'custom':
            return f"Custom load on {loc}: {self.n_points}-point quadrature"
        return f"Distributed load on {loc}"


@dataclass
class LumpedMass(Load):
    """
    Lumped (nodal) mass at a specific node or global x-coordinate.
    
    Affects natural frequencies in modal analysis.
    If `apply_gravity=True`, this mass also produces a downward force
    (-m * 9.80665 N) applied to the force vector during static analysis.
    
    Attributes:
    -----------
    node : int, optional
        Node ID where lumped mass is applied
    x : float, optional
        Global x-coordinate (mm). Used if node is None.
    m : float
        Mass (kg)
    Izz : float
        Rotational inertia (kg·mm²)
    apply_gravity : bool
        If True, applies a standard Earth gravity load automatically.
    """
    node: Optional[int] = None
    x: Optional[float] = None
    m: float = 0.0
    Izz: float = 0.0
    apply_gravity: bool = False
    
    def __post_init__(self):
        if self.node is not None and self.x is not None:
            warnings.warn("Both 'node' and 'x' specified for LumpedMass. 'node' will take precedence.", UserWarning, stacklevel=2)
        if self.node is None and self.x is None:
            raise ValueError("Must specify either 'node' or 'x' for LumpedMass.")

    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply gravity load if enabled."""
        if self.apply_gravity and self.m > 0:
            weight_N = -self.m * 9.80665
            # Delegate to PointLoad internal logic
            return PointLoad(node=self.node, x=self.x, fy=weight_N).apply_to_force_vector(F, mesh)
        return F

    def apply_to_mass_matrix(self, M, mesh):
        """Apply mass and inertia to global mass matrix."""
        from scipy.sparse import issparse
        
        is_sparse = issparse(M)
        if is_sparse:
            # LIL is efficient for changing sparsity pattern/direct indexing
            M = M.tolil()

        if self.node is not None:
            node_idx = self.node
        elif self.x is not None:
            # Look up closest node
            coords = mesh.nodes
            dist = np.abs(coords[:, 0] - self.x)
            node_idx = np.argmin(dist)
            if dist[node_idx] > 1e-6:
                warnings.warn(f"LumpedMass at x={self.x} moved to nearest node {node_idx} (x={coords[node_idx, 0]:.2f})", UserWarning, stacklevel=2)
        
        M[3 * node_idx, 3 * node_idx] += self.m
        M[3 * node_idx + 1, 3 * node_idx + 1] += self.m
        if self.Izz > 0:
            M[3 * node_idx + 2, 3 * node_idx + 2] += self.Izz
            
        if is_sparse:
            M = M.tocsr()
        return M

    def __str__(self):
        loc = f"node {self.node}" if self.node is not None else f"x={self.x:.1f}mm"
        gravity_str = " (with gravity)" if self.apply_gravity else ""
        return f"Lumped Mass at {loc}: m={self.m:.2f}kg, Izz={self.Izz:.2f}kg·mm²{gravity_str}"


class LoadCase:
    """Collection of loads representing a single load case."""
    
    def __init__(self, name: str = "Load Case 1"):
        """
        Initialize load case.
        
        Parameters:
        -----------
        name : str
            Name/identifier for this load case
        """
        self.name = name
        self.loads: List[Load] = []
    
    def add(self, load: Load):
        """Add a load object."""
        self.loads.append(load)

    def point_load(self, node: Optional[int] = None, x: Optional[float] = None, 
                   fx: Union[float, str] = 0, fy: Union[float, str] = 0):
        """Add a point load (force) to this load case."""
        self.loads.append(PointLoad(node=node, x=x, fx=fx, fy=fy))

    def moment(self, node: Optional[int] = None, x: Optional[float] = None,
               mz: Union[float, str] = 0):
        """Add a concentrated moment to this load case."""
        self.loads.append(ConcentratedMoment(node=node, x=x, mz=mz))

    def distributed_load(self, x_start: float = 0.0, x_end: float = 0.0,
                         distribution: str = 'uniform', **kwargs):
        """
        Add a distributed load to this load case.
        
        Parameters:
        -----------
        x_start, x_end : float
            Coordinate range (mm)
        distribution : str
            'uniform', 'linear', 'triangular', or 'custom'
        **kwargs
            Distribution-specific parameters (wy, wx, wy_start, wy_end, etc.)
        """
        self.loads.append(DistributedLoad(
            x_start=x_start, x_end=x_end, distribution=distribution, **kwargs
        ))

    def lumped_mass(self, node: Optional[int] = None, x: Optional[float] = None,
                    m: float = 0.0, Izz: float = 0.0, apply_gravity: bool = False):
        """Add a lumped mass to this load case."""
        self.loads.append(LumpedMass(
            node=node, x=x, m=m, Izz=Izz, apply_gravity=apply_gravity
        ))

    def has_mass_loads(self) -> bool:
        """Check if load case contains any lumped masses."""
        return any(isinstance(ld, LumpedMass) for ld in self.loads)

    def apply_to_mass_matrix(self, M, mesh):
        """
        Apply load properties that affect the global mass matrix.
        Modifies and returns M in-place.
        """
        for load in self.loads:
            M = load.apply_to_mass_matrix(M, mesh)
        return M
    def create_force_vector(self, num_dofs: int, mesh) -> np.ndarray:
        """
        Create global force vector from all loads in this case.
        
        Parameters:
        -----------
        num_dofs : int
            Total number of degrees of freedom
        mesh : Mesh
            Finite element mesh
            
        Returns:
        --------
        F : np.ndarray
            Global force vector
        """
        F = np.zeros(num_dofs)
        
        for load in self.loads:
            F = load.apply_to_force_vector(F, mesh)
        
        return F
    
    def __str__(self):
        load_summary = "\n  ".join(str(load) for load in self.loads)
        return f"Load Case: {self.name}\n  {load_summary}"


class LoadCombination:
    """Combination of multiple load cases with factors."""
    
    def __init__(self, name: str = "Combination 1"):
        """Initialize load combination."""
        self.name = name
        self.load_cases: List[Tuple[LoadCase, float]] = []
    
    def load_case(self, load_case: LoadCase, factor: float = 1.0):
        """
        Add a load case with a load factor.
        
        Parameters:
        -----------
        load_case : LoadCase
            Load case to add
        factor : float
            Load factor (e.g., 1.2 for dead load, 1.6 for live load)
        """
        self.load_cases.append((load_case, factor))
    
    def create_force_vector(self, num_dofs: int, mesh) -> np.ndarray:
        """Create combined force vector."""
        F_combined = np.zeros(num_dofs)
        
        for load_case, factor in self.load_cases:
            F_case = load_case.create_force_vector(num_dofs, mesh)
            F_combined += factor * F_case
        
        return F_combined

    def has_mass_loads(self) -> bool:
        """Check if combination contains any lumped masses."""
        return any(lc.has_mass_loads() for lc, _ in self.load_cases)

    def apply_to_mass_matrix(self, M, mesh):
        """Apply mass modifications, ignoring combination factors for mass matrix."""
        for load_case, _ in self.load_cases:
            M = load_case.apply_to_mass_matrix(M, mesh)
        return M
    
    def __str__(self):
        case_summary = "\n  ".join(f"{factor:.2f} × {lc.name}" 
                                  for lc, factor in self.load_cases)
        return f"Load Combination: {self.name}\n  {case_summary}"


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("LOAD DEFINITION EXAMPLES")
    print("="*70)
    
    # Example 1: Single load case
    print("\n1. Single Load Case:")
    lc1 = LoadCase("Dead Load")
    lc1.point_load(node=5, fy=-100)
    lc1.distributed_load(x_start=0, x_end=500, distribution='uniform', wy=-0.5)
    print(lc1)
    
    # Example 2: Load combination
    print("\n2. Load Combination (LRFD):")
    lc_dead = LoadCase("Dead Load")
    lc_dead.distributed_load(x_start=0, x_end=1000, distribution='uniform', wy=-1.0)
    
    lc_live = LoadCase("Live Load")
    lc_live.point_load(node=10, fy=-200)
    
    combo = LoadCombination("1.2D + 1.6L")
    combo.load_case(lc_dead, factor=1.2)
    combo.load_case(lc_live, factor=1.6)
    print(combo)
    
    # Example 3: Various load types
    print("\n3. Various Load Types:")
    lc_all = LoadCase("All Types")
    lc_all.point_load(node=5, fy=-100)
    lc_all.moment(node=5, mz=50)
    lc_all.distributed_load(x_start=0, x_end=500, distribution='uniform', wy=-2.0)
    lc_all.distributed_load(x_start=0, x_end=500, distribution='linear', wy_start=-1.0, wy_end=-3.0)
    lc_all.distributed_load(x_start=0, x_end=1000, distribution='triangular', w_peak=-5.0, peak_loc=500.0)
    lc_all.distributed_load(x_start=0, x_end=1000, distribution='custom', load_fn=lambda x: -x**2/1e6)
    
    print(lc_all)
