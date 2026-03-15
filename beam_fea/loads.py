"""
loads.py
========
Load definitions and application for finite element analysis.

Supports:
- Point loads
- Distributed loads (uniform and trapezoidal)
- Concentrated moments
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
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


@dataclass
class PointLoad(Load):
    """
    Point load at a specific node or global x-coordinate.
    
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
    mz : float or str
        Moment about z-axis (N·mm)
    """
    
    node: Optional[int] = None
    x: Optional[float] = None
    fx: Union[float, str] = 0.0
    fy: Union[float, str] = 0.0
    mz: Union[float, str] = 0.0
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply point load to global force vector."""
        if self.node is not None:
            # Use direct node index
            F[3 * self.node] += self.fx
            F[3 * self.node + 1] += self.fy
            F[3 * self.node + 2] += self.mz
        elif self.x is not None:
            # Resolve x to element and nodes
            elem_idx = mesh.find_element_at_x(self.x)
            if elem_idx == -1:
                # Outside mesh, ignore or warn? For now, ignore (standard FEA)
                return F
            
            coords = mesh.nodes
            node1, node2 = mesh.elements[elem_idx]
            p1, p2 = coords[node1], coords[node2]
            
            x1 = p1[0]
            L = np.linalg.norm(p2 - p1)
            xi = (self.x - x1) / L if L > 0 else 0
            xi = np.clip(xi, 0, 1) # Clamp to element
            
            # Axial components (linear)
            F[3 * node1] += self.fx * (1 - xi)
            F[3 * node2] += self.fx * xi
            
            # Transverse components (fy) using Hermite shape functions
            # N1, N2, N3, N4
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
            
            # Point moment (mz) using shape function derivatives (work-equivalent)
            # Ni' = dNi/dx = dNi/dxi * (1/L)
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
        components = []
        if abs(self.fx) > 1e-10:
            components.append(f"Fx={self.fx:.2f}N")
        if abs(self.fy) > 1e-10:
            components.append(f"Fy={self.fy:.2f}N")
        if abs(self.mz) > 1e-10:
            components.append(f"Mz={self.mz:.2f}N·mm")
        return f"Point load at {loc}: {', '.join(components)}"


@dataclass
class UniformDistributedLoad(Load):
    """
    Uniformly distributed load along element(s) or a coordinate range.
    
    Attributes:
    -----------
    element : int or list, optional
        Element ID(s) where load is applied
    x_start, x_end : float, optional
        Global x-coordinate range (mm). Used if element is None.
    wy : float or str
        Distributed force in y-direction (N/mm) - TRANSVERSE
    wx : float or str
        Distributed force in x-direction (N/mm) - AXIAL (optional)
    """
    
    element: Optional[Union[int, List[int]]] = None
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    wy: Union[float, str] = 0.0
    wx: Union[float, str] = 0.0
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Convert distributed load to equivalent nodal loads."""
        coords = mesh.nodes
        elements = mesh.elements
        
        if self.element is not None:
            # Traditional element-based logic
            elem_list = [self.element] if isinstance(self.element, int) else self.element
            for elem_id in elem_list:
                node1, node2 = elements[elem_id]
                L = np.linalg.norm(coords[node2] - coords[node1])
                self._apply_to_element(F, node1, node2, L, 0.0, 1.0)
        elif self.x_start is not None and self.x_end is not None:
            # Coordinate-based logic
            x_min_load = min(self.x_start, self.x_end)
            x_max_load = max(self.x_start, self.x_end)
            
            for eid in range(mesh.num_elements):
                node1, node2 = elements[eid]
                x1 = coords[node1, 0]
                x2 = coords[node2, 0]
                e_min = min(x1, x2)
                e_max = max(x1, x2)
                
                # Intersection of element [e_min, e_max] and load [x_min_load, x_max_load]
                inter_min = max(e_min, x_min_load)
                inter_max = min(e_max, x_max_load)
                
                if inter_max > inter_min:
                    L = np.linalg.norm(coords[node2] - coords[node1])
                    # Normalize intersection to local [xi1, xi2]
                    xi1 = (inter_min - e_min) / L
                    xi2 = (inter_max - e_min) / L
                    self._apply_to_element(F, node1, node2, L, xi1, xi2)
                    
        return F

    def _apply_to_element(self, F, node1, node2, L, xi1, xi2):
        """Helper to apply uniform load over a local xi range [xi1, xi2]."""
        dx = xi2 - xi1
        
        # Axial (linear)
        # Integral of (1-xi) from xi1 to xi2: [xi - xi^2/2]
        int_Na1 = (xi2 - 0.5*xi2**2) - (xi1 - 0.5*xi1**2)
        # Integral of xi from xi1 to xi2: [xi^2/2]
        int_Na2 = (0.5*xi2**2) - (0.5*xi1**2)
        
        F[3 * node1] += self.wx * L * int_Na1
        F[3 * node2] += self.wx * L * int_Na2
        
        # Transverse (Hermite)
        # N1 = 1 - 3xi^2 + 2xi^3  => Int = xi - xi^3 + 0.5xi^4
        # N2 = L(xi - 2xi^2 + xi^3) => Int = L(0.5xi^2 - 2/3xi^3 + 0.25xi^4)
        # N3 = 3xi^2 - 2xi^3      => Int = xi^3 - 0.5xi^4
        # N4 = L(-xi^2 + xi^3)    => Int = L(-1/3xi^3 + 0.25xi^4)
        
        def int_N(xi):
            return np.array([
                xi - xi**3 + 0.5*xi**4,
                L * (0.5*xi**2 - (2/3)*xi**3 + 0.25*xi**4),
                xi**3 - 0.5*xi**4,
                L * (-(1/3)*xi**3 + 0.25*xi**4)
            ])
        
        integrals = int_N(xi2) - int_N(xi1)
        
        F[3 * node1 + 1] += self.wy * L * integrals[0]
        F[3 * node1 + 2] += self.wy * L * integrals[1]
        F[3 * node2 + 1] += self.wy * L * integrals[2]
        F[3 * node2 + 2] += self.wy * L * integrals[3]
    
    def __str__(self):
        loc = f"element {self.element}" if self.element is not None else f"x=[{self.x_start}, {self.x_end}]"
        if abs(self.wx) > 1e-10:
             return f"UDL on {loc}: wy={self.wy:.4f}, wx={self.wx:.4f} N/mm"
        return f"UDL on {loc}: wy={self.wy:.4f} N/mm"


@dataclass
class TrapezoidalDistributedLoad(Load):
    """
    Linearly varying (trapezoidal) distributed load along an element or coordinate range.
    
    Attributes:
    -----------
    element : int, optional
        Element ID
    x_start, x_end : float, optional
        Global x-coordinate range (mm). Used if element is None.
    wy1, wy2 : float or str
        y-direction load at start and end (N/mm)
    wx1, wx2 : float or str
        x-direction load at start and end (N/mm) - Optional
    """
    
    element: Optional[int] = None
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    wy1: Union[float, str] = 0.0
    wy2: Union[float, str] = 0.0
    wx1: Union[float, str] = 0.0
    wx2: Union[float, str] = 0.0
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Convert linearly varying load to equivalent nodal loads."""
        coords = mesh.nodes
        elements = mesh.elements
        
        if self.element is not None:
            node1, node2 = elements[self.element]
            L = np.linalg.norm(coords[node2] - coords[node1])
            self._apply_to_element(F, node1, node2, L, 0.0, 1.0, self.wy1, self.wy2, self.wx1, self.wx2)
        elif self.x_start is not None and self.x_end is not None:
            x_min_load = min(self.x_start, self.x_end)
            x_max_load = max(self.x_start, self.x_end)
            total_L = x_max_load - x_min_load
            if total_L <= 0: return F

            for eid in range(mesh.num_elements):
                node1, node2 = elements[eid]
                x1 = coords[node1, 0]
                x2 = coords[node2, 0]
                e_min, e_max = min(x1, x2), max(x1, x2)
                
                inter_min = max(e_min, x_min_load)
                inter_max = min(e_max, x_max_load)
                
                if inter_max > inter_min:
                    L = np.linalg.norm(coords[node2] - coords[node1])
                    xi1 = (inter_min - e_min) / L
                    xi2 = (inter_max - e_min) / L
                    
                    # Interp intensities at inter_min and inter_max
                    def get_w(x, w1, w2):
                        return w1 + (w2 - w1) * (x - self.x_start) / (self.x_end - self.x_start)
                    
                    wy_a = get_w(inter_min, self.wy1, self.wy2)
                    wy_b = get_w(inter_max, self.wy1, self.wy2)
                    wx_a = get_w(inter_min, self.wx1, self.wx2)
                    wx_b = get_w(inter_max, self.wx1, self.wx2)
                    
                    self._apply_to_element(F, node1, node2, L, xi1, xi2, wy_a, wy_b, wx_a, wx_b)
                    
        return F

    def _apply_to_element(self, F, node1, node2, L, xi1, xi2, wya, wyb, wxa, wxb):
        """Helper to apply linear load over local range [xi1, xi2]."""
        # Load is linear in xi within [xi1, xi2]: w(xi) = A + B*xi
        # But it's easier to say w(xi) = wya + (wyb - wya) * (xi - xi1)/(xi2 - xi1)
        # Let's use w(xi) = C + D*xi for the whole element integration
        # w(xi1) = wya, w(xi2) = wyb
        if abs(xi2 - xi1) < 1e-12: return
        
        # wy(xi) = wy_base + wy_slope * xi
        wy_slope = (wyb - wya) / (xi2 - xi1)
        wy_base = wya - wy_slope * xi1
        
        # wx(xi) = wx_base + wx_slope * xi
        wx_slope = (wxb - wxa) / (xi2 - xi1)
        wx_base = wxa - wx_slope * xi1

        # Integrals of N_i: I1_i = integral(N_i) from xi1 to xi2
        # Integrals of xi*N_i: I2_i = integral(xi*N_i) from xi1 to xi2
        
        def get_integrals(xi):
            # [Integral(Ni), Integral(xi*Ni)]
            I1 = np.array([
                xi - xi**3 + 0.5*xi**4,
                L * (0.5*xi**2 - (2/3)*xi**3 + 0.25*xi**4),
                xi**3 - 0.5*xi**4,
                L * (-(1/3)*xi**3 + 0.25*xi**4)
            ])
            I2 = np.array([
                0.5*xi**2 - 0.75*xi**4 + 0.4*xi**5,
                L * ((1/3)*xi**3 - 0.5*xi**4 + 0.2*xi**5),
                0.75*xi**4 - 0.4*xi**5,
                L * (-0.25*xi**4 + 0.2*xi**5)
            ])
            # For axial (linear shape functions): Na1 = 1-xi, Na2 = xi
            # Integral(1-xi) = xi - 0.5xi^2
            # Integral(xi(1-xi)) = 0.5xi^2 - 1/3xi^3
            # Integral(xi) = 0.5xi^2
            # Integral(xi^2) = 1/3xi^3
            Ia = np.array([xi - 0.5*xi**2, 0.5*xi**2 - (1/3)*xi**3, 0.5*xi**2, (1/3)*xi**3])
            
            return I1, I2, Ia

        I1_b, I2_b, Ia_b = get_integrals(xi2)
        I1_a, I2_a, Ia_a = get_integrals(xi1)
        
        dI1, dI2, dIa = I1_b - I1_a, I2_b - I2_a, Ia_b - Ia_a
        
        # Transverse
        F[3 * node1 + 1] += L * (wy_base * dI1[0] + wy_slope * dI2[0])
        F[3 * node1 + 2] += L * (wy_base * dI1[1] + wy_slope * dI2[1])
        F[3 * node2 + 1] += L * (wy_base * dI1[2] + wy_slope * dI2[2])
        F[3 * node2 + 2] += L * (wy_base * dI1[3] + wy_slope * dI2[3])
        
        # Axial
        F[3 * node1] += L * (wx_base * dIa[0] + wx_slope * dIa[1])
        F[3 * node2] += L * (wx_base * dIa[2] + wx_slope * dIa[3])

    def __str__(self):
        loc = f"element {self.element}" if self.element is not None else f"x=[{self.x_start}, {self.x_end}]"
        return f"Trapezoidal load on {loc}: wy={self.wy1:.4f} to {self.wy2:.4f} N/mm"


@dataclass
class TriangularDistributedLoad(Load):
    """
    Triangularly distributed load along an element or coordinate range.
    
    Attributes:
    -----------
    element : int, optional
        Element ID
    x_start, x_end : float, optional
        Global x-coordinate range (mm). Used if element is None.
    w_peak : float or str
        Peak load intensity (N/mm)
    peak_loc : str
        'start' (peak at start of range) or 'end' (peak at end of range)
    """
    element: Optional[int] = None
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    w_peak: Union[float, str] = 0.0
    peak_loc: str = 'start'

    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Resolution of triangular load into nodal forces."""
        if self.peak_loc.lower() == 'start':
            wy1, wy2 = self.w_peak, 0.0
        else:
            wy1, wy2 = 0.0, self.w_peak
            
        temp_trap = TrapezoidalDistributedLoad(
            element=self.element, 
            x_start=self.x_start, x_end=self.x_end,
            wy1=wy1, wy2=wy2
        )
        return temp_trap.apply_to_force_vector(F, mesh)

    def __str__(self):
        loc = f"element {self.element}" if self.element is not None else f"x=[{self.x_start}, {self.x_end}]"
        return f"Triangular load on {loc}: peak {self.w_peak:.4f} at {self.peak_loc}"


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
                   fx: Union[float, str] = 0, fy: Union[float, str] = 0, mz: Union[float, str] = 0):
        """Add a point load to this load case."""
        self.loads.append(PointLoad(node=node, x=x, fx=fx, fy=fy, mz=mz))
    
    def concentrated_moment(self, node: Optional[int] = None, x: Optional[float] = None, mz: Union[float, str] = 0):
        """Add a concentrated moment to this load case."""
        self.loads.append(PointLoad(node=node, x=x, mz=mz))
    
    def moment(self, node: Optional[int] = None, x: Optional[float] = None, mz: Union[float, str] = 0):
        """Alias for concentrated_moment."""
        self.concentrated_moment(node=node, x=x, mz=mz)
    
    def uniform_load(self, element: Optional[Union[int, List[int]]] = None, 
                     x_start: Optional[float] = None, x_end: Optional[float] = None,
                     wy: Union[float, str] = 0.0, wx: Union[float, str] = 0.0):
        """
        Add a Uniformly Distributed Load (UDL).
        
        Parameters:
        -----------
        element : int or list, optional
            Element ID(s)
        x_start, x_end : float, optional
            Coordinate range (mm)
        wy : float or str
            Transverse load intensity (N/mm)
        wx : float or str, optional
            Axial load intensity (N/mm), default 0
        """
        self.loads.append(UniformDistributedLoad(element=element, x_start=x_start, x_end=x_end, wy=wy, wx=wx))
    
    def trapezoidal_load(self, element: Optional[int] = None, 
                         x_start: Optional[float] = None, x_end: Optional[float] = None,
                         wy1: Union[float, str] = 0.0, wy2: Union[float, str] = 0.0,
                         wx1: Union[float, str] = 0.0, wx2: Union[float, str] = 0.0):
        """
        Add a Trapezoidal (linearly varying) load.
        """
        self.loads.append(TrapezoidalDistributedLoad(
            element=element, x_start=x_start, x_end=x_end,
            wy1=wy1, wy2=wy2, wx1=wx1, wx2=wx2
        ))
    
    def triangular_load(self, element: Optional[int] = None, 
                        x_start: Optional[float] = None, x_end: Optional[float] = None,
                        w_peak: Union[float, str] = 0.0, peak_loc: str = 'start'):
        """
        Add a Triangular load.
        """
        self.loads.append(TriangularDistributedLoad(
            element=element, x_start=x_start, x_end=x_end,
            w_peak=w_peak, peak_loc=peak_loc
        ))

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
    lc1.uniform_load(element=[0, 1, 2], wy=-0.5)
    print(lc1)
    
    # Example 2: Load combination
    print("\n2. Load Combination (LRFD):")
    lc_dead = LoadCase("Dead Load")
    lc_dead.uniform_load(element=[0, 1], wy=-1.0)
    
    lc_live = LoadCase("Live Load")
    lc_live.point_load(node=10, fy=-200)
    
    combo = LoadCombination("1.2D + 1.6L")
    combo.load_case(lc_dead, factor=1.2)
    combo.load_case(lc_live, factor=1.6)
    print(combo)
    
    # Example 3: Different load types
    print("\n3. Various Load Types:")
    lc_all = LoadCase("All Types")
    lc_all.point_load(node=5, fy=-100, mz=50)
    lc_all.uniform_load(element=3, wy=-2.0)
    lc_all.trapezoidal_load(element=4, wy1=-1.0, wy2=-3.0)
    lc_all.triangular_load(element=5, w_peak=-5.0, peak_loc='start')
    
    print(lc_all)
