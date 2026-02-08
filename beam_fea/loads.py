"""
loads.py
========
Load definitions and application for finite element analysis.

Supports:
- Point loads
- Distributed loads
- Moments
- Temperature loads
- Initial displacements
"""

import numpy as np
from typing import List, Tuple, Optional
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
    Point load at a specific node.
    
    Attributes:
    -----------
    node : int
        Node ID where load is applied
    fx : float
        Force in x-direction (N)
    fy : float
        Force in y-direction (N)
    mz : float
        Moment about z-axis (N·mm)
    """
    
    node: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Apply point load to force vector."""
        F[3 * self.node] += self.fx
        F[3 * self.node + 1] += self.fy
        F[3 * self.node + 2] += self.mz
        return F
    
    def __str__(self):
        components = []
        if abs(self.fx) > 1e-10:
            components.append(f"Fx={self.fx:.2f}N")
        if abs(self.fy) > 1e-10:
            components.append(f"Fy={self.fy:.2f}N")
        if abs(self.mz) > 1e-10:
            components.append(f"Mz={self.mz:.2f}N·mm")
        return f"Point load at node {self.node}: {', '.join(components)}"


@dataclass
class UniformDistributedLoad(Load):
    """
    Uniformly distributed load along element(s).
    
    Attributes:
    -----------
    element : int or list
        Element ID(s) where load is applied
    wy : float
        Distributed force in y-direction (N/mm) - TRANSVERSE
    wx : float
        Distributed force in x-direction (N/mm) - AXIAL (optional)
    """
    
    element: int or List[int]
    wy: float = 0.0
    wx: float = 0.0
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Convert distributed load to equivalent nodal loads."""
        elements = [self.element] if isinstance(self.element, int) else self.element
        
        for elem_id in elements:
            elem = mesh.elements[elem_id]
            L = elem.length(mesh.nodes)
            
            # Axial component (if wx exists)
            if abs(self.wx) > 1e-12:
                Fx_node = self.wx * L / 2
                F[3 * elem.node1] += Fx_node
                F[3 * elem.node2] += Fx_node
            
            # Transverse component (wy) -- Primary
            Fy_node = self.wy * L / 2
            Mz_node1 = self.wy * L**2 / 12
            Mz_node2 = -self.wy * L**2 / 12
            
            F[3 * elem.node1 + 1] += Fy_node
            F[3 * elem.node1 + 2] += Mz_node1
            F[3 * elem.node2 + 1] += Fy_node
            F[3 * elem.node2 + 2] += Mz_node2
        
        return F
    
    def __str__(self):
        elem_str = f"element {self.element}" if isinstance(self.element, int) else f"elements {self.element}"
        if abs(self.wx) > 1e-10:
             return f"UDL on {elem_str}: wy={self.wy:.4f}, wx={self.wx:.4f} N/mm"
        return f"UDL on {elem_str}: wy={self.wy:.4f} N/mm"


@dataclass
class TrapezoidalDistributedLoad(Load):
    """
    Linearly varying (trapezoidal) distributed load.
    
    Attributes:
    -----------
    element : int
        Element ID
    wy1, wy2 : float
        y-direction load at start and end (N/mm)
    wx1, wx2 : float
        x-direction load at start and end (N/mm) - Optional
    """
    
    element: int
    wy1: float = 0.0
    wy2: float = 0.0
    wx1: float = 0.0
    wx2: float = 0.0
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """Convert linearly varying load to equivalent nodal loads."""
        elem = mesh.elements[self.element]
        L = elem.length(mesh.nodes)
        
        # Axial component
        if abs(self.wx1) > 1e-12 or abs(self.wx2) > 1e-12:
            Fx1 = (2*self.wx1 + self.wx2) * L / 6
            Fx2 = (self.wx1 + 2*self.wx2) * L / 6
            F[3 * elem.node1] += Fx1
            F[3 * elem.node2] += Fx2
        
        # Transverse component
        Fy1 = (7*self.wy1 + 3*self.wy2) * L / 20
        Fy2 = (3*self.wy1 + 7*self.wy2) * L / 20
        Mz1 = (3*self.wy1 + 2*self.wy2) * L**2 / 60
        Mz2 = -(2*self.wy1 + 3*self.wy2) * L**2 / 60
        
        F[3 * elem.node1 + 1] += Fy1
        F[3 * elem.node1 + 2] += Mz1
        F[3 * elem.node2 + 1] += Fy2
        F[3 * elem.node2 + 2] += Mz2
        
        return F
    
    def __str__(self):
        return f"Trapezoidal load on element {self.element}: wy={self.wy1:.4f} to {self.wy2:.4f} N/mm"


@dataclass
class ThermalLoad(Load):
    """
    Thermal load (temperature change).
    
    Attributes:
    -----------
    element : int or list
        Element ID(s)
    delta_T : float
        Temperature change (°C)
    alpha : float
        Coefficient of thermal expansion (1/°C)
    """
    
    element: int or List[int]
    delta_T: float
    alpha: float  # e.g., 1.2e-5 for steel
    
    def apply_to_force_vector(self, F: np.ndarray, mesh) -> np.ndarray:
        """
        Apply thermal load.
        """
        # Placeholder for thermal load implementation
        return F
    
    def __str__(self):
        return f"Thermal load: ΔT={self.delta_T}°C, α={self.alpha}"


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
    
    def point_load(self, node: int, fx: float = 0, fy: float = 0, mz: float = 0):
        """Add a point load to this load case."""
        self.loads.append(PointLoad(node=node, fx=fx, fy=fy, mz=mz))
    
    def uniform_load(self, element, wy: float, wx: float = 0):
        """
        Add a Uniformly Distributed Load (UDL).
        
        Parameters:
        -----------
        element : int or list
            Element ID(s)
        wy : float
            Transverse load intensity (N/mm)
        wx : float, optional
            Axial load intensity (N/mm), default 0
        """
        self.loads.append(UniformDistributedLoad(element=element, wy=wy, wx=wx))
    
    def trapezoidal_load(self, element: int, wy1: float, wy2: float, wx1: float = 0, wx2: float = 0):
        """
        Add a Trapezoidal (linearly varying) load.
        
        Parameters:
        -----------
        element : int
            Element ID
        wy1, wy2 : float
            Transverse load at start/end (N/mm)
        wx1, wx2 : float, optional
            Axial load at start/end (N/mm)
        """
        self.loads.append(TrapezoidalDistributedLoad(element=element, wy1=wy1, wy2=wy2, wx1=wx1, wx2=wx2))
    
    def triangular_load(self, element: int, w_peak: float, peak_loc: str = 'start'):
        """
        Add a Triangular load (special case of Trapezoidal).
        
        Parameters:
        -----------
        element : int
            Element ID
        w_peak : float
            Peak load intensity (N/mm)
        peak_loc : str
            'start' (peak at node 1) or 'end' (peak at node 2)
        """
        if peak_loc.lower() == 'start':
            self.loads.append(TrapezoidalDistributedLoad(element=element, wy1=w_peak, wy2=0.0))
        elif peak_loc.lower() == 'end':
            self.loads.append(TrapezoidalDistributedLoad(element=element, wy1=0.0, wy2=w_peak))
        else:
            raise ValueError("peak_loc must be 'start' or 'end'")

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


# Convenience functions removed as per user request to favor intentional "load_case.point_load()" style.

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
