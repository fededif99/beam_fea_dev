"""
boundary_conditions.py
======================
Boundary condition definitions and application.

Supports:
- Fixed supports
- Pinned supports
- Roller supports
- Spring supports
- Prescribed displacements
"""

import numpy as np
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
 
# Default penalty *multiplier* for relative penalty method.
# The actual penalty applied is: DEFAULT_PENALTY_MULTIPLIER * max(diag(K))
# This keeps the penalty numerically consistent regardless of the problem scale
# (unit system, material stiffness, structure size).
DEFAULT_PENALTY_MULTIPLIER = 1e10
# Minimum absolute fallback penalty if max(diag(K)) is near-zero
_PENALTY_ABS_MIN = 1e6


@dataclass
class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    
    @abstractmethod
    def get_constrained_dofs(self) -> List[int]:
        """Return list of constrained DOF indices."""
        pass
    
    @abstractmethod
    def __str__(self):
        """String representation."""
        pass


@dataclass
class FixedSupport(BoundaryCondition):
    """
    Fixed (clamped) support - all DOFs constrained.
    
    Attributes:
    -----------
    node : int
        Node ID
    """
    
    node: int
    
    def get_constrained_dofs(self) -> List[int]:
        """Return all DOFs at this node."""
        return [3*self.node, 3*self.node + 1, 3*self.node + 2]
    
    def __str__(self):
        return f"Fixed support at node {self.node}"


@dataclass
class PinnedSupport(BoundaryCondition):
    """
    Pinned (hinged) support - translations constrained, rotation free.
    
    Attributes:
    -----------
    node : int
        Node ID
    """
    
    node: int
    
    def get_constrained_dofs(self) -> List[int]:
        """Return translation DOFs only."""
        return [3*self.node, 3*self.node + 1]
    
    def __str__(self):
        return f"Pinned support at node {self.node}"


@dataclass
class RollerSupport(BoundaryCondition):
    """
    Roller support - one translation constrained, other DOFs free.
    
    Attributes:
    -----------
    node : int
        Node ID
    direction : str
        'x' or 'y' - direction of constraint
    """
    
    node: int
    direction: str = 'y'  # Default: vertical roller
    
    def get_constrained_dofs(self) -> List[int]:
        """Return constrained DOF."""
        if self.direction.lower() == 'x':
            return [3*self.node]  # Horizontal constraint
        elif self.direction.lower() == 'y':
            return [3*self.node + 1]  # Vertical constraint
        else:
            raise ValueError("Direction must be 'x' or 'y'")
    
    def __str__(self):
        return f"Roller support at node {self.node} (constrains {self.direction})"


@dataclass
class SpringSupport(BoundaryCondition):
    """
    Spring support - elastic constraint.
    
    Attributes:
    -----------
    node : int
        Node ID
    kx : float
        Spring stiffness in x-direction (N/mm)
    ky : float
        Spring stiffness in y-direction (N/mm)
    kr : float
        Rotational spring stiffness (N·mm/rad)
    """
    
    node: int
    kx: float = 0.0
    ky: float = 0.0
    kr: float = 0.0
    
    def get_constrained_dofs(self) -> List[int]:
        """Springs don't constrain DOFs, they modify stiffness."""
        return []
    
    def apply_to_stiffness(self, K: np.ndarray) -> np.ndarray:
        """
        Add spring stiffness to global stiffness matrix.
        
        Parameters:
        -----------
        K : np.ndarray
            Global stiffness matrix
            
        Returns:
        --------
        K : np.ndarray
            Modified stiffness matrix
        """
        if self.kx > 0:
            K[3*self.node, 3*self.node] += self.kx
        if self.ky > 0:
            K[3*self.node + 1, 3*self.node + 1] += self.ky
        if self.kr > 0:
            K[3*self.node + 2, 3*self.node + 2] += self.kr
        
        return K
    
    def __str__(self):
        springs = []
        if self.kx > 0:
            springs.append(f"kx={self.kx:.2e}")
        if self.ky > 0:
            springs.append(f"ky={self.ky:.2e}")
        if self.kr > 0:
            springs.append(f"kr={self.kr:.2e}")
        return f"Spring support at node {self.node}: {', '.join(springs)}"


@dataclass
class PrescribedDisplacement(BoundaryCondition):
    """
    Prescribed displacement at a DOF.
    
    Attributes:
    -----------
    node : int
        Node ID
    dx : float, optional
        Prescribed x-displacement (mm)
    dy : float, optional
        Prescribed y-displacement (mm)
    rotation : float, optional
        Prescribed rotation (rad)
    """
    
    node: int
    dx: Optional[float] = None
    dy: Optional[float] = None
    rotation: Optional[float] = None
    
    def get_constrained_dofs(self) -> List[int]:
        """Return DOFs with prescribed values."""
        dofs = []
        if self.dx is not None:
            dofs.append(3*self.node)
        if self.dy is not None:
            dofs.append(3*self.node + 1)
        if self.rotation is not None:
            dofs.append(3*self.node + 2)
        return dofs
    
    def get_prescribed_values(self) -> List[Tuple[int, float]]:
        """Return list of (DOF, value) tuples."""
        values = []
        if self.dx is not None:
            values.append((3*self.node, self.dx))
        if self.dy is not None:
            values.append((3*self.node + 1, self.dy))
        if self.rotation is not None:
            values.append((3*self.node + 2, self.rotation))
        return values
    
    def __str__(self):
        disps = []
        if self.dx is not None:
            disps.append(f"dx={self.dx}")
        if self.dy is not None:
            disps.append(f"dy={self.dy}")
        if self.rotation is not None:
            disps.append(f"θ={self.rotation}")
        return f"Prescribed displacement at node {self.node}: {', '.join(disps)}"


@dataclass
class SymmetryCondition(BoundaryCondition):
    """
    Symmetry boundary condition.
    
    Attributes:
    -----------
    nodes : list
        Node IDs on symmetry plane
    axis : str
        'x' or 'y' - axis of symmetry
    """
    
    nodes: List[int]
    axis: str = 'y'
    
    def get_constrained_dofs(self) -> List[int]:
        """Return DOFs constrained by symmetry."""
        dofs = []
        for node in self.nodes:
            if self.axis.lower() == 'y':
                # Symmetry about y-axis: constrain x-displacement and rotation
                dofs.extend([3*node, 3*node + 2])
            elif self.axis.lower() == 'x':
                # Symmetry about x-axis: constrain y-displacement and rotation
                dofs.extend([3*node + 1, 3*node + 2])
        return dofs
    
    def __str__(self):
        return f"Symmetry condition on {len(self.nodes)} nodes (axis: {self.axis})"


class BoundaryConditionSet:
    """Collection of boundary conditions."""
    
    def __init__(self, name: str = "BC Set"):
        """Initialize boundary condition set."""
        self.name = name
        self.conditions: List[BoundaryCondition] = []
        self.spring_supports: List[SpringSupport] = []
    
    def add(self, condition: BoundaryCondition):
        """Add a boundary condition object."""
        if isinstance(condition, SpringSupport):
            self.spring_supports.append(condition)
        else:
            self.conditions.append(condition)

    def fixed_support(self, node: int):
        """Add fixed support."""
        self.conditions.append(FixedSupport(node=node))
    
    def pinned_support(self, node: int):
        """Add pinned support."""
        self.conditions.append(PinnedSupport(node=node))
    
    def roller_support(self, node: int, direction: str = 'y'):
        """Add roller support."""
        self.conditions.append(RollerSupport(node=node, direction=direction))
    
    def spring_support(self, node: int, kx: float = 0, ky: float = 0, kr: float = 0):
        """Add spring support."""
        spring = SpringSupport(node=node, kx=kx, ky=ky, kr=kr)
        self.spring_supports.append(spring)
    
    def prescribed_displacement(self, node: int, dx=None, dy=None, rotation=None):
        """Add prescribed displacement."""
        self.conditions.append(PrescribedDisplacement(node=node, dx=dx, dy=dy, rotation=rotation))
    
    def get_all_constrained_dofs(self) -> Set[int]:
        """Get set of all constrained DOF indices."""
        constrained = set()
        for bc in self.conditions:
            constrained.update(bc.get_constrained_dofs())
        return constrained
    
    def apply_to_system(self, K: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply boundary conditions to stiffness matrix and force vector.
        
        Parameters:
        -----------
        K : np.ndarray
            Global stiffness matrix
        F : np.ndarray
            Global force vector
            
        Returns:
        --------
        K_bc : np.ndarray
            Modified stiffness matrix
        F_bc : np.ndarray
            Modified force vector
        """
        K_bc = K.copy()
        F_bc = F.copy()
        
        # Apply spring supports (add to stiffness)
        for spring in self.spring_supports:
            K_bc = spring.apply_to_stiffness(K_bc)
        
        # Get all constrained DOFs
        constrained_dofs = self.get_all_constrained_dofs()
        
        # Handle prescribed displacements
        prescribed_values = {}
        for bc in self.conditions:
            if isinstance(bc, PrescribedDisplacement):
                for dof, value in bc.get_prescribed_values():
                    prescribed_values[dof] = value
        
        # Apply constraints
        from scipy.sparse import issparse
        
        if issparse(K_bc):
            # For sparse matrices use the Penalty Method to preserve sparsity.
            # The penalty is scaled RELATIVE to the maximum stiffness diagonal entry.
            # This avoids ill-conditioning when problem units or section sizes vary
            # by many orders of magnitude (e.g., micro-scale vs. offshore structures).
            K_diag = K_bc.diagonal()
            K_max = float(np.max(np.abs(K_diag)))
            penalty = max(K_max * DEFAULT_PENALTY_MULTIPLIER, _PENALTY_ABS_MIN)

            K_bc = K_bc.tolil()  # Convert to LIL for efficient modification

            for dof in constrained_dofs:
                K_bc[dof, dof] += penalty

                # Update force vector for prescribed (non-zero) displacements
                if dof in prescribed_values:
                    F_bc[dof] += penalty * prescribed_values[dof]

            K_bc = K_bc.tocsr()  # Convert back for solver
            
        else:
            # For dense matrices, use the exact "Identity Method"
            # Zero out rows and columns
            for dof in constrained_dofs:
                K_bc[dof, :] = 0
                K_bc[:, dof] = 0
                K_bc[dof, dof] = 1
                
                if dof in prescribed_values:
                    F_bc[dof] = prescribed_values[dof]
                else:
                    F_bc[dof] = 0
        
        return K_bc, F_bc
    
    def __str__(self):
        bc_summary = "\n  ".join(str(bc) for bc in self.conditions)
        if self.spring_supports:
            spring_summary = "\n  ".join(str(spring) for spring in self.spring_supports)
            return f"{self.name}:\n  {bc_summary}\n  {spring_summary}"
        return f"{self.name}:\n  {bc_summary}"



if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("BOUNDARY CONDITION EXAMPLES")
    print("="*70)
    
    # Example 1: Simple beam configurations
    print("\n1. Common Support Configurations:")
    
    # Simply Supported (Pinned-Pinned)
    bc_simple = BoundaryConditionSet("Simply Supported")
    bc_simple.pinned_support(0)
    bc_simple.pinned_support(10)
    
    # Fixed-Fixed
    bc_fixed = BoundaryConditionSet("Fixed-Fixed")
    bc_fixed.fixed_support(0)
    bc_fixed.fixed_support(10)
    
    # Cantilever
    bc_cant = BoundaryConditionSet("Cantilever")
    bc_cant.fixed_support(0)

    configs = [
        ("Simply Supported", bc_simple),
        ("Fixed-Fixed", bc_fixed),
        ("Cantilever", bc_cant),
    ]
    
    for name, bc_set in configs:
        print(f"\n   {name}:")
        for bc in bc_set.conditions:
            print(f"     {bc}")
    
    # Example 2: Complex boundary conditions
    print("\n2. Complex Boundary Conditions:")
    bc_complex = BoundaryConditionSet("Complex Example")
    bc_complex.fixed_support(0)
    bc_complex.spring_support(5, ky=1000, kr=500)
    bc_complex.roller_support(10, direction='y')
    bc_complex.prescribed_displacement(7, dy=0.5)
    
    print(bc_complex)
    
    # Example 3: Constrained DOFs
    print("\n3. Constrained DOFs:")
    print(f"   Total constrained DOFs: {len(bc_complex.get_all_constrained_dofs())}")
    print(f"   DOF indices: {sorted(bc_complex.get_all_constrained_dofs())}")
