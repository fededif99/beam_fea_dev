"""
mesh.py
=======
Mesh generation and management for beam finite element analysis.

Handles node creation, element connectivity, and mesh refinement.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass


@dataclass
class Node:
    """
    Node in the finite element mesh.
    
    Attributes:
    -----------
    id : int
        Node identifier
    x : float
        x-coordinate (mm)
    y : float
        y-coordinate (mm)
    z : float
        z-coordinate (mm), default 0 for 2D
    """
    id: int
    x: float
    y: float
    z: float = 0.0
    
    def coordinates(self) -> np.ndarray:
        """Return coordinates as array."""
        return np.array([self.x, self.y, self.z])


@dataclass
class Element:
    """
    Element in the finite element mesh.
    
    Attributes:
    -----------
    id : int
        Element identifier
    node1 : int
        First node ID
    node2 : int
        Second node ID
    """
    id: int
    node1: int
    node2: int
    
    def length(self, nodes: List[Node]) -> float:
        """
        Calculate element length.
        
        Parameters:
        -----------
        nodes : list
            List of Node objects
            
        Returns:
        --------
        length : float
            Element length
        """
        n1 = nodes[self.node1]
        n2 = nodes[self.node2]
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        return np.sqrt(dx**2 + dy**2)


class Mesh:
    """
    Finite element mesh for beam structures.
    
    Manages nodes, elements, and provides mesh generation utilities.
    """
    
    def __init__(self):
        """Initialize empty mesh."""
        self.nodes: List[Node] = []
        self.elements: List[Element] = []
        self.waypoint_nodes: List[int] = [] # Node IDs at segment boundaries
        self._node_counter = 0
        self._element_counter = 0
        self._elem_starts = None  # Cache for element search
        self._node_coords = None  # Cache for node coordinates
    
    def add_node(self, x: float, y: float, z: float = 0.0) -> int:
        """
        Add a node to the mesh.
        
        Parameters:
        -----------
        x, y, z : float
            Node coordinates
            
        Returns:
        --------
        node_id : int
            ID of the created node
        """
        node = Node(id=self._node_counter, x=x, y=y, z=z)
        self.nodes.append(node)
        node_id = self._node_counter
        self._node_counter += 1
        
        # Invalidate coordinate cache
        self._node_coords = None
        
        return node_id
    
    def add_element(self, node1: int, node2: int) -> int:
        """
        Add an element to the mesh.
        
        Parameters:
        -----------
        node1, node2 : int
            Node IDs
            
        Returns:
        --------
        element_id : int
            ID of the created element
        """
        if node1 < 0 or node1 >= self.num_nodes:
            raise ValueError(f"Invalid node1 ID: {node1}. Must be between 0 and {self.num_nodes-1}")
        if node2 < 0 or node2 >= self.num_nodes:
            raise ValueError(f"Invalid node2 ID: {node2}. Must be between 0 and {self.num_nodes-1}")
        if node1 == node2:
            raise ValueError(f"Cannot create element with same node at both ends: {node1}")
        element = Element(id=self._element_counter, node1=node1, node2=node2)
        self.elements.append(element)
        element_id = self._element_counter
        self._element_counter += 1
        
        # Invalidate cache
        self._elem_starts = None
        
        return element_id
    
    def get_node_coords(self) -> np.ndarray:
        """
        Get all node coordinates as array.
        
        Returns:
        --------
        coords : np.ndarray
            Array of shape (num_nodes, 3)
        """
        if self._node_coords is None:
            self._node_coords = np.array([[n.x, n.y, n.z] for n in self.nodes])
        return self._node_coords
    
    def get_connectivity(self) -> np.ndarray:
        """
        Get element connectivity as array.
        
        Returns:
        --------
        connectivity : np.ndarray
            Array of shape (num_elements, 2)
        """
        return np.array([[e.node1, e.node2] for e in self.elements])
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in mesh."""
        return len(self.nodes)
    
    @property
    def num_elements(self) -> int:
        """Number of elements in mesh."""
        return len(self.elements)
    
    @property
    def num_dofs(self) -> int:
        """Total number of degrees of freedom (3 per node for 2D beam)."""
        return 3 * self.num_nodes
    
    def find_element_at_x(self, x: float) -> int:
        """
        Find the element ID containing the global coordinate x.
        
        Parameters:
        -----------
        x : float
            Global x-coordinate
            
        Returns:
        --------
        element_id : int
            ID of the containing element, or -1 if not found
        """
        coords = self.get_node_coords()
        
        # Optimization: Binary search for ordered meshes
        # Build or use cached index of element start positions
        if self._elem_starts is None:
            self._elem_starts = np.array([coords[e.node1, 0] for e in self.elements])
        
        # Check if elem_starts is roughly sorted (monotone increasing)
        # In most beam meshes generated by our tools, this is true.
        idx = np.searchsorted(self._elem_starts, x, side='right') - 1
        idx = np.clip(idx, 0, self.num_elements - 1)
        
        elem = self.elements[idx]
        x1, x2 = coords[elem.node1, 0], coords[elem.node2, 0]
        if min(x1, x2) <= x <= max(x1, x2):
            return idx
            
        # Fallback to linear search for irregular/unordered meshes
        for idx_lin, elem in enumerate(self.elements):
            x1 = coords[elem.node1, 0]
            x2 = coords[elem.node2, 0]
            if min(x1, x2) <= x <= max(x1, x2):
                return idx_lin
        
        # Tolerance check for precision at ends
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        if abs(x - x_min) < 1e-9: return 0
        if abs(x - x_max) < 1e-9: return self.num_elements - 1
            
        return -1

    def __str__(self):
        return f"Mesh: {self.num_nodes} nodes, {self.num_elements} elements"


class MeshGenerator:
    """Utilities for generating finite element meshes."""
    
    @staticmethod
    def line_mesh(start: Tuple[float, float], end: Tuple[float, float],
                  num_elements: int) -> 'Mesh':
        """
        Generate a uniform mesh along a straight line.
        
        Parameters:
        -----------
        start : tuple
            Starting coordinates (x, y)
        end : tuple
            Ending coordinates (x, y)
        num_elements : int
            Number of elements
            
        Returns:
        --------
        mesh : Mesh
            Generated mesh
        """
        mesh = Mesh()
        
        # Generate nodes
        x_coords = np.linspace(start[0], end[0], num_elements + 1)
        y_coords = np.linspace(start[1], end[1], num_elements + 1)
        
        for x, y in zip(x_coords, y_coords):
            mesh.add_node(x, y)
        
        # Generate elements
        for i in range(num_elements):
            mesh.add_element(i, i + 1)
        
        return mesh
    
    @staticmethod
    def beam_mesh_1d(length: float, num_elements: int,
                     orientation: str = 'horizontal') -> 'Mesh':
        """
        Generate a 1D beam mesh (horizontal or vertical).
        
        Parameters:
        -----------
        length : float
            Total beam length (mm)
        num_elements : int
            Number of elements
        orientation : str
            'horizontal' or 'vertical'
            
        Returns:
        --------
        mesh : Mesh
            Generated mesh
        """
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        if num_elements < 1:
            raise ValueError(f"Number of elements must be at least 1, got {num_elements}")
        
        if orientation.lower() == 'horizontal':
            start = (0, 0)
            end = (length, 0)
        elif orientation.lower() == 'vertical':
            start = (0, 0)
            end = (0, length)
        else:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'")
        
        return MeshGenerator.line_mesh(start, end, num_elements)
    
    @staticmethod
    def graded_mesh(start: Tuple[float, float], end: Tuple[float, float],
                    num_elements: int, grading_ratio: float = 1.0) -> 'Mesh':
        """
        Generate a mesh with graded element sizes.
        
        Parameters:
        -----------
        start : tuple
            Starting coordinates (x, y)
        end : tuple
            Ending coordinates (x, y)
        num_elements : int
            Number of elements
        grading_ratio : float
            Ratio of last element size to first element size
            > 1: elements get larger
            < 1: elements get smaller
            = 1: uniform mesh
            
        Returns:
        --------
        mesh : Mesh
            Generated mesh
        """
        mesh = Mesh()
        
        if grading_ratio == 1.0:
            # Uniform mesh
            return MeshGenerator.line_mesh(start, end, num_elements)
        
        # Calculate geometric series spacing
        total_length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        # Solve for first element size
        if grading_ratio != 1.0:
            L1 = total_length * (grading_ratio - 1) / (grading_ratio**num_elements - 1)
        else:
            L1 = total_length / num_elements
        
        # Generate node positions
        positions = [0.0]
        for i in range(num_elements):
            positions.append(positions[-1] + L1 * grading_ratio**i)
        
        # Normalize to actual length
        positions = np.array(positions) * (total_length / positions[-1])
        
        # Convert to x, y coordinates
        direction = np.array([end[0] - start[0], end[1] - start[1]]) / total_length
        
        for pos in positions:
            x = start[0] + direction[0] * pos
            y = start[1] + direction[1] * pos
            mesh.add_node(x, y)
        
        # Generate elements
        for i in range(num_elements):
            mesh.add_element(i, i + 1)
        
        return mesh
    
    @staticmethod
    def path_mesh(points: List[Tuple[float, float]],
                  elements_per_segment: Union[int, List[int]],
                  grading_ratios: Optional[Union[float, List[float]]] = None) -> 'Mesh':
        """
        Generate a 2D mesh along a multi-point path (angled beams).

        Parameters:
        -----------
        points : list of (x, y) tuples
            Waypoints for the beam structure.
        elements_per_segment : int or list of int
            Number of elements in each segment between points.
        grading_ratios : float or list of float, optional
            Grading ratio for each segment.

        Returns:
        --------
        mesh : Mesh
            The generated 2D beam mesh. `mesh.waypoint_nodes` contains IDs.
        """
        if len(points) < 2:
            raise ValueError("Path must have at least 2 points.")

        num_segments = len(points) - 1

        # Standardize inputs to lists
        if isinstance(elements_per_segment, int):
            elements_per_segment = [elements_per_segment] * num_segments
        if len(elements_per_segment) != num_segments:
            raise ValueError(f"Expected {num_segments} element counts, got {len(elements_per_segment)}")

        if grading_ratios is None:
            grading_ratios = [1.0] * num_segments
        elif isinstance(grading_ratios, float):
            grading_ratios = [grading_ratios] * num_segments

        mesh = Mesh()

        # Add first waypoint
        first_id = mesh.add_node(points[0][0], points[0][1])
        mesh.waypoint_nodes.append(first_id)

        current_node_id = first_id

        for i in range(num_segments):
            p1, p2 = points[i], points[i+1]
            num_elem = elements_per_segment[i]
            ratio = grading_ratios[i]

            # Use graded_mesh logic for each segment
            seg_mesh = MeshGenerator.graded_mesh(p1, p2, num_elem, ratio)

            # Merge nodes and elements (skipping the first node of seg_mesh as it's the current_node_id)
            # seg_mesh nodes: [0, 1, ..., num_elem]
            # ID mapping for new nodes
            node_map = {0: current_node_id}

            for j in range(1, seg_mesh.num_nodes):
                n = seg_mesh.nodes[j]
                node_map[j] = mesh.add_node(n.x, n.y)

            for e in seg_mesh.elements:
                mesh.add_element(node_map[e.node1], node_map[e.node2])

            current_node_id = node_map[seg_mesh.num_nodes - 1]
            mesh.waypoint_nodes.append(current_node_id)

        return mesh

    @staticmethod
    def multi_span_beam(span_lengths: List[Union[float, Tuple[float, float]]],
                        elements_per_span: List[int],
                        orientation: str = 'horizontal') -> 'Mesh':
        """
        Generate mesh for a multi-span continuous beam.

        Parameters:
        -----------
        span_lengths : list of float or tuple
            If float: Length of each span (mm). Orientation applies.
            If tuple: (dx, dy) relative vector for each span.
        elements_per_span : list of int
            Number of elements in each span.
        orientation : str
            'horizontal' or 'vertical' (only used if span_lengths are floats).

        Returns:
        --------
        mesh : Mesh
            Generated mesh. `mesh.waypoint_nodes` contains support IDs.
        """
        if len(span_lengths) != len(elements_per_span):
            raise ValueError("span_lengths and elements_per_span must have same length")

        # Convert span definitions to waypoints
        points = [(0.0, 0.0)]
        curr_x, curr_y = 0.0, 0.0

        for val in span_lengths:
            if isinstance(val, (int, float)):
                if orientation.lower() == 'horizontal':
                    curr_x += val
                else:
                    curr_y += val
            elif isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
                curr_x += val[0]
                curr_y += val[1]
            else:
                raise TypeError(f"Invalid span definition: {val}. Expected float or (dx, dy).")

            points.append((curr_x, curr_y))

        return MeshGenerator.path_mesh(points, elements_per_span)
    
    @staticmethod
    def curved_beam(radius: float, start_angle: float, end_angle: float,
                    num_elements: int) -> 'Mesh':
        """
        Generate mesh for a curved beam.
        
        Parameters:
        -----------
        radius : float
            Radius of curvature (mm)
        start_angle : float
            Starting angle (degrees)
        end_angle : float
            Ending angle (degrees)
        num_elements : int
            Number of elements
            
        Returns:
        --------
        mesh : Mesh
            Generated mesh
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        if num_elements < 1:
            raise ValueError(f"Number of elements must be at least 1, got {num_elements}")
        
        mesh = Mesh()
        
        # Convert to radians
        theta_start = np.radians(start_angle)
        theta_end = np.radians(end_angle)
        
        # Generate nodes along arc
        theta = np.linspace(theta_start, theta_end, num_elements + 1)
        
        for angle in theta:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            mesh.add_node(x, y)
        
        # Generate elements
        for i in range(num_elements):
            mesh.add_element(i, i + 1)
        
        return mesh


class MeshRefinement:
    """Mesh refinement utilities."""
    
    @staticmethod
    def refine_uniform(mesh: 'Mesh', refinement_level: int = 1) -> 'Mesh':
        """
        Uniformly refine mesh by splitting each element.
        
        Parameters:
        -----------
        mesh : Mesh
            Original mesh
        refinement_level : int
            Number of refinement iterations
            
        Returns:
        --------
        refined_mesh : Mesh
            Refined mesh
        """
        current_mesh = mesh
        
        for _ in range(refinement_level):
            new_mesh = Mesh()
            
            # Copy existing nodes
            for node in current_mesh.nodes:
                new_mesh.add_node(node.x, node.y, node.z)
            
            # Split each element
            for elem in current_mesh.elements:
                n1 = current_mesh.nodes[elem.node1]
                n2 = current_mesh.nodes[elem.node2]
                
                # Midpoint
                mid_x = (n1.x + n2.x) / 2
                mid_y = (n1.y + n2.y) / 2
                mid_z = (n1.z + n2.z) / 2
                
                mid_node = new_mesh.add_node(mid_x, mid_y, mid_z)
                
                # Create two new elements
                new_mesh.add_element(elem.node1, mid_node)
                new_mesh.add_element(mid_node, elem.node2)
            
            current_mesh = new_mesh
        
        return current_mesh
    
    @staticmethod
    def get_element_sizes(mesh: Mesh) -> np.ndarray:
        """
        Calculate size of each element.
        
        Parameters:
        -----------
        mesh : Mesh
            Mesh object
            
        Returns:
        --------
        sizes : np.ndarray
            Array of element sizes
        """
        sizes = []
        for elem in mesh.elements:
            length = elem.length(mesh.nodes)
            sizes.append(length)
        return np.array(sizes)


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("MESH GENERATION EXAMPLES")
    print("="*70)
    
    # Example 1: Simple beam
    print("\n1. Simple horizontal beam:")
    mesh1 = MeshGenerator.beam_mesh_1d(length=1000, num_elements=10)
    print(f"   {mesh1}")
    
    # Example 2: Multi-span beam
    print("\n2. Multi-span continuous beam:")
    mesh2 = MeshGenerator.multi_span_beam(
        span_lengths=[500, 800, 500],
        elements_per_span=[5, 8, 5]
    )
    print(f"   {mesh2}")
    print(f"   Total length: {mesh2.nodes[-1].x} mm")
    
    # Example 3: Graded mesh
    print("\n3. Graded mesh (fine at start):")
    mesh3 = MeshGenerator.graded_mesh((0, 0), (1000, 0), num_elements=10, grading_ratio=2.0)
    sizes = MeshRefinement.get_element_sizes(mesh3)
    print(f"   {mesh3}")
    print(f"   Element sizes: {sizes[0]:.2f} to {sizes[-1]:.2f} mm")
    
    # Example 4: Uniform refinement
    print("\n4. Uniform refinement:")
    mesh4 = MeshGenerator.beam_mesh_1d(length=100, num_elements=5)
    print(f"   Original: {mesh4}")
    mesh4_refined = MeshRefinement.refine_uniform(mesh4, refinement_level=2)
    print(f"   After 2 refinements: {mesh4_refined}")
