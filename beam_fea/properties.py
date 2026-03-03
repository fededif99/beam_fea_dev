"""
properties.py
=============
Property collector for materials and cross-sections.
Allows assigning specific properties to elements or groups of elements.
"""

from typing import Dict, Optional, Union, List
from .materials import Material
from .cross_sections import SectionProperties

class PropertySet:
    """
    A collector for material and section properties across the beam.

    Supports global defaults and per-element overrides.
    """

    def __init__(self, default_material: Optional[Material] = None,
                 default_section: Optional[SectionProperties] = None):
        """
        Initialize property set.

        Parameters:
        -----------
        default_material : Material, optional
            Global default material
        default_section : SectionProperties, optional
            Global default section
        """
        self.default_material = default_material
        self.default_section = default_section

        # Mapping: element_id -> Material
        self.material_overrides: Dict[int, Material] = {}

        # Mapping: element_id -> SectionProperties
        self.section_overrides: Dict[int, SectionProperties] = {}

    def assign_material(self, material: Material, elements: Optional[Union[int, List[int]]] = None):
        """
        Assign a material to specific elements.

        Parameters:
        -----------
        material : Material
            The material to assign
        elements : int or list, optional
            Element ID(s). If None, sets the global default.
        """
        if elements is None:
            self.default_material = material
        else:
            e_ids = [elements] if isinstance(elements, int) else elements
            for eid in e_ids:
                self.material_overrides[eid] = material

    def assign_section(self, section: SectionProperties, elements: Optional[Union[int, List[int]]] = None):
        """
        Assign a section to specific elements.

        Parameters:
        -----------
        section : SectionProperties
            The section properties to assign
        elements : int or list, optional
            Element ID(s). If None, sets the global default.
        """
        if elements is None:
            self.default_section = section
        else:
            e_ids = [elements] if isinstance(elements, int) else elements
            for eid in e_ids:
                self.section_overrides[eid] = section

    def get_material(self, element_id: int) -> Material:
        """Get material for a specific element."""
        mat = self.material_overrides.get(element_id, self.default_material)
        if mat is None:
            raise ValueError(f"No material defined for element {element_id} and no default set.")
        return mat

    def get_section(self, element_id: int) -> SectionProperties:
        """Get section for a specific element."""
        sec = self.section_overrides.get(element_id, self.default_section)
        if sec is None:
            raise ValueError(f"No section defined for element {element_id} and no default set.")
        return sec

    def has_multiple_sections(self, num_elements: int) -> bool:
        """Check if different sections are used across the structure."""
        if not self.section_overrides:
            return False

        # If any element has an override different from default, return True
        first_sec = self.get_section(0)
        for i in range(1, num_elements):
            if self.get_section(i) is not first_sec:
                return True
        return False
