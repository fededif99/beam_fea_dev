"""
properties.py
=============
Property collector for materials and cross-sections.
Allows assigning specific properties to elements or groups of elements.
"""

import numpy as np
from typing import Dict, Optional, Union, List
from .materials import Material
from .cross_sections import SectionProperties

class PropertySet:
    """
    A collector for material and cross-section properties across the beam.
    
    Acts as both a single property assignment and a container for multiple 
    assignments (similar to LoadCase or BoundaryConditionSet).
    """

    def __init__(self, material: Optional[Material] = None,
                 section: Optional[SectionProperties] = None,
                 elements: Optional[Union[int, List[int], range]] = None,
                 name: str = "Property Set"):
        """
        Initialize property set.

        Parameters:
        -----------
        material : Material, optional
            The material assigned to these elements.
        section : SectionProperties, optional
            The cross-section assigned to these elements.
        elements : int, list, or range, optional
            Element ID(s) this set applies to. If None, it implies a global 
            fallback/default set for the whole beam.
        name : str
            Identifier for this property collector.
        """
        self.name = name
        self._assignments = []
        
        # Internal resolved state
        self._mat_map = {}
        self._sec_map = {}
        self._is_resolved = False

        # Pre-calculated element arrays
        self.EA = None
        self.EI = None
        self.ES = None
        self.GA_s = None
        self.rho_lin = None

        # If initialized with arguments, add them as the first assignment
        if material is not None or section is not None:
            self.add(material, section, elements)

    def add(self, material: Optional[Material] = None,
            section: Optional[SectionProperties] = None,
            elements: Optional[Union[int, List[int], range]] = None):
        """
        Add a property assignment to the set.

        Parameters:
        -----------
        material : Material, optional
        section : SectionProperties, optional
        elements : int, list, or range, optional
            If None, applies to all elements (global default).
        """
        self._assignments.append({
            'material': material,
            'section': section,
            'elements': elements
        })
        self._is_resolved = False

    def resolve(self, num_elements: int):
        """
        Pre-calculate mapping for all elements and validate coverage.
        
        Precedence: Later assignments in the set override earlier ones.
        """
        self._mat_map = {}
        self._sec_map = {}
        
        for assignment in self._assignments:
            material = assignment['material']
            section = assignment['section']
            elements = assignment['elements']
            
            e_ids = elements if elements is not None else range(num_elements)
            # Handle single int or range
            if isinstance(e_ids, int):
                e_ids = [e_ids]
            elif isinstance(e_ids, range):
                e_ids = list(e_ids)
                
            for eid in e_ids:
                if material is not None:
                    self._mat_map[eid] = material
                if section is not None:
                    self._sec_map[eid] = section
        
        self.validate(num_elements)
        
        # Pre-calculate and cache element properties
        self.EA = np.zeros(num_elements)
        self.EI = np.zeros(num_elements)
        self.ES = np.zeros(num_elements)
        self.GA_s = np.zeros(num_elements)
        self.rho_lin = np.zeros(num_elements)
        
        for i in range(num_elements):
            mat = self._mat_map[i]
            sec = self._sec_map[i]
            
            stiff = mat.get_sectional_stiffness(sec)
            self.EA[i] = stiff['EA']
            self.EI[i] = stiff['EI']
            self.ES[i] = stiff['ES']
            self.GA_s[i] = stiff['GA_s'] * sec.shear_factor
            self.rho_lin[i] = mat.get_linear_density(sec)
            
            # Cross-check Laminate thickness vs Section height/width
            from .composites import Laminate
            if isinstance(mat, Laminate):
                h_sec = sec.y_top - sec.y_bottom
                w_sec = sec.z_right - sec.z_left

                # Check height
                if not np.isclose(mat.total_thickness, h_sec, rtol=1e-3):
                    import warnings
                    warnings.warn(
                        f"Element {i}: Laminate '{mat.name}' thickness ({mat.total_thickness:.2f} mm) "
                        f"mismatches Section height ({h_sec:.2f} mm). This may yield inconsistent results."
                    )

                # Check area consistency (width * height vs actual A)
                # Significant deviation suggests non-rectangular geometry where 1D CLT is less accurate.
                theoretical_area = w_sec * h_sec
                if not np.isclose(theoretical_area, sec.A, rtol=0.05):
                    import warnings
                    warnings.warn(
                        f"Element {i}: Section '{sec.name}' is non-rectangular (Area deviation > 5%). "
                        f"Laminate calculations using robust width ({w_sec:.2f} mm) may be approximate."
                    )

        self._is_resolved = True

    def validate(self, num_elements: int):
        """
        Ensure every element has both a material and a section assigned.
        """
        missing_mats = []
        missing_secs = []
        
        for i in range(num_elements):
            if i not in self._mat_map:
                missing_mats.append(i)
            if i not in self._sec_map:
                missing_secs.append(i)
                
        if missing_mats or missing_secs:
            msg = f"Property Assignment Incomplete in '{self.name}':\n"
            if missing_mats:
                msg += f" - Elements missing Material: {self._format_list(missing_mats)}\n"
            if missing_secs:
                msg += f" - Elements missing Section:  {self._format_list(missing_secs)}\n"
            raise ValueError(msg)

    def _format_list(self, ids: List[int], max_show: int = 5) -> str:
        if len(ids) <= max_show:
            return str(ids)
        return f"{ids[:max_show]}... (+{len(ids)-max_show} more)"

    def get_material(self, element_id: int) -> Material:
        if not self._is_resolved:
            raise RuntimeError("PropertySet must be resolved via solver before access.")
        return self._mat_map[element_id]

    def get_section(self, element_id: int) -> SectionProperties:
        if not self._is_resolved:
            raise RuntimeError("PropertySet must be resolved via solver before access.")
        return self._sec_map[element_id]
        
    def has_multiple_sections(self, num_elements: int) -> bool:
        """Check if different section objects are used across the structure."""
        if num_elements <= 1:
            return False
            
        first_sec = self.get_section(0)
        for i in range(1, num_elements):
            if self.get_section(i) is not first_sec:
                return True
        return False

    def has_multiple_materials(self, num_elements: int) -> bool:
        """Check if different material objects are used across the structure."""
        if num_elements <= 1:
            return False
            
        first_mat = self.get_material(0)
        for i in range(1, num_elements):
            if self.get_material(i) is not first_mat:
                return True
        return False

    def is_uniform(self, num_elements: int) -> bool:
        """Check if the structure has uniform properties everywhere."""
        return not (self.has_multiple_sections(num_elements) or 
                    self.has_multiple_materials(num_elements))

    @property
    def default_material(self):
        # Compatibility helper (returns material of first element)
        return self.get_material(0) if self._mat_map else None

    @property
    def default_section(self):
        # Compatibility helper (returns section of first element)
        return self.get_section(0) if self._sec_map else None
