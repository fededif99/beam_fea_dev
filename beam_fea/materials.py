"""
materials.py
============
Material property definitions for structural analysis.

Contains predefined materials and a Material class for custom materials.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Material:
    """
    Material properties for structural analysis.
    
    Attributes:
    -----------
    name : str
        Material name/identifier
    E : float
        Young's modulus (MPa)
    G : float, optional
        Shear modulus (MPa), calculated from E and nu if not provided
    nu : float
        Poisson's ratio (dimensionless)
    rho : float
        Density (kg/mm³)
    yield_strength : float, optional
        Yield strength (MPa)
    ultimate_strength : float, optional
        Ultimate tensile strength (MPa)
    """
    
    name: str
    E: float
    nu: float
    rho: float
    G: Optional[float] = None
    yield_strength: Optional[float] = None
    ultimate_strength: Optional[float] = None
    
    def __post_init__(self):
        """Calculate shear modulus if not provided and validate inputs."""
        # Validate Young's modulus
        if self.E <= 0:
            raise ValueError(f"Young's modulus E must be positive, got {self.E}")
        
        # Validate Poisson's ratio (must be between -1 and 0.5 for isotropic materials)
        if not (-1.0 <= self.nu <= 0.5):
            raise ValueError(f"Poisson's ratio must be between -1 and 0.5, got {self.nu}")
        
        # Validate density
        if self.rho <= 0:
            raise ValueError(f"Density rho must be positive, got {self.rho}")
        
        # Validate shear modulus if provided
        if self.G is not None and self.G <= 0:
            raise ValueError(f"Shear modulus G must be positive, got {self.G}")
        
        # Calculate shear modulus if not provided
        if self.G is None:
            self.G = self.E / (2 * (1 + self.nu))
        
        # Validate strength values if provided
        if self.yield_strength is not None and self.yield_strength <= 0:
            raise ValueError(f"Yield strength must be positive, got {self.yield_strength}")
        
        if self.ultimate_strength is not None and self.ultimate_strength <= 0:
            raise ValueError(f"Ultimate strength must be positive, got {self.ultimate_strength}")
        
        # Check that ultimate strength >= yield strength if both provided
        if (self.yield_strength is not None and self.ultimate_strength is not None 
            and self.ultimate_strength < self.yield_strength):
            raise ValueError(f"Ultimate strength ({self.ultimate_strength}) must be >= yield strength ({self.yield_strength})")
    
    def get_sectional_stiffness(self, section) -> dict:
        """
        Calculate sectional stiffness for isotropic material.

        Parameters:
        -----------
        section : SectionProperties
            Cross-section properties

        Returns:
        --------
        stiffness : dict
            {'EA': float, 'ES': 0.0, 'EI': float, 'GA_s': float}
        """
        EA = self.E * section.A
        ES = 0.0
        EI = self.E * section.Iy
        # Note: Shear correction factor (kappa) is applied at the element/solver level
        GA_s = self.G * section.A

        return {'EA': EA, 'ES': ES, 'EI': EI, 'GA_s': GA_s}

    def get_linear_density(self, section) -> float:
        """
        Calculate mass per unit length.

        Parameters:
        -----------
        section : SectionProperties
            Cross-section properties

        Returns:
        --------
        rho_lin : float
            Linear mass density (kg/mm)
        """
        return self.rho * section.A

    def __str__(self):
        return (f"Material: {self.name}\n"
                f"  E = {self.E:.2e} MPa\n"
                f"  G = {self.G:.2e} MPa\n"
                f"  ν = {self.nu:.3f}\n"
                f"  ρ = {self.rho:.2e} kg/mm³")


# ============================================================================
# STRUCTURAL STEELS (Generic & Standards-Based)
# ============================================================================

# Generic structural steel
STEEL = Material(
    name="Steel (Generic Structural)",
    E=200000,
    nu=0.30,
    rho=7.85e-6,
    yield_strength=250,
    ultimate_strength=400
)

# ASTM Standards
STEEL_A36 = Material(
    name="Steel ASTM A36",
    E=200000,
    nu=0.30,
    rho=7.85e-6,
    yield_strength=250,
    ultimate_strength=400
)

STEEL_A572_GR50 = Material(
    name="Steel ASTM A572 Grade 50",
    E=200000,
    nu=0.30,
    rho=7.85e-6,
    yield_strength=345,
    ultimate_strength=450
)

# High-strength steels
STEEL_4340 = Material(
    name="Steel AISI 4340 (Alloy, Quenched & Tempered)",
    E=205000,
    nu=0.29,
    rho=7.85e-6,
    yield_strength=710,
    ultimate_strength=1110
)

STEEL_300M = Material(
    name="Steel 300M (Ultra High-Strength)",
    E=205000,
    nu=0.29,
    rho=7.87e-6,
    yield_strength=1520,
    ultimate_strength=1900
)

# Stainless steels
STEEL_STAINLESS_304 = Material(
    name="Stainless Steel 304 (Austenitic)",
    E=193000,
    nu=0.29,
    rho=8.00e-6,
    yield_strength=215,
    ultimate_strength=505
)

STEEL_STAINLESS_17_4PH = Material(
    name="Stainless Steel 17-4 PH (Precipitation Hardened)",
    E=196500,
    nu=0.27,
    rho=7.80e-6,
    yield_strength=1172,
    ultimate_strength=1310
)

# ============================================================================
# ALUMINUM ALLOYS (Generic & Standards-Based)
# ============================================================================

# Generic aluminum
ALUMINUM = Material(
    name="Aluminum (Generic)",
    E=69000,
    nu=0.33,
    rho=2.70e-6,
    yield_strength=276,
    ultimate_strength=310
)

# 6000 series (Mg-Si, good formability)
ALUMINUM_6061_T6 = Material(
    name="Aluminum 6061-T6 (AMS-QQ-A-250/11)",
    E=68900,
    nu=0.33,
    rho=2.70e-6,
    yield_strength=276,
    ultimate_strength=310
)

# 7000 series (Zn-Mg, aerospace grade)
ALUMINUM_7075_T6 = Material(
    name="Aluminum 7075-T6 (AMS-QQ-A-250/12)",
    E=71700,
    nu=0.33,
    rho=2.81e-6,
    yield_strength=503,
    ultimate_strength=572
)

ALUMINUM_7050_T7451 = Material(
    name="Aluminum 7050-T7451 (AMS 4050)",
    E=71700,
    nu=0.33,
    rho=2.83e-6,
    yield_strength=455,
    ultimate_strength=524
)

# 2000 series (Cu, aircraft structures)
ALUMINUM_2024_T3 = Material(
    name="Aluminum 2024-T3 (AMS-QQ-A-250/4)",
    E=73100,
    nu=0.33,
    rho=2.78e-6,
    yield_strength=345,
    ultimate_strength=483
)

# ============================================================================
# TITANIUM ALLOYS (Generic & Standards-Based)
# ============================================================================

# Generic titanium
TITANIUM = Material(
    name="Titanium (Generic)",
    E=110000,
    nu=0.34,
    rho=4.51e-6,
    yield_strength=830,
    ultimate_strength=900
)

# Ti-6Al-4V (most common aerospace titanium)
TITANIUM_TI_6AL_4V = Material(
    name="Titanium Ti-6Al-4V (Grade 5, AMS 4911)",
    E=113800,
    nu=0.34,
    rho=4.43e-6,
    yield_strength=880,
    ultimate_strength=950
)

# Ti-6Al-4V ELI (Extra Low Interstitial, higher toughness)
TITANIUM_TI_6AL_4V_ELI = Material(
    name="Titanium Ti-6Al-4V ELI (Grade 23, AMS 4907)",
    E=113800,
    nu=0.34,
    rho=4.43e-6,
    yield_strength=795,
    ultimate_strength=860
)

# Commercially pure titanium
TITANIUM_CP_GRADE2 = Material(
    name="Titanium CP Grade 2 (AMS 4921)",
    E=102700,
    nu=0.34,
    rho=4.51e-6,
    yield_strength=275,
    ultimate_strength=345
)

# ============================================================================
# COMPOSITES
# ============================================================================

CARBON_FIBER_EPOXY = Material(
    name="Carbon Fiber/Epoxy Composite (UD, 0°)",
    E=150000,
    nu=0.30,
    rho=1.60e-6,
    yield_strength=1500,
    ultimate_strength=2000
)

CARBON_FIBER_EPOXY_QI = Material(
    name="Carbon Fiber/Epoxy Quasi-Isotropic [0/±45/90]",
    E=55000,
    nu=0.30,
    rho=1.60e-6,
    yield_strength=600,
    ultimate_strength=800
)

GLASS_FIBER_EPOXY = Material(
    name="Glass Fiber/Epoxy Composite (UD, 0°)",
    E=45000,
    nu=0.28,
    rho=2.00e-6,
    yield_strength=1000,
    ultimate_strength=1500
)

# ============================================================================
# SPECIAL ALLOYS
# ============================================================================

INCONEL_718 = Material(
    name="Inconel 718 (AMS 5662, Ni-Cr Superalloy)",
    E=200000,
    nu=0.29,
    rho=8.19e-6,
    yield_strength=1035,
    ultimate_strength=1275
)

MAGNESIUM_AZ31B = Material(
    name="Magnesium AZ31B (ASTM B90/B90M)",
    E=45000,
    nu=0.35,
    rho=1.77e-6,
    yield_strength=220,
    ultimate_strength=290
)

# ============================================================================
# MATERIAL LIBRARY DICTIONARY
# ============================================================================

MATERIAL_LIBRARY = {
    # Generic materials
    'steel': STEEL,
    'aluminum': ALUMINUM,
    'aluminium': ALUMINUM,
    'titanium': TITANIUM,
    
    # Steels - ASTM
    'steel_a36': STEEL_A36,
    'steel_a572': STEEL_A572_GR50,
    'steel_4340': STEEL_4340,
    'steel_300m': STEEL_300M,
    
    # Stainless steels
    'stainless_304': STEEL_STAINLESS_304,
    'stainless_17_4ph': STEEL_STAINLESS_17_4PH,
    
    # Aluminum alloys
    'aluminum_6061': ALUMINUM_6061_T6,
    'aluminum_7075': ALUMINUM_7075_T6,
    'aluminum_7050': ALUMINUM_7050_T7451,
    'aluminum_2024': ALUMINUM_2024_T3,
    'aluminium_6061': ALUMINUM_6061_T6,
    'aluminium_7075': ALUMINUM_7075_T6,
    
    # Titanium alloys
    'titanium_ti64': TITANIUM_TI_6AL_4V,
    'ti_6al_4v': TITANIUM_TI_6AL_4V,
    'titanium_grade5': TITANIUM_TI_6AL_4V,
    'titanium_grade2': TITANIUM_CP_GRADE2,
    
    # Composites
    'carbon_fiber': CARBON_FIBER_EPOXY,
    'carbon_fiber_qi': CARBON_FIBER_EPOXY_QI,
    'glass_fiber': GLASS_FIBER_EPOXY,
    
    # Special alloys
    'inconel_718': INCONEL_718,
    'magnesium_az31b': MAGNESIUM_AZ31B,
}


def get_material(name: str) -> Material:
    """
    Get a predefined material by name.
    
    Parameters:
    -----------
    name : str
        Material name (case insensitive)
        
    Returns:
    --------
    material : Material
        Material object
        
    Raises:
    -------
    ValueError
        If material name not found
    """
    name_lower = name.lower().replace(' ', '_').replace('-', '_')
    
    if name_lower not in MATERIAL_LIBRARY:
        available = ', '.join(sorted(MATERIAL_LIBRARY.keys()))
        raise ValueError(f"Material '{name}' not found. Available: {available}")
    
    return MATERIAL_LIBRARY[name_lower]


def list_materials():
    """Print all available materials."""
    print("\n" + "="*80)
    print("AVAILABLE MATERIALS - AEROSPACE & MECHANICAL ENGINEERING")
    print("="*80)
    
    categories = {
        'Generic Materials': ['steel', 'aluminum', 'titanium'],
        'Structural Steels (ASTM)': ['steel_a36', 'steel_a572', 'steel_4340', 'steel_300m'],
        'Stainless Steels': ['stainless_304', 'stainless_17_4ph'],
        'Aluminum Alloys (AMS/ASTM)': ['aluminum_6061', 'aluminum_7075', 'aluminum_7050', 'aluminum_2024'],
        'Titanium Alloys (AMS)': ['titanium_ti64', 'titanium_grade2'],
        'Composites': ['carbon_fiber', 'carbon_fiber_qi', 'glass_fiber'],
        'Special Alloys': ['inconel_718', 'magnesium_az31b']
    }
    
    for category, materials in categories.items():
        print(f"\n{category}:")
        for mat_key in materials:
            mat = MATERIAL_LIBRARY[mat_key]
            yield_str = f"σy = {mat.yield_strength:5.0f}" if mat.yield_strength else "       "
            print(f"  {mat_key:25s} E = {mat.E:8.0f} MPa, ρ = {mat.rho:.2e} kg/mm³, {yield_str} MPa")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Demonstration
    list_materials()
    
    print("\nExample usage:")
    steel = get_material('steel')
    print(steel)
