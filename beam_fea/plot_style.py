"""
plot_style.py
=============
Standardised plot settings and unit-scaling utilities for Beam FEA visualizations.

Provides:
- PlotStyle  : dataclass with colours, sizes, and line settings used by all plots
- smart_units: helper that chooses a human-readable scale + unit label
               based on magnitude (e.g. 'mm' vs 'm', 'N' vs 'kN', 'N·mm' vs 'kN·m')
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PlotStyle:
    """
    Central repository for matplotlib styling constants used across all
    Beam FEA plots.  Import once and pass to a plotting function, or use
    the module-level default instance ``DEFAULT_STYLE``.

    Colours follow an accessible palette with distinct roles:
    - primary (blue)        – shear force, structure beam line
    - moment (green)        – bending moment fill
    - max (red)             – peak-value markers and dotted lines
    - bc (dark green)       – boundary condition symbols
    - load (dark red)       – transverse point-load arrows
    - load_fx (orange)      – axial point-load arrows
    - udl (purple)          – distributed load combs
    - moment_load (amber)   – concentrated moment arcs
    """

    # --- Figure sizes ---------------------------------------------------
    dpi: int = 150
    figsize_wide: Tuple[int, int] = field(default_factory=lambda: (14, 5))
    figsize_structure: Tuple[int, int] = field(default_factory=lambda: (14, 6))
    figsize_square: Tuple[int, int] = field(default_factory=lambda: (8, 8))

    # --- Line / fill weights --------------------------------------------
    line_width: float = 3.0
    beam_line_width: float = 3.0
    fill_alpha: float = 0.3
    grid_alpha: float = 0.3
    
    # --- Component Line Weights -----------------------------------------
    bc_line_width: float = 2.5
    bc_hatch_width: float = 1.0
    load_line_width: float = 2.25
    load_mutation_scale: float = 20.0
    reaction_line_width: float = 2.25
    reaction_mutation_scale: float = 20.0

    # --- Font sizes & Positioning ---------------------------------------
    title_fontsize: int = 18
    label_fontsize: int = 12
    tick_fontsize: int = 12
    annotation_fontsize: int = 12
    
    # Text annotation offsets (multipliers of arrow_scale or radius)
    annot_offset_fy_vert: float = 1.0    # Vertical offset for Fy labels
    annot_offset_fy_horz: float = 1.5   # Horizontal offset from arrow shaft
    annot_offset_fx_vert: float = 0.2    # Vertical offset from horz arrow shaft
    annot_offset_fx_horz: float = 0.5    # Horizontal gap from tip
    annot_offset_mz: float = 1.5         # Radial offset for moment labels
    annot_offset_udl: float = 0.25       # Vertical offset for UDL labels

    # --- Colours --------------------------------------------------------
    colour_primary: str = "#2563EB"       # blue  – shear / beam
    colour_moment: str = "#16A34A"        # green – bending moment
    colour_zero_line: str = "#111827"     # near-black – datum lines
    colour_max: str = "#DC2626"           # red   – max marker & dotted line

    colour_bc: str = "#15803D"            # dark green – support glyphs
    colour_bc_hatch: str = "#166534"      # slightly darker for hatch
    colour_load: str = "#B91C1C"          # dark red   – Fy arrows
    colour_load_fx: str = "#EA580C"       # orange     – Fx arrows
    colour_udl: str = "#7C3AED"           # purple     – distributed load
    colour_moment_load: str = "#D97706"   # amber      – concentrated moment arcs

    colour_deformed: str = "#DC2626"      # red   – deformed shape
    colour_undeformed: str = "#2563EB"    # blue  – undeformed (ghost)


# Module-level default instance – import and use directly
DEFAULT_STYLE = PlotStyle()


# ---------------------------------------------------------------------------
# Smart unit scaling
# ---------------------------------------------------------------------------

def smart_units(value_max: float, quantity: str) -> Tuple[float, str]:
    """
    Choose a human-readable scale factor and unit label for a given quantity.

    Parameters
    ----------
    value_max : float
        Maximum absolute value of the quantity (in base SI units used by the
        package: mm, N, N·mm, MPa).
    quantity : str
        One of ``'length'``, ``'force'``, ``'moment'``, ``'stress'``,
        ``'udl'`` (distributed load, N/mm).

    Returns
    -------
    scale : float
        Multiply raw values by ``1/scale`` before plotting.
    label : str
        Unit string for axis labels (e.g. ``'kN'``, ``'kN·m'``).

    Examples
    --------
    >>> scale, label = smart_units(15000, 'length')   # => (1000, 'm')
    >>> scale, label = smart_units(85000, 'force')    # => (1000, 'kN')
    >>> scale, label = smart_units(5e7, 'moment')     # => (1e6, 'kN·m')
    """
    abs_val = abs(value_max)

    if quantity == 'length':
        if abs_val >= 1_000:
            return 1_000.0, 'm'
        return 1.0, 'mm'

    elif quantity == 'force':
        if abs_val >= 1_000:
            return 1_000.0, 'kN'
        return 1.0, 'N'

    elif quantity == 'moment':
        if abs_val >= 1_000_000:
            return 1_000_000.0, 'kN·m'
        elif abs_val >= 1_000:
            return 1_000.0, 'kN·mm'
        return 1.0, 'N·mm'

    elif quantity == 'stress':
        return 1.0, 'MPa'

    elif quantity == 'udl':
        # N/mm  →  kN/m  (1 N/mm = 1 kN/m, so scale is 1 but label changes)
        if abs_val >= 1:
            return 1.0, 'kN/m'
        return 1.0, 'N/mm'

    else:
        return 1.0, ''


if __name__ == "__main__":
    print("PlotStyle module")
    style = PlotStyle()
    print(f"  Primary colour : {style.colour_primary}")
    print(f"  Default DPI    : {style.dpi}")
    print()
    tests = [
        (500, 'length'), (15000, 'length'),
        (800, 'force'), (85000, 'force'),
        (500, 'moment'), (2e5, 'moment'), (8e7, 'moment'),
        (150, 'stress'),
        (0.5, 'udl'), (2.5, 'udl'),
    ]
    for val, qty in tests:
        scale, label = smart_units(val, qty)
        print(f"  smart_units({val:>10}, {qty:<8})  →  ÷{scale:.0f}  [{label}]")
