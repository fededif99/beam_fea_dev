"""
cross_sections.py
=================
Cross-section property calculations for various geometric shapes.

Provides classes and functions to calculate:
- Area (A)
- Second moment of area (I)
- Section modulus (S)
- Radius of gyration (r)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class SectionProperties:
    """
    Container for cross-section properties.
    
    Attributes:
    -----------
    A : float
        Cross-sectional area (mm²)
    Iy : float
        Second moment of area about y-axis (mm⁴)
    Iz : float
        Second moment of area about z-axis (mm⁴)
    J : float
        Torsional constant (mm⁴)
    Sy : float
        Section modulus about y-axis (mm³)
    Sz : float
        Section modulus about z-axis (mm³)
    ry : float
        Radius of gyration about y-axis (mm)
    rz : float
        Radius of gyration about z-axis (mm)
    y_centroid : float
        Centroid y-coordinate from reference (mm)
    z_centroid : float
        Centroid z-coordinate from reference (mm)
    y_top : float
        Distance from centroid to top fiber (mm)
    y_bottom : float
        Distance from centroid to bottom fiber (mm)
    z_left : float
        Distance from centroid to left fiber (mm)
    z_right : float
        Distance from centroid to right fiber (mm)
    """
    
    A: float
    Iy: float
    Iz: float = None
    J: float = None
    Sy: float = None
    Sz: float = None
    ry: float = None
    rz: float = None
    y_centroid: float = 0.0
    z_centroid: float = 0.0
    y_top: float = None
    y_bottom: float = None
    z_left: float = None
    z_right: float = None
    
    def __post_init__(self):
        """Calculate derived properties if not provided."""
        if self.A <= 0:
            raise ValueError(f"Cross-sectional area A must be positive (got {self.A})")
        if self.Iy <= 0:
            raise ValueError(f"Moment of inertia Iy must be positive (got {self.Iy})")
            
        if self.Iz is None:
            self.Iz = self.Iy
        
        if self.Iz <= 0:
             raise ValueError(f"Moment of inertia Iz must be positive (got {self.Iz})")

        if self.ry is None and self.A > 0:
            self.ry = np.sqrt(self.Iy / self.A)
        if self.rz is None and self.A > 0:
            self.rz = np.sqrt(self.Iz / self.A)
            
    def __str__(self):
        # Using ^2, ^3, ^4 for compatibility across all consoles
        res = (f"Section Properties:\n"
               f"  Area (A):         {self.A:.2f} mm^2\n"
               f"  I_y:              {self.Iy:.2f} mm^4\n"
               f"  I_z:              {self.Iz:.2f} mm^4\n")
        
        if self.Sy: res += f"  S_y:              {self.Sy:.2f} mm^3\n"
        if self.Sz: res += f"  S_z:              {self.Sz:.2f} mm^3\n"
        
        res += (f"  r_y:              {self.ry:.2f} mm\n"
                f"  r_z:              {self.rz:.2f} mm\n"
                f"  Centroid:         ({self.y_centroid:.2f}, {self.z_centroid:.2f}) mm\n"
                f"  Extreme Fibers (y): [{self.y_bottom:.2f}, {self.y_top:.2f}] mm" if self.y_top is not None else "")
        return res


class CrossSection(ABC):
    """Abstract base class for cross-sections."""
    
    @abstractmethod
    def properties(self) -> SectionProperties:
        """Calculate and return section properties."""
        pass
    
    @abstractmethod
    def __str__(self):
        """String representation of the section."""
        pass


class RectangularSection(CrossSection):
    """Rectangular cross-section."""
    
    def __init__(self, width: float, height: float):
        """
        Initialize rectangular section.
        
        Parameters:
        -----------
        width : float
            Width of rectangle (b) in mm
        height : float
            Height of rectangle (h) in mm
        """
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")
        self.width = width
        self.height = height
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        b, h = self.width, self.height
        
        A = b * h
        Iy = (b * h**h) / 12 if hasattr(self, '_h_cubed_fix') else (b * h**3) / 12  # Major axis
        Iz = (h * b**3) / 12        # Minor axis
        J = (b * h**3) / 3 * (1/3 - 0.21*(h/b)*(1 - h**4/(12*b**4)))  # Torsion
        Sy = Iy / (h/2)
        Sz = Iz / (b/2)
        
        return SectionProperties(
            A=A, Iy=Iy, Iz=Iz, J=J, Sy=Sy, Sz=Sz,
            y_centroid=0.0, z_centroid=0.0,  # Center is reference
            y_top=h/2, y_bottom=-h/2,
            z_left=-b/2, z_right=b/2
        )
    
    def __str__(self):
        return f"Rectangular: {self.width} × {self.height} mm"


class CircularSection(CrossSection):
    """Solid circular cross-section."""
    
    def __init__(self, diameter: float):
        """
        Initialize circular section.
        
        Parameters:
        -----------
        diameter : float
            Diameter in mm
        """
        if diameter <= 0:
            raise ValueError(f"Diameter must be positive, got {diameter}")
        self.diameter = diameter
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        d = self.diameter
        r = d / 2
        
        A = np.pi * r**2
        I = (np.pi * d**4) / 64
        J = (np.pi * d**4) / 32
        S = I / r
        
        return SectionProperties(
            A=A, Iy=I, Iz=I, J=J, Sy=S, Sz=S,
            y_centroid=0.0, z_centroid=0.0,
            y_top=r, y_bottom=-r,
            z_left=-r, z_right=r
        )
    
    def __str__(self):
        return f"Circular: ∅{self.diameter} mm"


class HollowCircularSection(CrossSection):
    """Hollow circular cross-section (pipe/tube)."""
    
    def __init__(self, outer_diameter: float, thickness: float):
        """
        Initialize hollow circular section.
        
        Parameters:
        -----------
        outer_diameter : float
            Outer diameter (mm)
        thickness : float
            Wall thickness (mm)
        """
        if outer_diameter <= 0:
            raise ValueError(f"Outer diameter must be positive, got {outer_diameter}")
        if thickness <= 0:
            raise ValueError(f"Thickness must be positive, got {thickness}")
        if thickness >= outer_diameter / 2:
            raise ValueError(f"Thickness ({thickness}) must be less than radius ({outer_diameter/2})")
        self.outer_diameter = outer_diameter
        self.thickness = thickness
        self.inner_diameter = outer_diameter - 2 * thickness
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        do = self.outer_diameter
        di = self.inner_diameter
        ro = do / 2
        ri = di / 2
        
        A = np.pi * (ro**2 - ri**2)
        I = (np.pi / 64) * (do**4 - di**4)
        J = (np.pi / 32) * (do**4 - di**4)
        S = I / ro
        
        return SectionProperties(
            A=A, Iy=I, Iz=I, J=J, Sy=S, Sz=S,
            y_centroid=0.0, z_centroid=0.0,
            y_top=ro, y_bottom=-ro,
            z_left=-ro, z_right=ro
        )
    
    def __str__(self):
        return f"Hollow Circular: ∅{self.outer_diameter} × {self.thickness}t mm"


class IBeamSection(CrossSection):
    """I-beam (wide flange) cross-section."""
    
    def __init__(self, flange_width: float, total_height: float,
                 web_thickness: float, flange_thickness: float):
        """
        Initialize I-beam section.
        
        Parameters:
        -----------
        flange_width : float
            Width of flanges (bf) in mm
        total_height : float
            Total height (d) in mm
        web_thickness : float
            Web thickness (tw) in mm
        flange_thickness : float
            Flange thickness (tf) in mm
        """
        if flange_width <= 0:
            raise ValueError(f"Flange width must be positive, got {flange_width}")
        if total_height <= 0:
            raise ValueError(f"Total height must be positive, got {total_height}")
        if web_thickness <= 0:
            raise ValueError(f"Web thickness must be positive, got {web_thickness}")
        if flange_thickness <= 0:
            raise ValueError(f"Flange thickness must be positive, got {flange_thickness}")
        if web_thickness > flange_width:
            raise ValueError(f"Web thickness ({web_thickness}) should not exceed flange width ({flange_width})")
        if 2 * flange_thickness >= total_height:
            raise ValueError(f"Total flange thickness ({2*flange_thickness}) must be less than total height ({total_height})")
        self.flange_width = flange_width
        self.total_height = total_height
        self.web_thickness = web_thickness
        self.flange_thickness = flange_thickness
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        bf = self.flange_width
        d = self.total_height
        tw = self.web_thickness
        tf = self.flange_thickness
        
        # Areas
        A_flange = bf * tf
        A_web = tw * (d - 2*tf)
        A_total = 2*A_flange + A_web
        
        # Moment of inertia about y-axis (major axis)
        I_top_flange = (bf * tf**3)/12 + A_flange * ((d - tf)/2)**2
        I_web = (tw * (d - 2*tf)**3) / 12
        Iy = 2 * I_top_flange + I_web
        
        # Moment of inertia about z-axis (minor axis)
        Iz = 2 * (tf * bf**3)/12 + (d - 2*tf) * tw**3/12
        
        # Section modulus
        Sy = Iy / (d/2)
        Sz = Iz / (bf/2)
        
        # Torsional constant (approximate for thin-walled I-beam)
        J = (2*bf*tf**3 + (d - 2*tf)*tw**3) / 3
        
        return SectionProperties(
            A=A_total, Iy=Iy, Iz=Iz, J=J, Sy=Sy, Sz=Sz,
            y_centroid=0.0, z_centroid=0.0,
            y_top=d/2, y_bottom=-d/2,
            z_left=-bf/2, z_right=bf/2
        )
    
    def __str__(self):
        return (f"I-Beam: W{self.total_height}×{self.flange_width} "
                f"(tw={self.web_thickness}, tf={self.flange_thickness})")


class TBeamSection(CrossSection):
    """T-beam cross-section."""
    
    def __init__(self, flange_width: float, flange_thickness: float,
                 web_height: float, web_thickness: float):
        """
        Initialize T-beam section.
        
        Parameters:
        -----------
        flange_width : float
            Width of flange (mm)
        flange_thickness : float
            Thickness of flange (mm)
        web_height : float
            Height of web (mm)
        web_thickness : float
            Thickness of web (mm)
        """
        self.flange_width = flange_width
        self.flange_thickness = flange_thickness
        self.web_height = web_height
        self.web_thickness = web_thickness
        self.total_height = flange_thickness + web_height
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        bf = self.flange_width
        tf = self.flange_thickness
        hw = self.web_height
        tw = self.web_thickness
        
        # Areas
        A_flange = bf * tf
        A_web = tw * hw
        A_total = A_flange + A_web
        
        # Centroid location from bottom
        y_flange = hw + tf/2
        y_web = hw/2
        y_bar = (A_flange*y_flange + A_web*y_web) / A_total
        
        # Moment of inertia about centroid
        I_flange = (bf*tf**3)/12 + A_flange*(y_flange - y_bar)**2
        I_web = (tw*hw**3)/12 + A_web*(y_web - y_bar)**2
        Iy = I_flange + I_web
        
        # Minor axis
        Iz = (tf*bf**3)/12 + (hw*tw**3)/12
        
        return SectionProperties(
            A=A_total, Iy=Iy, Iz=Iz, Sy=Sy,
            y_centroid=y_bar, z_centroid=0.0,
            y_top=c_top, y_bottom=-c_bottom,
            z_left=-bf/2, z_right=bf/2
        )
    
    def __str__(self):
        return f"T-Beam: {self.flange_width}×{self.flange_thickness} flange, {self.web_height}×{self.web_thickness} web"


class BoxSection(CrossSection):
    """Rectangular hollow section (box section)."""
    
    def __init__(self, width: float, height: float, thickness: float):
        """
        Initialize box section.
        
        Parameters:
        -----------
        width : float
            Outer width (mm)
        height : float
            Outer height (mm)
        thickness : float
            Wall thickness (mm)
        """
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")
        if thickness <= 0:
            raise ValueError(f"Thickness must be positive, got {thickness}")
        if thickness >= width / 2:
            raise ValueError(f"Thickness ({thickness}) must be less than half the width ({width/2})")
        if thickness >= height / 2:
            raise ValueError(f"Thickness ({thickness}) must be less than half the height ({height/2})")
        self.width = width
        self.height = height
        self.thickness = thickness
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        B = self.width
        H = self.height
        t = self.thickness
        
        b = B - 2*t
        h = H - 2*t
        
        A = B*H - b*h
        Iy = (B*H**3 - b*h**3) / 12
        Iz = (H*B**3 - h*b**3) / 12

        if A <= 0:
            raise ValueError(f"Calculated area must be positive (got {A}). Check dimensions.")
        if Iy <= 0:
            raise ValueError(f"Calculated moment of inertia Iy must be positive (got {Iy}). Check dimensions.")
        if Iz <= 0:
            raise ValueError(f"Calculated moment of inertia Iz must be positive (got {Iz}). Check dimensions.")
        
        # Torsional constant for thin-walled box
        Am = b * h  # Mean area
        perimeter = 2*(B + H)
        J = (4 * Am**2 * t) / perimeter
        
        return SectionProperties(
            A=A, Iy=Iy, Iz=Iz, J=J, Sy=Sy, Sz=Sz,
            y_centroid=0.0, z_centroid=0.0,
            y_top=H/2, y_bottom=-H/2,
            z_left=-B/2, z_right=B/2
        )
    
    def __str__(self):
        return f"Box: {self.width}×{self.height}×{self.thickness}t mm"


class CChannelSection(CrossSection):
    """C-channel cross-section."""
    
    def __init__(self, height: float, flange_width: float,
                 web_thickness: float, flange_thickness: float):
        """
        Initialize C-channel section.
        
        Parameters:
        -----------
        height : float
            Total height (mm)
        flange_width : float
            Flange width (mm)
        web_thickness : float
            Web thickness (mm)
        flange_thickness : float
            Flange thickness (mm)
        """
        self.height = height
        self.flange_width = flange_width
        self.web_thickness = web_thickness
        self.flange_thickness = flange_thickness
    
    def properties(self) -> SectionProperties:
        """Calculate section properties."""
        h = self.height
        bf = self.flange_width
        tw = self.web_thickness
        tf = self.flange_thickness
        
        # Areas
        A_flange = bf * tf
        A_web = tw * (h - 2*tf)
        A_total = 2*A_flange + A_web
        
        # Centroid (measured from back of web)
        x_flange = bf/2
        x_web = 0
        x_bar = (2*A_flange*x_flange + A_web*x_web) / A_total
        
        # Major axis (y-axis)
        Iy_flange = (bf*tf**3)/12 + A_flange*((h - tf)/2)**2
        Iy_web = (tw*(h - 2*tf)**3)/12
        Iy = 2*Iy_flange + Iy_web
        
        # Minor axis (z-axis) - about centroid
        Iz_flange = (tf*bf**3)/12 + A_flange*(x_flange - x_bar)**2
        Iz_web = (h - 2*tf)*tw**3/12 + A_web*(x_web - x_bar)**2
        Iz = 2*Iz_flange + Iz_web
        
        return SectionProperties(
            A=A_total, Iy=Iy, Iz=Iz, Sy=Sy,
            y_centroid=0.0, z_centroid=x_bar,
            y_top=h/2, y_bottom=-h/2,
            z_left=-x_bar, z_right=bf-x_bar
        )
    
    def __str__(self):
        return f"C-Channel: C{self.height}×{self.flange_width}"


class LSection(CrossSection):
    """L-section (angle) cross-section.
    
    Note: Properties calculated about centroidal axes.
    For equal angles, use equal leg lengths.
    """
    
    def __init__(self, leg_length_vertical: float, leg_length_horizontal: float,
                 thickness: float):
        """
        Initialize L-section (angle).
        
        Parameters:
        -----------
        leg_length_vertical : float
            Vertical leg length (mm)
        leg_length_horizontal : float
            Horizontal leg length (mm)
        thickness : float
            Leg thickness (mm)
        
        Note: For equal angle, set both legs equal (e.g., L50x50x5)
        """
        if leg_length_vertical <= 0:
            raise ValueError(f"Vertical leg length must be positive, got {leg_length_vertical}")
        if leg_length_horizontal <= 0:
            raise ValueError(f"Horizontal leg length must be positive, got {leg_length_horizontal}")
        if thickness <= 0:
            raise ValueError(f"Thickness must be positive, got {thickness}")
        if thickness >= min(leg_length_vertical, leg_length_horizontal):
            raise ValueError(f"Thickness must be less than leg lengths")
            
        self.leg_length_vertical = leg_length_vertical
        self.leg_length_horizontal = leg_length_horizontal
        self.thickness = thickness
    
    def properties(self) -> SectionProperties:
        """Calculate section properties about centroidal axes."""
        h = self.leg_length_vertical
        b = self.leg_length_horizontal
        t = self.thickness
        
        # Area
        A_total = h*t + b*t - t*t  # Two legs minus overlap
        
        # Centroid location from corner (assuming L-shape with corner at origin)
        # Vertical leg: centroid at (t/2, h/2)
        # Horizontal leg: centroid at (b/2, t/2)
        A_vert = h * t
        A_horiz = (b - t) * t
        
        x_bar = (A_vert * (t/2) + A_horiz * (t + (b-t)/2)) / A_total
        y_bar = (A_vert * (h/2) + A_horiz * (t/2)) / A_total
        
        # Moments of inertia about own axes (parallel axis theorem)
        # Vertical leg
        Ix_vert_local = t * h**3 / 12
        Iy_vert_local = h * t**3 / 12
        
        # Horizontal leg (excluding overlap)
        Ix_horiz_local = (b - t) * t**3 / 12
        Iy_horiz_local = t * (b - t)**3 / 12
        
        # Transfer to centroid using parallel axis theorem
        # I = I_local + A*d²
        Ix_vert = Ix_vert_local + A_vert * (h/2 - y_bar)**2
        Iy_vert = Iy_vert_local + A_vert * (t/2 - x_bar)**2
        
        Ix_horiz = Ix_horiz_local + A_horiz * (t/2 - y_bar)**2
        Iy_horiz = Iy_horiz_local + A_horiz * (t + (b-t)/2 - x_bar)**2
        
        # Total moments
        Ix = Ix_vert + Ix_horiz
        Iy = Iy_vert + Iy_horiz
        
        return SectionProperties(
            A=A_total, Iy=Ix, Iz=Iy,
            y_centroid=y_bar, z_centroid=x_bar,
            y_top=h-y_bar, y_bottom=-y_bar,
            z_left=-x_bar, z_right=b-x_bar
        )
    
    def __str__(self):
        if self.leg_length_vertical == self.leg_length_horizontal:
            return f"L-Section (Equal): L{self.leg_length_vertical}×{self.thickness}mm"
        else:
            return f"L-Section (Unequal): L{self.leg_length_vertical}×{self.leg_length_horizontal}×{self.thickness}mm"


def offset_section(section_props: SectionProperties, 
                   offset_y: float = 0.0, 
                   offset_z: float = 0.0) -> SectionProperties:
    """
    Offset section properties to a new reference axis.
    
    Uses parallel axis theorem: I_new = I_centroid + A * d^2
    
    Parameters:
    -----------
    section_props : SectionProperties
        Original section properties (assumed about centroid)
    offset_y : float
        Offset distance in y-direction (mm)
    offset_z : float
        Offset distance in z-direction (mm)
        
    Returns:
    --------
    SectionProperties
        New properties about the offset axis
    """
    A = section_props.A
    
    # Parallel axis theorem
    Iy_new = section_props.Iy + A * offset_z**2
    Iz_new = section_props.Iz + A * offset_y**2
    
    # Update centroid location relative to the NEW reference
    # d_y = (y_centroid_new - y_centroid_old) -> y_centroid_new = y_centroid_old - offset_y
    y_cent_new = section_props.y_centroid - offset_y
    z_cent_new = section_props.z_centroid - offset_z
    
    # Section modulus and extreme fibers remain relative to the geometry
    return SectionProperties(
        A=A,
        Iy=Iy_new,
        Iz=Iz_new,
        J=section_props.J,
        Sy=section_props.Sy,
        Sz=section_props.Sz,
        y_centroid=y_cent_new,
        z_centroid=z_cent_new,
        y_top=section_props.y_top,
        y_bottom=section_props.y_bottom,
        z_left=section_props.z_left,
        z_right=section_props.z_right
    )


# --- PRIMARY API HELPER FUNCTIONS ---
# These functions are the recommended way to create cross-sections.

def rectangular(width: float, height: float) -> SectionProperties:
    """
    Create a rectangular cross-section.
    
    Parameters:
    -----------
    width : float
        Width (mm)
    height : float
        Height (mm)
    """
    return RectangularSection(width, height).properties()

def circular(diameter: float) -> SectionProperties:
    """
    Create a solid circular cross-section.
    
    Parameters:
    -----------
    diameter : float
        Diameter (mm)
    """
    return CircularSection(diameter).properties()

def hollow_circular(outer_diameter: float, thickness: float) -> SectionProperties:
    """
    Create a hollow circular (pipe) cross-section.
    
    Parameters:
    -----------
    outer_diameter : float
        Outer diameter (mm)
    thickness : float
        Wall thickness (mm)
    """
    return HollowCircularSection(outer_diameter, thickness).properties()

def i_beam(flange_width: float, total_height: float,
           web_thickness: float, flange_thickness: float) -> SectionProperties:
    """
    Create an I-beam (wide flange) cross-section.
    
    Parameters:
    -----------
    flange_width : float
        Flange width (mm)
    total_height : float
        Total height (mm)
    web_thickness : float
        Web thickness (mm)
    flange_thickness : float
        Flange thickness (mm)
    """
    return IBeamSection(flange_width, total_height, web_thickness, flange_thickness).properties()

def box(width: float, height: float, thickness: float) -> SectionProperties:
    """
    Create a rectangular box cross-section.
    
    Parameters:
    -----------
    width : float
        Outer width (mm)
    height : float
        Outer height (mm)
    thickness : float
        Wall thickness (mm)
    """
    return BoxSection(width, height, thickness).properties()

def t_beam(flange_width: float, flange_thickness: float,
           web_height: float, web_thickness: float) -> SectionProperties:
    """
    Create a T-beam cross-section.
    
    Parameters:
    -----------
    flange_width : float
        Flange width (mm)
    flange_thickness : float
        Flange thickness (mm)
    web_height : float
        Web height (mm)
    web_thickness : float
        Web thickness (mm)
    """
    return TBeamSection(flange_width, flange_thickness, web_height, web_thickness).properties()

def c_channel(height: float, flange_width: float,
              web_thickness: float, flange_thickness: float) -> SectionProperties:
    """
    Create a C-channel cross-section.
    
    Parameters:
    -----------
    height : float
        Total height (mm)
    flange_width : float
        Flange width (mm)
    web_thickness : float
        Web thickness (mm)
    flange_thickness : float
        Flange thickness (mm)
    """
    return CChannelSection(height, flange_width, web_thickness, flange_thickness).properties()

def l_section(leg_vertical: float, leg_horizontal: float, thickness: float) -> SectionProperties:
    """
    Create an L-section (angle) cross-section.
    
    Parameters:
    -----------
    leg_vertical : float
        Vertical leg length (mm)
    leg_horizontal : float
        Horizontal leg length (mm)  
    thickness : float
        Leg thickness (mm)
    
    Note: For equal angle, use same value for both legs (e.g., L50x50x5)
    """
    return LSection(leg_vertical, leg_horizontal, thickness).properties()


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("CROSS-SECTION EXAMPLES")
    print("="*70)
    
    # Using the simplified functional API
    sections = {
        "Rectangle": rectangular(50, 100),
        "Circle": circular(50),
        "Pipe": hollow_circular(60, 5),
        "I-Beam": i_beam(100, 200, 8, 12),
        "Box": box(80, 120, 6),
        "L-Section (Equal)": l_section(50, 50, 5),
        "L-Section (Unequal)": l_section(75, 50, 6)
    }
    
    for name, props in sections.items():
        print(f"\n{name}:")
        print(f"  A  = {props.A:10.2f} mm^2")
        print(f"  Iy = {props.Iy:10.2f} mm^4")
        print(f"  Iz = {props.Iz:10.2f} mm^4")
        if props.J:
            print(f"  J  = {props.J:10.2f} mm^4")
