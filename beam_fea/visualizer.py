"""
visualizer.py
=============
Visualization tools for beam FEA results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from .plot_style import PlotStyle, smart_units, DEFAULT_STYLE


class BeamVisualizer:
    """Visualization for beam analysis results."""
    
    def __init__(self, mesh, results=None):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        mesh : Mesh
            Finite element mesh
        results : dict, optional
            Analysis results dictionary
        """
        self.mesh = mesh
        self.results = results or {}
    
    def _auto_scale_factor(self, displacements: np.ndarray) -> float:
        """Calculate automatic scaling factor."""
        coords = self.mesh.get_node_coords()
        # Characteristic dimension for scaling (max span)
        dx_max = np.max(coords[:, 0]) - np.min(coords[:, 0])
        dy_max = np.max(coords[:, 1]) - np.min(coords[:, 1])
        beam_length = max(dx_max, dy_max)
        
        # Consider absolute displacement magnitude for general 2D structures
        u = displacements[0::3]
        v = displacements[1::3]
        mag = np.sqrt(u**2 + v**2)
        max_def = np.max(mag)
        
        if max_def < 1e-10:
            return 1.0
        
        target = 0.05 * beam_length
        scale = target / max_def
        
        # Round to nice value
        magnitude = 10 ** np.floor(np.log10(scale))
        normalized = scale / magnitude
        
        if normalized < 1.5:
            nice = 1
        elif normalized < 3.5:
            nice = 2
        elif normalized < 7.5:
            nice = 5
        else:
            nice = 10
        
        return nice * magnitude
    
    def plot_deformed_shape(self, displacements: np.ndarray,
                           scale_factor: Optional[float] = None,
                           show_undeformed: bool = True,
                           figsize: Tuple = (12, 6),
                           dpi: int = 100,
                           output_path: Optional[str] = None):
        """
        Plot deformed shape of beam.
        
        Parameters:
        -----------
        displacements : np.ndarray
            Displacement vector
        scale_factor : float, optional
            Scaling factor (auto if None)
        show_undeformed : bool
            Show original shape
        output_path : str, optional
            Path to save the plot (if None, displays interactively)
        """
        if scale_factor is None:
            scale_factor = self._auto_scale_factor(displacements)
        
        st = DEFAULT_STYLE
        coords = self.mesh.get_node_coords()
        beam_length = np.max(coords[:, 0]) - np.min(coords[:, 0])
        x_scale, x_unit = smart_units(beam_length, 'length')

        u = displacements[::3]
        v = displacements[1::3]

        deformed = coords.copy()
        deformed[:, 0] += scale_factor * u
        deformed[:, 1] += scale_factor * v

        # Use the same scale for Y as for X to maintain aspect ratio with equal axis
        y_scale = x_scale
        y_unit = x_unit

        # Adjust figsize based on beam aspect ratio if default is used
        dx_max = np.max(coords[:, 0]) - np.min(coords[:, 0])
        dy_max = np.max(coords[:, 1]) - np.min(coords[:, 1])
        if dy_max > 1.5 * dx_max and figsize == (12, 6):
            figsize = (6, 10) # Swap to vertical portrait layout

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if show_undeformed:
            ax.plot(coords[:, 0] / x_scale, coords[:, 1] / y_scale, '--o',
                   color=st.colour_undeformed, alpha=0.3, label='Undeformed', markersize=4)

        ax.plot(deformed[:, 0] / x_scale, deformed[:, 1] / y_scale, '-o',
               color=st.colour_deformed, linewidth=2,
               label=f'Deformed ({scale_factor:.0f}×)', markersize=6)

        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        if abs(scale_factor - 1.0) < 1e-6:
            ax.set_ylabel(f'Y Position ({y_unit})', fontsize=st.label_fontsize)
        else:
            ax.set_ylabel(f'Scaled Y Position ({y_unit})', fontsize=st.label_fontsize)
        ax.set_title('Beam Deformation', fontsize=st.title_fontsize, weight='bold')
        ax.legend(fontsize=st.tick_fontsize)
        ax.grid(True, alpha=st.grid_alpha)
        ax.axis('equal')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_mode_shape(self, mode_shape: np.ndarray, mode_num: int,
                        frequency: float, scale: float = 1.0, 
                        figsize: Optional[Tuple] = None, dpi: int = 150,
                        output_path: Optional[str] = None):
        """Plot a mode shape (2D supported)."""
        st = DEFAULT_STYLE
        fs = figsize or st.figsize_wide
        coords = self.mesh.get_node_coords()
        
        # Characteristic length for normalization
        dx_max = np.max(coords[:, 0]) - np.min(coords[:, 0])
        dy_max = np.max(coords[:, 1]) - np.min(coords[:, 1])
        ref_length = max(dx_max, dy_max)
        x_scale, x_unit = smart_units(ref_length, 'length')

        # Modal displacements
        u_mode = mode_shape[0::3] * scale
        v_mode = mode_shape[1::3] * scale
        
        deformed = coords.copy()
        deformed[:, 0] += u_mode
        deformed[:, 1] += v_mode

        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        
        # Plot undeformed center-line
        ax.plot(coords[:, 0] / x_scale, coords[:, 1] / x_scale, '--',
                color=st.colour_undeformed, alpha=0.3, linewidth=1)
        
        # Plot mode shape
        ax.plot(deformed[:, 0] / x_scale, deformed[:, 1] / x_scale,
                color=st.colour_deformed, linewidth=st.line_width, label=f'Mode {mode_num}')

        ax.set_xlabel(f'X Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel(f'Y Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_title(f'Mode Shape {mode_num} - {frequency:.2f} Hz', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_bending_moment(self, moments: np.ndarray, positions: np.ndarray,
                           output_path: Optional[str] = None, figsize: Optional[Tuple] = None, 
                           dpi: int = 150):
        """Plot bending moment diagram."""
        st = DEFAULT_STYLE
        fs = figsize or st.figsize_wide
        
        max_M = np.max(np.abs(moments))
        beam_L = np.max(positions) - np.min(positions)
        
        m_scale, m_unit = smart_units(max_M, 'moment')
        x_scale, x_unit = smart_units(beam_L, 'length')
        
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        
        # Plot data
        ax.plot(positions / x_scale, moments / m_scale, color=st.colour_moment, linewidth=st.line_width)
        ax.fill_between(positions / x_scale, 0, moments / m_scale, 
                        color=st.colour_moment, alpha=st.fill_alpha)
        ax.axhline(0, color=st.colour_zero_line, linewidth=1.5)
        
        # Mark Maximum
        if max_M > 1e-10:
            max_idx = np.argmax(np.abs(moments))
            x_peak = positions[max_idx] / x_scale
            y_peak = moments[max_idx] / m_scale
            
            # Peak marker
            ax.plot(x_peak, y_peak, 'o', color=st.colour_max, markersize=8)
            
            # Dotted line to Y-axis
            ax.axhline(y_peak, xmax=x_peak/ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else 0, 
                       color=st.colour_max, linestyle=':', linewidth=1.5)
            
            # Add value to Y-ticks to ensure it's displayed on axis as requested
            existing_ticks = list(ax.get_yticks())
            # Filter out ticks that are too close to our peak value to avoid overlap
            new_ticks = [t for t in existing_ticks if abs(t - y_peak) > 0.1 * max(abs(y_peak), 1)]
            new_ticks.append(y_peak)
            ax.set_yticks(new_ticks)
            
            # Small annotation near marker (without box)
            ax.annotate(f'{y_peak:.2f}', (x_peak, y_peak), 
                        xytext=(5, 5), textcoords='offset points',
                        color=st.colour_max, weight='bold', fontsize=st.annotation_fontsize)
        
        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel(f'Bending Moment ({m_unit})', fontsize=st.label_fontsize)
        ax.set_title('Bending Moment Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_shear_force(self, forces: np.ndarray, positions: np.ndarray,
                         output_path: Optional[str] = None, figsize: Optional[Tuple] = None,
                         dpi: int = 150):
        """Plot shear force diagram."""
        st = DEFAULT_STYLE
        fs = figsize or st.figsize_wide
        
        max_V = np.max(np.abs(forces))
        beam_L = np.max(positions) - np.min(positions)
        
        v_scale, v_unit = smart_units(max_V, 'force')
        x_scale, x_unit = smart_units(beam_L, 'length')
        
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        
        # Step plot for shear
        ax.step(positions / x_scale, forces / v_scale, color=st.colour_primary, 
                linewidth=st.line_width, where='post')
        ax.fill_between(positions / x_scale, 0, forces / v_scale, 
                        color=st.colour_primary, alpha=st.fill_alpha, step='post')
        ax.axhline(0, color=st.colour_zero_line, linewidth=1.5)
        
        # Mark Maxima (often two for shear)
        if max_V > 1e-10:
            # Find index of max absolute value
            max_idx = np.argmax(np.abs(forces))
            # Just mark the primary max for clarity
            x_peak = positions[max_idx] / x_scale
            y_peak = forces[max_idx] / v_scale
            
            ax.plot(x_peak, y_peak, 'o', color=st.colour_max, markersize=8)
            
            # Dotted line to Y-axis
            ax.axhline(y_peak, xmax=x_peak/ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else 0,
                       color=st.colour_max, linestyle=':', linewidth=1.5)
            
            # Add value to Y-ticks
            existing_ticks = list(ax.get_yticks())
            new_ticks = [t for t in existing_ticks if abs(t - y_peak) > 0.1 * max(abs(y_peak), 1)]
            new_ticks.append(y_peak)
            ax.set_yticks(new_ticks)
            
            # Small annotation
            ax.annotate(f'{y_peak:.2f}', (x_peak, y_peak), 
                        xytext=(5, 5), textcoords='offset points',
                        color=st.colour_max, weight='bold', fontsize=st.annotation_fontsize)
        
        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel(f'Shear Force ({v_unit})', fontsize=st.label_fontsize)
        ax.set_title('Shear Force Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_section_properties(self, props, title: str = "Cross-Section Analysis",
                               output_path: Optional[str] = None,
                               dpi: int = 100):
        """
        Visualize cross-section properties, centroid, and explicit geometry.
        
        Parameters:
        -----------
        props : SectionProperties
            The properties object to visualize
        title : str
            Title for the plot
        """
        from matplotlib.patches import Rectangle, Circle, Polygon, PathPatch
        import matplotlib.path as mpath

        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
        st = DEFAULT_STYLE
        
        yc, zc = props.y_centroid, props.z_centroid
        
        # Explicit Geometry Drawing
        drawn = False
        if hasattr(props, 'parent_section') and props.parent_section is not None:
            sec = props.parent_section
            from .cross_sections import (RectangularSection, CircularSection, HollowCircularSection,
                                       IBeamSection, TBeamSection, BoxSection, CChannelSection, LSection)

            color = st.colour_primary
            alpha = 0.7

            if isinstance(sec, RectangularSection):
                w, h = sec.width, sec.height
                ax.add_patch(Rectangle((zc - w/2, yc - h/2), w, h, color=color, alpha=alpha, label=sec.name))
                drawn = True
            elif isinstance(sec, CircularSection):
                ax.add_patch(Circle((zc, yc), sec.diameter/2, color=color, alpha=alpha, label=sec.name))
                drawn = True
            elif isinstance(sec, HollowCircularSection):
                ax.add_patch(Circle((zc, yc), sec.outer_diameter/2, color=color, alpha=alpha, label=sec.name))
                ax.add_patch(Circle((zc, yc), sec.inner_diameter/2, color='white', zorder=2))
                drawn = True
            elif isinstance(sec, IBeamSection):
                d, bf, tw, tf = sec.total_height, sec.flange_width, sec.web_thickness, sec.flange_thickness
                # Top flange
                ax.add_patch(Rectangle((zc - bf/2, yc + d/2 - tf), bf, tf, color=color, alpha=alpha))
                # Bottom flange
                ax.add_patch(Rectangle((zc - bf/2, yc - d/2), bf, tf, color=color, alpha=alpha))
                # Web
                ax.add_patch(Rectangle((zc - tw/2, yc - d/2 + tf), tw, d - 2*tf, color=color, alpha=alpha, label=sec.name))
                drawn = True
            elif isinstance(sec, TBeamSection):
                bf, tf, hw, tw = sec.flange_width, sec.flange_thickness, sec.web_height, sec.web_thickness
                y_top = yc + props.y_top
                y_bot = yc + props.y_bottom
                # Flange
                ax.add_patch(Rectangle((zc - bf/2, y_top - tf), bf, tf, color=color, alpha=alpha))
                # Web
                ax.add_patch(Rectangle((zc - tw/2, y_bot), tw, hw, color=color, alpha=alpha, label=sec.name))
                drawn = True
            elif isinstance(sec, BoxSection):
                W, H, t = sec.width, sec.height, sec.thickness
                ax.add_patch(Rectangle((zc - W/2, yc - H/2), W, H, color=color, alpha=alpha, label=sec.name))
                ax.add_patch(Rectangle((zc - (W-2*t)/2, yc - (H-2*t)/2), W-2*t, H-2*t, color='white', zorder=2))
                drawn = True
            elif isinstance(sec, CChannelSection):
                h, bf, tw, tf = sec.height, sec.flange_width, sec.web_thickness, sec.flange_thickness
                z_back = zc - sec.properties().z_centroid # back of web is at z=0 in local coords? No, z_centroid is distance from back of web.
                # back of web is at zc - z_centroid_from_back
                z0 = zc - sec.properties().z_centroid
                # Web
                ax.add_patch(Rectangle((z0, yc - h/2), tw, h, color=color, alpha=alpha, label=sec.name))
                # Top flange
                ax.add_patch(Rectangle((z0 + tw, yc + h/2 - tf), bf - tw, tf, color=color, alpha=alpha))
                # Bottom flange
                ax.add_patch(Rectangle((z0 + tw, yc - h/2), bf - tw, tf, color=color, alpha=alpha))
                drawn = True
            elif isinstance(sec, LSection):
                hv, hh, t = sec.leg_length_vertical, sec.leg_length_horizontal, sec.thickness
                z0 = zc - sec.properties().z_centroid
                y0 = yc - sec.properties().y_centroid
                # Vertical leg
                ax.add_patch(Rectangle((z0, y0), t, hv, color=color, alpha=alpha, label=sec.name))
                # Horizontal leg
                ax.add_patch(Rectangle((z0 + t, y0), hh - t, t, color=color, alpha=alpha))
                drawn = True

        if not drawn:
            # Fallback to bounding box if no explicit geometry is available
            y_top = yc + props.y_top
            y_bot = yc + props.y_bottom
            z_left = zc + props.z_left
            z_right = zc + props.z_right
            box_y = [y_bot, y_top, y_top, y_bot, y_bot]
            box_z = [z_left, z_left, z_right, z_right, z_left]
            ax.fill(box_z, box_y, color='lightblue', alpha=0.3, label='Bounding Box (Fallback)')
            ax.plot(box_z, box_y, 'k--', alpha=0.5)
        
        # Neutral Axes (passing through centroid)
        ax.axhline(yc, color='gray', linestyle=':', alpha=0.6, label='Neutral Axis (z)')
        ax.axvline(zc, color='gray', linestyle=':', alpha=0.6, label='Neutral Axis (y)')
        
        # Centroid Marker
        ax.plot(zc, yc, 'ro', markersize=10, label=f'Centroid')
        
        # Reference Origin
        ax.plot(0, 0, 'kx', markersize=12, label='Reference (0,0)')
        
        # Labels and formatting
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(title, fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        ax.set_aspect('equal', adjustable='datalim')
        
        # Add property text box - include units clearly
        props_text = (f"Area: {props.A:.1f} mm²\n"
                     f"Iy: {props.Iy:.2e} mm⁴\n"
                     f"Iz: {props.Iz:.2e} mm⁴\n"
                     f"y_top: {props.y_top:+.1f} mm\n"
                     f"y_bot: {props.y_bottom:+.1f} mm")
        
        ax.text(1.05, 0.5, props_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_multiple_sections(self, property_set, output_path: Optional[str] = None, dpi: int = 150):
        """
        Plot all unique cross-sections in a PropertySet as subplots.
        """
        unique_secs = []
        seen_ids = set()

        # We need to know how many elements to check.
        # But PropertySet might not be resolved yet.
        # Let's assume it is resolved if we are calling this.
        if not property_set._is_resolved:
             # Try to resolve with a dummy count if possible, or just look at assignments
             for assignment in property_set._assignments:
                 sec = assignment['section']
                 if sec and id(sec) not in seen_ids:
                     seen_ids.add(id(sec))
                     unique_secs.append(sec)
        else:
            for eid in range(len(property_set._sec_map)):
                sec = property_set.get_section(eid)
                if id(sec) not in seen_ids:
                    seen_ids.add(id(sec))
                    unique_secs.append(sec)

        num_secs = len(unique_secs)
        if num_secs == 0:
            return

        cols = min(num_secs, 3)
        rows = (num_secs + cols - 1) // cols

        fig = plt.figure(figsize=(5 * cols, 5 * rows), dpi=dpi)
        st = DEFAULT_STYLE

        from matplotlib.patches import Rectangle, Circle
        from .cross_sections import (RectangularSection, CircularSection, HollowCircularSection,
                                   IBeamSection, TBeamSection, BoxSection, CChannelSection, LSection)

        for i, sec_props in enumerate(unique_secs):
            ax = fig.add_subplot(rows, cols, i + 1)

            yc, zc = sec_props.y_centroid, sec_props.z_centroid
            color = st.colour_primary
            alpha = 0.7

            drawn = False
            if hasattr(sec_props, 'parent_section') and sec_props.parent_section is not None:
                sec = sec_props.parent_section
                if isinstance(sec, RectangularSection):
                    w, h = sec.width, sec.height
                    ax.add_patch(Rectangle((zc - w/2, yc - h/2), w, h, color=color, alpha=alpha))
                    drawn = True
                elif isinstance(sec, CircularSection):
                    ax.add_patch(Circle((zc, yc), sec.diameter/2, color=color, alpha=alpha))
                    drawn = True
                elif isinstance(sec, HollowCircularSection):
                    ax.add_patch(Circle((zc, yc), sec.outer_diameter/2, color=color, alpha=alpha))
                    ax.add_patch(Circle((zc, yc), sec.inner_diameter/2, color='white', zorder=2))
                    drawn = True
                elif isinstance(sec, IBeamSection):
                    d, bf, tw, tf = sec.total_height, sec.flange_width, sec.web_thickness, sec.flange_thickness
                    ax.add_patch(Rectangle((zc - bf/2, yc + d/2 - tf), bf, tf, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((zc - bf/2, yc - d/2), bf, tf, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((zc - tw/2, yc - d/2 + tf), tw, d - 2*tf, color=color, alpha=alpha))
                    drawn = True
                elif isinstance(sec, TBeamSection):
                    bf, tf, hw, tw = sec.flange_width, sec.flange_thickness, sec.web_height, sec.web_thickness
                    y_top = yc + sec_props.y_top
                    y_bot = yc + sec_props.y_bottom
                    ax.add_patch(Rectangle((zc - bf/2, y_top - tf), bf, tf, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((zc - tw/2, y_bot), tw, hw, color=color, alpha=alpha))
                    drawn = True
                elif isinstance(sec, BoxSection):
                    W, H, t = sec.width, sec.height, sec.thickness
                    ax.add_patch(Rectangle((zc - W/2, yc - H/2), W, H, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((zc - (W-2*t)/2, yc - (H-2*t)/2), W-2*t, H-2*t, color='white', zorder=2))
                    drawn = True
                elif isinstance(sec, CChannelSection):
                    h, bf, tw, tf = sec.height, sec.flange_width, sec.web_thickness, sec.flange_thickness
                    z0 = zc - sec_props.z_centroid
                    ax.add_patch(Rectangle((z0, yc - h/2), tw, h, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((z0 + tw, yc + h/2 - tf), bf - tw, tf, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((z0 + tw, yc - h/2), bf - tw, tf, color=color, alpha=alpha))
                    drawn = True
                elif isinstance(sec, LSection):
                    hv, hh, t = sec.leg_length_vertical, sec.leg_length_horizontal, sec.thickness
                    z0 = zc - sec_props.z_centroid
                    y0 = yc - sec_props.y_centroid
                    ax.add_patch(Rectangle((z0, y0), t, hv, color=color, alpha=alpha))
                    ax.add_patch(Rectangle((z0 + t, y0), hh - t, t, color=color, alpha=alpha))
                    drawn = True

            if not drawn:
                y_top, y_bot = yc + sec_props.y_top, yc + sec_props.y_bottom
                z_l, z_r = zc + sec_props.z_left, zc + sec_props.z_right
                ax.fill([z_l, z_r, z_r, z_l, z_l], [y_bot, y_bot, y_top, y_top, y_bot], color='lightblue', alpha=0.3)
                ax.plot([z_l, z_r, z_r, z_l, z_l], [y_bot, y_bot, y_top, y_top, y_bot], 'k--', alpha=0.5)

            ax.axhline(yc, color='gray', linestyle=':', alpha=0.6)
            ax.axvline(zc, color='gray', linestyle=':', alpha=0.6)
            ax.plot(zc, yc, 'ro', markersize=6)
            ax.set_title(sec_props.name, fontsize=st.label_fontsize)
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(True, alpha=st.grid_alpha)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_laminate_stackup(self, laminate, output_path: Optional[str] = None, dpi: int = 150):
        """
        Visualize the laminate stack-up and fiber orientation.

        Abaqus-inspired stack-up plot with a HyperMesh-style orientation rosette.
        """
        from matplotlib.patches import Rectangle, Wedge
        import matplotlib.cm as cm

        st = DEFAULT_STYLE
        fig = plt.figure(figsize=(10, 8), dpi=dpi)

        # Grid layout: left for stackup, right for rosette and info
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
        ax_stack = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1])

        # 1. Map Colors to Unique Fiber Angles for consistency
        plies = laminate.plies
        total_t = laminate.total_thickness
        
        unique_angles = sorted(list(set(angle for ply, angle in plies)))
        
        # Professional "Engineering" Palette (Muted but distinct)
        # Inspired by standard technical plotting libraries (e.g., Seaborn Deep or ColorBrewer Set2)
        eng_colors = [
            '#4C72B0', # Muted Blue
            '#C44E52', # Muted Red
            '#55A868', # Muted Green
            '#8172B2', # Muted Purple
            '#CCB974', # Muted Gold
            '#64B5CD', # Muted Cyan
            '#8C8C8C', # Medium Gray
            '#E1974C', # Muted Orange
            '#B07AA1'  # Muted Pink
        ]
        color_map = {angle: eng_colors[k % len(eng_colors)] for k, angle in enumerate(unique_angles)}

        # 1. Stack-up Plot (Professional style)
        y_bottom = -total_t / 2
        for i, (ply, angle) in enumerate(plies):
            t = ply.thickness
            color = color_map[angle]
            rect = Rectangle((0, y_bottom), 1.0, t, facecolor=color, alpha=0.9, edgecolor='black', linewidth=0.8)
            ax_stack.add_patch(rect)

            # Label with orientation (Professional font size and weight)
            # Use white text for dark colors, black for light
            r, g, b = [int(color[i:i+2], 16)/255.0 for i in (1, 3, 5)]
            lum = 0.299*r + 0.587*g + 0.114*b # Standard luminance
            ax_stack.text(0.5, y_bottom + t/2, f"P{i+1}: {angle}°",
                         ha='center', va='center', fontsize=9, weight='semibold', 
                         color='white' if lum < 0.6 else 'black')
            y_bottom += t

        ax_stack.set_xlim(-0.2, 1.2)
        ax_stack.set_ylim(-total_t/2 - 0.05*total_t, total_t/2 + 0.05*total_t)
        ax_stack.set_xticks([])
        ax_stack.set_ylabel("Thickness (mm)", fontsize=st.label_fontsize, family='serif')
        ax_stack.set_title(f"Laminate Stack-up: {laminate.name}", fontsize=st.title_fontsize, weight='bold', pad=20)
        ax_stack.grid(axis='y', alpha=0.2, linestyle='-')

        # 2. Orientation Rosette (Engineering Standard)
        ax_info.set_aspect('equal')
        ax_info.axis('off')
        from matplotlib.patches import FancyArrowPatch

        # Background grid (Muted and professional)
        for r in [0.25, 0.5, 0.75, 1.0]:
            circle = plt.Circle((0, 0), r, color='#DDDDDD', fill=False, linestyle='-', alpha=0.6, linewidth=0.6)
            ax_info.add_patch(circle)

        # Draw main orientations (0, 45, 90, 135...)
        for a in range(0, 360, 45):
            th = np.radians(a)
            ax_info.plot([0, np.cos(th)], [0, np.sin(th)], color='#EEEEEE', linewidth=0.5, zorder=1)

        # Draw axes labels (Further out to avoid clash)
        ax_info.text(1.3, 0, 'X (0°)', ha='center', va='center', fontsize=9, color='#555555', weight='bold')
        ax_info.text(0, 1.3, 'Y (90°)', ha='center', va='center', fontsize=9, color='#555555', weight='bold')
        ax_info.plot([-1.1, 1.1], [0, 0], color='#BBBBBB', alpha=0.4, linewidth=0.8, zorder=2)
        ax_info.plot([0, 0], [-1.1, 1.1], color='#BBBBBB', alpha=0.4, linewidth=0.8, zorder=2)

        # Draw orientations with Unified Arrow Patches
        for angle in unique_angles:
            theta = np.radians(angle)
            color = color_map[angle]
            
            # Fiber Vector
            arrow = FancyArrowPatch((0, 0), (np.cos(theta), np.sin(theta)),
                                    arrowstyle='-|>', mutation_scale=15,
                                    color=color, linewidth=2.5, antialiased=True, zorder=10)
            ax_info.add_patch(arrow)
            
            # Radial degree labels removed per user request (redundant with legend)

        # Centered Orientation Legend (Professional markers)
        # Fix markers going outside: use Square patches
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='none', marker='s', markersize=10, 
                                  markerfacecolor=color_map[a], markeredgecolor='black', 
                                  label=f"{a}° Orientation") for a in unique_angles]
        
        leg = ax_info.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.4), 
                             title="Fiber Orientations", title_fontsize=10, ncol=2, 
                             frameon=True, fancybox=False, edgecolor='#CCCCCC', fontsize=8)
        leg.get_frame().set_linewidth(0.8)

        # Effective properties summary (Technical Specs Box)
        # Positioned relative to ax_info to ensure it stays within figure bounds
        props = laminate.get_effective_properties()
        summary_text = (f"LAMINATE TECHNICAL DATA\n"
                       f"=======================\n"
                       f"Nominal Thickness: {total_t:6.3f} mm\n"
                       f"Young's Moduli:\n"
                       f"  Ex (Axial):   {props['Ex']/1000:7.1f} GPa\n"
                       f"  Ey (Trans):   {props['Ey']/1000:7.1f} GPa\n"
                       f"  Eb (Flex):    {props['Eb']/1000:7.1f} GPa\n"
                       f"Shear Modulus:\n"
                       f"  Gxy:          {props['Gxy']/1000:7.1f} GPa\n"
                       f"Poisson's Ratio:\n"
                       f"  nu_xy:         {props['nu_xy']:7.3f}")

        ax_info.text(0.5, -0.05, summary_text, transform=ax_info.transAxes,
                    bbox=dict(facecolor='#FFFFFF', alpha=1.0, edgecolor='#AAAAAA', boxstyle='square,pad=1'),
                    ha='center', va='top', family='monospace', fontsize=9, zorder=20)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    print("\nVisualization Module")
    print("Provides plotting tools for FEA results")
