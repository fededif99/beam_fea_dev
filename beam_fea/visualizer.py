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
        beam_length = np.max(coords[:, 0]) - np.min(coords[:, 0])
        
        v_displ = displacements[1::3]
        max_def = np.max(np.abs(v_displ))
        
        if max_def < 1e-10:
            return 1.0
        
        target = 0.02 * beam_length
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
        """Plot a mode shape."""
        st = DEFAULT_STYLE
        fs = figsize or st.figsize_wide
        coords = self.mesh.get_node_coords()
        v_mode = mode_shape[1::3] * scale
        
        beam_length = np.max(coords[:, 0]) - np.min(coords[:, 0])
        x_scale, x_unit = smart_units(beam_length, 'length')
        
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)
        
        ax.plot(coords[:, 0] / x_scale, v_mode, color=st.colour_deformed, linewidth=st.line_width)
        
        ax.axhline(0, color=st.colour_zero_line, linewidth=1)
        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel('Modal Amplitude', fontsize=st.label_fontsize)
        ax.set_title(f'Mode Shape {mode_num} - {frequency:.2f} Hz', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        
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

        # 1. Stack-up Plot (Abaqus style)
        plies = laminate.plies # list of (Ply, angle)
        total_t = laminate.total_thickness

        y_bottom = -total_t / 2
        colors = cm.get_cmap('viridis')(np.linspace(0, 0.8, len(plies)))

        for i, (ply, angle) in enumerate(plies):
            t = ply.thickness
            rect = Rectangle((0, y_bottom), 1.0, t, facecolor=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
            ax_stack.add_patch(rect)

            # Label with orientation
            ax_stack.text(0.5, y_bottom + t/2, f"Ply {i+1}: {angle}° ({ply.name})",
                         ha='center', va='center', weight='bold', color='white' if i < len(plies)/2 else 'black')
            y_bottom += t

        ax_stack.set_xlim(-0.5, 1.5)
        ax_stack.set_ylim(-total_t/2 - 0.1*total_t, total_t/2 + 0.1*total_t)
        ax_stack.set_xticks([])
        ax_stack.set_ylabel("Thickness (mm)", fontsize=st.label_fontsize)
        ax_stack.set_title(f"Stack-up: {laminate.name}", fontsize=st.label_fontsize, weight='bold')
        ax_stack.grid(axis='y', alpha=0.3)

        # 2. Orientation Rosette (HyperMesh style)
        ax_info.set_aspect('equal')
        ax_info.axis('off')

        # Draw background circles
        for r in [0.5, 0.8, 1.0]:
            circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.3)
            ax_info.add_patch(circle)

        # Draw axes
        ax_info.plot([-1.1, 1.1], [0, 0], 'k-', alpha=0.5)
        ax_info.plot([0, 0], [-1.1, 1.1], 'k-', alpha=0.5)
        ax_info.text(1.2, 0, 'x (0°)', ha='left', va='center')
        ax_info.text(0, 1.2, 'y (90°)', ha='center', va='bottom')

        # Draw ply orientation lines
        for i, (ply, angle) in enumerate(plies):
            theta = np.radians(angle)
            ax_info.plot([0, np.cos(theta)], [0, np.sin(theta)], color=colors[i],
                        linewidth=2, label=f"Ply {i+1} ({angle}°)")
            # Small arrowhead
            ax_info.annotate('', xy=(np.cos(theta), np.sin(theta)), xytext=(0.9*np.cos(theta), 0.9*np.sin(theta)),
                            arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))

        # Effective properties summary
        props = laminate.get_effective_properties()
        summary_text = (f"Laminate Properties:\n"
                       f"Total Thick: {total_t:.3f} mm\n"
                       f"Ex: {props['Ex']/1000:.1f} GPa\n"
                       f"Ey: {props['Ey']/1000:.1f} GPa\n"
                       f"Eb: {props['Eb']/1000:.1f} GPa\n"
                       f"Gxy: {props['Gxy']/1000:.1f} GPa\n"
                       f"nu_xy: {props['nu_xy']:.3f}")

        plt.figtext(0.75, 0.15, summary_text, bbox=dict(facecolor='white', alpha=0.8),
                    ha='center', va='center', family='monospace')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    print("\nVisualization Module")
    print("Provides plotting tools for FEA results")
