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
        
        # ax.plot(coords[:, 0] / x_scale, coords[:, 1], '--', color=st.colour_undeformed, 
        #         alpha=0.3, label='Undeformed')
        ax.plot(coords[:, 0] / x_scale, v_mode, color=st.colour_deformed, linewidth=st.line_width)
        
        ax.axhline(0, color=st.colour_zero_line, linewidth=1)
        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel('Modal Amplitude', fontsize=st.label_fontsize)
        ax.set_title(f'Mode Shape {mode_num} - {frequency:.2f} Hz', fontsize=st.title_fontsize, weight='bold')
        # ax.legend(fontsize=st.tick_fontsize)
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
        Visualize cross-section properties, centroid, and extreme fibers.
        
        Parameters:
        -----------
        props : SectionProperties
            The properties object to visualize
        title : str
            Title for the plot
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
        
        yc, zc = props.y_centroid, props.z_centroid
        
        # Fiber positions in reference system
        y_top = yc + props.y_top
        y_bot = yc + props.y_bottom
        z_left = zc + props.z_left
        z_right = zc + props.z_right
        
        # Draw bounding box
        box_y = [y_bot, y_top, y_top, y_bot, y_bot]
        box_z = [z_left, z_left, z_right, z_right, z_left]
        ax.fill(box_z, box_y, color='lightblue', alpha=0.3, label='Bounding Box')
        ax.plot(box_z, box_y, 'k--', alpha=0.5, label='Bounding Box')
        
        # Neutral Axes (passing through centroid)
        ax.axhline(yc, color='gray', linestyle=':', alpha=0.6, label='Neutral Axis (z)')
        ax.axvline(zc, color='gray', linestyle=':', alpha=0.6, label='Neutral Axis (y)')
        
        # Centroid Marker
        ax.plot(zc, yc, 'ro', markersize=10, label=f'Centroid ({zc:.1f}, {yc:.1f})')
        
        # Reference Origin
        ax.plot(0, 0, 'kx', markersize=12, label='Reference (0,0)')
        
        # Dimensions as annotations (cleaner than labels under dimensions)
        width = z_right - z_left
        height = y_top - y_bot
        
        # Add dimension lines with arrows
        ax.annotate('', xy=(z_left, y_bot - (height*0.1)), xytext=(z_right, y_bot - (height*0.1)),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        ax.text((z_left+z_right)/2, y_bot - (height*0.15), f'w={width:.1f}', ha='center', verticalalignment='top')
        
        ax.annotate('', xy=(z_right + (width*0.1), y_bot), xytext=(z_right + (width*0.1), y_top),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        ax.text(z_right + (width*0.15), (y_bot+y_top)/2, f'h={height:.1f}', va='center', rotation=90)
        
        # Labels and formatting
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.axis('equal')
        
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


if __name__ == "__main__":
    print("\nVisualization Module")
    print("Provides plotting tools for FEA results")
