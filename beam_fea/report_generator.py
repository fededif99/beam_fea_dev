"""
report_generator.py
===================
Professional report generation for beam FEA analysis results.

Generates markdown reports with linked visualizations including:
- Structural diagrams with boundary conditions and loads
- Deformed shape plots
- Shear force diagrams
- Bending moment diagrams
- Cross-section details
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import os
from .loads import PointLoad, UniformDistributedLoad, TrapezoidalDistributedLoad, TriangularDistributedLoad
from .boundary_conditions import PinnedSupport, RollerSupport, FixedSupport
from .plot_style import PlotStyle, smart_units, DEFAULT_STYLE


class BeamReportGenerator:
    """
    Generate professional markdown reports for beam analysis results.
    """
    
    def __init__(self, solver, mesh, material, section, load_case, bc_set,
                 displacements, reactions):
        """
        Initialize report generator.
        
        Parameters
        ----------
        solver : BeamSolver
            Beam solver instance
        mesh : Mesh
            Finite element mesh
        material : Material
            Material properties
        section : SectionProperties
            Cross-section properties
        load_case : LoadCase
            Applied load case
        bc_set : BoundaryConditionSet
            Boundary conditions
        displacements : np.ndarray
            Displacement vector
        reactions : np.ndarray
            Reaction forces
        """
        self.solver = solver
        self.mesh = mesh
        self.material = material
        self.section = section
        self.load_case = load_case
        self.bc_set = bc_set
        self.displacements = displacements
        self.reactions = reactions
        
        # Will store results
        self.positions = None
        self.shear_forces = None
        self.bending_moments = None
        self.stresses = None
        
    def calculate_internal_forces(self, num_points: int = 100):
        """
        Calculate shear force and bending moment along the beam using centralized solver logic.
        """
        results = self.solver.calculate_internal_forces(num_points)
        self.positions = results['positions']
        self.shear_forces = results['shear_forces']
        self.bending_moments = results['bending_moments']
    
    # ------------------------------------------------------------------
    # Private helpers – boundary condition glyphs
    # ------------------------------------------------------------------

    def _draw_fixed_support(self, ax, x, y, beam_h, side='left'):
        """Draw a hatched wall fixed support at (x, y)."""
        st = DEFAULT_STYLE
        w = beam_h * 0.6   # wall half-width
        # Thick vertical line at the connection point
        ax.plot([x, x], [y - w, y + w], color=st.colour_bc, linewidth=3, zorder=4)
        # Add 45 degree hatch lines
        n_hatch = 5
        hatch_len = beam_h * 0.3
        y_hatches = np.linspace(y - w, y + w, n_hatch)
        direction = -1 if side == 'left' else 1
        for yh in y_hatches:
            ax.plot([x, x + direction * hatch_len], [yh, yh - hatch_len],
                    color=st.colour_bc, linewidth=1.5, zorder=3)

    def _draw_pinned_support(self, ax, x, y, beam_h):
        """Draw a triangle pinned support beneath (x, y)."""
        st = DEFAULT_STYLE
        tri_h = beam_h * 0.55
        tri_w = beam_h * 0.55
        triangle = mpatches.Polygon(
            [[x, y], [x - tri_w / 2, y - tri_h], [x + tri_w / 2, y - tri_h]],
            closed=True, linewidth=1.5,
            edgecolor=st.colour_bc, facecolor='#d1fae5', zorder=3
        )
        ax.add_patch(triangle)
        # Ground line with hatching below
        ground_y = y - tri_h
        ax.plot([x - tri_w * 0.7, x + tri_w * 0.7], [ground_y, ground_y],
                color=st.colour_bc, linewidth=2, zorder=3)
        # Small hatch ticks below ground line
        n_ticks = 5
        tick_len = tri_h * 0.25
        xs_ticks = np.linspace(x - tri_w * 0.55, x + tri_w * 0.55, n_ticks)
        for xt in xs_ticks:
            ax.plot([xt, xt - tick_len * 0.5], [ground_y, ground_y - tick_len],
                    color=st.colour_bc, linewidth=1, zorder=3)

    def _draw_roller_support(self, ax, x, y, beam_h):
        """Draw a roller (circle-on-triangle) support beneath (x, y)."""
        st = DEFAULT_STYLE
        tri_h = beam_h * 0.45
        tri_w = beam_h * 0.50
        r_circle = beam_h * 0.10
        triangle = mpatches.Polygon(
            [[x, y], [x - tri_w / 2, y - tri_h], [x + tri_w / 2, y - tri_h]],
            closed=True, linewidth=1.5,
            edgecolor=st.colour_bc, facecolor='#d1fae5', zorder=3
        )
        ax.add_patch(triangle)
        # Small circles (wheels) at base corners
        for cx in [x - tri_w * 0.25, x, x + tri_w * 0.25]:
            circle = mpatches.Circle(
                (cx, y - tri_h - r_circle), r_circle,
                linewidth=1.2, edgecolor=st.colour_bc, facecolor='white', zorder=4
            )
            ax.add_patch(circle)
        # Ground line below wheels
        ground_y = y - tri_h - 2 * r_circle
        ax.plot([x - tri_w * 0.7, x + tri_w * 0.7], [ground_y, ground_y],
                color=st.colour_bc, linewidth=2, zorder=3)
        # Small hatch ticks
        n_ticks = 5
        tick_len = tri_h * 0.22
        xs_ticks = np.linspace(x - tri_w * 0.55, x + tri_w * 0.55, n_ticks)
        for xt in xs_ticks:
            ax.plot([xt, xt - tick_len * 0.5], [ground_y, ground_y - tick_len],
                    color=st.colour_bc, linewidth=1, zorder=3)

    # ------------------------------------------------------------------
    # Private helpers – load glyphs
    # ------------------------------------------------------------------

    def _draw_point_force_arrow(self, ax, x, y, fy, max_fy, arrow_scale,
                                colour, label_str):
        """Draw a vertical force arrow at (x, y)."""
        length = max(abs(fy) / max_fy * arrow_scale, arrow_scale * 0.25)
        hw = arrow_scale * 0.08
        hl = arrow_scale * 0.15
        if fy < 0:   # downward
            ax.annotate('', xy=(x, y), xytext=(x, y + length),
                        arrowprops=dict(arrowstyle=f'->', color=colour, lw=2,
                                        mutation_scale=12))
            ax.text(x + hw * 1.2, y + length * 0.5, label_str, color=colour,
                    fontsize=DEFAULT_STYLE.annotation_fontsize, weight='bold', va='center')
        else:         # upward
            ax.annotate('', xy=(x, y), xytext=(x, y - length),
                        arrowprops=dict(arrowstyle=f'->', color=colour, lw=2,
                                        mutation_scale=12))
            ax.text(x + hw * 1.2, y - length * 0.5, label_str, color=colour,
                    fontsize=DEFAULT_STYLE.annotation_fontsize, weight='bold', va='center')

    def _draw_moment_arc(self, ax, x, y, mz, radius, colour, label_str):
        """Draw a curved arc arrow to represent a concentrated moment."""
        import numpy as np
        # CCW for positive mz (sagging), CW for negative
        theta = np.linspace(0.2 * np.pi, 1.7 * np.pi, 60)
        if mz > 0:
            arc_x = x + radius * np.cos(theta)
            arc_y = y + radius * np.sin(theta)
            dx = arc_x[-1] - arc_x[-2]
            dy = arc_y[-1] - arc_y[-2]
        else:
            arc_x = x + radius * np.cos(-theta)
            arc_y = y + radius * np.sin(-theta)
            dx = arc_x[-1] - arc_x[-2]
            dy = arc_y[-1] - arc_y[-2]
        ax.plot(arc_x, arc_y, color=colour, linewidth=2.0, zorder=4)
        # Arrowhead at end of arc
        ax.annotate('', xy=(arc_x[-1] + dx * 0.01, arc_y[-1] + dy * 0.01),
                    xytext=(arc_x[-1], arc_y[-1]),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=2, mutation_scale=10))
        # Central dot
        ax.plot(x, y, 'o', color=colour, markersize=5, zorder=5)
        # Label
        ax.text(x + radius * 1.3, y, label_str, color=colour,
                fontsize=DEFAULT_STYLE.annotation_fontsize, weight='bold', va='center')

    def _draw_udl_comb(self, ax, x_start, x_end, wy, beam_h, colour,
                       label_str, wy2=None):
        """
        Draw a distributed-load arrow comb between x_start and x_end.
        If wy2 is given, the comb is tapered (trapezoidal / triangular).
        """
        span = abs(x_end - x_start)
        comb_h = beam_h * 1.1    # base comb height
        n_arrows = max(int(span / (beam_h * 0.6)), 3)
        xs = np.linspace(x_start, x_end, n_arrows)

        if wy2 is None:
            heights = np.full(n_arrows, comb_h)
        else:
            # Linear taper: height proportional to intensity
            max_w = max(abs(wy), abs(wy2))
            if max_w < 1e-12:
                return
            h1 = comb_h * abs(wy) / max_w
            h2 = comb_h * abs(wy2) / max_w
            heights = np.linspace(h1, h2, n_arrows)

        # Sign determines direction
        sign = -1 if wy < 0 else 1
        arrow_tips = np.zeros(n_arrows)
        arrow_bases = np.zeros(n_arrows)

        hw = comb_h * 0.04
        for i, (xi, hi) in enumerate(zip(xs, heights)):
            eff_h = hi if hi > comb_h * 0.05 else comb_h * 0.05
            tip_y = 0.0
            base_y = tip_y - sign * eff_h
            arrow_tips[i] = tip_y
            arrow_bases[i] = base_y
            ax.annotate('', xy=(xi, tip_y), xytext=(xi, base_y),
                        arrowprops=dict(arrowstyle='->', color=colour, lw=1.5,
                                       mutation_scale=8))

        # Cap line connecting arrow bases
        ax.plot(xs, arrow_bases, color=colour, linewidth=2, zorder=3)
        # Intensity label
        mid_x = (x_start + x_end) / 2
        label_y = arrow_bases.mean() - sign * comb_h * 0.25
        ax.text(mid_x, label_y, label_str, color=colour,
                fontsize=DEFAULT_STYLE.annotation_fontsize - 1,
                weight='bold', ha='center', va='center')

    def plot_structure_diagram(self, output_path: str, dpi: int = 150):
        """
        Plot beam structure with boundary conditions and loads.

        Boundary conditions are drawn as engineering-standard glyphs:
        - Fixed  : hatched wall rectangle
        - Pinned : triangle with ground hatching
        - Roller : triangle with wheels and ground line

        Loads use consistent arrow symbology:
        - Point Fy  : vertical solid arrow (scaled by max Fy)
        - Point Fx  : horizontal solid arrow (scaled by max Fx)
        - Moment Mz : curved arc arrow with direction sense
        - UDL       : uniform arrow comb with intensity label
        - Trapezoidal/Triangular: tapered arrow comb
        """
        st = DEFAULT_STYLE
        coords = self.mesh.get_node_coords()
        beam_length = coords[:, 0].max() - coords[:, 0].min()

        # Scale things relative to beam length in scaled units
        x_scale, x_unit = smart_units(beam_length, 'length')
        L_scaled = beam_length / x_scale
        
        arrow_scale = L_scaled * 0.08      # reference height for Fy arrows
        moment_radius = L_scaled * 0.03    # arc radius for Mz

        fig, ax = plt.subplots(figsize=st.figsize_structure, dpi=dpi)

        # ---- Beam line -------------------------------------------------
        ax.plot(coords[:, 0] / x_scale, coords[:, 1], '-',
                color=st.colour_primary, linewidth=st.beam_line_width, label='Beam', zorder=2)
        ax.plot(coords[:, 0] / x_scale, coords[:, 1], 'o',
                color=st.colour_primary, markersize=5, zorder=2)

        # ---- Characteristic height for glyph sizing --------------------
        # Use a fraction of beam length as a reference dimension
        glyph_h = L_scaled * 0.04

        # ---- Boundary conditions ---------------------------------------
        shown_bc_labels = set()
        for bc in self.bc_set.conditions:
            node_id = bc.node
            x = coords[node_id, 0] / x_scale
            y = coords[node_id, 1]

            if isinstance(bc, FixedSupport):
                # Decide which side the wall is on
                side = 'left' if node_id <= self.mesh.num_nodes // 2 else 'right'
                self._draw_fixed_support(ax, x, y, glyph_h, side=side)
                label = 'Fixed Support'
                if label not in shown_bc_labels:
                    ax.plot([], [], 's', color=st.colour_bc, markersize=10,
                            label=label, markerfacecolor='#d1fae5')
                    shown_bc_labels.add(label)
            elif isinstance(bc, PinnedSupport):
                self._draw_pinned_support(ax, x, y, glyph_h)
                label = 'Pinned Support'
                if label not in shown_bc_labels:
                    ax.plot([], [], '^', color=st.colour_bc, markersize=10,
                            label=label, markerfacecolor='#d1fae5')
                    shown_bc_labels.add(label)
            elif isinstance(bc, RollerSupport):
                self._draw_roller_support(ax, x, y, glyph_h)
                label = 'Roller Support'
                if label not in shown_bc_labels:
                    ax.plot([], [], 'o', color=st.colour_bc, markersize=10,
                            label=label, markerfacecolor='#d1fae5')
                    shown_bc_labels.add(label)

        # ---- Load scaling (per-component) ------------------------------
        all_fy = [abs(l.fy) for l in self.load_case.loads
                  if isinstance(l, PointLoad) and abs(l.fy) > 1e-10]
        all_fx = [abs(l.fx) for l in self.load_case.loads
                  if isinstance(l, PointLoad) and abs(l.fx) > 1e-10]
        all_mz = [abs(l.mz) for l in self.load_case.loads
                  if isinstance(l, PointLoad) and abs(l.mz) > 1e-10]
        max_fy = max(all_fy) if all_fy else 1.0
        max_fx = max(all_fx) if all_fx else 1.0
        max_mz = max(all_mz) if all_mz else 1.0

        # ---- Draw loads ------------------------------------------------
        for load in self.load_case.loads:
            if isinstance(load, PointLoad):
                if load.node is not None:
                    x_raw, y_pt = coords[load.node, 0], coords[load.node, 1]
                elif load.x is not None:
                    x_raw, y_pt = load.x, 0.0
                else:
                    continue
                if not (coords[:, 0].min() <= x_raw <= coords[:, 0].max()):
                    continue
                x_pt = x_raw / x_scale

                if abs(load.fy) > 1e-10:
                    f_scale, f_unit = smart_units(max_fy, 'force')
                    label_str = f"{load.fy / f_scale:+.2g} {f_unit}"
                    self._draw_point_force_arrow(
                        ax, x_pt, y_pt, load.fy, max_fy,
                        arrow_scale,  # already in scaled units
                        st.colour_load, label_str
                    )
                if abs(load.fx) > 1e-10:
                    f_scale, f_unit = smart_units(max_fx, 'force')
                    label_str = f"{load.fx / f_scale:+.2g} {f_unit} (axial)"
                    # Horizontal arrow
                    h_len = max(abs(load.fx) / max_fx * arrow_scale,
                                arrow_scale * 0.25)
                    if load.fx > 0:
                        ax.annotate('', xy=(x_pt, y_pt),
                                    xytext=(x_pt - h_len / x_scale, y_pt),
                                    arrowprops=dict(arrowstyle='->', lw=2,
                                                   color=st.colour_load_fx,
                                                   mutation_scale=12))
                    else:
                        ax.annotate('', xy=(x_pt, y_pt),
                                    xytext=(x_pt + h_len / x_scale, y_pt),
                                    arrowprops=dict(arrowstyle='->', lw=2,
                                                   color=st.colour_load_fx,
                                                   mutation_scale=12))
                    ax.text(x_pt, y_pt + arrow_scale * 0.12, label_str,
                            color=st.colour_load_fx,
                            fontsize=st.annotation_fontsize - 1, ha='center')
                if abs(load.mz) > 1e-10:
                    m_scale, m_unit = smart_units(max_mz, 'moment')
                    label_str = f"{load.mz / m_scale:+.2g} {m_unit}"
                    self._draw_moment_arc(
                        ax, x_pt, y_pt, load.mz,
                        moment_radius,
                        st.colour_moment_load, label_str
                    )

            elif isinstance(load, (UniformDistributedLoad,
                                   TrapezoidalDistributedLoad,
                                   TriangularDistributedLoad)):
                # Resolve span
                if hasattr(load, 'element') and load.element is not None:
                    elements = ([load.element] if isinstance(load.element, int)
                                else load.element)
                    x_s = coords[self.mesh.elements[elements[0]].node1, 0]
                    x_e = coords[self.mesh.elements[elements[-1]].node2, 0]
                elif (hasattr(load, 'x_start') and load.x_start is not None):
                    x_s, x_e = load.x_start, load.x_end
                else:
                    continue

                # Get intensities
                if isinstance(load, UniformDistributedLoad):
                    wy1 = load.wy
                    wy2 = None
                elif isinstance(load, TrapezoidalDistributedLoad):
                    wy1, wy2 = load.wy1, load.wy2
                else:  # Triangular
                    if load.peak_loc == 'start':
                        wy1, wy2 = load.w_peak, 0.0
                    else:
                        wy1, wy2 = 0.0, load.w_peak

                if abs(wy1) < 1e-12 and (wy2 is None or abs(wy2) < 1e-12):
                    continue

                ref_wy = max(abs(wy1), abs(wy2 or wy1))
                w_scale, w_unit = smart_units(ref_wy, 'udl')
                if wy2 is None:
                    label_str = f"{wy1 / w_scale:.3g} {w_unit}"
                else:
                    label_str = (f"{wy1 / w_scale:.3g}–"
                                 f"{wy2 / w_scale:.3g} {w_unit}")

                self._draw_udl_comb(
                    ax, x_s / x_scale, x_e / x_scale,
                    wy1, glyph_h,
                    st.colour_udl, label_str, wy2=wy2
                )

        # ---- Axes, grid, legend ----------------------------------------
        # Ensure aspect ratio is equal so 1m of beam looks physically identical to 1m of arrow/BC
        ax.set_aspect('equal', adjustable='datalim')
        # Add padding so large annotations don't get cut off
        ax.margins(y=0.4, x=0.05)

        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_title('Structure & Loading Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.legend(fontsize=st.tick_fontsize, loc='upper right')
        ax.grid(True, alpha=st.grid_alpha)

        # Hide y-axis since there's no meaning to the physical y-span
        ax.get_yaxis().set_visible(False)

        # Draw line representing the ground for visual grounding
        y_min = ax.get_ylim()[0]
        ax.hlines(0, coords[:, 0].min() / x_scale, coords[:, 0].max() / x_scale,
                  colors='gray', linestyles='--', alpha=0.3, zorder=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_deformed_shape(self, output_path: str, scale_factor: Union[float, str] = 'auto', dpi: int = 150):
        """
        Plot deformed shape compared to undeformed.
        """
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        
        if scale_factor == 'auto':
            s = viz._auto_scale_factor(self.displacements)
        else:
            s = float(scale_factor)
            
        viz.plot_deformed_shape(
            self.displacements,
            scale_factor=s,
            dpi=dpi,
            output_path=output_path
        )
    
    def plot_shear_diagram(self, output_path: str, dpi: int = 150):
        """
        Plot shear force diagram.

        The maximum absolute shear is annotated with a red dot, a horizontal
        dotted line extending to the y-axis, and a custom y-tick showing the
        value — no yellow annotation box.
        """
        st = DEFAULT_STYLE
        if self.shear_forces is None:
            self.calculate_internal_forces()

        beam_length = self.positions[-1] - self.positions[0]
        x_scale, x_unit = smart_units(beam_length, 'length')
        max_abs_V = np.max(np.abs(self.shear_forces))
        v_scale, v_unit = smart_units(max_abs_V, 'force')

        pos_scaled = self.positions / x_scale
        V_scaled   = self.shear_forces / v_scale

        fig, ax = plt.subplots(figsize=st.figsize_wide, dpi=dpi)

        ax.plot(pos_scaled, V_scaled, '-', color=st.colour_primary,
                linewidth=st.line_width)
        ax.fill_between(pos_scaled, 0, V_scaled,
                        alpha=st.fill_alpha, color=st.colour_primary)
        ax.axhline(0, color=st.colour_zero_line, linewidth=1)

        # --- Max annotation: dot + dotted line to y-axis + axis tick ---
        max_idx    = np.argmax(np.abs(self.shear_forces))
        x_max      = pos_scaled[max_idx]
        v_max      = V_scaled[max_idx]
        x_left     = pos_scaled[0]

        ax.plot(x_max, v_max, 'o', color=st.colour_max, markersize=8, zorder=5)
        ax.hlines(v_max, x_left, x_max,
                  colors=st.colour_max, linestyles=':', linewidth=1.5, zorder=4)

        # Filter ticks that clash with the max value
        existing_ticks = ax.get_yticks()
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        min_dist = y_range * 0.05
        filtered_ticks = [t for t in existing_ticks if abs(t - v_max) > min_dist]
        filtered_ticks.append(v_max)
        ax.set_yticks(filtered_ticks)

        # Apply formatting before assigning colours to ensure labels exist
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        fig.canvas.draw()
        
        for val, lbl in zip(ax.get_yticks(), ax.get_yticklabels()):
            if abs(val - v_max) < abs(v_max) * 1e-3 + 1e-6:
                lbl.set_color(st.colour_max)
                lbl.set_weight('bold')

        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel(f'Shear Force ({v_unit})', fontsize=st.label_fontsize)
        ax.set_title('Shear Force Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_moment_diagram(self, output_path: str, dpi: int = 150):
        """
        Plot bending moment diagram.

        The maximum absolute moment is annotated with a red dot, a horizontal
        dotted line extending to the y-axis, and a custom y-tick showing the
        value — no yellow annotation box.
        """
        st = DEFAULT_STYLE
        if self.bending_moments is None:
            self.calculate_internal_forces()

        beam_length = self.positions[-1] - self.positions[0]
        x_scale, x_unit = smart_units(beam_length, 'length')
        max_abs_M = np.max(np.abs(self.bending_moments))
        m_scale, m_unit = smart_units(max_abs_M, 'moment')

        pos_scaled = self.positions / x_scale
        M_scaled   = self.bending_moments / m_scale

        fig, ax = plt.subplots(figsize=st.figsize_wide, dpi=dpi)

        ax.plot(pos_scaled, M_scaled, '-', color=st.colour_moment,
                linewidth=st.line_width)
        ax.fill_between(pos_scaled, 0, M_scaled,
                        alpha=st.fill_alpha, color=st.colour_moment)
        ax.axhline(0, color=st.colour_zero_line, linewidth=1)

        # --- Max annotation: dot + dotted line to y-axis + axis tick ---
        max_idx    = np.argmax(np.abs(self.bending_moments))
        x_max      = pos_scaled[max_idx]
        m_max      = M_scaled[max_idx]
        x_left     = pos_scaled[0]

        ax.plot(x_max, m_max, 'o', color=st.colour_max, markersize=8, zorder=5)
        ax.hlines(m_max, x_left, x_max,
                  colors=st.colour_max, linestyles=':', linewidth=1.5, zorder=4)

        # Filter ticks that clash with the max value
        existing_ticks = ax.get_yticks()
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        min_dist = y_range * 0.05
        filtered_ticks = [t for t in existing_ticks if abs(t - m_max) > min_dist]
        filtered_ticks.append(m_max)
        ax.set_yticks(filtered_ticks)

        # Apply formatting before assigning colours to ensure labels exist
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        fig.canvas.draw()

        for val, lbl in zip(ax.get_yticks(), ax.get_yticklabels()):
            if abs(val - m_max) < abs(m_max) * 1e-3 + 1e-6:
                lbl.set_color(st.colour_max)
                lbl.set_weight('bold')

        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel(f'Bending Moment ({m_unit})', fontsize=st.label_fontsize)
        ax.set_title('Bending Moment Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_stress_distributions(self, output_path: str, dpi: int = 150):
        """
        Plot peak stress distributions (Bending, Shear, von Mises) along the beam.
        """
        if self.stresses is None:
            # Default resolution for report plots
            self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)
        
        x_coords = self.stresses['x']
        
        # Extract peak values at each x-station
        # axes: (x, y, z)
        vm_peak = np.max(self.stresses['von_mises'], axis=(1, 2))
        bending_peak = np.max(np.abs(self.stresses['bending']), axis=(1, 2))
        shear_peak = np.max(np.abs(self.stresses['shear']), axis=(1, 2))
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
        
        ax.plot(x_coords, vm_peak, 'k-', linewidth=2.5, label='Peak von Mises')
        ax.plot(x_coords, bending_peak, 'r--', linewidth=1.5, label='Peak Bending (abs)')
        ax.plot(x_coords, shear_peak, 'b:', linewidth=1.5, label='Peak Shear (abs)')
        
        ax.fill_between(x_coords, 0, vm_peak, color='gray', alpha=0.1)
        
        # Highlight absolute maximum
        max_vm_val = np.max(vm_peak)
        max_vm_idx = np.argmax(vm_peak)
        ax.plot(x_coords[max_vm_idx], max_vm_val, 'ro', markersize=8)
        
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Stress (MPa)', fontsize=12)
        ax.set_title('Peak Stress Distribution along Beam Length', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_cross_section(self, output_path: str, dpi: int = 150):
        """
        Plot cross-section using BeamVisualizer.
        """
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        viz.plot_section_properties(
            self.section,
            output_path=output_path,
            dpi=dpi
        )
    
    def generate_report(self, output_path: str, deformation_scale: Union[float, str] = 'auto'):
        """
        Generate complete markdown report. Images are saved to a
        ``<report_name>_images/`` folder alongside the report file.
        
        Parameters
        ----------
        output_path : str
            Path for the markdown report file (e.g., 'report.md')
        deformation_scale : float or 'auto'
            Scale factor for deformation visualization
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        report_name = output_path.stem
        
        images_dir = output_dir / f"{report_name}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        images_rel_path = f"{report_name}_images"
        
        # Stress and image generation pipeline
        self.calculate_internal_forces()
        
        # [Fix]: Compute stresses once and cache it
        if self.stresses is None:
            self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)

        # Generate all plots
        self.plot_structure_diagram(str(images_dir / "structure_diagram.png"))
        self.plot_cross_section(str(images_dir / "cross_section.png"))
        self.plot_deformed_shape(str(images_dir / "deformed_shape.png"), scale_factor=deformation_scale)
        self.plot_shear_diagram(str(images_dir / "shear_diagram.png"))
        self.plot_moment_diagram(str(images_dir / "moment_diagram.png"))
        self.plot_stress_distributions(str(images_dir / "stress_distribution.png"))
        
        # Calculate key results
        v_displacements = self.displacements[1::3]
        max_deflection = abs(min(v_displacements))
        max_deflection_node = v_displacements.argmin()
        max_deflection_pos = self.mesh.nodes[max_deflection_node].x
        
        max_shear = np.max(np.abs(self.shear_forces))
        max_moment = np.max(np.abs(self.bending_moments))
        
        # Determine actual scale used for deformation plot label
        if deformation_scale == 'auto':
            from .visualizer import BeamVisualizer
            _s = BeamVisualizer(self.mesh)._auto_scale_factor(self.displacements)
        else:
            _s = float(deformation_scale)
        
        # Generate markdown content
        md_content = f"""# Beam FEA Analysis Report

**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

---

## 1. Analysis Overview

This report presents the results of a finite element analysis of a beam structure subjected to static loading.

### Load Case
- **Name:** {self.load_case.name}
- **Number of Point Loads:** {len([l for l in self.load_case.loads if isinstance(l, PointLoad)])}
- **Number of Distributed Loads:** {len([l for l in self.load_case.loads if isinstance(l, UniformDistributedLoad)])}

---

## 2. Model Information

### Mesh Details

| Property | Value |
|----------|-------|
| Number of Nodes | {self.mesh.num_nodes} |
| Number of Elements | {self.mesh.num_elements} |
| Total DOFs | {self.mesh.num_dofs} |
| Element Type | Euler-Bernoulli Beam |

### Material Properties

| Property | Value | Units |
|----------|-------|-------|
| Material | {self.material.name} | - |
| Young's Modulus (E) | {self.material.E:,.0f} | MPa |
| Shear Modulus (G) | {self.material.G:,.0f} | MPa |
| Density (ρ) | {self.material.rho:.2e} | kg/mm³ |
| Poisson's Ratio (ν) | {self.material.nu:.3f} | - |

### Cross-Section Properties

| Property | Value | Units |
|----------|-------|-------|
| Area (A) | {self.section.A:,.2f} | mm² |
| Moment of Inertia (Iy) | {self.section.Iy:.2e} | mm⁴ |
| Moment of Inertia (Iz) | {self.section.Iz:.2e} | mm⁴ |

---

## 3. Structural Model

### Beam Geometry with Boundary Conditions and Loads

![Structure Diagram]({images_rel_path}/structure_diagram.png)

### Cross-Section Details

![Cross-Section]({images_rel_path}/cross_section.png)

---

## 4. Analysis Results

### Deformed Shape

The following plot shows the deformed shape of the beam (exaggerated by {_s:.0f}× for visualization):

![Deformed Shape]({images_rel_path}/deformed_shape.png)

### Maximum Deflection

| Property | Value |
|----------|-------|
| Maximum Deflection | {max_deflection:.4f} mm |
| Location | Node {max_deflection_node} (x = {max_deflection_pos:.1f} mm) |

### Reaction Forces

"""
        
        # Add reaction forces
        for bc in self.bc_set.conditions:
            node_id = bc.node
            if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
                fy_reaction = self.reactions[3*node_id + 1]
                md_content += f"- **Node {node_id}:** Vertical reaction = {fy_reaction:.2f} N ({fy_reaction/1000:.2f} kN)\n"
        
        # Calculate total reaction and applied load
        total_fy = sum(
            self.reactions[3*bc.node + 1]
            for bc in self.bc_set.conditions
            if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport))
        )
        total_load = sum(abs(getattr(load, 'fy', 0)) 
                        for load in self.load_case.loads if hasattr(load, 'fy'))
        for load in self.load_case.loads:
            if hasattr(load, 'wy'):
                wy = getattr(load, 'wy', 0)
                if wy == 0: continue
                
                if hasattr(load, 'element') and load.element is not None:
                    elements = [load.element] if isinstance(load.element, int) else load.element
                    for elem_id in elements:
                        elem_length = self.mesh.elements[elem_id].length(self.mesh.nodes)
                        total_load += abs(wy) * elem_length
                elif hasattr(load, 'x_start') and load.x_start is not None:
                    total_load += abs(wy) * abs(load.x_end - load.x_start)

        md_content += f"""
### Equilibrium Check

| Property | Value |
|----------|-------|
| Total Vertical Reaction | {total_fy:.2f} N ({total_fy/1000:.2f} kN) |
| Total Applied Load | {abs(total_load):.2f} N ({abs(total_load)/1000:.2f} kN) |
| Difference | {abs(total_fy - abs(total_load)):.2e} N |

---

## 5. Internal Forces

### Shear Force Diagram

![Shear Force Diagram]({images_rel_path}/shear_diagram.png)

**Maximum Shear Force:** {max_shear/1000:.2f} kN

### Bending Moment Diagram

![Bending Moment Diagram]({images_rel_path}/moment_diagram.png)

**Maximum Bending Moment:** {max_moment/1e6:.2f} kN·m

---

- The maximum deflection of **{max_deflection:.4f} mm** occurs at x = {max_deflection_pos:.1f} mm
- The maximum shear force is **{max_shear/1000:.2f} kN**
- The maximum bending moment is **{max_moment/1e6:.2f} kN·m**
- **Peak von Mises Stress: {np.max(self.stresses['von_mises']):.2f} MPa**
- Equilibrium is satisfied with a residual of {abs(total_fy - abs(total_load)):.2e} N

---

## 6. Stress Analysis

### Peak Stress Distributions

The following plot illustrates the peak internal stresses (von Mises, Bending, and Shear) as they vary along the length of the beam. 

![Stress Distribution]({images_rel_path}/stress_distribution.png)

### Summary of Peak Stresses

| Stress Component | Maximum Value | Units |
|------------------|---------------|-------|
| von Mises (Peak) | {np.max(self.stresses['von_mises']):.2f} | MPa |
| Bending (Max)    | {np.max(np.abs(self.stresses['bending'])):.2f} | MPa |
| Shear (Max)      | {np.max(np.abs(self.stresses['shear'])):.2f} | MPa |
| Axial (Max)      | {np.max(np.abs(self.stresses['axial'])):.2f} | MPa |

### Structural Integrity

| Criterion | Value |
|-----------|-------|
| Material Yield Strength | {self.material.yield_strength:,.1f} MPa |
| Peak von Mises Stress | {np.max(self.stresses['von_mises']):.2f} MPa |
| **Factor of Safety** | **{(self.material.yield_strength / np.max(self.stresses['von_mises'])) if np.max(self.stresses['von_mises']) > 0 else 0:.2f}** |

---

## 7. Summary

The finite element analysis has been successfully completed. Key findings:

- The maximum deflection of **{max_deflection:.4f} mm** occurs at x = {max_deflection_pos:.1f} mm
- The maximum shear force is **{max_shear/1000:.2f} kN**
- The maximum bending moment is **{max_moment/1e6:.2f} kN·m**
- The peak internal stress (von Mises) is **{np.max(self.stresses['von_mises']):.2f} MPa**
- Equilibrium is satisfied with a residual of {abs(total_fy - abs(total_load)):.2e} N

---

*Report generated by Beam FEA Analysis Tool*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(output_path)


if __name__ == "__main__":
    print("\nBeam Report Generator Module")
    print("Generates professional markdown reports for beam analysis")
