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
        
        # Modal Results
        self.frequencies = None
        self.mode_shapes = None
        
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
        ax.plot([x, x], [y - w, y + w], color=st.colour_bc, linewidth=st.bc_line_width, zorder=4)
        # Add 45 degree hatch lines
        n_hatch = 5
        hatch_len = beam_h * 0.3
        y_hatches = np.linspace(y - w, y + w, n_hatch)
        direction = -1 if side == 'left' else 1
        for yh in y_hatches:
            ax.plot([x, x + direction * hatch_len], [yh, yh - hatch_len],
                    color=st.colour_bc, linewidth=st.bc_hatch_width, zorder=3)

    def _draw_pinned_support(self, ax, x, y, beam_h):
        """Draw a triangle pinned support beneath (x, y)."""
        st = DEFAULT_STYLE
        tri_h = beam_h * 0.55
        tri_w = beam_h * 0.55
        triangle = mpatches.Polygon(
            [[x, y], [x - tri_w / 2, y - tri_h], [x + tri_w / 2, y - tri_h]],
            closed=True, linewidth=st.bc_line_width * 0.75,
            edgecolor=st.colour_bc, facecolor='#d1fae5', zorder=3
        )
        ax.add_patch(triangle)
        # Ground line with hatching below
        ground_y = y - tri_h
        ax.plot([x - tri_w * 0.7, x + tri_w * 0.7], [ground_y, ground_y],
                color=st.colour_bc, linewidth=st.bc_line_width, zorder=3)
        # Small hatch ticks below ground line
        n_ticks = 5
        tick_len = tri_h * 0.25
        xs_ticks = np.linspace(x - tri_w * 0.55, x + tri_w * 0.55, n_ticks)
        for xt in xs_ticks:
            ax.plot([xt, xt - tick_len * 0.5], [ground_y, ground_y - tick_len],
                    color=st.colour_bc, linewidth=st.bc_hatch_width, zorder=3)

    def _draw_roller_support(self, ax, x, y, beam_h):
        """Draw a roller (circle-on-triangle) support beneath (x, y)."""
        st = DEFAULT_STYLE
        tri_h = beam_h * 0.45
        tri_w = beam_h * 0.50
        r_circle = beam_h * 0.10
        triangle = mpatches.Polygon(
            [[x, y], [x - tri_w / 2, y - tri_h], [x + tri_w / 2, y - tri_h]],
            closed=True, linewidth=st.bc_line_width * 0.75,
            edgecolor=st.colour_bc, facecolor='#d1fae5', zorder=3
        )
        ax.add_patch(triangle)
        # Small circles (wheels) at base corners
        for cx in [x - tri_w * 0.25, x, x + tri_w * 0.25]:
            circle = mpatches.Circle(
                (cx, y - tri_h - r_circle), r_circle,
                linewidth=st.bc_hatch_width * 1.2, edgecolor=st.colour_bc, facecolor='white', zorder=4
            )
            ax.add_patch(circle)
        # Ground line below wheels
        ground_y = y - tri_h - 2 * r_circle
        ax.plot([x - tri_w * 0.7, x + tri_w * 0.7], [ground_y, ground_y],
                color=st.colour_bc, linewidth=st.bc_line_width, zorder=3)
        # Small hatch ticks
        n_ticks = 5
        tick_len = tri_h * 0.22
        xs_ticks = np.linspace(x - tri_w * 0.55, x + tri_w * 0.55, n_ticks)
        for xt in xs_ticks:
            ax.plot([xt, xt - tick_len * 0.5], [ground_y, ground_y - tick_len],
                    color=st.colour_bc, linewidth=st.bc_hatch_width, zorder=3)

    # ------------------------------------------------------------------
    # Private helpers – load glyphs
    # ------------------------------------------------------------------

    def _draw_point_force_arrow(self, ax, x, y, fy, max_fy, arrow_scale,
                                colour, label_str, is_reaction=False, x_max=None, stack_level=0):
        """Draw a vertical force arrow at (x, y)."""
        st = DEFAULT_STYLE
        length = max(abs(fy) / max_fy * arrow_scale, arrow_scale * 0.25)
        hw = arrow_scale * 0.08
        
        # Standardize weights and sizes
        if is_reaction:
            lw = st.reaction_line_width
            mut = st.reaction_mutation_scale
        else:
            lw = st.load_line_width
            mut = st.load_mutation_scale
            
        fs = st.annotation_fontsize - 1

        # Determine alignment based on position
        ha_val = 'left'
        x_offset = hw * st.annot_offset_fy_horz
        if x_max is not None and x >= x_max - 1e-4:
            ha_val = 'right'
            x_offset = -hw * st.annot_offset_fy_horz

        y_shift = arrow_scale * ((st.annot_offset_fy_vert - 1.0) + stack_level * 0.4)

        if fy < 0:   # downward load (text goes UP above the tail)
            ax.annotate('', xy=(x, y), xytext=(x, y + length),
                        arrowprops=dict(arrowstyle=f'->', color=colour, lw=lw,
                                        mutation_scale=mut))
            ax.text(x + x_offset, y + length + y_shift + arrow_scale * 0.05, label_str, color=colour,
                    fontsize=fs, weight='bold', va='center', ha=ha_val)
        else:         # upward load/reaction (text goes DOWN below the tail)
            ax.annotate('', xy=(x, y), xytext=(x, y - length),
                        arrowprops=dict(arrowstyle=f'->', color=colour, lw=lw,
                                        mutation_scale=mut))
            ax.text(x + x_offset, y - length - y_shift - arrow_scale * 0.15, label_str, color=colour,
                    fontsize=fs, weight='bold', va='center', ha=ha_val)

    def _draw_moment_arc(self, ax, x, y, mz, radius, colour, label_str, is_reaction=False, stack_level=0):
        """Draw a curved arc arrow to represent a concentrated moment."""
        import numpy as np
        st = DEFAULT_STYLE
        # CCW for positive mz (sagging), CW for negative
        theta = np.linspace(0.2 * np.pi, 1.7 * np.pi, 60)
        
        if is_reaction:
            lw = st.reaction_line_width
            mut = st.reaction_mutation_scale
        else:
            lw = st.load_line_width
            mut = st.load_mutation_scale
            
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
        ax.plot(arc_x, arc_y, color=colour, linewidth=lw, zorder=4)
        # Arrowhead at end of arc
        ax.annotate('', xy=(arc_x[-1] + dx * 0.01, arc_y[-1] + dy * 0.01),
                    xytext=(arc_x[-1], arc_y[-1]),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=lw, mutation_scale=mut))
        # Central dot
        ax.plot(x, y, 'o', color=colour, markersize=5, zorder=5)
        
        # Label - Shifting vertically ABOVE the beam
        y_offset = radius * (st.annot_offset_mz + stack_level * 0.8)
        ax.text(x, y + y_offset, label_str, color=colour,
                fontsize=st.annotation_fontsize, weight='bold', ha='center', va='center')

    def _draw_udl_comb(self, ax, x_start, x_end, wy, beam_h, colour,
                       label_str, wy2=None):
        """
        Draw a distributed-load arrow comb between x_start and x_end.
        If wy2 is given, the comb is tapered (trapezoidal / triangular).
        """
        st = DEFAULT_STYLE
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
                        arrowprops=dict(arrowstyle='->', color=colour, 
                                        lw=st.load_line_width * 0.8,
                                        mutation_scale=st.load_mutation_scale * 0.8))

        # Cap line connecting arrow bases
        ax.plot(xs, arrow_bases, color=colour, linewidth=st.load_line_width * 1.2, zorder=3)
        # Intensity label
        mid_x = (x_start + x_end) / 2
        label_y = arrow_bases.mean() - sign * comb_h * st.annot_offset_udl
        ax.text(mid_x, label_y, label_str, color=colour,
                fontsize=st.annotation_fontsize - 1,
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
        if self.load_case:
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
            stack_fy = {}
            stack_fx = {}
            stack_mz = {}
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

                    x_key = round(x_pt, 4)

                    if abs(load.fy) > 1e-10:
                        stack = stack_fy.get(x_key, 0)
                        f_scale, f_unit = smart_units(max_fy, 'force')
                        label_str = f"{load.fy / f_scale:+.3g} {f_unit}"
                        self._draw_point_force_arrow(
                            ax, x_pt, y_pt, load.fy, max_fy,
                            arrow_scale, st.colour_load, label_str, is_reaction=False,
                            x_max=coords[:, 0].max() / x_scale,
                            stack_level=stack
                        )
                        stack_fy[x_key] = stack + 1
                        
                    if abs(load.fx) > 1e-10:
                        f_scale, f_unit = smart_units(max_fx, 'force')
                        label_str = f"{load.fx / f_scale:+.3g} {f_unit} (axial)"
                        h_len = max(abs(load.fx) / max_fx * arrow_scale, arrow_scale * 0.25)
                        if load.fx > 0:
                            ax.annotate('', xy=(x_pt, y_pt), xytext=(x_pt - h_len / x_scale, y_pt),
                                        arrowprops=dict(arrowstyle='->', lw=st.load_line_width, color=st.colour_load,
                                                       mutation_scale=st.load_mutation_scale))
                        else:
                            ax.annotate('', xy=(x_pt, y_pt), xytext=(x_pt + h_len / x_scale, y_pt),
                                        arrowprops=dict(arrowstyle='->', lw=st.load_line_width, color=st.colour_load,
                                                       mutation_scale=st.load_mutation_scale))
                        # Dynamic text positioning for horizontal loads
                        ha_val = 'center'
                        x_label = x_pt
                        if x_pt >= coords[:, 0].max() / x_scale - 1e-4:
                            ha_val = 'right'
                            x_label = x_pt - st.annot_offset_fx_horz * arrow_scale
                        elif x_pt <= coords[:, 0].min() / x_scale + 1e-4:
                            ha_val = 'left'
                            x_label = x_pt + st.annot_offset_fx_horz * arrow_scale

                        stack = stack_fx.get(x_key, 0)
                        y_text = y_pt - arrow_scale * (st.annot_offset_fx_vert + stack * 0.4)

                        ax.text(x_label, y_text, label_str, color=st.colour_load,
                                fontsize=st.annotation_fontsize - 1, ha=ha_val)
                        stack_fx[x_key] = stack + 1
                        
                    if abs(load.mz) > 1e-10:
                        stack = stack_mz.get(x_key, 0)
                        m_scale, m_unit = smart_units(max_mz, 'moment')
                        label_str = f"{load.mz / m_scale:+.3g} {m_unit}"
                        self._draw_moment_arc(
                            ax, x_pt, y_pt, load.mz, moment_radius, st.colour_moment_load, label_str,
                            is_reaction=False, stack_level=stack
                        )
                        stack_mz[x_key] = stack + 1

                elif isinstance(load, (UniformDistributedLoad, TrapezoidalDistributedLoad, TriangularDistributedLoad)):
                    if hasattr(load, 'element') and load.element is not None:
                        elements = ([load.element] if isinstance(load.element, int) else load.element)
                        x_s = coords[self.mesh.elements[elements[0]].node1, 0]
                        x_e = coords[self.mesh.elements[elements[-1]].node2, 0]
                    elif (hasattr(load, 'x_start') and load.x_start is not None):
                        x_s, x_e = load.x_start, load.x_end
                    else: continue

                    if isinstance(load, UniformDistributedLoad): wy1, wy2 = load.wy, None
                    elif isinstance(load, TrapezoidalDistributedLoad): wy1, wy2 = load.wy1, load.wy2
                    else: 
                        if load.peak_loc == 'start': wy1, wy2 = load.w_peak, 0.0
                        else: wy1, wy2 = 0.0, load.w_peak

                    if abs(wy1) < 1e-12 and (wy2 is None or abs(wy2) < 1e-12): continue
                    ref_wy = max(abs(wy1), abs(wy2 or wy1))
                    w_scale, w_unit = smart_units(ref_wy, 'udl')
                    label_str = f"{wy1/w_scale:.3g} {w_unit}" if wy2 is None else f"{wy1/w_scale:.3g}–{wy2/w_scale:.3g} {w_unit}"
                    self._draw_udl_comb(ax, x_s/x_scale, x_e/x_scale, wy1, glyph_h, st.colour_udl, label_str, wy2=wy2)

        # ---- Axes, grid, legend ----------------------------------------
        ax.set_aspect('equal', adjustable='datalim')
        ax.margins(y=0.4, x=0.05)
        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_title('Structure & Loading Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.legend(fontsize=st.tick_fontsize, loc='upper right')
        ax.grid(True, alpha=st.grid_alpha)
        ax.get_yaxis().set_visible(False)
        ax.hlines(0, coords[:, 0].min() / x_scale, coords[:, 0].max() / x_scale, colors='gray', linestyles='--', alpha=0.3, zorder=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_reaction_diagram(self, output_path: str, dpi: int = 150):
        """
        Plot Free Body Diagram (FBD) showing applied loads and reaction forces.
        Boundary condition glyphs are replaced by reaction force and moment vectors.
        """
        st = DEFAULT_STYLE
        coords = self.mesh.get_node_coords()
        beam_length = coords[:, 0].max() - coords[:, 0].min()

        # Scale relative to scaled beam length
        x_scale, x_unit = smart_units(beam_length, 'length')
        L_scaled = beam_length / x_scale
        
        arrow_scale = L_scaled * 0.08
        moment_radius = L_scaled * 0.03
        glyph_h = L_scaled * 0.04

        fig, ax = plt.subplots(figsize=st.figsize_structure, dpi=dpi)

        # ---- Beam line -------------------------------------------------
        ax.plot(coords[:, 0] / x_scale, coords[:, 1], '-',
                color=st.colour_primary, linewidth=st.beam_line_width, label='Beam', zorder=2)
        ax.plot(coords[:, 0] / x_scale, coords[:, 1], 'o',
                color=st.colour_primary, markersize=5, zorder=2)

        # ---- Load scaling (per-component) including reactions ----------
        all_fy = []
        all_fx = []
        all_mz = []
        
        if self.load_case:
            all_fy = [abs(l.fy) for l in self.load_case.loads if isinstance(l, PointLoad)]
            all_fx = [abs(l.fx) for l in self.load_case.loads if isinstance(l, PointLoad)]
            all_mz = [abs(l.mz) for l in self.load_case.loads if isinstance(l, PointLoad)]
        
        for bc in self.bc_set.conditions:
            node_id = bc.node
            if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
                all_fx.append(abs(self.reactions[3*node_id]))
                all_fy.append(abs(self.reactions[3*node_id + 1]))
                all_mz.append(abs(self.reactions[3*node_id + 2]))

        max_fy = max(all_fy) if all_fy and max(all_fy) > 1e-10 else 1.0
        max_fx = max(all_fx) if all_fx and max(all_fx) > 1e-10 else 1.0
        max_mz = max(all_mz) if all_mz and max(all_mz) > 1e-10 else 1.0

        # ---- Draw Reaction Forces --------------------------------------
        reaction_color = '#28a745' # Success green for stability
        threshold = 1e-5
        fs_react = st.annotation_fontsize - 1
        stack_fy = {}
        stack_fx = {}
        stack_mz = {}
        
        for i, bc in enumerate(self.bc_set.conditions):
            node_id = bc.node
            x_pt = coords[node_id, 0] / x_scale
            y_pt = coords[node_id, 1]
            x_key = round(x_pt, 4)

            if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
                fx = self.reactions[3*node_id]
                fy = self.reactions[3*node_id + 1]
                mz = self.reactions[3*node_id + 2]
                
                if abs(fy) > threshold:
                    stack = stack_fy.get(x_key, 0)
                    f_scale, f_unit = smart_units(max_fy, 'force')
                    label_str = rf"$\mathbf{{R_{{y,{i+1}}}}}$={fy / f_scale:+.3g} {f_unit}"
                    self._draw_point_force_arrow(ax, x_pt, y_pt, fy, max_fy, arrow_scale, reaction_color, label_str, is_reaction=True, x_max=coords[:, 0].max() / x_scale, stack_level=stack)
                    stack_fy[x_key] = stack + 1
                if abs(fx) > threshold:
                    f_scale, f_unit = smart_units(max_fx, 'force')
                    label_str = rf"$\mathbf{{R_{{x,{i+1}}}}}$={fx / f_scale:+.3g} {f_unit}"
                    h_len = max(abs(fx) / max_fx * arrow_scale, arrow_scale * 0.25)
                    tip_x = x_pt + h_len / x_scale if fx > 0 else x_pt - h_len / x_scale
                    ha_val = 'left' if fx > 0 else 'right'
                    
                    x_label = tip_x
                    if x_pt >= coords[:, 0].max() / x_scale - 1e-4:
                        ha_val, x_label = 'right', tip_x - st.annot_offset_fx_horz * arrow_scale
                    elif x_pt <= coords[:, 0].min() / x_scale + 1e-4:
                        ha_val, x_label = 'left', tip_x + st.annot_offset_fx_horz * arrow_scale
                        
                    ax.annotate('', xy=(tip_x, y_pt), xytext=(x_pt, y_pt), arrowprops=dict(arrowstyle='->', lw=st.reaction_line_width, color=reaction_color, mutation_scale=st.reaction_mutation_scale))
                    stack = stack_fx.get(x_key, 0)
                    y_text = y_pt - arrow_scale * (st.annot_offset_fx_vert + stack * 0.4)
                    ax.text(x_label, y_text, label_str, color=reaction_color, fontsize=fs_react, ha=ha_val, va='top', weight='bold')
                    stack_fx[x_key] = stack + 1
                    
                if abs(mz) > threshold:
                    stack = stack_mz.get(x_key, 0)
                    m_scale, m_unit = smart_units(max_mz, 'moment')
                    label_str = rf"$\mathbf{{M_{{z,{i+1}}}}}$={mz / m_scale:+.3g} {m_unit}"
                    self._draw_moment_arc(ax, x_pt, y_pt, mz, moment_radius, reaction_color, label_str, is_reaction=True, stack_level=stack)
                    stack_mz[x_key] = stack + 1

        ax.plot([], [], 's', color=reaction_color, label='Reactions', markerfacecolor='#fca5a5')

        # ---- Draw Loads ------------------------------------------------
        if self.load_case:
            for load in self.load_case.loads:
                if isinstance(load, PointLoad):
                    if load.node is not None: x_raw, y_pt = coords[load.node, 0], coords[load.node, 1]
                    elif load.x is not None: x_raw, y_pt = load.x, 0.0
                    else: continue
                    if not (coords[:, 0].min() <= x_raw <= coords[:, 0].max()): continue
                    x_pt = x_raw / x_scale
                    x_key = round(x_pt, 4)

                    if abs(load.fy) > 1e-10:
                        stack = stack_fy.get(x_key, 0)
                        f_scale, f_unit = smart_units(max_fy, 'force')
                        label_str = f"{load.fy / f_scale:+.3g} {f_unit}"
                        self._draw_point_force_arrow(ax, x_pt, y_pt, load.fy, max_fy, arrow_scale, st.colour_load, label_str, is_reaction=False, x_max=coords[:, 0].max() / x_scale, stack_level=stack)
                        stack_fy[x_key] = stack + 1
                    if abs(load.fx) > 1e-10:
                        f_scale, f_unit = smart_units(max_fx, 'force')
                        label_str = f"{load.fx / f_scale:+.3g} {f_unit} (axial)"
                        h_len = max(abs(load.fx) / max_fx * arrow_scale, arrow_scale * 0.25)
                        tip_x = x_pt + h_len / x_scale if load.fx > 0 else x_pt - h_len / x_scale
                        ha_val = 'left' if load.fx > 0 else 'right'
                        x_label = tip_x
                        if x_pt >= coords[:, 0].max() / x_scale - 1e-4:
                            ha_val, x_label = 'right', tip_x - st.annot_offset_fx_horz * arrow_scale
                        elif x_pt <= coords[:, 0].min() / x_scale + 1e-4:
                            ha_val, x_label = 'left', tip_x + st.annot_offset_fx_horz * arrow_scale
                        ax.annotate('', xy=(tip_x, y_pt), xytext=(x_pt, y_pt), arrowprops=dict(arrowstyle='->', lw=st.load_line_width, color=st.colour_load, mutation_scale=st.load_mutation_scale))
                        stack = stack_fx.get(x_key, 0)
                        y_text = y_pt - arrow_scale * (st.annot_offset_fx_vert + stack * 0.4)
                        ax.text(x_label, y_text, label_str, color=st.colour_load, fontsize=st.annotation_fontsize - 1, ha=ha_val, weight='bold')
                        stack_fx[x_key] = stack + 1
                    if abs(load.mz) > 1e-10:
                        stack = stack_mz.get(x_key, 0)
                        m_scale, m_unit = smart_units(max_mz, 'moment')
                        label_str = f"{load.mz / m_scale:+.3g} {m_unit}"
                        self._draw_moment_arc(ax, x_pt, y_pt, load.mz, moment_radius, st.colour_moment_load, label_str, is_reaction=False, stack_level=stack)
                        stack_mz[x_key] = stack + 1

                elif isinstance(load, (UniformDistributedLoad, TrapezoidalDistributedLoad, TriangularDistributedLoad)):
                    if hasattr(load, 'element') and load.element is not None:
                        elements = ([load.element] if isinstance(load.element, int) else load.element)
                        x_s, x_e = coords[self.mesh.elements[elements[0]].node1, 0], coords[self.mesh.elements[elements[-1]].node2, 0]
                    elif hasattr(load, 'x_start') and load.x_start is not None: x_s, x_e = load.x_start, load.x_end
                    else: continue
                    if isinstance(load, UniformDistributedLoad): wy1, wy2 = load.wy, None
                    elif isinstance(load, TrapezoidalDistributedLoad): wy1, wy2 = load.wy1, load.wy2
                    else: wy1, wy2 = (load.w_peak, 0.0) if load.peak_loc == 'start' else (0.0, load.w_peak)
                    w_scale, w_unit = smart_units(max(abs(wy1), abs(wy2 or wy1)), 'udl')
                    label_str = f"{wy1/w_scale:.3g} {w_unit}" if wy2 is None else f"{wy1/w_scale:.3g}–{wy2/w_scale:.3g} {w_unit}"
                    self._draw_udl_comb(ax, x_s/x_scale, x_e/x_scale, wy1, glyph_h, st.colour_udl, label_str, wy2=wy2)

        ax.set_aspect('equal', adjustable='datalim')
        ax.margins(y=0.4, x=0.05)
        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_title('Reaction Force Diagram', fontsize=st.title_fontsize, weight='bold')
        ax.legend(fontsize=st.tick_fontsize, loc='upper right')
        ax.grid(True, alpha=st.grid_alpha)
        ax.get_yaxis().set_visible(False)
        ax.hlines(0, coords[:, 0].min() / x_scale, coords[:, 0].max() / x_scale, colors='gray', linestyles='--', alpha=0.3, zorder=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    def _plot_udl_segment(self, ax, x1, x2, wy, label=False):
        """Helper to plot a UDL segment with arrows."""
        num_arrows = 5
        x_arrows = np.linspace(x1, x2, num_arrows)
        
        # Determine arrow direction and height
        # Simple heuristic for scaling
        h = 500  # Base height for arrows
        if wy < 0:
            y_start = h
            y_end = 0
        else:
            y_start = -h
            y_end = 0
            
        for x in x_arrows:
            ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        # Connect arrow tops
        ax.plot([x1, x2], [y_start, y_start], 'r-', alpha=0.5, 
                label='Uniform Load' if label else "")

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
        st = DEFAULT_STYLE
        if self.stresses is None:
            # Default resolution for report plots
            self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)
        
        x_coords = self.stresses['x']
        beam_length = x_coords[-1] - x_coords[0]
        x_scale, x_unit = smart_units(beam_length, 'length')
        
        # Extract peak values at each x-station
        # axes: (x, y, z)
        vm_peak = np.max(self.stresses['von_mises'], axis=(1, 2))
        bending_peak = np.max(np.abs(self.stresses['bending']), axis=(1, 2))
        shear_peak = np.max(np.abs(self.stresses['shear']), axis=(1, 2))
        
        fig, ax = plt.subplots(figsize=st.figsize_wide, dpi=dpi)
        
        ax.plot(x_coords / x_scale, vm_peak, '-', color='#111827', linewidth=2.5, label='Peak von Mises')
        ax.plot(x_coords / x_scale, bending_peak, '--', color=st.colour_max, linewidth=1.5, label='Peak Bending (abs)')
        ax.plot(x_coords / x_scale, shear_peak, ':', color=st.colour_primary, linewidth=1.5, label='Peak Shear (abs)')
        
        ax.fill_between(x_coords / x_scale, 0, vm_peak, color='gray', alpha=0.1)
        
        # Highlight absolute maximum
        max_vm_val = np.max(vm_peak)
        if max_vm_val > 1e-10:
            max_vm_idx = np.argmax(vm_peak)
            x_peak = x_coords[max_vm_idx] / x_scale
            
            ax.plot(x_peak, max_vm_val, 'o', color=st.colour_max, markersize=8, zorder=5)
            # Dotted line to Y-axis
            ax.hlines(max_vm_val, x_coords[0]/x_scale, x_peak,
                      colors=st.colour_max, linestyles=':', linewidth=1.5, zorder=4)
            
            # Custom Y-tick
            existing_ticks = list(ax.get_yticks())
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            new_ticks = [t for t in existing_ticks if abs(t - max_vm_val) > y_range*0.05]
            new_ticks.append(max_vm_val)
            ax.set_yticks(new_ticks)
            
            fig.canvas.draw()
            for val, lbl in zip(ax.get_yticks(), ax.get_yticklabels()):
                if abs(val - max_vm_val) < max_vm_val*1e-3 + 1e-6:
                    lbl.set_color(st.colour_max)
                    lbl.set_weight('bold')

        ax.set_xlabel(f'Position ({x_unit})', fontsize=st.label_fontsize)
        ax.set_ylabel('Stress (MPa)', fontsize=st.label_fontsize)
        ax.set_title('Peak Stress Distribution along Beam Length', fontsize=st.title_fontsize, weight='bold')
        ax.grid(True, alpha=st.grid_alpha)
        ax.legend(fontsize=st.tick_fontsize)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
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
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        report_name = output_path.stem
        
        images_dir = output_dir / f"{report_name}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        images_rel_path = f"{report_name}_images"
        
        # Branch based on analysis type
        if self.load_case is not None:
            return self._generate_static_report(output_path, images_dir, images_rel_path, deformation_scale)
        elif self.frequencies is not None:
            return self._generate_modal_report(output_path, images_dir, images_rel_path)
        else:
            raise ValueError("No analysis results found to generate report")

    def _generate_static_report(self, output_path, images_dir, images_rel_path, deformation_scale):
        # Stress and image generation pipeline
        self.calculate_internal_forces()
        
        if self.stresses is None:
            self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)

        # Generate all plots
        self.plot_structure_diagram(str(images_dir / "structure_diagram.png"))
        self.plot_reaction_diagram(str(images_dir / "reaction_diagram.png"))
        self.plot_deformed_shape(str(images_dir / "deformed_shape.png"), scale_factor=deformation_scale)
        self.plot_shear_diagram(str(images_dir / "shear_diagram.png"))
        self.plot_moment_diagram(str(images_dir / "moment_diagram.png"))
        self.plot_stress_distributions(str(images_dir / "stress_distribution.png"))
        
        # Calculate key results
        v_displacements = self.displacements[1::3]
        max_deflection = abs(min(v_displacements))
        max_deflection_node = v_displacements.argmin()
        max_deflection_pos = self.mesh.nodes[max_deflection_node].x
        
        if deformation_scale == 'auto':
            from .visualizer import BeamVisualizer
            _s = BeamVisualizer(self.mesh)._auto_scale_factor(self.displacements)
        else:
            _s = float(deformation_scale)

        def get_max_stress_info(stress_array):
            if stress_array.size == 0: return 0.0, 0.0, 0.0, 0.0
            max_idx = np.unravel_index(np.argmax(np.abs(stress_array)), stress_array.shape)
            idx_x, idx_y, idx_z = max_idx
            max_val = np.abs(stress_array[idx_x, idx_y, idx_z])
            x_loc = self.stresses['x'][idx_x]
            y_loc = self.stresses['y'][idx_y, idx_z]
            z_loc = self.stresses['z'][idx_y, idx_z]
            return max_val, x_loc, y_loc, z_loc

        max_vm, vm_x, vm_y, vm_z = get_max_stress_info(self.stresses['von_mises'])
        max_bend_val, bend_x, bend_y, bend_z = get_max_stress_info(self.stresses['bending'])
        max_shear_val, shear_x, shear_y, shear_z = get_max_stress_info(self.stresses['shear'])
        max_axial_val, axial_x, axial_y, axial_z = get_max_stress_info(self.stresses['axial'])

        # Margin of Safety
        yield_strength = getattr(self.material, 'yield_strength', 0.0)
        if yield_strength and max_vm > 0:
            mos = (yield_strength / max_vm) - 1.0
            status_text = "PASS" if mos >= 0 else "FAIL"
            mos_text = f"{mos:.2f}"
            status_color = "🟢" if mos >= 0 else "🔴"
        else:
            mos = float('inf')
            status_text = "N/A"
            mos_text = "N/A"
            status_color = "⚪"

        # Equilibrium Check
        total_fy_reaction = 0.0
        total_fx_reaction = 0.0
        total_mz_res = 0.0
        for bc in self.bc_set.conditions:
            node_id = bc.node
            if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
                fx = self.reactions[3*node_id]
                fy = self.reactions[3*node_id + 1]
                mz = self.reactions[3*node_id + 2]
                total_fx_reaction += fx
                total_fy_reaction += fy
                x_pos = self.mesh.nodes[node_id].x
                total_mz_res += mz + (fy * x_pos)
                
        total_fy_load = 0.0
        total_fx_load = 0.0
        for load in self.load_case.loads:
            if hasattr(load, 'fy'): total_fy_load += getattr(load, 'fy', 0)
            if hasattr(load, 'fx'): total_fx_load += getattr(load, 'fx', 0)
            
            if isinstance(load, PointLoad):
                x_pos = self.mesh.nodes[load.node].x if load.node is not None else load.x
                total_mz_res += getattr(load, 'mz', 0.0) + (getattr(load, 'fy', 0.0) * x_pos)
            elif isinstance(load, UniformDistributedLoad):
                wy = getattr(load, 'wy', 0)
                if wy != 0:
                    if hasattr(load, 'element') and load.element is not None:
                        elements = [load.element] if isinstance(load.element, int) else load.element
                        for elem_id in elements:
                            n1 = self.mesh.elements[elem_id].node1
                            n2 = self.mesh.elements[elem_id].node2
                            x1, x2 = self.mesh.nodes[n1].x, self.mesh.nodes[n2].x
                            L_el = abs(x2 - x1)
                            total_fy_load += wy * L_el
                            total_mz_res += (wy * L_el) * ((x1 + x2) / 2)
                    elif hasattr(load, 'x_start') and load.x_start is not None:
                        L_el = abs(load.x_end - load.x_start)
                        total_fy_load += wy * L_el
                        total_mz_res += (wy * L_el) * ((load.x_start + load.x_end) / 2)

        residual_fx = abs(total_fx_reaction + total_fx_load)
        residual_fy = abs(total_fy_reaction + total_fy_load)
        residual_mz = abs(total_mz_res)
        
        coords = self.mesh.get_node_coords()
        beam_length = coords[:, 0].max() - coords[:, 0].min()

        num_pt_loads = len([l for l in self.load_case.loads if isinstance(l, PointLoad)])
        num_dist_loads = len([l for l in self.load_case.loads if isinstance(l, UniformDistributedLoad)])

        load_details_list = []
        for i, load in enumerate(self.load_case.loads, 1):
            if isinstance(load, PointLoad):
                loc_str = f"Node {load.node}" if load.node is not None else f"x = {load.x} mm"
                vals = []
                if load.fx: vals.append(f"Fx={load.fx} N")
                if load.fy: vals.append(f"Fy={load.fy} N")
                if load.mz: vals.append(f"Mz={load.mz} N·mm")
                load_details_list.append(f"{i}. **Point Load** at {loc_str}: {', '.join(vals)}")
            elif isinstance(load, UniformDistributedLoad):
                span_str = f"Elements {load.element}" if load.element is not None else f"x = {load.x_start} to {load.x_end} mm"
                load_details_list.append(f"{i}. **Uniform Load** on {span_str}: wy={load.wy} N/mm")
            elif hasattr(load, 'wy1') and hasattr(load, 'wy2'):
                span_str = f"Elements {load.element}" if load.element is not None else f"x = {load.x_start} to {load.x_end} mm"
                load_details_list.append(f"{i}. **Trapezoidal Load** on {span_str}: wy1={load.wy1}, wy2={load.wy2} N/mm")
            elif hasattr(load, 'w_peak'):
                span_str = f"Elements {load.element}" if load.element is not None else f"x = {load.x_start} to {load.x_end} mm"
                load_details_list.append(f"{i}. **Triangular Load** on {span_str}: wy_peak={load.w_peak} N/mm at {load.peak_loc}")
        
        load_details_str = "\n".join(load_details_list) if load_details_list else "None"

        md_content = rf"""# Beam FEA Analysis Report (Static)

**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

---

## 1. Executive Summary

- **Analysis Overview**: Static structural analysis of a {beam_length:,.0f} mm beam subjected to {num_pt_loads} point load(s) and {num_dist_loads} distributed load(s).
- **Peak Deflection**: {max_deflection:.4f} mm at x = {max_deflection_pos:.1f} mm.
- **Critical Margin of Safety (MoS)**: {mos_text} {status_color} **{status_text}** (based on Material Yield Stress vs Max von Mises).

---

## 2. Model Setup & Inputs

### Structure & Loading Diagram

![Structure Diagram]({images_rel_path}/structure_diagram.png)

### Load Cases
- **Name:** {self.load_case.name}
- **Summary:** {num_pt_loads} point load(s), {num_dist_loads} distributed load(s)

**Detailed Applied Loads:**
{load_details_str}

### Cross-Section Properties

| Property | Value | Units |
|----------|-------|-------|
| Geometric Profile | {getattr(self.section, 'name', 'Custom Profile')} | - |
| Area (A) | {self.section.A:,.2f} | mm² |
| Moment of Inertia (Iy) | {self.section.Iy:.2e} | mm⁴ |
| Moment of Inertia (Iz) | {self.section.Iz:.2e} | mm⁴ |

### Material Properties

| Property | Value | Units |
|----------|-------|-------|
| Material | {self.material.name} | - |
| Young's Modulus (E) | {self.material.E:,.0f} | MPa |
| Shear Modulus (G) | {self.material.G:,.0f} | MPa |
| Density (ρ) | {self.material.rho:.2e} | kg/mm³ |
| Poisson's Ratio (ν) | {self.material.nu:.3f} | - |
| Yield Strength | {self.material.yield_strength:,.1f} | MPa |

---

## 3. Mesh & Solver Details

| Property | Value |
|----------|-------|
| Number of Nodes | {self.mesh.num_nodes} |
| Number of Elements | {self.mesh.num_elements} |
| Total DOFs | {self.mesh.num_dofs} |
| Element Formulation | Euler-Bernoulli Beam |

---

## 4. Analysis Results

### Reaction Forces & FBD Diagram

![Reaction Diagram]({images_rel_path}/reaction_diagram.png)

*Equilibrium Check*: 
- $\Sigma F_x$ Residual = {residual_fx:.2e} N
- $\Sigma F_y$ Residual = {residual_fy:.2e} N
- $\Sigma M_z$ Residual (about x=0) = {residual_mz:.2e} N·mm

### Displacements

The following plot shows the deformed shape of the beam (exaggerated by {_s:.0f}× for visualization):

![Deformed Shape]({images_rel_path}/deformed_shape.png)

### Internal Forces

![Shear Force Diagram]({images_rel_path}/shear_diagram.png)

![Bending Moment Diagram]({images_rel_path}/moment_diagram.png)

### Stress Analysis & Structural Integrity

*The distributions below plot the peak stresses at the extreme top and bottom fibers of the cross-section evaluated along the length of the beam.*

![Stress Distribution]({images_rel_path}/stress_distribution.png)

| Stress Component | Maximum Magnitude | Location (x, y, z) [mm] | Units |
|------------------|---------------|-------------------------|-------|
| von Mises (Peak) | {max_vm:.2f} | ({vm_x:.1f}, {vm_y:.1f}, {vm_z:.1f}) | MPa |
| Bending (Max)    | {max_bend_val:.2f} | ({bend_x:.1f}, {bend_y:.1f}, {bend_z:.1f}) | MPa |
| Shear (Max)      | {max_shear_val:.2f} | ({shear_x:.1f}, {shear_y:.1f}, {shear_z:.1f}) | MPa |
| Axial (Max)      | {max_axial_val:.2f} | ({axial_x:.1f}, {axial_y:.1f}, {axial_z:.1f}) | MPa |

---

*Report generated by Beam FEA Analysis Tool*
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        return str(output_path)

    def _generate_modal_report(self, output_path, images_dir, images_rel_path):
        # Generate Structure Plot
        self.plot_structure_diagram(str(images_dir / "structure_diagram.png"))
        
        # Generate Mode Shape Plots (Top 3)
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        mode_shape_plots = []
        for i in range(min(3, len(self.frequencies))):
            f_val = self.frequencies[i]
            img_name = f"mode_{i+1}_shape.png"
            viz.plot_mode_shape(self.mode_shapes[:, i], i+1, f_val, output_path=str(images_dir / img_name))
            mode_shape_plots.append((i+1, f_val, img_name))

        # Frequency Table
        freq_rows = []
        for i, f in enumerate(self.frequencies):
            freq_rows.append(f"| {i+1} | {f:8.2f} | {f*60:12.0f} |")
        freq_table = "\n".join(freq_rows)

        coords = self.mesh.get_node_coords()
        beam_length = coords[:, 0].max() - coords[:, 0].min()

        md_content = f"""# Beam FEA Analysis Report (Modal)

**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

---

## 1. Executive Summary

- **Analysis Overview**: Modal analysis of a {beam_length:,.0f} mm beam.
- **Fundamental Frequency**: {self.frequencies[0]:.2f} Hz ({self.frequencies[0]*60:.0f} RPM).
- **Number of Modes Calculated**: {len(self.frequencies)}

---

## 2. Model Setup & Inputs

### Structure Diagram

![Structure Diagram]({images_rel_path}/structure_diagram.png)

### Cross-Section Properties

| Property | Value | Units |
|----------|-------|-------|
| Area (A) | {self.section.A:,.2f} | mm² |
| Moment of Inertia (Iy) | {self.section.Iy:.2e} | mm⁴ |

### Material Properties

| Property | Value | Units |
|----------|-------|-------|
| Material | {self.material.name} | - |
| Young's Modulus (E) | {self.material.E:,.0f} | MPa |
| Density (ρ) | {self.material.rho:.2e} | kg/mm³ |

---

## 3. Modal Results

### Natural Frequencies

| Mode | Frequency (Hz) | Frequency (RPM) |
|------|----------------|-----------------|
{freq_table}

### Mode Shapes (First 3)

"""
        for mode_idx, freq, img in mode_shape_plots:
            md_content += f"#### Mode {mode_idx} ({freq:.2f} Hz)\n\n![Mode {mode_idx}]({images_rel_path}/{img})\n\n"

        md_content += "\n---\n\n*Report generated by Beam FEA Analysis Tool*\n"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        return str(output_path)



if __name__ == "__main__":
    print("\nBeam Report Generator Module")
    print("Generates professional markdown reports for beam analysis")
