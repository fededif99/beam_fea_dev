"""
diagrams.py
===========
Engineering diagram and visualization generation for beam FEA results.

Provides functions for rendering:
- Structure and applied loading diagrams with standard support glyphs
- Free-body reaction diagrams (FBD)
- Continuous shear force diagrams (SFD)
- Continuous bending moment diagrams (BMD)
- Stress distribution curves along the beam axis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Union, Optional, List
from .loads import PointLoad, ConcentratedMoment, DistributedLoad
from .boundary_conditions import PinnedSupport, RollerSupport, FixedSupport
from .plot_style import smart_units, DEFAULT_STYLE


# ------------------------------------------------------------------
# Boundary condition drawing glyphs
# ------------------------------------------------------------------

def draw_fixed_support(ax, x, y, beam_h, side='left'):
    """Draw a hatched wall fixed support at (x, y)."""
    st = DEFAULT_STYLE
    w = beam_h * 0.6
    ax.plot([x, x], [y - w, y + w], color=st.colour_bc, linewidth=st.bc_line_width, zorder=4)
    n_hatch = 5
    hatch_len = beam_h * 0.3
    y_hatches = np.linspace(y - w, y + w, n_hatch)
    direction = -1 if side == 'left' else 1
    for yh in y_hatches:
        ax.plot([x, x + direction * hatch_len], [yh, yh - hatch_len],
                color=st.colour_bc, linewidth=st.bc_hatch_width, zorder=3)


def draw_pinned_support(ax, x, y, beam_h):
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
    ground_y = y - tri_h
    ax.plot([x - tri_w * 0.7, x + tri_w * 0.7], [ground_y, ground_y],
            color=st.colour_bc, linewidth=st.bc_line_width, zorder=3)
    n_ticks = 5
    tick_len = tri_h * 0.25
    xs_ticks = np.linspace(x - tri_w * 0.55, x + tri_w * 0.55, n_ticks)
    for xt in xs_ticks:
        ax.plot([xt, xt - tick_len * 0.5], [ground_y, ground_y - tick_len],
                color=st.colour_bc, linewidth=st.bc_hatch_width, zorder=3)


def draw_roller_support(ax, x, y, beam_h):
    """Draw a roller support beneath (x, y)."""
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
    for cx in [x - tri_w * 0.25, x, x + tri_w * 0.25]:
        circle = mpatches.Circle(
            (cx, y - tri_h - r_circle), r_circle,
            linewidth=st.bc_hatch_width * 1.2, edgecolor=st.colour_bc, facecolor='white', zorder=4
        )
        ax.add_patch(circle)
    ground_y = y - tri_h - 2 * r_circle
    ax.plot([x - tri_w * 0.7, x + tri_w * 0.7], [ground_y, ground_y],
            color=st.colour_bc, linewidth=st.bc_line_width, zorder=3)
    n_ticks = 5
    tick_len = tri_h * 0.22
    xs_ticks = np.linspace(x - tri_w * 0.55, x + tri_w * 0.55, n_ticks)
    for xt in xs_ticks:
        ax.plot([xt, xt - tick_len * 0.5], [ground_y, ground_y - tick_len],
                color=st.colour_bc, linewidth=st.bc_hatch_width, zorder=3)


# ------------------------------------------------------------------
# Load drawing glyphs
# ------------------------------------------------------------------

def draw_point_force_arrow(ax, x, y, fy, max_fy, arrow_scale,
                           colour, label_str, is_reaction=False, x_max=None, stack_level=0):
    """Draw a vertical force arrow at (x, y)."""
    st = DEFAULT_STYLE
    length = max(abs(fy) / max_fy * arrow_scale, arrow_scale * 0.25)
    hw = arrow_scale * 0.08
    
    if is_reaction:
        lw = st.reaction_line_width
        mut = st.reaction_mutation_scale
    else:
        lw = st.load_line_width
        mut = st.load_mutation_scale
        
    fs = st.annotation_fontsize - 1

    ha_val = 'left'
    x_offset = hw * st.annot_offset_fy_horz
    if x_max is not None and x >= x_max - 1e-4:
        ha_val = 'right'
        x_offset = -hw * st.annot_offset_fy_horz

    y_shift = arrow_scale * ((st.annot_offset_fy_vert - 1.0) + stack_level * 0.4)

    if fy < 0:   # downward load (text above tail)
        ax.annotate('', xy=(x, y), xytext=(x, y + length),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=lw,
                                    mutation_scale=mut))
        ax.text(x + x_offset, y + length + y_shift + arrow_scale * 0.05, label_str, color=colour,
                fontsize=fs, weight='bold', va='center', ha=ha_val)
    else:         # upward load/reaction (text below tail)
        ax.annotate('', xy=(x, y), xytext=(x, y - length),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=lw,
                                    mutation_scale=mut))
        ax.text(x + x_offset, y - length - y_shift - arrow_scale * 0.15, label_str, color=colour,
                fontsize=fs, weight='bold', va='center', ha=ha_val)


def draw_moment_arc(ax, x, y, mz, radius, colour, label_str, is_reaction=False, stack_level=0):
    """Draw a curved arc arrow to represent a concentrated moment."""
    st = DEFAULT_STYLE
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
    ax.annotate('', xy=(arc_x[-1] + dx * 0.01, arc_y[-1] + dy * 0.01),
                xytext=(arc_x[-1], arc_y[-1]),
                arrowprops=dict(arrowstyle='->', color=colour, lw=lw, mutation_scale=mut))
    ax.plot(x, y, 'o', color=colour, markersize=5, zorder=5)
    
    y_offset = radius * (st.annot_offset_mz + stack_level * 0.8)
    ax.text(x, y + y_offset, label_str, color=colour,
            fontsize=st.annotation_fontsize, weight='bold', ha='center', va='center')


def draw_udl_comb(ax, x_start, x_end, wy, beam_h, colour, label_str, wy2=None):
    """
    Draw a distributed-load arrow comb between x_start and x_end.
    If wy2 is given, the comb is tapered (trapezoidal / triangular).
    """
    st = DEFAULT_STYLE
    span = abs(x_end - x_start)
    comb_h = beam_h * 1.1
    n_arrows = max(int(span / (beam_h * 0.6)), 3)
    xs = np.linspace(x_start, x_end, n_arrows)

    if wy2 is None:
        heights = np.full(n_arrows, comb_h)
    else:
        max_w = max(abs(wy), abs(wy2))
        if max_w < 1e-12:
            return
        h1 = comb_h * abs(wy) / max_w
        h2 = comb_h * abs(wy2) / max_w
        heights = np.linspace(h1, h2, n_arrows)

    sign = -1 if wy < 0 else 1
    arrow_tips = np.zeros(n_arrows)
    arrow_bases = np.zeros(n_arrows)

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

    ax.plot(xs, arrow_bases, color=colour, linewidth=st.load_line_width * 1.2, zorder=3)
    mid_x = (x_start + x_end) / 2
    label_y = arrow_bases.mean() - sign * comb_h * st.annot_offset_udl
    ax.text(mid_x, label_y, label_str, color=colour,
            fontsize=st.annotation_fontsize - 1,
            weight='bold', ha='center', va='center')


# ------------------------------------------------------------------
# High-level diagram plotters
# ------------------------------------------------------------------

def plot_structure_diagram(mesh, bc_set, load_case, output_path: str, dpi: int = 150):
    """
    Plot beam structure with boundary conditions and loads.
    """
    st = DEFAULT_STYLE
    coords = mesh.get_node_coords()
    dx_max = coords[:, 0].max() - coords[:, 0].min()
    dy_max = coords[:, 1].max() - coords[:, 1].min()
    beam_length = max(dx_max, dy_max)

    x_scale, x_unit = smart_units(beam_length, 'length')
    L_scaled = beam_length / x_scale
    
    arrow_scale = L_scaled * 0.08
    moment_radius = L_scaled * 0.03

    fig, ax = plt.subplots(figsize=st.figsize_structure, dpi=dpi)

    # Beam line
    ax.plot(coords[:, 0] / x_scale, coords[:, 1] / x_scale, '-',
            color=st.colour_primary, linewidth=st.beam_line_width, label='Beam', zorder=2)
    ax.plot(coords[:, 0] / x_scale, coords[:, 1] / x_scale, 'o',
            color=st.colour_primary, markersize=5, zorder=2)

    glyph_h = L_scaled * 0.04

    # Boundary conditions
    shown_bc_labels = set()
    for bc in bc_set.conditions:
        node_id = bc.node
        x = coords[node_id, 0] / x_scale
        y = coords[node_id, 1] / x_scale

        if isinstance(bc, FixedSupport):
            side = 'left' if node_id <= mesh.num_nodes // 2 else 'right'
            draw_fixed_support(ax, x, y, glyph_h, side=side)
            label = 'Fixed Support'
            if label not in shown_bc_labels:
                ax.plot([], [], 's', color=st.colour_bc, markersize=10,
                        label=label, markerfacecolor='#d1fae5')
                shown_bc_labels.add(label)
        elif isinstance(bc, PinnedSupport):
            draw_pinned_support(ax, x, y, glyph_h)
            label = 'Pinned Support'
            if label not in shown_bc_labels:
                ax.plot([], [], '^', color=st.colour_bc, markersize=10,
                        label=label, markerfacecolor='#d1fae5')
                shown_bc_labels.add(label)
        elif isinstance(bc, RollerSupport):
            draw_roller_support(ax, x, y, glyph_h)
            label = 'Roller Support'
            if label not in shown_bc_labels:
                ax.plot([], [], 'o', color=st.colour_bc, markersize=10,
                        label=label, markerfacecolor='#d1fae5')
                shown_bc_labels.add(label)

    # Loads
    if load_case:
        all_fy = [abs(getattr(l, 'fy', 0.0)) for l in load_case.loads
                  if isinstance(l, (PointLoad, ConcentratedMoment)) and abs(getattr(l, 'fy', 0.0)) > 1e-10]
        all_fx = [abs(getattr(l, 'fx', 0.0)) for l in load_case.loads
                  if isinstance(l, (PointLoad, ConcentratedMoment)) and abs(getattr(l, 'fx', 0.0)) > 1e-10]
        all_mz = [abs(getattr(l, 'mz', 0.0)) for l in load_case.loads
                  if isinstance(l, (PointLoad, ConcentratedMoment)) and abs(getattr(l, 'mz', 0.0)) > 1e-10]
        max_fy = max(all_fy) if all_fy else 1.0
        max_fx = max(all_fx) if all_fx else 1.0
        max_mz = max(all_mz) if all_mz else 1.0

        stack_fy = {}
        stack_fx = {}
        stack_mz = {}
        for load in load_case.loads:
            if isinstance(load, (PointLoad, ConcentratedMoment)):
                if load.node is not None:
                    x_raw, y_raw = coords[load.node, 0], coords[load.node, 1]
                elif load.x is not None:
                    x_raw, y_raw = load.x, 0.0
                else:
                    continue

                x_pt = x_raw / x_scale
                y_pt = y_raw / x_scale
                x_key = round(x_pt, 4)

                fy_val = getattr(load, 'fy', 0.0)
                fx_val = getattr(load, 'fx', 0.0)
                mz_val = getattr(load, 'mz', 0.0)

                if abs(fy_val) > 1e-10:
                    stack = stack_fy.get(x_key, 0)
                    f_scale, f_unit = smart_units(max_fy, 'force')
                    label_str = f"{fy_val / f_scale:+.3g} {f_unit}"
                    draw_point_force_arrow(
                        ax, x_pt, y_pt, fy_val, max_fy,
                        arrow_scale, st.colour_load, label_str, is_reaction=False,
                        x_max=coords[:, 0].max() / x_scale,
                        stack_level=stack
                    )
                    stack_fy[x_key] = stack + 1
                    
                if abs(fx_val) > 1e-10:
                    f_scale, f_unit = smart_units(max_fx, 'force')
                    label_str = f"{fx_val / f_scale:+.3g} {f_unit} (axial)"
                    h_len = max(abs(fx_val) / max_fx * arrow_scale, arrow_scale * 0.25)
                    if fx_val > 0:
                        ax.annotate('', xy=(x_pt, y_pt), xytext=(x_pt - h_len / x_scale, y_pt),
                                    arrowprops=dict(arrowstyle='->', lw=st.load_line_width, color=st.colour_load,
                                                   mutation_scale=st.load_mutation_scale))
                    else:
                        ax.annotate('', xy=(x_pt, y_pt), xytext=(x_pt + h_len / x_scale, y_pt),
                                    arrowprops=dict(arrowstyle='->', lw=st.load_line_width, color=st.colour_load,
                                                   mutation_scale=st.load_mutation_scale))
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
                    
                if abs(mz_val) > 1e-10:
                    stack = stack_mz.get(x_key, 0)
                    m_scale, m_unit = smart_units(max_mz, 'moment')
                    label_str = f"{mz_val / m_scale:+.3g} {m_unit}"
                    draw_moment_arc(
                        ax, x_pt, y_pt, mz_val, moment_radius, st.colour_moment_load, label_str,
                        is_reaction=False, stack_level=stack
                    )
                    stack_mz[x_key] = stack + 1

            elif isinstance(load, DistributedLoad):
                if hasattr(load, 'element') and load.element is not None:
                    elements = ([load.element] if isinstance(load.element, int) else load.element)
                    x_s = coords[mesh.elements[elements[0], 0], 0]
                    x_e = coords[mesh.elements[elements[-1], 1], 0]
                elif hasattr(load, 'x_start') and load.x_start is not None:
                    x_s, x_e = load.x_start, load.x_end
                else:
                    continue

                dist = getattr(load, 'distribution', 'uniform').lower()
                if dist == 'uniform': wy1, wy2 = load.wy, None
                elif dist == 'linear': wy1, wy2 = load.wy_start, load.wy_end
                elif dist == 'triangular': 
                    if str(load.peak_loc).lower() == 'start': wy1, wy2 = load.w_peak, 0.0
                    else: wy1, wy2 = 0.0, load.w_peak
                else:
                    continue

                if abs(wy1) < 1e-12 and (wy2 is None or abs(wy2) < 1e-12): continue
                ref_wy = max(abs(wy1), abs(wy2 or wy1))
                w_scale, w_unit = smart_units(ref_wy, 'udl')
                label_str = f"{wy1/w_scale:.3g} {w_unit}" if wy2 is None else f"{wy1/w_scale:.3g}–{wy2/w_scale:.3g} {w_unit}"
                draw_udl_comb(ax, x_s/x_scale, x_e/x_scale, wy1, glyph_h, st.colour_udl, label_str, wy2=wy2)

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


def plot_reaction_diagram(mesh, bc_set, load_case, reactions, output_path: str, dpi: int = 150):
    """
    Plot Free Body Diagram (FBD) showing applied loads and reaction forces.
    """
    st = DEFAULT_STYLE
    coords = mesh.get_node_coords()
    dx_max = coords[:, 0].max() - coords[:, 0].min()
    dy_max = coords[:, 1].max() - coords[:, 1].min()
    beam_length = max(dx_max, dy_max)

    x_scale, x_unit = smart_units(beam_length, 'length')
    L_scaled = beam_length / x_scale
    
    arrow_scale = L_scaled * 0.08
    moment_radius = L_scaled * 0.03
    glyph_h = L_scaled * 0.04

    fig, ax = plt.subplots(figsize=st.figsize_structure, dpi=dpi)

    ax.plot(coords[:, 0] / x_scale, coords[:, 1] / x_scale, '-',
            color=st.colour_primary, linewidth=st.beam_line_width, label='Beam', zorder=2)
    ax.plot(coords[:, 0] / x_scale, coords[:, 1] / x_scale, 'o',
            color=st.colour_primary, markersize=5, zorder=2)

    all_fy = []
    all_fx = []
    all_mz = []
    
    if load_case:
        all_fy = [abs(getattr(l, 'fy', 0.0)) for l in load_case.loads if isinstance(l, (PointLoad, ConcentratedMoment))]
        all_fx = [abs(getattr(l, 'fx', 0.0)) for l in load_case.loads if isinstance(l, (PointLoad, ConcentratedMoment))]
        all_mz = [abs(getattr(l, 'mz', 0.0)) for l in load_case.loads if isinstance(l, (PointLoad, ConcentratedMoment))]
    
    for bc in bc_set.conditions:
        node_id = bc.node
        if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
            all_fx.append(abs(reactions[3*node_id]))
            all_fy.append(abs(reactions[3*node_id + 1]))
            all_mz.append(abs(reactions[3*node_id + 2]))

    max_fy = max(all_fy) if all_fy and max(all_fy) > 1e-10 else 1.0
    max_fx = max(all_fx) if all_fx and max(all_fx) > 1e-10 else 1.0
    max_mz = max(all_mz) if all_mz and max(all_mz) > 1e-10 else 1.0

    reaction_color = '#28a745'
    threshold = 1e-5
    fs_react = st.annotation_fontsize - 1
    stack_fy = {}
    stack_fx = {}
    stack_mz = {}
    
    for i, bc in enumerate(bc_set.conditions):
        node_id = bc.node
        x_pt = coords[node_id, 0] / x_scale
        y_pt = coords[node_id, 1] / x_scale
        x_key = round(x_pt, 4)

        if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
            fx = reactions[3*node_id]
            fy = reactions[3*node_id + 1]
            mz = reactions[3*node_id + 2]
            
            if abs(fy) > threshold:
                stack = stack_fy.get(x_key, 0)
                f_scale, f_unit = smart_units(max_fy, 'force')
                label_str = rf"$\mathbf{{R_{{y,{i+1}}}}}$={fy / f_scale:+.3g} {f_unit}"
                draw_point_force_arrow(ax, x_pt, y_pt, fy, max_fy, arrow_scale, reaction_color, label_str, is_reaction=True, x_max=coords[:, 0].max() / x_scale, stack_level=stack)
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
                draw_moment_arc(ax, x_pt, y_pt, mz, moment_radius, reaction_color, label_str, is_reaction=True, stack_level=stack)
                stack_mz[x_key] = stack + 1

    ax.plot([], [], 's', color=reaction_color, label='Reactions', markerfacecolor='#fca5a5')

    if load_case:
        for load in load_case.loads:
            if isinstance(load, (PointLoad, ConcentratedMoment)):
                if load.node is not None: x_raw, y_raw = coords[load.node, 0], coords[load.node, 1]
                elif load.x is not None: x_raw, y_raw = load.x, 0.0
                else: continue
                x_pt = x_raw / x_scale
                y_pt = y_raw / x_scale
                x_key = round(x_pt, 4)

                fy_val = getattr(load, 'fy', 0.0)
                fx_val = getattr(load, 'fx', 0.0)
                mz_val = getattr(load, 'mz', 0.0)

                if abs(fy_val) > 1e-10:
                    stack = stack_fy.get(x_key, 0)
                    f_scale, f_unit = smart_units(max_fy, 'force')
                    label_str = f"{fy_val / f_scale:+.3g} {f_unit}"
                    draw_point_force_arrow(ax, x_pt, y_pt, fy_val, max_fy, arrow_scale, st.colour_load, label_str, is_reaction=False, x_max=coords[:, 0].max() / x_scale, stack_level=stack)
                    stack_fy[x_key] = stack + 1
                if abs(fx_val) > 1e-10:
                    f_scale, f_unit = smart_units(max_fx, 'force')
                    label_str = f"{fx_val / f_scale:+.3g} {f_unit} (axial)"
                    h_len = max(abs(fx_val) / max_fx * arrow_scale, arrow_scale * 0.25)
                    tip_x = x_pt + h_len / x_scale if fx_val > 0 else x_pt - h_len / x_scale
                    ha_val = 'left' if fx_val > 0 else 'right'
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
                if abs(mz_val) > 1e-10:
                    stack = stack_mz.get(x_key, 0)
                    m_scale, m_unit = smart_units(max_mz, 'moment')
                    label_str = f"{mz_val / m_scale:+.3g} {m_unit}"
                    draw_moment_arc(ax, x_pt, y_pt, mz_val, moment_radius, st.colour_moment_load, label_str, is_reaction=False, stack_level=stack)
                    stack_mz[x_key] = stack + 1

            elif isinstance(load, DistributedLoad):
                if hasattr(load, 'element') and load.element is not None:
                    elements = ([load.element] if isinstance(load.element, int) else load.element)
                    x_s, x_e = coords[mesh.elements[elements[0], 0], 0], coords[mesh.elements[elements[-1], 1], 0]
                elif hasattr(load, 'x_start') and load.x_start is not None: x_s, x_e = load.x_start, load.x_end
                else: continue
                dist = getattr(load, 'distribution', 'uniform').lower()
                if dist == 'uniform': wy1, wy2 = load.wy, None
                elif dist == 'linear': wy1, wy2 = load.wy_start, load.wy_end
                elif dist == 'triangular': wy1, wy2 = (load.w_peak, 0.0) if str(load.peak_loc).lower() == 'start' else (0.0, load.w_peak)
                else: continue
                w_scale, w_unit = smart_units(max(abs(wy1), abs(wy2 or wy1)), 'udl')
                label_str = f"{wy1/w_scale:.3g} {w_unit}" if wy2 is None else f"{wy1/w_scale:.3g}–{wy2/w_scale:.3g} {w_unit}"
                draw_udl_comb(ax, x_s/x_scale, x_e/x_scale, wy1, glyph_h, st.colour_udl, label_str, wy2=wy2)

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


def plot_shear_diagram(path_positions, shear_forces, output_path: str, dpi: int = 150):
    """
    Plot continuous shear force diagram with peak highlighting.
    """
    st = DEFAULT_STYLE
    beam_length = path_positions[-1] - path_positions[0]
    x_scale, x_unit = smart_units(beam_length, 'length')
    max_abs_V = np.max(np.abs(shear_forces))
    v_scale, v_unit = smart_units(max_abs_V, 'force')

    pos_scaled = path_positions / x_scale
    V_scaled = shear_forces / v_scale

    fig, ax = plt.subplots(figsize=st.figsize_wide, dpi=dpi)

    ax.plot(pos_scaled, V_scaled, '-', color=st.colour_primary, linewidth=st.line_width)
    ax.fill_between(pos_scaled, 0, V_scaled, alpha=st.fill_alpha, color=st.colour_primary)
    ax.axhline(0, color=st.colour_zero_line, linewidth=1)

    max_idx = np.argmax(np.abs(shear_forces))
    x_max = pos_scaled[max_idx]
    v_max = V_scaled[max_idx]
    x_left = pos_scaled[0]

    ax.plot(x_max, v_max, 'o', color=st.colour_max, markersize=8, zorder=5)
    ax.hlines(v_max, x_left, x_max, colors=st.colour_max, linestyles=':', linewidth=1.5, zorder=4)

    existing_ticks = ax.get_yticks()
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_dist = y_range * 0.05
    filtered_ticks = [t for t in existing_ticks if abs(t - v_max) > min_dist]
    filtered_ticks.append(v_max)
    ax.set_yticks(filtered_ticks)

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


def plot_moment_diagram(path_positions, bending_moments, output_path: str, dpi: int = 150):
    """
    Plot continuous bending moment diagram with peak highlighting.
    """
    st = DEFAULT_STYLE
    beam_length = path_positions[-1] - path_positions[0]
    x_scale, x_unit = smart_units(beam_length, 'length')
    max_abs_M = np.max(np.abs(bending_moments))
    m_scale, m_unit = smart_units(max_abs_M, 'moment')

    pos_scaled = path_positions / x_scale
    M_scaled = bending_moments / m_scale

    fig, ax = plt.subplots(figsize=st.figsize_wide, dpi=dpi)

    ax.plot(pos_scaled, M_scaled, '-', color=st.colour_moment, linewidth=st.line_width)
    ax.fill_between(pos_scaled, 0, M_scaled, alpha=st.fill_alpha, color=st.colour_moment)
    ax.axhline(0, color=st.colour_zero_line, linewidth=1)

    max_idx = np.argmax(np.abs(bending_moments))
    x_max = pos_scaled[max_idx]
    m_max = M_scaled[max_idx]
    x_left = pos_scaled[0]

    ax.plot(x_max, m_max, 'o', color=st.colour_max, markersize=8, zorder=5)
    ax.hlines(m_max, x_left, x_max, colors=st.colour_max, linestyles=':', linewidth=1.5, zorder=4)

    existing_ticks = ax.get_yticks()
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    min_dist = y_range * 0.05
    filtered_ticks = [t for t in existing_ticks if abs(t - m_max) > min_dist]
    filtered_ticks.append(m_max)
    ax.set_yticks(filtered_ticks)

    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    fig.canvas.draw()

    for val, lbl in zip(ax.get_yticks(), ax.get_yticklabels()):
        if abs(val - m_max) < abs(m_max) * 1e-3 + 1e-6:
            lbl.set_color(st.colour_max)
            lbl.set_weight('bold')

    ax.set_xlabel(f'Path Position ({x_unit})', fontsize=st.label_fontsize)
    ax.set_ylabel(f'Bending Moment ({m_unit})', fontsize=st.label_fontsize)
    ax.set_title('Bending Moment Diagram', fontsize=st.title_fontsize, weight='bold')
    ax.grid(True, alpha=st.grid_alpha)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_stress_distributions(stresses, output_path: str, dpi: int = 150):
    """
    Plot peak stress distributions (Bending, Shear, von Mises) along the beam.
    """
    st = DEFAULT_STYLE
    x_coords = stresses['x']
    beam_length = x_coords[-1] - x_coords[0]
    x_scale, x_unit = smart_units(beam_length, 'length')
    
    vm_peak = np.max(stresses['von_mises'], axis=(1, 2))
    bending_peak = np.max(np.abs(stresses['bending']), axis=(1, 2))
    shear_peak = np.max(np.abs(stresses['shear']), axis=(1, 2))
    
    fig, ax = plt.subplots(figsize=st.figsize_wide, dpi=dpi)
    
    ax.plot(x_coords / x_scale, vm_peak, '-', color='#111827', linewidth=2.5, label='Peak von Mises')
    ax.plot(x_coords / x_scale, bending_peak, '--', color=st.colour_max, linewidth=1.5, label='Peak Bending (abs)')
    ax.plot(x_coords / x_scale, shear_peak, ':', color=st.colour_primary, linewidth=1.5, label='Peak Shear (abs)')
    
    ax.fill_between(x_coords / x_scale, 0, vm_peak, color='gray', alpha=0.1)
    
    max_vm_val = np.max(vm_peak)
    if max_vm_val > 1e-10:
        max_vm_idx = np.argmax(vm_peak)
        x_peak = x_coords[max_vm_idx] / x_scale
        
        ax.plot(x_peak, max_vm_val, 'o', color=st.colour_max, markersize=8, zorder=5)
        ax.hlines(max_vm_val, x_coords[0]/x_scale, x_peak,
                  colors=st.colour_max, linestyles=':', linewidth=1.5, zorder=4)
        
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
