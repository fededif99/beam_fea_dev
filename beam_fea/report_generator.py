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

from pathlib import Path
from datetime import datetime
from typing import Union
import numpy as np
import pandas as pd

from .loads import PointLoad, ConcentratedMoment, DistributedLoad, LumpedMass
from .batch_report import BatchReportGenerator
from .diagrams import (
    plot_structure_diagram,
    plot_reaction_diagram,
    plot_shear_diagram,
    plot_moment_diagram,
    plot_stress_distributions,
    draw_fixed_support,
    draw_pinned_support,
    draw_roller_support,
    draw_point_force_arrow,
    draw_moment_arc,
    draw_udl_comb,
)

__all__ = ['BatchReportGenerator', 'BeamReportGenerator']


class BeamReportGenerator:
    """
    Generate professional markdown reports for beam analysis results.
    """
    
    def __init__(self, solver, mesh, load_case, bc_set,
                 displacements, reactions, failure_criterion: str = 'tsai_wu'):
        """
        Initialize report generator.
        
        Parameters
        ----------
        solver : BeamSolver
            Beam solver instance
        mesh : Mesh
            Finite element mesh
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
        self.material = solver.properties.default_material
        self.section = solver.properties.default_section
        self.load_case = load_case
        self.bc_set = bc_set
        self.displacements = displacements
        self.reactions = reactions
        self.failure_criterion = failure_criterion
        
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
        Calculate internal forces using consistent recovery for reporting.
        """
        results = self.solver.calculate_internal_forces(num_points, strategy='consistent')
        self.positions = results['positions']
        self.path_positions = results.get('path_positions', self.positions)
        self.shear_forces = results['shear_forces']
        self.bending_moments = results['bending_moments']

    # ------------------------------------------------------------------
    # Diagram plotting methods (delegated to diagrams module)
    # ------------------------------------------------------------------

    def plot_structure_diagram(self, output_path: str, dpi: int = 150):
        """Plot beam structure with boundary conditions and loads."""
        plot_structure_diagram(self.mesh, self.bc_set, self.load_case, output_path=output_path, dpi=dpi)

    def plot_reaction_diagram(self, output_path: str, dpi: int = 150):
        """Plot Free Body Diagram (FBD) showing applied loads and reactions."""
        plot_reaction_diagram(self.mesh, self.bc_set, self.load_case, self.reactions, output_path=output_path, dpi=dpi)

    def plot_shear_diagram(self, output_path: str, dpi: int = 150):
        """Plot shear force diagram."""
        if self.shear_forces is None:
            self.calculate_internal_forces()
        path_pos = getattr(self, 'path_positions', self.positions)
        plot_shear_diagram(path_pos, self.shear_forces, output_path=output_path, dpi=dpi)

    def plot_moment_diagram(self, output_path: str, dpi: int = 150):
        """Plot bending moment diagram."""
        if self.bending_moments is None:
            self.calculate_internal_forces()
        path_pos = getattr(self, 'path_positions', self.positions)
        plot_moment_diagram(path_pos, self.bending_moments, output_path=output_path, dpi=dpi)

    def plot_stress_distributions(self, output_path: str, dpi: int = 150):
        """Plot peak stress distributions along the beam length."""
        if self.stresses is None:
            self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)
        plot_stress_distributions(self.stresses, output_path=output_path, dpi=dpi)

    def plot_deformed_shape(self, output_path: str, scale_factor: Union[float, str] = 'auto', dpi: int = 150):
        """Plot deformed shape compared to undeformed."""
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

    def plot_cross_sections(self, output_path: str, dpi: int = 150):
        """Plot unique cross-sections using BeamVisualizer."""
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        if self.solver.properties.has_multiple_sections(self.mesh.num_elements):
            viz.plot_multiple_sections(self.solver.properties, output_path=output_path, dpi=dpi)
        else:
            viz.plot_section_properties(self.section, output_path=output_path, dpi=dpi)

    def plot_laminate_stackup(self, output_path: str, dpi: int = 150):
        """Plot laminate stack-up if the material is a laminate."""
        from .composites import Laminate
        from .visualizer import BeamVisualizer
        if hasattr(self.material, '_is_laminate') or isinstance(self.material, Laminate):
            viz = BeamVisualizer(self.mesh)
            viz.plot_laminate_stackup(self.material, output_path=output_path, dpi=dpi)

    # ------------------------------------------------------------------
    # Report orchestration & markdown generation
    # ------------------------------------------------------------------
    
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
        
        # Determine analysis types present
        has_static = self.displacements is not None
        has_modal = self.frequencies is not None
        
        if not has_static and not has_modal:
            raise ValueError("No analysis results found to generate report")

        # Start generating content
        md_content = self._generate_header()
        md_content += self._generate_executive_summary(has_static, has_modal)
        md_content += self._generate_model_setup(images_dir, images_rel_path)
        
        # Cross-section visualization (Enhanced)
        self.plot_cross_sections(str(images_dir / "cross_sections.png"))
        self.plot_laminate_stackup(str(images_dir / "laminate_stackup.png"))

        if has_static:
            md_content += self._generate_static_results(images_dir, images_rel_path, deformation_scale)
            
        if has_modal:
            md_content += self._generate_modal_results(images_dir, images_rel_path)
            
        md_content += self._generate_footer()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        return str(output_path)

    def _generate_header(self):
        """Generate report header section."""
        title = "Beam FEA Analysis Report"
        if self.displacements is not None and self.frequencies is not None:
            title += " (Static & Modal)"
        elif self.displacements is not None:
            title += " (Static)"
        else:
            title += " (Modal)"
            
        return rf"""# {title}

**Generated:** {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

---
"""

    def _generate_executive_summary(self, has_static, has_modal):
        """Generate executive summary based on available results."""
        coords = self.mesh.get_node_coords()
        beam_length = coords[:, 0].max() - coords[:, 0].min()
        
        summary = f"\n## 1. Executive Summary\n\n- **Structure**: {beam_length:,.0f} mm beam with {self.mesh.num_nodes} nodes and {self.mesh.num_elements} elements.\n"
        
        if has_static:
            v_displacements = self.displacements[1::3]
            max_deflection = abs(min(v_displacements))
            max_deflection_node = v_displacements.argmin()
            max_deflection_pos = self.mesh.nodes[max_deflection_node, 0]
            
            # Stress highlights
            if self.stresses is None:
                self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)
            
            max_vm = np.max(self.stresses['von_mises'])
            yield_strength = getattr(self.material, 'yield_strength', None) or 0.0
            
            summary += f"- **Static Results**: Peak deflection of {max_deflection:.4f} mm at x = {max_deflection_pos:.1f} mm.\n"
            if yield_strength > 0:
                mos = (yield_strength / max_vm) - 1.0 if max_vm > 0 else float('inf')
                status = "PASS" if mos >= 0 else "FAIL"
                summary += f"- **Structural Integrity**: {status} (Margin of Safety: {mos:.2f}).\n"

        if has_modal:
            summary += f"- **Modal Results**: Found {len(self.frequencies)} natural frequencies. Fundamental frequency: {self.frequencies[0]:.2f} Hz.\n"
            
        summary += "\n---\n"
        return summary

    def _generate_model_setup(self, images_dir, images_rel_path):
        """Generate shared model setup section."""
        self.plot_structure_diagram(str(images_dir / "structure_diagram.png"))
        
        # Load details
        num_pt_loads = 0
        num_dist_loads = 0
        num_mass_loads = 0
        load_details_str = "None"
        
        if self.load_case:
            num_pt_loads = len([l for l in self.load_case.loads if isinstance(l, (PointLoad, ConcentratedMoment))])
            num_dist_loads = len([l for l in self.load_case.loads if isinstance(l, DistributedLoad)])
            num_mass_loads = len([l for l in self.load_case.loads if isinstance(l, LumpedMass)])
            
            load_details_list = []
            for i, load in enumerate(self.load_case.loads, 1):
                if isinstance(load, ConcentratedMoment):
                    loc_str = f"Node {load.node}" if load.node is not None else f"x = {load.x} mm"
                    load_details_list.append(f"{i}. **Moment** at {loc_str}: Mz={load.mz} N·mm")
                elif isinstance(load, PointLoad):
                    loc_str = f"Node {load.node}" if load.node is not None else f"x = {load.x} mm"
                    vals = []
                    if getattr(load, 'fx', 0.0): vals.append(f"Fx={load.fx} N")
                    if getattr(load, 'fy', 0.0): vals.append(f"Fy={load.fy} N")
                    load_details_list.append(f"{i}. **Point Load** at {loc_str}: {', '.join(vals)}")
                elif isinstance(load, DistributedLoad):
                    if load.element is not None:
                        span_str = f"Elements {load.element}"
                    elif load.x_start is not None:
                        span_str = f"x = {load.x_start} to {load.x_end} mm"
                    else:
                        span_str = "Full Span"
                    dist = load.distribution.lower()
                    if dist == 'uniform':
                        load_details_list.append(f"{i}. **Uniform Load** on {span_str}: wy={load.wy} N/mm")
                    elif dist == 'linear':
                        load_details_list.append(f"{i}. **Linear Load** on {span_str}: wy={load.wy_start} to {load.wy_end} N/mm")
                    elif dist == 'triangular':
                        load_details_list.append(f"{i}. **Triangular Load** on {span_str}: peak={load.w_peak} N/mm at {load.peak_loc}")
                    elif dist == 'custom':
                        load_details_list.append(f"{i}. **Custom Load** on {span_str}: {load.n_points}-pt quadrature")
                elif isinstance(load, LumpedMass):
                    loc_str = f"Node {load.node}" if load.node is not None else f"x = {load.x} mm"
                    grav_str = " (gravity enabled)" if load.apply_gravity else ""
                    load_details_list.append(f"{i}. **Lumped Mass** at {loc_str}: m = {load.m} kg, Izz = {load.Izz} kg·mm²{grav_str}")
            
            if load_details_list:
                load_details_str = "\n".join(load_details_list)

        if hasattr(self.material, 'get_effective_properties'):
            props = self.material.get_effective_properties()
            mat_df = pd.DataFrame([
                ["Laminate", f"{self.material.name}", "-"],
                ["Ex (Longitudinal)", f"{props['Ex']:,.0f}", "MPa"],
                ["Ey (Transverse)", f"{props['Ey']:,.0f}", "MPa"],
                ["Eb (Bending)", f"{props['Eb']:,.0f}", "MPa"],
                ["Gxy (Shear)", f"{props['Gxy']:,.0f}", "MPa"],
                ["Density (ρ)", f"{props['rho']:.2e}", "kg/mm³"],
                ["Poisson's Ratio (ν_xy)", f"{props['nu_xy']:.3f}", "-"]
            ], columns=["Property", "Value", "Units"])
            mat_props_md = "### Material Properties (Composite Laminate)\n\n" + mat_df.to_markdown(index=False) + "\n"
        else:
            yield_str = f"{self.material.yield_strength:,.1f}" if getattr(self.material, 'yield_strength', None) is not None else "N/A"
            mat_df = pd.DataFrame([
                ["Material", f"{self.material.name}", "-"],
                ["Young's Modulus (E)", f"{self.material.E:,.0f}", "MPa"],
                ["Density (ρ)", f"{self.material.rho:.2e}", "kg/mm³"],
                ["Poisson's Ratio (ν)", f"{self.material.nu:.3f}", "-"],
                ["Yield Strength", f"{yield_str}", "MPa"]
            ], columns=["Property", "Value", "Units"])
            mat_props_md = "### Material Properties\n\n" + mat_df.to_markdown(index=False) + "\n"

        sec_df = pd.DataFrame([
            ["Geometric Profile", f"{getattr(self.section, 'name', 'Custom Profile')}", "-"],
            ["Area (A)", f"{self.section.A:,.2f}", "mm²"],
            ["Moment of Inertia (Iy)", f"{self.section.Iy:.2e}", "mm⁴"],
            ["Moment of Inertia (Iz)", f"{self.section.Iz:.2e}", "mm⁴"]
        ], columns=["Property", "Value", "Units"])

        return rf"""
## 2. Model Setup & Inputs

### Structure & Loading Diagram

![Structure Diagram]({images_rel_path}/structure_diagram.png)

### Load Case Information
- **Name:** {self.load_case.name if self.load_case else 'N/A'}
- **Applied Loads:** {num_pt_loads} point load(s), {num_dist_loads} distributed load(s), {num_mass_loads} lumped mass(es)

**Detailed Loads:**
{load_details_str}

### Cross-Section Geometry

![Cross Sections]({images_rel_path}/cross_sections.png)

{sec_df.to_markdown(index=False)}

{mat_props_md.strip()}

{self._generate_laminate_section(images_rel_path)}

---
"""

    def _generate_static_results(self, images_dir, images_rel_path, deformation_scale):
        """Generate static analysis results section."""
        self.calculate_internal_forces()
        if self.stresses is None:
            self.stresses = self.solver.calculate_stresses(num_x_points=100, num_y_points=20, num_z_points=10)

        # Generate plots
        self.plot_reaction_diagram(str(images_dir / "reaction_diagram.png"))
        self.plot_deformed_shape(str(images_dir / "deformed_shape.png"), scale_factor=deformation_scale)
        self.plot_shear_diagram(str(images_dir / "shear_diagram.png"))
        self.plot_moment_diagram(str(images_dir / "moment_diagram.png"))
        self.plot_stress_distributions(str(images_dir / "stress_distribution.png"))

        # Equilibrium calculations (Centralized)
        eq = self.solver.verify_equilibrium()
        residual_fx = eq['residual_fx']
        residual_fy = eq['residual_fy']
        residual_mz = eq['residual_mz']

        # Deflection / Auto-scale
        v_displacements = self.displacements[1::3]
        max_deflection = abs(min(v_displacements))
        max_deflection_node = v_displacements.argmin()
        max_deflection_pos = self.mesh.nodes[max_deflection_node, 0]

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

        ply_stress_table = ""
        from .composites import Laminate
        if (hasattr(self.material, '_is_laminate') or isinstance(self.material, Laminate)) and hasattr(self.solver, 'laminate_results'):
            crit_label = self.failure_criterion.replace('_', ' ').title()
            master_ply_data = []
            
            sf_key = f'min_sf_{self.failure_criterion}'
            if self.failure_criterion == 'max_stress': sf_key = 'min_sf_max'

            ply_indices = sorted(next(iter(self.solver.laminate_results.values()))['ply_data'], key=lambda x: x['index'])
            for p_meta in ply_indices:
                pid = p_meta['index']
                d = {
                    'Ply': pid + 1, 'Name': p_meta['name'], 'Angle': f"{p_meta['angle']}°",
                    'Max σ1': 0.0, 'Max σ2': 0.0, 'Max τ12': 0.0, 'Max τxz': 0.0,
                    'Max σx': 0.0, 'Max τxy': 0.0, f'Min {crit_label} SF': float('inf')
                }

                for eid, results in self.solver.laminate_results.items():
                    ply = next(p for p in results['ply_data'] if p['index'] == pid)
                    d['Max σ1'] = max(d['Max σ1'], np.max(ply['peak_sigma_1']))
                    d['Max σ2'] = max(d['Max σ2'], np.max(ply.get('peak_sigma_2', 0.0)))
                    d['Max τ12'] = max(d['Max τ12'], np.max(ply.get('peak_tau_12', 0.0)))
                    d['Max τxz'] = max(d['Max τxz'], np.max(ply.get('peak_tau_xz', 0.0)))
                    d['Max σx'] = max(d['Max σx'], np.max(ply['peak_sigma_x']))
                    d['Max τxy'] = max(d['Max τxy'], np.max(ply['peak_tau_xy']))
                    d[f'Min {crit_label} SF'] = min(d[f'Min {crit_label} SF'], np.min(ply.get(sf_key, float('inf'))))

                master_ply_data.append(d)

            ply_df = pd.DataFrame(master_ply_data)
            ply_stress_table = "\n### Ply-by-Ply Maximum Stresses\n\n" + ply_df.to_markdown(index=False)
            
            beam_assump = "Narrow Beam (sigma_y=0)" if self.material.beam_type == 'narrow' else "Wide Beam (epsilon_y=0)"
            ply_stress_table += f"\n\n*Note: All stress values reported in MPa. Safety Factor (SF) calculated using **{crit_label}** criterion. Values represent absolute peaks across all stations and ply thickness (Top/Mid/Bot). Model assumes: **{beam_assump}**.*"

        stress_summary_df = pd.DataFrame([
            ["von Mises (Peak)", f"{max_vm:.2f}", f"({vm_x:.1f}, {vm_y:.1f}, {vm_z:.1f})", "MPa"],
            ["Bending (Max)", f"{max_bend_val:.2f}", f"({bend_x:.1f}, {bend_y:.1f}, {bend_z:.1f})", "MPa"],
            ["Shear (Max)", f"{max_shear_val:.2f}", f"({shear_x:.1f}, {shear_y:.1f}, {shear_z:.1f})", "MPa"],
            ["Axial (Max)", f"{max_axial_val:.2f}", f"({axial_x:.1f}, {axial_y:.1f}, {axial_z:.1f})", "MPa"]
        ], columns=["Stress Component", "Maximum Magnitude", "Location (x, y, z) [mm]", "Units"])

        return rf"""
## 3. Static Analysis Results

### Reaction Forces & Equilibrium Check

![Reaction Diagram]({images_rel_path}/reaction_diagram.png)

*Equilibrium Check*: 
- $\Sigma F_x$ Residual = {residual_fx:.2e} N
- $\Sigma F_y$ Residual = {residual_fy:.2e} N
- $\Sigma M_z$ Residual (about x=0) = {residual_mz:.2e} N·mm

### Deformed Shape
The following plot shows the deformed shape of the beam (exaggerated by {_s:.0f}× for visualization):

![Deformed Shape]({images_rel_path}/deformed_shape.png)

- **Max Deflection**: {max_deflection:.4f} mm at x = {max_deflection_pos:.1f} mm.

### Internal Force Recovery
Shear Force (V) and Bending Moment (M) distributions:

![Shear Force Diagram]({images_rel_path}/shear_diagram.png)

![Bending Moment Diagram]({images_rel_path}/moment_diagram.png)

### Stress Analysis & Integrity

![Stress Distribution]({images_rel_path}/stress_distribution.png)

{stress_summary_df.to_markdown(index=False)}

{ply_stress_table}

---
"""

    def _generate_modal_results(self, images_dir, images_rel_path):
        """Generate modal analysis results section."""
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        mode_shape_plots = []
        for i in range(min(3, len(self.frequencies))):
            f_val = self.frequencies[i]
            img_name = f"mode_{i+1}_shape.png"
            viz.plot_mode_shape(self.mode_shapes[:, i], i+1, f_val, output_path=str(images_dir / img_name))
            mode_shape_plots.append((i+1, f_val, img_name))

        freq_rows = []
        cum_x = 0.0
        cum_y = 0.0

        has_part = hasattr(self.solver, 'last_modal_participation') and self.solver.last_modal_participation

        for i, f in enumerate(self.frequencies):
            period = 1/f if f > 0 else 0

            if has_part:
                m_data = self.solver.last_modal_participation['modes'][i]
                eff_x = m_data['percent_x'] * 100
                eff_y = m_data['percent_y'] * 100
                cum_x += eff_x
                cum_y += eff_y
                freq_rows.append(f"| {i+1} | {f:8.2f} | {period:8.4f} | {eff_x:10.2f}% | {eff_y:10.2f}% | {cum_y:12.2f}% |")
            else:
                freq_rows.append(f"| {i+1} | {f:8.2f} | {period:8.4f} | - | - | - |")

        freq_table = "\n".join(freq_rows)

        total_mass_str = ""
        if has_part:
            tm_x = self.solver.last_modal_participation['total_mass_x']
            tm_y = self.solver.last_modal_participation['total_mass_y']
            total_mass_str = f"\n**Total Structural Mass**: {tm_y:.4e} kg (Y), {tm_x:.4e} kg (X)\n"

        results_md = rf"""
## 4. Modal Analysis Results

### Modal Properties & Mass Participation

| Mode | Frequency (Hz) | Period (s) | Effective X (%) | Effective Y (%) | Cumulative Y (%) |
|------|----------------|------------|-----------------|-----------------|------------------|
{freq_table}
{total_mass_str}

### Mode Shapes (First 3)

"""
        for mode_idx, freq, img in mode_shape_plots:
            results_md += f"#### Mode {mode_idx} ({freq:.2f} Hz)\n\n![Mode {mode_idx}]({images_rel_path}/{img})\n\n"

        return results_md + "---\n"

    def _generate_laminate_section(self, images_rel_path):
        """Generate additional section for laminate details."""
        from .composites import Laminate
        if not (hasattr(self.material, '_is_laminate') or isinstance(self.material, Laminate)):
            return ""

        return rf"""
### Composite Laminate Details

![Laminate Stackup]({images_rel_path}/laminate_stackup.png)

The following stack-up defines the laminate properties and fiber orientations.
"""

    def _generate_footer(self):
        """Generate report footer."""
        return "\n\n*Report generated by Beam FEA Analysis Tool*\n"
