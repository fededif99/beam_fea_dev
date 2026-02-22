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
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import os
from .loads import PointLoad, UniformDistributedLoad
from .boundary_conditions import PinnedSupport, RollerSupport, FixedSupport


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
    
    def plot_structure_diagram(self, output_path: str, dpi: int = 150):
        """
        Plot beam structure with boundary conditions and loads.
        
        Parameters
        ----------
        output_path : str
            Path to save the plot
        dpi : int
            Resolution in dots per inch
        """
        coords = self.mesh.get_node_coords()
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
        
        # Plot beam
        ax.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=3, label='Beam')
        ax.plot(coords[:, 0], coords[:, 1], 'bo', markersize=6)
        
        # Plot boundary conditions
        for bc in self.bc_set.conditions:
            node_id = bc.node
            x, y = coords[node_id, 0], coords[node_id, 1]
            
            if isinstance(bc, PinnedSupport):
                ax.plot(x, y, 'g^', markersize=15, label='Pinned Support' if node_id == 0 else '')
                ax.annotate('Pinned', (x, y), xytext=(0, -30), 
                           textcoords='offset points', ha='center',
                           fontsize=9, color='green')
            elif isinstance(bc, RollerSupport):
                ax.plot(x, y, 'go', markersize=12, label='Roller Support' if node_id == 0 else '')
                ax.annotate('Roller', (x, y), xytext=(0, -30),
                           textcoords='offset points', ha='center',
                           fontsize=9, color='green')
            elif isinstance(bc, FixedSupport):
                ax.plot(x, y, 'gs', markersize=15, label='Fixed Support' if node_id == 0 else '')
                ax.annotate('Fixed', (x, y), xytext=(0, -30),
                           textcoords='offset points', ha='center',
                           fontsize=9, color='green', weight='bold')
        
        # Precompute max point load for scaling arrows
        point_loads = [l for l in self.load_case.loads if isinstance(l, PointLoad)]
        max_fy = max([abs(l.fy) for l in point_loads] + [1.0])
        
        # Plot loads
        for load in self.load_case.loads:
            if isinstance(load, PointLoad):
                node_id = load.node
                x, y = coords[node_id, 0], coords[node_id, 1]
                
                if load.fy != 0:
                    fy = load.fy
                    arrow_length = (abs(fy) / max_fy) * 500
                    
                    if fy < 0:
                        start_y = y + arrow_length
                        dy = -arrow_length
                        text_y_offset = 10
                    else:
                        start_y = y - arrow_length
                        dy = arrow_length
                        text_y_offset = -20
                    
                    ax.arrow(x, start_y, 0, dy * 0.8,
                            head_width=100, head_length=abs(dy)*0.2, fc='red', ec='red',
                            linewidth=2)
                    ax.annotate(f'{abs(fy)/1000:.1f} kN', (x, start_y),
                               xytext=(10, text_y_offset), textcoords='offset points',
                               fontsize=10, color='red', weight='bold')
            
            elif isinstance(load, UniformDistributedLoad):
                elements = [load.element] if isinstance(load.element, int) else load.element
                for elem_id in elements:
                    elem = self.mesh.elements[elem_id]
                    x1 = coords[elem.node1, 0]
                    x2 = coords[elem.node2, 0]
                    wy = load.wy
                    
                    if wy != 0:
                        height = 150
                        box_y_start = height if wy < 0 else -height
                        
                        rect = plt.Rectangle((x1, 0), x2-x1, box_y_start,
                                           facecolor='red', alpha=0.1, edgecolor='red', linestyle='--')
                        ax.add_patch(rect)
                        
                        num_arrows = 3
                        for x_pos in np.linspace(x1 + (x2-x1)*0.1, x2 - (x2-x1)*0.1, num_arrows):
                            ax.arrow(x_pos, box_y_start, 0, -box_y_start * 0.9,
                                    head_width=40, head_length=abs(box_y_start)*0.2, fc='red', ec='red',
                                    alpha=0.6)
                        
                        if elem_id == elements[0]:
                            ax.annotate(f'{abs(wy):.2f} N/mm', ((x1+x2)/2, box_y_start),
                                       xytext=(0, 10 if wy < 0 else -20), 
                                       textcoords='offset points', ha='center',
                                       fontsize=9, color='red', weight='bold')
        
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title('Beam Structure with Boundary Conditions and Loads', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
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
        
        Parameters
        ----------
        output_path : str
            Path to save the plot
        dpi : int
            Resolution
        """
        if self.shear_forces is None:
            self.calculate_internal_forces()
        
        fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
        
        ax.plot(self.positions, self.shear_forces / 1000, 'b-', linewidth=2.5)
        ax.fill_between(self.positions, 0, self.shear_forces / 1000, alpha=0.3, color='blue')
        ax.axhline(0, color='k', linewidth=1)
        
        max_idx = np.argmax(np.abs(self.shear_forces))
        ax.plot(self.positions[max_idx], self.shear_forces[max_idx] / 1000, 'ro', markersize=8)
        ax.annotate(f'Max: {self.shear_forces[max_idx]/1000:.2f} kN',
                   (self.positions[max_idx], self.shear_forces[max_idx] / 1000),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='red', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Shear Force (kN)', fontsize=12)
        ax.set_title('Shear Force Diagram', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def plot_moment_diagram(self, output_path: str, dpi: int = 150):
        """
        Plot bending moment diagram.
        
        Parameters
        ----------
        output_path : str
            Path to save the plot
        dpi : int
            Resolution
        """
        if self.bending_moments is None:
            self.calculate_internal_forces()
        
        fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
        
        ax.plot(self.positions, self.bending_moments / 1e6, 'g-', linewidth=2.5)
        ax.fill_between(self.positions, 0, self.bending_moments / 1e6, alpha=0.3, color='green')
        ax.axhline(0, color='k', linewidth=1)
        
        max_idx = np.argmax(np.abs(self.bending_moments))
        ax.plot(self.positions[max_idx], self.bending_moments[max_idx] / 1e6, 'ro', markersize=8)
        ax.annotate(f'Max: {self.bending_moments[max_idx]/1e6:.2f} kN·m',
                   (self.positions[max_idx], self.bending_moments[max_idx] / 1e6),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='red', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Bending Moment (kN·m)', fontsize=12)
        ax.set_title('Bending Moment Diagram', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        
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
                elements = [load.element] if isinstance(load.element, int) else load.element
                for elem_id in elements:
                    elem_length = self.mesh.elements[elem_id].length(self.mesh.nodes)
                    total_load += abs(getattr(load, 'wy', 0)) * elem_length

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
