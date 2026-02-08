"""
report_generator.py
===================
Professional report generation for beam FEA analysis results.

Generates markdown reports with embedded visualizations including:
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
from typing import Optional, Tuple, List, Union
import os
import base64
import io
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
        
        # Will store calculated internal forces
        self.positions = None
        self.shear_forces = None
        self.bending_moments = None
        
    def calculate_internal_forces(self, num_points: int = 100):
        """
        Calculate shear force and bending moment along the beam using centralized solver logic.
        """
        results = self.solver.calculate_internal_forces(num_points)
        self.positions = results['positions']
        self.shear_forces = results['shear_forces']
        self.bending_moments = results['bending_moments']
    
    def plot_structure_diagram(self, output_path: Optional[str] = None, dpi: int = 150, return_base64: bool = False) -> Optional[str]:
        """
        Plot beam structure with boundary conditions and loads.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot
        dpi : int
            Resolution in dots per inch
        return_base64 : bool
            If True, returns the plot as a base64 string
            
        Returns
        -------
        str or None
            Base64 encoded string if return_base64 is True
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
                # Pinned support
                ax.plot(x, y, 'g^', markersize=15, label='Pinned Support' if node_id == 0 else '')
                ax.annotate('Pinned', (x, y), xytext=(0, -30), 
                           textcoords='offset points', ha='center',
                           fontsize=9, color='green')
            elif isinstance(bc, RollerSupport):
                # Roller support
                ax.plot(x, y, 'go', markersize=12, label='Roller Support' if node_id == 0 else '')
                ax.annotate('Roller', (x, y), xytext=(0, -30),
                           textcoords='offset points', ha='center',
                           fontsize=9, color='green')
            elif isinstance(bc, FixedSupport):
                # Fixed support
                ax.plot(x, y, 'gs', markersize=15, label='Fixed Support' if node_id == 0 else '')
                ax.annotate('Fixed', (x, y), xytext=(0, -30),
                           textcoords='offset points', ha='center',
                           fontsize=9, color='green', weight='bold')
        
        # Plot loads
        for load in self.load_case.loads:
            if isinstance(load, PointLoad):
                node_id = load.node
                x, y = coords[node_id, 0], coords[node_id, 1]
                
                # Draw force arrow
                if load.fy != 0:
                    fy = load.fy
                    # Normalize arrow length relative to max load
                    max_fy = max([abs(l.fy) for l in self.load_case.loads if isinstance(l, PointLoad)] + [1])
                    arrow_length = (abs(fy) / max_fy) * 500
                    
                    if fy < 0:
                        # Downward force: Arrow starts above beam and points down
                        start_y = y + arrow_length
                        dy = -arrow_length
                        text_y_offset = 10
                    else:
                        # Upward force: Arrow starts below beam and points up
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
                        direction = -1 if wy < 0 else 1
                        height = 150 # Visualization height for the block
                        
                        # Academic standard: downward loads are shown above the beam pointing down
                        # OR if wy < 0, direction = -1 means it draws below.
                        # We want arrows pointing TO the beam.
                        
                        box_y_start = height if wy < 0 else -height
                        
                        # Draw rectangle for UDL
                        rect = plt.Rectangle((x1, 0), x2-x1, box_y_start,
                                           facecolor='red', alpha=0.1, edgecolor='red', linestyle='--')
                        ax.add_patch(rect)
                        
                        # Add arrows pointing to the beam (y=0)
                        num_arrows = 3
                        for x_pos in np.linspace(x1 + (x2-x1)*0.1, x2 - (x2-x1)*0.1, num_arrows):
                            # Arrow starts at box_y_start and points to 0
                            ax.arrow(x_pos, box_y_start, 0, -box_y_start * 0.9,
                                    head_width=40, head_length=abs(box_y_start)*0.2, fc='red', ec='red',
                                    alpha=0.6)
                        
                        if elem_id == elements[0]:
                            ax.annotate(f'{abs(wy):.2f} N/mm', ( (x1+x2)/2, box_y_start),
                                       xytext=(0, 10 if wy < 0 else -20), 
                                       textcoords='offset points', ha='center',
                                       fontsize=9, color='red', weight='bold')
        
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Height (mm)', fontsize=12)
        ax.set_title('Beam Structure with Boundary Conditions and Loads', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            
        result = None
        if return_base64:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
            buffer.seek(0)
            result = base64.b64encode(buffer.read()).decode()
            
        plt.close()
        return result
    
    def plot_deformed_shape(self, output_path: Optional[str] = None, scale_factor: Union[float, str] = 'auto', dpi: int = 150, return_base64: bool = False) -> Optional[str]:
        """
        Plot deformed shape compared to undeformed using BeamVisualizer logic.
        """
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        
        # Determine scale factor
        if scale_factor == 'auto':
            s = viz._auto_scale_factor(self.displacements)
        else:
            s = float(scale_factor)
            
        return viz.plot_deformed_shape(
            self.displacements,
            scale_factor=s,
            dpi=dpi,
            output_path=output_path,
            return_base64=return_base64
        )
    
    def plot_shear_diagram(self, output_path: Optional[str] = None, dpi: int = 150, return_base64: bool = False) -> Optional[str]:
        """
        Plot shear force diagram.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot
        dpi : int
            Resolution
        return_base64 : bool
            If True, returns the plot as a base64 string
        """
        if self.shear_forces is None:
            self.calculate_internal_forces()
        
        fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
        
        ax.plot(self.positions, self.shear_forces / 1000, 'b-', linewidth=2.5)
        ax.fill_between(self.positions, 0, self.shear_forces / 1000, alpha=0.3, color='blue')
        ax.axhline(0, color='k', linewidth=1)
        
        # Mark maximum values
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
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            
        result = None
        if return_base64:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
            buffer.seek(0)
            result = base64.b64encode(buffer.read()).decode()
            
        plt.close()
        return result
    
    def plot_moment_diagram(self, output_path: Optional[str] = None, dpi: int = 150, return_base64: bool = False) -> Optional[str]:
        """
        Plot bending moment diagram.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the plot
        dpi : int
            Resolution
        return_base64 : bool
            If True, returns the plot as a base64 string
        """
        if self.bending_moments is None:
            self.calculate_internal_forces()
        
        fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
        
        ax.plot(self.positions, self.bending_moments / 1e6, 'g-', linewidth=2.5)
        ax.fill_between(self.positions, 0, self.bending_moments / 1e6, alpha=0.3, color='green')
        ax.axhline(0, color='k', linewidth=1)
        
        # Mark maximum values
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
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            
        result = None
        if return_base64:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
            buffer.seek(0)
            result = base64.b64encode(buffer.read()).decode()
            
        plt.close()
        return result
    
    def plot_cross_section(self, output_path: Optional[str] = None, dpi: int = 150, return_base64: bool = False) -> Optional[str]:
        """
        Plot cross-section using the enhanced BeamVisualizer logic.
        """
        from .visualizer import BeamVisualizer
        viz = BeamVisualizer(self.mesh)
        return viz.plot_section_properties(
            self.section,
            output_path=output_path,
            return_base64=return_base64,
            dpi=dpi
        )
    
    def generate_report(self, output_path: str, deformation_scale: Union[float, str] = 'auto', embed_images: bool = False):
        """
        Generate complete markdown report.
        
        Parameters
        ----------
        output_path : str
            Path for the markdown report file
        deformation_scale : float
            Scale factor for deformation visualization
        embed_images : bool
            If True, images are embedded as base64 strings
        """
        # Create output directory
        output_path = Path(output_path)
        output_dir = output_path.parent
        report_name = output_path.stem
        
        images_dir = output_dir / f"{report_name}_images"
        if not embed_images:
            images_dir.mkdir(parents=True, exist_ok=True)
            images_rel_path = f"{report_name}_images"
        
        # Helper to handle plot generation and embedding
        def get_image_markdown(plot_func, filename, alt_text, **kwargs):
            if embed_images:
                b64_str = plot_func(return_base64=True, **kwargs)
                return f"![{alt_text}](data:image/png;base64,{b64_str})"
            else:
                img_path = images_dir / filename
                plot_func(str(img_path), **kwargs)
                return f"![{alt_text}]({images_rel_path}/{filename})"
        
        # Generate all plots
        # Generating structure diagram...
        structure_md = get_image_markdown(self.plot_structure_diagram, "structure_diagram.png", "Structure Diagram")
        
        # Generating cross-section plot...
        section_md = get_image_markdown(self.plot_cross_section, "cross_section.png", "Cross-Section")
        
        # Generating deformed shape plot...
        deformed_md = get_image_markdown(self.plot_deformed_shape, "deformed_shape.png", "Deformed Shape", scale_factor=deformation_scale)
        
        # Calculating internal forces...
        self.calculate_internal_forces()
        
        # Generating shear force diagram...
        shear_md = get_image_markdown(self.plot_shear_diagram, "shear_diagram.png", "Shear Force Diagram")
        
        # Generating bending moment diagram...
        moment_md = get_image_markdown(self.plot_moment_diagram, "moment_diagram.png", "Bending Moment Diagram")
        
        # Calculate key results
        v_displacements = self.displacements[1::3]
        max_deflection = abs(min(v_displacements))
        max_deflection_node = v_displacements.argmin()
        max_deflection_pos = self.mesh.nodes[max_deflection_node].x
        
        max_shear = np.max(np.abs(self.shear_forces))
        max_moment = np.max(np.abs(self.bending_moments))
        
        # Generate markdown content
        md_content = f"""# Beam FEA Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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

{structure_md}

### Cross-Section Details

{section_md}

---

## 4. Analysis Results

### Deformed Shape

The following plot shows the deformed shape of the beam (exaggerated by {deformation_scale:.0f}× for visualization):

{deformed_md}

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
        
        # Calculate total reaction
        total_fy = 0
        for bc in self.bc_set.conditions:
            if isinstance(bc, (PinnedSupport, RollerSupport, FixedSupport)):
                total_fy += self.reactions[3*bc.node + 1]
                # Calculate total applied load for summary
        total_load = sum(abs(getattr(load, 'fy', 0)) 
                        for load in self.load_case.loads if hasattr(load, 'fy'))
        
        # Add distributed loads
        for load in self.load_case.loads:
            if hasattr(load, 'wy'):
                # Handle both single element and list of elements
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

{shear_md}

**Maximum Shear Force:** {max_shear/1000:.2f} kN

### Bending Moment Diagram

{moment_md}

**Maximum Bending Moment:** {max_moment/1e6:.2f} kN·m

---

## 6. Summary

The finite element analysis has been successfully completed. Key findings:

- The maximum deflection of **{max_deflection:.4f} mm** occurs at x = {max_deflection_pos:.1f} mm
- The maximum shear force is **{max_shear/1000:.2f} kN**
- The maximum bending moment is **{max_moment/1e6:.2f} kN·m**
- Equilibrium is satisfied with a residual of {abs(total_fy - abs(total_load)):.2e} N

---

*Report generated by Beam FEA Analysis Tool*
"""
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # Report generation complete
        
        return str(output_path)


if __name__ == "__main__":
    print("\nBeam Report Generator Module")
    print("Generates professional markdown reports for beam analysis")
