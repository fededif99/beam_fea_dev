"""
Versatile Beam Analysis Template
================================
This script serves as a general-purpose template for 1D Beam FEA.
It supports multiple cross-sections, materials, and various boundary conditions.

How to use:
-----------
1. Scroll down to the 'USER CONFIGURATION' section at the bottom.
2. Select an ANALYSIS_TYPE (CANTILEVER, SIMPLY_SUPPORTED, etc.).
3. Define your loads (Point and Distributed).
4. Run the script: `python examples/example_walkthrough.py`
"""

from typing import List, Dict, Any
import os

from beam_fea import (
    BeamSolver, MeshGenerator, get_material, 
    LoadCase, BoundaryConditionSet
)
from beam_fea.cross_sections import (
    rectangular, circular, i_beam, hollow_circular, box, t_beam
)

def apply_preset_bc(bc_set: BoundaryConditionSet, preset: str, num_nodes: int):
    """Apply standard engineering boundary condition presets."""
    last_node = num_nodes - 1
    
    if preset == "CANTILEVER":
        bc_set.fixed_support(0)
    elif preset == "SIMPLY_SUPPORTED":
        bc_set.pinned_support(0)
        bc_set.pinned_support(last_node)
    elif preset == "FIXED_FIXED":
        bc_set.fixed_support(0)
        bc_set.fixed_support(last_node)
    elif preset == "PROPPED_CANTILEVER":
        bc_set.fixed_support(0)
        bc_set.pinned_support(last_node)
    elif preset == "PINNED_FIXED":
        bc_set.pinned_support(0)
        bc_set.fixed_support(last_node)

def run_analysis(config: Dict[str, Any]):
    """Orchestrates the FEA analysis based on user configuration."""
    print("\n" + "="*80)
    print(f" INITIALIZING ANALYSIS: {config['title']}")
    print("="*80)

    # 1. Mesh Generation
    mesh = MeshGenerator.beam_mesh_1d(
        length=config['length'], 
        num_elements=config['num_elements']
    )
    print(f"[*] Mesh: {mesh.num_elements} elements, {mesh.num_nodes} nodes.")

    # 2. Material & Section
    material = get_material(config['material'])
    
    # Section selector
    st = config['section_type'].lower()
    sp = config['section_params']
    if st == 'rectangular': section = rectangular(sp['width'], sp['height'])
    elif st == 'i_beam': section = i_beam(sp['fw'], sp['h'], sp['tw'], sp['tf'])
    elif st == 'circular': section = circular(sp['d'])
    elif st == 'hollow_circular': section = hollow_circular(sp['od'], sp['t'])
    else: raise ValueError(f"Unsupported section type: {st}")
    
    print(f"[*] Material: {material.name}")
    print(f"[*] Section: {config['section_type']} (Iy = {section.Iy:.2e} mm^4)")

    # 3. Solver Setup
    solver = BeamSolver(mesh, material, section)

    # 4. Boundary Conditions (Presets or Custom)
    bc_set = BoundaryConditionSet("User Supports")
    if config['analysis_type'] in ["CANTILEVER", "SIMPLY_SUPPORTED", "FIXED_FIXED", "PROPPED_CANTILEVER"]:
        apply_preset_bc(bc_set, config['analysis_type'], mesh.num_nodes)
        print(f"[*] Applied Preset BC: {config['analysis_type']}")
    else:
        # Custom BC implementation (users can add logic here)
        print("[!] Using custom/unspecified BC logic.")
        bc_set.fixed_support(0)

    # 5. Load Case (Multi-load support)
    lc = LoadCase("User Loads")
    
    # Point Loads: [{'node': 10, 'fy': -5000}, ...]
    for pl in config.get('point_loads', []):
        lc.point_load(node=pl.get('node', mesh.num_nodes-1), fy=pl['fy'])
    
    # Distributed Loads: [{'elements': [0,1,2], 'wy': -5.0}, ...]
    for dl in config.get('distributed_loads', []):
        lc.uniform_load(dl['elements'], wy=dl['wy'])

    print(f"[*] Loads Applied: {len(config.get('point_loads', []))} points, {len(config.get('distributed_loads', []))} distributed.")

    # 6. Solve
    print("\nSolving System...")
    solver.solve_static(lc, bc_set)
    disp_max, node_max = solver.get_max_deflection()
    print(f"[SUCCESS] Max Deflection: {disp_max:.4f} mm at node {node_max}")
    
    # 7. Stress Analysis
    if config.get('calculate_stresses', True):
        print("\nCalculating 3D Stress Field...")
        res = config.get('stress_resolution', {'x': 50, 'y': 20, 'z': 20})
        stresses = solver.calculate_stresses(
            num_x_points=res['x'], 
            num_y_points=res['y'], 
            num_z_points=res['z']
        )
        import numpy as np
        max_vm = np.max(stresses['von_mises'])
        max_shear = np.max(np.abs(stresses['shear']))
        print(f"[*] Peak von Mises Stress: {max_vm:.2f} MPa")
        print(f"[*] Peak Shear Stress:    {max_shear:.2f} MPa")
        
        # Check against yield
        margin = (material.yield_strength / max_vm) - 1.0 if max_vm > 0 else float('inf')
        print(f"[*] Factor of Safety (von Mises): {margin + 1.0:.2f}")
        if margin < 0:
            print("[WARNING] Stress exceeds material yield strength!")

    # 8. Report & Visualization
    if config.get('generate_report', True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, config.get('report_name', 'analysis_report.md'))
        solver.generate_report(path)
        print(f"[*] Report saved to: {path}")

    if config.get('show_plots', False):
        print("\nDisplaying Visualization...")
        solver.visualize('static', scale_factor=config.get('plot_scale', None))
        solver.visualize('shear')
        solver.visualize('moment')

    # 8. Modal Analysis (Optional)
    if config.get('num_modes', 0) > 0:
        print(f"\nPerforming Modal Analysis ({config['num_modes']} modes)...")
        freqs, _ = solver.solve_modal(bc_set, num_modes=config['num_modes'])
        for i, f in enumerate(freqs, 1):
            print(f"  Mode {i}: {f:.2f} Hz")

if __name__ == "__main__":
    # ============================================================================
    # 🚩 USER CONFIGURATION AREA
    # ============================================================================
    
    USER_CONFIG = {
        'title': "Standard Wing Spar Verification",
        
        # Geometry
        'length': 3000.0,       # mm
        'num_elements': 50,    # discretization density
        
        # Analysis Type (Presets: CANTILEVER, SIMPLY_SUPPORTED, FIXED_FIXED, PROPPED_CANTILEVER)
        'analysis_type': "CANTILEVER",
        
        # Material (predefined in MATERIAL_LIBRARY)
        'material': "aluminium_7075",
        
        # Section Selection & Params
        'section_type': "rectangular",   # Options: rectangular, i_beam, circular, hollow_circular
        'section_params': {
            'width': 100, 'height': 200
        },
        
        # Loading (Point Loads at nodes)
        'point_loads': [
            {'node': 10, 'fy': -5000.0},
            {'node': 25, 'fy': -10000.0},
            {'node': 40, 'fy': -5000.0}
        ],
        
        # Loading (Distributed Loads on lists of elements)
        'distributed_loads': [
            {'elements': list(range(50)), 'wy': -2.0} # 2 N/mm along entire beam
        ],
        
        # Analysis Options
        'num_modes': 5,            # Set > 0 for modal analysis
        'calculate_stresses': True, # Detailed 3D stress field extraction
        'stress_resolution': {'x': 100, 'y': 20, 'z': 20},
        'generate_report': True,
        'report_name': "spar_analysis_report.md",
        'show_plots': False        # Requires Matplotlib
    }

    # ============================================================================
    # RUN TEMPLATE
    # ============================================================================
    run_analysis(USER_CONFIG)
