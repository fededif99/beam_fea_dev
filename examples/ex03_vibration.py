"""
Example 3: Rotating Shaft Vibration Analysis
=============================================

Objective:
----------
Calculate natural frequencies and critical speeds of a shaft supported
at both ends (typical turbomachinery configuration).

Application:
------------
This analysis is critical for:
- Gas turbine rotors
- Electric motor shafts
- Propeller shafts
- Any rotating machinery

The first natural frequency gives the first critical speed, which must
be avoided during operation.

This example demonstrates:
- Modal analysis
- Natural frequency calculation
- Critical speed determination
"""

from beam_fea import (
    BeamSolver, Mesh, get_material, 
    BoundaryConditionSet, circular
)

def run_example():
    import os
    print(f"[*] Running {os.path.basename(__file__)}...")
    
    # 1. Parameters
    L = 1500.0  # mm (1.5 m shaft)
    D = 50.0    # mm diameter
    
    # Material (Aluminum from database)
    material = get_material('aluminum_6061')
    
    # Section: Solid circular shaft
    section = circular(D)
    
    # 2. Mesh
    num_elems = 30
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=num_elems)
    solver = BeamSolver(mesh, material, section)
    
    # 3. Supports: Simply supported (bearings at ends)
    bc = BoundaryConditionSet("Bearings")
    bc.pinned_support(0)
    bc.pinned_support(num_elems)
    
    # 4. Modal Analysis
    print("  Solving...")
    
    num_modes = 10
    frequencies, mode_shapes = solver.solve_modal(bc, num_modes)
    
    # 5. Results (Results logged for visibility, but could be in report)
    # The user might want some terminal output for vibration, but let's keep it minimal
    print(f"[SUCCESS] Calculated {len(frequencies)} natural frequencies.")
    
    return frequencies


if __name__ == "__main__":
    run_example()
