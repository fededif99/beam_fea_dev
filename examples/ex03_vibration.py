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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beam_fea import (
    BeamSolver, MeshGenerator, Material, 
    BoundaryConditionSet
)
from beam_fea.cross_sections import circular

def run_example():
    print("Running Example 3: Rotating Shaft Vibration Analysis")
    print("---------------------------------------------------")
    
    # 1. Parameters
    L = 1500.0  # mm (1.5 m shaft)
    D = 50.0    # mm diameter
    
    # Material: Steel AISI 4340 (common for high-speed shafts)
    steel_4340 = Material(
        "Steel AISI 4340",
        E=205000,
        nu=0.29,
        rho=7.85e-6,
        yield_strength=710
    )
    
    # Section: Solid circular shaft
    section = circular(D)
    print(f"Shaft Properties:")
    print(f"  Diameter = {D} mm")
    print(f"  I = {section.Iy:.2e} mm^4")
    
    # 2. Mesh
    num_elems = 30
    mesh = MeshGenerator.beam_mesh_1d(L, num_elems)
    solver = BeamSolver(mesh, steel_4340, section)
    
    # 3. Supports: Simply supported (bearings at ends)
    bc = BoundaryConditionSet("Bearings")
    bc.pinned_support(0)
    bc.pinned_support(num_elems)
    
    # 4. Modal Analysis
    print("\nCalculating first 10 natural frequencies...")
    
    num_modes = 10
    frequencies, mode_shapes = solver.solve_modal(bc, num_modes)
    
    # 5. Results
    print("\nNatural Frequencies:")
    for i, freq in enumerate(frequencies, 1):
        rpm = freq * 60  # Convert Hz to RPM
        print(f"  Mode {i:2d}: {freq:10.2f} Hz  ({rpm:10.0f} RPM)")
    
    # Critical speeds
    print("\nCritical Speeds:")
    print(f"  1st Critical: {frequencies[0]*60:.0f} RPM")
    print(f"  2nd Critical: {frequencies[1]*60:.0f} RPM")
    
    print("\n*** WARNING: Operating speed must avoid these critical speeds!")
    print("    Typical safety margin: +/-20% of critical speed")
    
    return frequencies


if __name__ == "__main__":
    run_example()
