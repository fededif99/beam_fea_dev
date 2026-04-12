"""
visual_test_configurations.py
==============================
Visual stress tests for Beam FEA report generator.

Runs 8 diverse beam configurations that collectively cover:
  - All primary cross-section types (rectangular, i_beam, c_channel, box,
    circular, hollow_circular, t_beam, l_section)
  - All support types (Fixed, Pinned, Roller) including multi-BC per beam
  - Multiple load types (PointLoad Fy, PointLoad Fx+Mz, UDL, Trapezoidal,
    combined loads)
  - Extreme scale ranges (500 mm → 15 000 mm) to test smart unit labelling

Reports and images are saved to:
    <project_root>/visual_stress_tests/<config_name>_report.md
    <project_root>/visual_stress_tests/<config_name>_report_images/
"""

import os
import traceback
import numpy as np

# ROOT is used for defining OUTPUT_ROOT
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from beam_fea import (
    BeamSolver, Mesh, Material, get_material,
    LoadCase, BoundaryConditionSet
)
from beam_fea.cross_sections import (
    rectangular, i_beam, c_channel, box as box_section,
    circular, hollow_circular, t_beam, l_section
)
from beam_fea.composites import Ply, Laminate

OUTPUT_ROOT = os.path.join(ROOT, 'visual_stress_tests')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared materials
# ---------------------------------------------------------------------------
steel     = get_material('steel')
aluminium = get_material('aluminium_7075')

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def run_config(name: str, mesh, material, section, lc, bc,
               deformation_scale='auto') -> bool:
    """Solve, generate report, print result. Returns True on success."""
    try:
        solver = BeamSolver(mesh, material, section, element_type='euler')
        solver.solve_static(lc, bc)

        report_path = os.path.join(OUTPUT_ROOT, f'{name}_report.md')
        solver.generate_report(report_path, deformation_scale=deformation_scale)

        # Quick sanity: max deflection should be finite
        max_rec = solver.get_max_deflection()
        print(f"  [PASS] {name:55s}  d_max = {max_rec['res']:.4f} mm")
        return True
    except Exception:
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 1 — Short simply-supported beam, single central point load
# Cross-section: rectangular (small 30 × 50 mm)
# BCs: Pinned (node 0) + Pinned (node 5)
# ---------------------------------------------------------------------------
def config_1():
    L, n = 500.0, 5
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = rectangular(width=30.0, height=50.0)

    lc = LoadCase("Central point load")
    lc.point_load(x=L / 2, fy=-5000.0)          # 5 kN downward

    bc = BoundaryConditionSet("Pin-Pin")
    bc.pinned_support(node=0)
    bc.pinned_support(node=n)

    return run_config("01_short_pinpin_rect", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 2 — Long simply-supported beam, full-span UDL (kN·m scale)
# Cross-section: I-beam (IPE 300 approx.)
# BCs: Pinned (node 0) + Pinned (end node)
# ---------------------------------------------------------------------------
def config_2():
    L, n = 10_000.0, 40
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = i_beam(flange_width=150, total_height=300,
                     web_thickness=7, flange_thickness=10.7)

    lc = LoadCase("Full UDL")
    lc.distributed_load(x_start=0.0, x_end=L, distribution='uniform', wy=-5.0)   # 5 N/mm = 5 kN/m

    bc = BoundaryConditionSet("Pin-Pin long")
    bc.pinned_support(node=0)
    bc.pinned_support(node=n)

    return run_config("02_long_pinpin_ibeam_udl", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 3 — Cantilever, tip point load
# Cross-section: C-channel
# BCs: Fixed (node 0)
# ---------------------------------------------------------------------------
def config_3():
    L, n = 2_000.0, 20
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = c_channel(height=160, flange_width=80,
                        web_thickness=6, flange_thickness=9)

    lc = LoadCase("Tip load")
    lc.point_load(node=n, fy=-8000.0)              # 8 kN

    bc = BoundaryConditionSet("Cantilever")
    bc.fixed_support(node=0)

    return run_config("03_cantilever_cchannel", mesh, aluminium, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 4 — Propped cantilever (Fixed root + Roller tip), full UDL
# Cross-section: box section
# Multi-BC: Fixed at root, Roller at tip
# ---------------------------------------------------------------------------
def config_4():
    L, n = 3_000.0, 30
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = box_section(width=120, height=180, thickness=8)

    lc = LoadCase("Full UDL propped")
    lc.distributed_load(x_start=0.0, x_end=L, distribution='uniform', wy=-3.0)   # 3 N/mm

    bc = BoundaryConditionSet("Fixed–Roller propped cantilever")
    bc.fixed_support(node=0)
    bc.roller_support(node=n, direction='y')

    return run_config("04_propped_cantilever_box", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 5 — Long cantilever, extreme scale (15 m), tip point load
# Cross-section: solid circular (100 mm diameter)
# BCs: Fixed (node 0)  — tests 'm' unit label
# ---------------------------------------------------------------------------
def config_5():
    L, n = 15_000.0, 50
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = circular(diameter=100.0)

    lc = LoadCase("Tip load extreme")
    lc.point_load(node=n, fy=-2000.0)              # 2 kN

    bc = BoundaryConditionSet("Long cantilever")
    bc.fixed_support(node=0)

    return run_config("05_long_cantilever_circular", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 6 — 3-support continuous beam (multi-BC)
# Cross-section: hollow circular tube
# BCs: Pin (node 0) + Roller (mid node) + Pin (end node)
# Loads: UDL on left half + point load at mid
# ---------------------------------------------------------------------------
def config_6():
    L, n = 4_000.0, 20
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    mid_node = n // 2

    section = hollow_circular(outer_diameter=100.0, thickness=10.0)

    lc = LoadCase("Multi-span loads")
    lc.distributed_load(x_start=0.0, x_end=L / 2, distribution='uniform', wy=-2.0)   # left half UDL
    lc.point_load(x=L / 2, fy=-6000.0)                    # mid point load

    bc = BoundaryConditionSet("Pin–Roller–Pin (3 supports)")
    bc.pinned_support(node=0)
    bc.roller_support(node=mid_node, direction='y')
    bc.pinned_support(node=n)

    return run_config("06_3support_hollow_circ", mesh, aluminium, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 7 — Simply-supported, combined loads including concentrated moment
# Cross-section: T-beam
# BCs: Pin–Pin
# Loads: Point Fy + UDL + Concentrated Mz  (tests moment arc symbol)
# ---------------------------------------------------------------------------
def config_7():
    L, n = 4_000.0, 25
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = t_beam(flange_width=150, web_height=185,
                     web_thickness=10, flange_thickness=15)

    lc = LoadCase("Combined loads + moment")
    lc.point_load(x=L * 0.3, fy=-10_000.0)               # 10 kN
    lc.distributed_load(x_start=L * 0.5, x_end=L, distribution='uniform', wy=-1.5)   # right-half UDL
    lc.moment(x=L * 0.7, mz=5_000_000.0)    # 5 kN·m CCW

    bc = BoundaryConditionSet("Pin–Pin combined")
    bc.pinned_support(node=0)
    bc.pinned_support(node=n)

    return run_config("07_combined_tbeam", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 8 — Simply-supported, trapezoidal distributed load
# Cross-section: L-section (angle section)
# BCs: Pin–Pin
# ---------------------------------------------------------------------------
def config_8():
    L, n = 6_000.0, 20
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = l_section(leg_vertical=80, leg_horizontal=80, thickness=8)

    lc = LoadCase("Trapezoidal UDL")
    lc.distributed_load(x_start=0.0, x_end=L, distribution='linear',
                        wy_start=-0.5, wy_end=-3.0)    # 0.5→3 N/mm tapered

    bc = BoundaryConditionSet("Pin–Pin trapezoidal")
    bc.pinned_support(node=0)
    bc.pinned_support(node=n)

    return run_config("08_trap_udl_lsection", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 9 — Fixed-Fixed with Axial Force
# Cross-section: Rectangular
# BCs: Fixed at both ends (node 0 and node n)
# Loads: Multiple point loads + 1 axial component
# ---------------------------------------------------------------------------
def config_9():
    L, n = 5_000.0, 50
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
    section = rectangular(width=200, height=400)

    lc = LoadCase("Fixed-Fixed Axial")
    # Vertical loads
    lc.point_load(node=int(n/2), fy=-50000.0)      # 50 kN mid-span
    lc.point_load(node=int(n/4), fy=-25000.0)      # 25 kN
    # Axial load - 100 kN at node n/4 acting horizontally
    lc.point_load(node=int(n/4), fx=100000.0)

    bc = BoundaryConditionSet("Fixed-Fixed")
    bc.fixed_support(node=0)
    bc.fixed_support(node=n)

    return run_config("09_fixed_fixed_axial", mesh, steel, section, lc, bc)


# ---------------------------------------------------------------------------
# Test 10 — Modal Analysis
# Cross-section: Rectangular
# BCs: Pinned-Pinned
# Tests: solve_modal outputs, mode shape plots, freq tables
# ---------------------------------------------------------------------------
def config_10():
    try:
        L, n = 2_000.0, 40
        mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
        section = rectangular(width=50, height=100)
        
        bc = BoundaryConditionSet("Modal Restraint")
        bc.pinned_support(node=0)
        bc.pinned_support(node=n)

        lc = LoadCase("Free Vibration")

        solver = BeamSolver(mesh, steel, section, element_type='timoshenko')
        solver.solve_modal(bc, num_modes=3)

        report_path = os.path.join(OUTPUT_ROOT, '10_modal_analysis_report.md')
        solver.generate_report(report_path)
        print(f"  [PASS] {'10_modal_analysis':55s}  Modes = 3")
        return True
    except Exception:
        print(f"  [FAIL] 10_modal_analysis")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 11 — Composite Laminate Static
# Cross-section: Rectangular (Thin lam)
# BCs: Cantilever
# Tests: Ply-by-Ply stesses, Polar Rosette, Tsai-Wu Failure
# ---------------------------------------------------------------------------
def config_11():
    try:
        L, n = 1_000.0, 20
        mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
        
        carbon_ply = Ply(
            name="T300_Epoxy", E1=135000, E2=10000, nu12=0.3,
            G12=5000, G13=5000, G23=4000, thickness=0.2, rho=1.6e-6,
            Xt=1500, Xc=1200, Yt=50, Yc=250, S=70, S13=50, S23=40
        )
        lam = Laminate("CFRP_Skin", beam_type='narrow', stack=[
            (carbon_ply, [0, 45, -45, 90, 90, -45, 45, 0])
        ])
        
        section = rectangular(width=25, height=lam.total_thickness)
        
        lc = LoadCase("Composite Tip Load")
        lc.point_load(node=n, fy=-100.0)
        
        bc = BoundaryConditionSet("Cantilever")
        bc.fixed_support(0)
        
        solver = BeamSolver(mesh, lam, section, element_type='timoshenko')
        solver.solve_static(lc, bc)
        
        report_path = os.path.join(OUTPUT_ROOT, '11_composite_laminate_report.md')
        solver.generate_report(report_path, failure_criterion='tsai_wu')
        
        max_rec = solver.get_max_deflection()
        print(f"  [PASS] {'11_composite_laminate':55s}  d_max = {max_rec['res']:.4f} mm")
        return True
    except Exception:
        print(f"  [FAIL] 11_composite_laminate")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 12 — Advanced Loads & Springs
# BCs: Spring root, prescribed displacement tip
# Loads: LumpedMass + Custom Equation
# ---------------------------------------------------------------------------
def config_12():
    try:
        L, n = 2_000.0, 40
        mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=n)
        section = circular(diameter=80)
        
        lc = LoadCase("Advanced Loads")
        lc.distributed_load(x_start=0, x_end=L, distribution='custom', load_fn=lambda x: -1e-6 * x**2)
        lc.lumped_mass(x=L/2, m=50.0, Izz=0.0, apply_gravity=True)
        
        bc = BoundaryConditionSet("Spring and Displacement")
        bc.spring_support(0, ky=1e5, kr=1e7)
        bc.prescribed_displacement(n, dy=-5.0)
        
        solver = BeamSolver(mesh, aluminium, section)
        solver.solve_static(lc, bc)
        
        report_path = os.path.join(OUTPUT_ROOT, '12_advanced_loads_bc_report.md')
        solver.generate_report(report_path)
        
        max_rec = solver.get_max_deflection()
        print(f"  [PASS] {'12_advanced_loads_bc':55s}  d_max = {max_rec['res']:.4f} mm")
        return True
    except Exception:
        print(f"  [FAIL] 12_advanced_loads_bc")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 13 — Angled Frame Geometry
# Shape: Diagonal beam
# Tests: 2D geometry projection in Deformation charts
# ---------------------------------------------------------------------------
def config_13():
    try:
        mesh = Mesh.from_path([(0, 0), (1000, 500)], elements_per_segment=20)
        section = i_beam(flange_width=100, total_height=200, web_thickness=5, flange_thickness=8)
        
        lc = LoadCase("Vertical load on diagonal beam")
        # Distributed load along the exact mapped element length (1118.03)
        lc.distributed_load(x_start=0, x_end=1118.03, distribution='uniform', wy=-2.0)
        
        bc = BoundaryConditionSet("Fixed-Free")
        bc.fixed_support(0)
        
        solver = BeamSolver(mesh, steel, section)
        solver.solve_static(lc, bc)
        
        report_path = os.path.join(OUTPUT_ROOT, '13_angled_frame_report.md')
        solver.generate_report(report_path)
        
        max_rec = solver.get_max_deflection()
        print(f"  [PASS] {'13_angled_frame':55s}  d_max = {max_rec['res']:.4f} mm")
        return True
    except Exception:
        print(f"  [FAIL] 13_angled_frame")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    configs = [
        config_1,
        config_2,
        config_3,
        config_4,
        config_5,
        config_6,
        config_7,
        config_8,
        config_9,
        config_10,
        config_11,
        config_12,
        config_13,
    ]

    print("=" * 75)
    print("Beam FEA — Visual Stress Test Suite")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 75)

    results = [fn() for fn in configs]

    passed = sum(results)
    total  = len(results)
    print("=" * 75)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("Review the FAIL stack traces above.")
    else:
        print("All configurations produced reports. Open the images to review.")
    print(f"Reports saved to: {OUTPUT_ROOT}")
