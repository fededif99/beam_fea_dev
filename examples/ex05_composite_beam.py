"""
examples/ex05_composite_beam.py
==============================
Example of using Classical Laminate Theory (CLT) to model a composite beam.
"""

from beam_fea import BeamSolver, Mesh, BoundaryConditionSet, LoadCase, rectangular
from beam_fea.composites import Ply, Laminate
from beam_fea.failure_criteria import TsaiWuCriterion, MaximumStressCriterion
import numpy as np

def run_composite_analysis():
    # 1. Define Ply Properties 
    # Carbon/Epoxy UD (Stiff, lightweight skins)
    carbon_ply = Ply(
        name="T300_Epoxy",
        E1=135000,   # Longitudinal Modulus (MPa)
        E2=10000,    # Transverse Modulus (MPa)
        nu12=0.3,    # Poisson's ratio
        G12=5000,    # Shear Modulus (MPa)
        G13=5000, G23=4000, # Transverse Shear
        thickness=0.125, # mm
        rho=1.6e-6,  # kg/mm^3
        Xt=1500, Xc=1200, Yt=50, Yc=250, S=70 # MPa
    )

    # Aluminum Code (Isotropic properties entered as orthotropic)
    aluminum_core = Ply(
        name="Aluminum_Core",
        E1=71000,    # Isotropic E
        E2=71000,    # Isotropic E
        nu12=0.33,   # Poisson's ratio
        G12=26691,   # G = E / (2*(1+nu))
        G13=26691, G23=26691,
        thickness=5.0,   # 5mm thick core
        rho=2.7e-6,  # kg/mm^3
        Xt=310, Xc=310, Yt=310, Yc=310, S=190 # 6061-T6 approx
    )

    # 2. Create Laminate Stack-up: Sandwich Panel
    # [0/45/-45/90/Core/90/-45/45/0]
    lam = Laminate("Sandwich_Panel", stack=[
        (carbon_ply, [0, 45, -45, 90]), # Bottom skin
        (aluminum_core, 0),             # Core
        (carbon_ply, [90, -45, 45, 0])  # Top skin
    ])

    print("\n" + "="*50)
    print(f"Laminate Analysis: {lam.name}")
    print("="*50)
    print(lam)

    props = lam.get_effective_properties()
    print(f"\nEffective Properties:")
    print(f"  Ex (Axial):   {props['Ex']:.1f} MPa")
    print(f"  Eb (Bending): {props['Eb']:.1f} MPa")
    print(f"  Gxy (Shear):  {props['Gxy']:.1f} MPa")
    print(f"  Thickness:    {props['thickness']:.3f} mm")

    # 3. Create Multi-Property FEA Model
    # A 1000mm beam. Ends are wider (100mm) for supports, middle is standard (50mm).
    L = 1000
    h = props['thickness']
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=50)

    # Define sections
    sec_wide = rectangular(width=100, height=h)
    sec_mid = rectangular(width=50, height=h)

    # Create PropertySet collector
    from beam_fea import PropertySet
    p_set = PropertySet()
    
    # Apply sandwich material everywhere
    p_set.add(material=lam)
    
    # Apply sections
    p_set.add(section=sec_mid) # Default everywhere
    
    # Override ends (First 10 and last 10 elements)
    end_elements = list(range(10)) + list(range(40, 50))
    p_set.add(section=sec_wide, elements=end_elements)

    # Note: element_type='timoshenko' is highly recommended for thick sandwiches
    solver = BeamSolver(mesh, p_set, element_type='timoshenko')

    # 4. Boundary Conditions & Loads (Simply Supported with UDL)
    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(0)
    bc.roller_support(50) # node at x=L

    load = LoadCase("UDL Pressure")
    load.distributed_load(x_start=0, x_end=L, distribution='uniform', wy=-2.0) # 2 N/mm downward

    # 5. Solve Static
    solver.solve_static(load, bc)

    # 6. Results & Ply Stress Recovery
    max_def_record = solver.get_max_deflection()
    max_def_v = max_def_record['v']
    print(f"\nFEA Results:")
    print(f"  Max Transverse Deflection (v): {max_def_v:.4f} mm")

    # Trigger 3D stress field generation (required for ply recovery)
    print("\nExtracting High-Fidelity Ply Stresses...")
    # Calculate stresses (we don't need the returned grid matrices right now, it caches ply data internally)
    solver.calculate_stresses(num_x_points=100, num_y_points=int(lam.total_thickness*2)) # Double points for fine resolution
    
    # We want to check the stress right in the middle where bending is maximum (Element 25, Station 0 which is left side of elem)
    from beam_fea.post_processing import StressEngine
    peak_element_id = 25 
    
    # get_ply_stresses returns a list of dictionaries for each ply at a specific station index
    # We typically sample internal forces at 2 stations (start/end) per element if num_x_points matches element boundaries loosely, 
    # but solver.calculate_stresses(100) on 50 elements means 2 points per element. Index 0 is the start of the element.
    ply_stresses = StressEngine.get_ply_stresses(solver, element_id=peak_element_id, x_station_idx=0)
    
    if ply_stresses:
        print(f"\nPly-by-Ply Stresses near Midspan (Element {peak_element_id}):")
        print(f"{'Ply':<4} | {'Material':<13} | {'Angle':<5} | {'Sigma_1':<10} | {'Sigma_2':<10} | {'Tau_12':<10} | {'Tau_xz':<10} | {'TW SF':<8}")
        print("-" * 90)
        for ply in ply_stresses:
            s1 = (ply['sigma_1_bot'] + ply['sigma_1_top']) / 2.0
            s2 = (ply['sigma_2_bot'] + ply['sigma_2_top']) / 2.0
            t12 = max(abs(ply['tau_12_bot']), abs(ply['tau_12_top']))
            txz = max(abs(ply['tau_xz_bot']), abs(ply['tau_xz_top']))
            sf = ply['sf_tsai_wu']
            print(f"{ply['ply_index']:<4} | {ply['ply_name']:<13} | {ply['angle']:<5.1f} | {s1:>10.1f} | {s2:>10.1f} | {t12:>10.1f} | {txz:>10.1f} | {sf:>8.3f}")
    else:
        print("Ply stresses not available. Ensure laminate properties are correctly assigned.")

    # --- 8. Failure Criteria Evaluation ---
    # Demonstrate applying Tsai-Wu and Max Stress criteria to ply stresses.
    if ply_stresses:
        print("\n" + "="*50)
        print("Failure Criteria Evaluation (carbon_ply properties)")
        print("="*50)

        tw  = TsaiWuCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
        ms  = MaximumStressCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)

        # Collect mid-plane ply stresses as arrays for vectorised evaluation
        s1_arr  = np.array([(p['sigma_1_bot'] + p['sigma_1_top']) / 2 for p in ply_stresses])
        s2_arr  = np.array([(p['sigma_2_bot'] + p['sigma_2_top']) / 2 for p in ply_stresses])
        t12_arr = np.array([max(abs(p['tau_12_bot']), abs(p['tau_12_top'])) for p in ply_stresses])

        tw_result  = tw.evaluate(sigma_1=s1_arr,  sigma_2=s2_arr, tau_12=t12_arr)
        ms_result  = ms.evaluate(sigma_1=s1_arr,  sigma_2=s2_arr, tau_12=t12_arr)

        print(f"\n{'Ply':<4} | {'TW Stress':<12} | {'TW SF':<10} | {'Max Str SF':<14} | {'Status'}")
        print("-" * 65)
        for i, p in enumerate(ply_stresses):
            stress_tw = tw_result['stress'][i]
            sf_tw  = tw_result['SF'][i]
            sf_ms  = ms_result['SF'][i]
            status = "PASS" if tw_result['passed'][i] else "FAIL"
            print(f"{p['ply_index']:<4} | {stress_tw:<12.4f} | {sf_tw:<10.4f} | {sf_ms:<14.4f} | {status}")

    # 9. Generate Report
    # The report will automatically feature the stack-up (with rosettes and the core) and ply tables
    import os
    report_path = os.path.join(os.path.dirname(__file__), "composite_beam_report.md")
    solver.generate_report(report_path)
    print(f"\nReport generated: {report_path}")

if __name__ == "__main__":
    run_composite_analysis()
