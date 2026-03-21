"""
Validation Script: Analytical Comparison
========================================

Compares Beam FEA results against exact analytical solutions from 
Roark's Formulas for Stress and Strain and Timoshenko's Vibration Problems.

Cases:
1. Cantilever Point Load (Roark Table 8.1 Case 1a)
2. Pinned-Pinned UDL (Roark Table 8.1 Case 2e)
3. Fixed-Fixed UDL (Roark Table 8.1 Case 2d)
4. Natural Frequencies (Timoshenko)
"""

import numpy as np

from beam_fea import BeamSolver, Mesh, Material, rectangular, LoadCase, BoundaryConditionSet

def header(title):
    print(f"\n{'='*70}")
    print(f"VALIDATION: {title}")
    print(f"{'='*70}")

def check_error(name, computed, exact, tol_percent=1.0):
    error = abs((computed - exact) / exact) * 100
    status = "PASS" if error < tol_percent else "FAIL"
    print(f"{name:<25} | FEA: {computed:>10.4f} | Exact: {exact:>10.4f} | Error: {error:>6.4f}% | [{status}]")
    return error < tol_percent

def validate_static_cases():
    # Common properties
    L = 1000.0  # mm
    E = 200000.0  # MPa
    b = 50.0  # mm
    h = 100.0  # mm
    
    # Calculate I
    I = b * h**3 / 12  # 4,166,666.67 mm^4
    A_rect = b * h
    c = h / 2.0
    
    material = Material("Steel", E, 0.3, 7850e-9)
    section = rectangular(b, h)
    
    # 100 elements for high accuracy
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=100)
    
    # Use Euler-Bernoulli for direct comparison with elementary beam theory
    solver = BeamSolver(mesh, material, section, element_type='euler')
    solver.assemble_global_matrices()
    
    # ---------------------------------------------------------
    # Case 1: Cantilever with Tip Load
    # ---------------------------------------------------------
    header("Cantilever Beam + Tip Load (Roark Case 1a)")
    P = -1000.0  # N
    
    bc_cantilever = BoundaryConditionSet("Cantilever")
    bc_cantilever.fixed_support(0)
    
    lc_point = LoadCase("Tip Load")
    lc_point.point_load(100, fy=P) # Node 100 is at tip
    
    solver.solve_static(lc_point, bc_cantilever)
    max_rec = solver.get_max_deflection()
    max_def = max_rec['res']
    
    # Exact: P L^3 / (3 E I)
    deflection_exact = abs(P * L**3 / (3 * E * I))
    check_error("Tip Deflection", abs(max_def), deflection_exact)
    
    # Stress Validation
    stresses = solver.calculate_stresses(num_x_points=101, num_y_points=21, num_z_points=5)
    
    # Max Bending Stress: M*c/I = |P*L|*c/I
    sigma_exact = abs(P * L) * c / I
    check_error("Max Bending Stress", np.max(np.abs(stresses['bending'])), sigma_exact)
    
    # Max Shear Stress (Rectangular): 1.5 * V / A
    tau_exact = 1.5 * abs(P) / A_rect
    check_error("Max Shear Stress", np.max(np.abs(stresses['shear'])), tau_exact)
    
    # ---------------------------------------------------------
    # Case 2: Simply Supported + UDL
    # ---------------------------------------------------------
    header("Pinned-Pinned Beam + UDL (Roark Case 2e)")
    w = -1.0  # N/mm
    
    bc_pinned = BoundaryConditionSet("Pinned-Pinned")
    bc_pinned.pinned_support(0)
    bc_pinned.pinned_support(100)
    
    lc_udl = LoadCase("UDL")
    lc_udl.distributed_load(element=list(range(100)), distribution='uniform', wy=w)
    
    solver.solve_static(lc_udl, bc_pinned)
    max_rec = solver.get_max_deflection()
    max_def = max_rec['res']
    
    # Exact: 5 w L^4 / (384 E I)
    deflection_exact = abs(5 * w * L**4 / (384 * E * I))
    check_error("Mid-span Deflection", abs(max_def), deflection_exact)
    
    # Stress Validation
    stresses = solver.calculate_stresses(num_x_points=101, num_y_points=21, num_z_points=5)
    
    # Max Bending Stress: M_max * c / I = (w*L^2 / 8) * c / I
    sigma_exact = abs(w * L**2 / 8.0) * c / I
    check_error("Max Bending Stress", np.max(np.abs(stresses['bending'])), sigma_exact)
    
    # Max Shear Stress: 1.5 * V_max / A = 1.5 * (w*L/2) / A
    tau_exact = 1.5 * abs(w * L / 2.0) / A_rect
    check_error("Max Shear Stress", np.max(np.abs(stresses['shear'])), tau_exact)
    
    # ---------------------------------------------------------
    # Case 3: Fixed-Fixed + UDL
    # ---------------------------------------------------------
    header("Fixed-Fixed Beam + UDL (Roark Case 2d)")
    
    bc_fixed = BoundaryConditionSet("Fixed-Fixed")
    bc_fixed.fixed_support(0)
    bc_fixed.fixed_support(100)
    
    solver.solve_static(lc_udl, bc_fixed)
    max_rec = solver.get_max_deflection()
    max_def = max_rec['res']
    
    # Exact: w L^4 / (384 E I)
    deflection_exact = abs(w * L**4 / (384 * E * I))
    check_error("Mid-span Deflection", abs(max_def), deflection_exact)
    
    # Stress Validation
    stresses = solver.calculate_stresses(num_x_points=101, num_y_points=21, num_z_points=5)
    
    # Max Bending Stress: M_max * c / I = (w*L^2 / 12) * c / I
    sigma_exact = abs(w * L**2 / 12.0) * c / I
    check_error("Max Bending Stress", np.max(np.abs(stresses['bending'])), sigma_exact)
    
    # Max Shear Stress: 1.5 * V_max / A = 1.5 * (w*L/2) / A
    tau_exact = 1.5 * abs(w * L / 2.0) / A_rect
    check_error("Max Shear Stress", np.max(np.abs(stresses['shear'])), tau_exact)

def validate_modal_cases():
    # ---------------------------------------------------------
    # Case 4: Natural Frequencies
    # ---------------------------------------------------------
    header("Natural Frequencies (Timoshenko)")
    
    L = 1000.0
    E = 200000.0
    b = 50.0
    h = 100.0
    rho = 7.85e-9 # tonne/mm^3 (steel density ~7850 kg/m^3)
    
    I = b * h**3 / 12
    A = b * h
    
    material = Material("Steel", E, 0.3, rho)
    section = rectangular(b, h)
    
    # Use finer mesh for modal analysis accuracy
    mesh = Mesh.from_path([(0, 0), (L, 0)], elements_per_segment=200)
    solver = BeamSolver(mesh, material, section, element_type='euler')
    
    bc_pinned = BoundaryConditionSet("Pinned-Pinned")
    bc_pinned.pinned_support(0)
    bc_pinned.pinned_support(200)
    
    freqs, _ = solver.solve_modal(bc_pinned, num_modes=3)
    
    # Exact: f_n = (n^2 * pi / 2*L^2) * sqrt(EI / rhoA)
    # Note: Timoshenko gives omega = (n*pi/L)^2 * sqrt(EI/rhoA) for simply supported
    # f = omega / 2pi = (n^2 * pi / 2*L^2) * sqrt(EI/rhoA)
    
    term = (np.pi / (2 * L**2)) * np.sqrt(E * I / (rho * A))
    
    for n in range(1, 4):
        f_exact = (n**2) * term
        f_computed = freq_exact = freqs[n-1]
        check_error(f"Mode {n} Frequency", f_computed, f_exact, tol_percent=0.5)

if __name__ == "__main__":
    print("Running Verification Suite...")
    validate_static_cases()
    validate_modal_cases()
    print("\nVerification Complete.")
