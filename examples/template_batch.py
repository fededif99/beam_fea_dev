"""
Template: Batch Static Analysis
===============================
This template demonstrates how to perform multiple static analyses
using a parametric table (CSV) to define load magnitudes.

Workflow:
1. Define Structural Model (Mesh, Material, Section)
2. Define Template Load Case with string placeholders
3. Create/Load Parameter CSV
4. Run Batch Solution
5. Generate Summary Report and Export Results
"""

import os
import pandas as pd
from beam_fea import (
    BeamSolver, Mesh, get_material,
    LoadCase, BoundaryConditionSet, rectangular
)
from beam_fea.batch import BatchProcessor

def main():
    # 1. MODEL SETUP
    material = get_material('steel_a36')
    section = rectangular(width=50.0, height=100.0)
    mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=20)

    solver = BeamSolver(mesh, material, section)

    # Boundary Conditions (Shared across all batch cases)
    bc = BoundaryConditionSet("Cantilever")
    bc.fixed_support(node=0)

    # 2. TEMPLATE LOAD CASE
    # Use string tags (e.g., 'P_tip') as placeholders for magnitudes
    lc_template = LoadCase("Parametric Study")
    lc_template.point_load(node=20, fy="P_tip")
    lc_template.distributed_load(x_start=0, x_end=1000, distribution='uniform', wy="snow_load")

    # 3. PARAMETER TABLE
    # In a real workflow, you would load this from a CSV: df = pd.read_csv('params.csv')
    param_data = {
        'case_name': ['Case_1', 'Case_2', 'Case_3'],
        'P_tip': [-1000, -2000, -5000],
        'snow_load': [-0.5, -1.0, -2.5]
    }
    df_params = pd.DataFrame(param_data)
    csv_path = "batch_parameters.csv"
    df_params.to_csv(csv_path, index=False)

    print(f"[*] Running batch analysis with {len(df_params)} cases...")

    # 4. SOLVE BATCH
    # load_from_table substitutes the CSV values into the template
    load_cases = BatchProcessor.load_from_table(lc_template, csv_path)

    # mode='light' (default) stores only peaks; mode='full' stores all nodal results
    summary = solver.solve_batch(load_cases, bc, mode='light')

    # 5. REPORT & EXPORT
    report_path = "batch_summary_report.md"
    solver.generate_batch_report(report_path)

    export_path = "batch_results_export.csv"
    solver.export_results(export_path)

    print(f"[SUCCESS] Batch summary report saved to: {report_path}")
    print(f"[SUCCESS] Results exported to: {export_path}")

    # Cleanup temporary CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)

if __name__ == "__main__":
    main()
