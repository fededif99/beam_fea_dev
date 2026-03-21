
import pandas as pd
import numpy as np
from beam_fea import BeamSolver, Mesh, get_material, rectangular, LoadCase, BoundaryConditionSet, BatchProcessor

def run_batch_demo():
    print("Beam FEA - Batch Analysis Demo")
    print("==============================\n")

    # Setup Common Model
    mesh = Mesh.from_path([(0, 0), (2000, 0)], elements_per_segment=20)
    mat = get_material('aluminum_7075')
    sec = rectangular(50, 100)
    solver = BeamSolver(mesh, mat, sec)

    bc = BoundaryConditionSet("Simply Supported")
    bc.pinned_support(0)
    bc.roller_support(20)

    # --- Workflow 2 Demo: Parametric Table ---
    print("Running Workflow 2: Parametric Study...")

    # 1. Define Template with Placeholders
    template = LoadCase("Parametric Template")
    template.point_load(node=10, fy="P_mid")      # Center point load
    template.distributed_load(x_start=0, x_end=2000, distribution='uniform', wy="W_dist") # UDL

    # 2. Create Parameter Table (Simulating loading from CSV)
    data = {
        'case_name': ['Nominal', 'Heavy_Point', 'Heavy_UDL', 'Extreme'],
        'P_mid': [-5000, -10000, -5000, -15000],
        'W_dist': [-2.0, -2.0, -10.0, -12.0]
    }
    param_df = pd.DataFrame(data)
    param_df.to_csv("parametric_study.csv", index=False)

    # 3. Load and Solve
    load_cases = BatchProcessor.load_from_table(template, "parametric_study.csv")
    summary = solver.solve_batch(load_cases, bc)

    print("\nBatch Summary Table:")
    print(summary.to_markdown(index=False))

    # 4. Export and Report
    solver.export_results("batch_results.csv")
    solver.generate_batch_report("batch_analysis_summary.md")

    print("\nResults exported to 'batch_results.csv'")
    print("Summary report generated: 'batch_analysis_summary.md'")

if __name__ == "__main__":
    run_batch_demo()
