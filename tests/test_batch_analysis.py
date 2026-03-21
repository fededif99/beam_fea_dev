
import os
import pandas as pd
from beam_fea import BeamSolver, Mesh, get_material, rectangular, LoadCase, BoundaryConditionSet
from beam_fea.batch import BatchProcessor

def test_batch_workflow_1():
    print("\n--- Testing Workflow 1: Load List ---")
    # Setup model
    mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=10)
    mat = get_material('steel_a36')
    sec = rectangular(50, 100)
    solver = BeamSolver(mesh, mat, sec)
    bc = BoundaryConditionSet()
    bc.fixed_support(0)

    # Create CSV for Workflow 1
    csv_path = "test_batch_list.csv"
    df = pd.DataFrame([
        ['LC1', 10, 'node', 'fy', -1000, 0, 0, 0, 0],
        ['LC2', 10, 'node', 'fy', -2000, 0, 0, 0, 0],
        ['LC3', 0, 'range', 'udl', 0, 1000, -5, 0, 0]
    ], columns=['case_name', 'target_id', 'target_type', 'load_type', 'v1', 'v2', 'v3', 'v4', 'v5'])
    df.to_csv(csv_path, index=False)

    # Load and Solve
    lcs = BatchProcessor.load_from_list(csv_path)
    summary = solver.solve_batch(lcs, bc)

    print(summary.to_markdown(index=False))

    # Export
    solver.export_results("batch_results.csv")
    # Report
    solver.generate_batch_report("batch_report.md")

    os.remove(csv_path)
    print("Workflow 1 Test Passed")

def test_batch_workflow_2():
    print("\n--- Testing Workflow 2: Parametric Table ---")
    # Setup model
    mesh = Mesh.from_path([(0, 0), (1000, 0)], elements_per_segment=10)
    mat = get_material('aluminum_7075')
    sec = rectangular(40, 80)
    solver = BeamSolver(mesh, mat, sec)
    bc = BoundaryConditionSet()
    bc.pinned_support(0)
    bc.roller_support(10)

    # Create template
    template = LoadCase("Template")
    template.point_load(node=5, fy="P_mid")
    template.distributed_load(x_start=0, x_end=1000, distribution="uniform", wy="W_dist")

    # Create CSV for Workflow 2
    csv_path = "test_batch_table.csv"
    df = pd.DataFrame([
        ['Case_A', -1000, -1.0],
        ['Case_B', -2000, -2.0]
    ], columns=['case_name', 'P_mid', 'W_dist'])
    df.to_csv(csv_path, index=False)

    # Load and Solve
    lcs = BatchProcessor.load_from_table(template, csv_path)
    summary = solver.solve_batch(lcs, bc)

    print(summary.to_markdown(index=False))

    os.remove(csv_path)
    print("Workflow 2 Test Passed")

if __name__ == "__main__":
    test_batch_workflow_1()
    test_batch_workflow_2()
