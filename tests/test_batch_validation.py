
import pytest
import os
import pandas as pd
from beam_fea import BeamSolver, Mesh, get_material, rectangular, LoadCase, BoundaryConditionSet
from beam_fea.batch import BatchProcessor

def test_batch_file_not_found():
    with pytest.raises(FileNotFoundError):
        BatchProcessor.load_from_list("non_existent.csv")

    with pytest.raises(FileNotFoundError):
        BatchProcessor.load_from_table(LoadCase(), "non_existent.csv")

def test_batch_invalid_columns():
    csv_path = "invalid_cols.csv"
    pd.DataFrame({'wrong': [1, 2]}).to_csv(csv_path, index=False)
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            BatchProcessor.load_from_list(csv_path)
    finally:
        os.remove(csv_path)

def test_batch_unresolved_placeholders():
    template = LoadCase()
    template.point_load(node=0, fy="missing_param")

    csv_path = "params.csv"
    pd.DataFrame({'case_name': ['Case1'], 'other': [10]}).to_csv(csv_path, index=False)
    try:
        with pytest.raises(ValueError, match="unresolved placeholders"):
            BatchProcessor.load_from_table(template, csv_path)
    finally:
        os.remove(csv_path)

def test_solve_batch_validation():
    mesh = Mesh.from_path([(0, 0), (100, 0)], elements_per_segment=2)
    solver = BeamSolver(mesh, get_material('steel'), rectangular(10, 10))
    bc = BoundaryConditionSet()

    with pytest.raises(ValueError, match="non-empty list"):
        solver.solve_batch([], bc)

    with pytest.raises(TypeError, match="must be LoadCase objects"):
        solver.solve_batch([1, 2, 3], bc)

if __name__ == "__main__":
    pytest.main([__file__])
