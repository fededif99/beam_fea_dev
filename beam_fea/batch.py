"""
batch.py
========
Batch processing engine for performing multiple analyses.
Supports loading load cases from CSV/TXT files and parametric studies.
"""

import pandas as pd
import numpy as np
import copy
import os
from typing import List, Union, Dict, Any
from .loads import LoadCase

class BatchProcessor:
    """
    Coordinates loading and preparation of multiple load cases.
    """

    @staticmethod
    def load_from_list(filepath: str) -> List[LoadCase]:
        """
        Workflow 1: Load a list of independent load cases from a CSV.

        CSV Structure:
        case_name, target_id, target_type, load_type, v1, v2, v3, v4, v5

        target_type: 'node', 'element', or 'range'
        load_type: 'point', 'fy', 'fx', 'mz', 'udl', 'trap', 'tri'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Batch load file not found: {filepath}")

        df = pd.read_csv(filepath)

        required_cols = ['case_name', 'target_id', 'target_type', 'load_type']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Batch CSV missing required columns: {missing}")

        load_cases = {}

        for _, row in df.iterrows():
            name = str(row['case_name'])
            if name not in load_cases:
                load_cases[name] = LoadCase(name)

            lc = load_cases[name]
            target_id = row['target_id']
            target_type = str(row['target_type']).lower()
            ltype = str(row['load_type']).lower()

            v1 = row.get('v1', 0.0)
            v2 = row.get('v2', 0.0)
            v3 = row.get('v3', 0.0)
            v4 = row.get('v4', 0.0)
            v5 = row.get('v5', 0.0)

            if ltype == 'point':
                if target_type == 'node':
                    lc.point_load(node=int(target_id), fx=float(v1), fy=float(v2), mz=float(v3))
                else:
                    lc.point_load(x=float(target_id), fx=float(v1), fy=float(v2), mz=float(v3))
            elif ltype == 'fy':
                if target_type == 'node':
                    lc.point_load(node=int(target_id), fy=float(v1))
                else:
                    lc.point_load(x=float(target_id), fy=float(v1))
            elif ltype == 'fx':
                if target_type == 'node':
                    lc.point_load(node=int(target_id), fx=float(v1))
                else:
                    lc.point_load(x=float(target_id), fx=float(v1))
            elif ltype == 'mz':
                if target_type == 'node':
                    lc.point_load(node=int(target_id), mz=float(v1))
                else:
                    lc.point_load(x=float(target_id), mz=float(v1))
            elif ltype == 'udl':
                if target_type == 'element':
                    elems = [int(e) for e in str(target_id).split(',')] if ',' in str(target_id) else int(target_id)
                    lc.uniform_load(element=elems, wy=float(v1), wx=float(v2))
                else:
                    lc.uniform_load(x_start=float(v1), x_end=float(v2), wy=float(v3), wx=float(v4))
            elif ltype == 'trap':
                if target_type == 'element':
                    lc.trapezoidal_load(element=int(target_id), wy1=float(v1), wy2=float(v2), wx1=float(v3), wx2=float(v4))
                else:
                    lc.trapezoidal_load(x_start=float(v1), x_end=float(v2), wy1=float(v3), wy2=float(v4), wx1=float(v5))
            elif ltype == 'tri':
                peak_loc = 'start' if str(v2).lower() in ['0', 'start'] else 'end'
                if target_type == 'element':
                    lc.triangular_load(element=int(target_id), w_peak=float(v1), peak_loc=peak_loc)
                else:
                    # Assuming v1=x_start, v2=x_end, v3=w_peak, v4=peak_loc
                    lc.triangular_load(x_start=float(v1), x_end=float(v2), w_peak=float(v3), peak_loc=str(v4))

        return list(load_cases.values())

    @staticmethod
    def load_from_table(template_lc: LoadCase, filepath: str) -> List[LoadCase]:
        """
        Workflow 2: Parametric Table.

        Each row in the table (CSV) represents a load case.
        Columns match the string placeholders in the template_lc.
        """
        if not isinstance(template_lc, LoadCase):
            raise TypeError(f"template_lc must be a LoadCase object, got {type(template_lc)}")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Batch parameter file not found: {filepath}")

        df = pd.read_csv(filepath)
        load_cases = []

        for _, row in df.iterrows():
            # Create a deep copy of the template
            case_name = str(row.get('case_name', f"Case_{len(load_cases)+1}"))
            lc = copy.deepcopy(template_lc)
            lc.name = case_name

            # Substitute placeholders
            for load in lc.loads:
                BatchProcessor._substitute_placeholders(load, row)

            unresolved = BatchProcessor._get_unresolved_placeholders(lc)
            if unresolved:
                raise ValueError(f"Case '{case_name}' has unresolved placeholders: {unresolved}")

            load_cases.append(lc)

        return load_cases

    @staticmethod
    def _substitute_placeholders(load: Any, parameters: pd.Series):
        """Recursively substitute string placeholders in a Load object."""
        # Check all attributes
        for attr in ['fx', 'fy', 'mz', 'wy', 'wx', 'wy1', 'wy2', 'wx1', 'wx2', 'w_peak']:
            if hasattr(load, attr):
                val = getattr(load, attr)
                if isinstance(val, str):
                    if val in parameters:
                        setattr(load, attr, float(parameters[val]))
                    else:
                        # Placeholder not found in CSV
                        pass

    @staticmethod
    def _get_unresolved_placeholders(load_case: LoadCase) -> List[str]:
        """Find any remaining string placeholders in a LoadCase."""
        unresolved = []
        for load in load_case.loads:
            for attr in ['fx', 'fy', 'mz', 'wy', 'wx', 'wy1', 'wy2', 'wx1', 'wx2', 'w_peak']:
                if hasattr(load, attr):
                    val = getattr(load, attr)
                    if isinstance(val, str):
                        unresolved.append(val)
        return unresolved
