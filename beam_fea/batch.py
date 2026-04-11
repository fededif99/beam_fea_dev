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

        for row in df.itertuples(index=False):
            # Fast dictionary access from namedtuple
            row_dict = row._asdict()
            name = str(row_dict['case_name'])
            if name not in load_cases:
                load_cases[name] = LoadCase(name)

            lc = load_cases[name]
            target_id = row_dict['target_id']
            target_type = str(row_dict['target_type']).lower()
            ltype = str(row_dict['load_type']).lower()

            v1 = row_dict.get('v1', 0.0)
            v2 = row_dict.get('v2', 0.0)
            v3 = row_dict.get('v3', 0.0)
            v4 = row_dict.get('v4', 0.0)
            v5 = row_dict.get('v5', 0.0)

            # Support target_type == 'range' explicitly
            def _parse_elements(tid):
                if ',' in str(tid):
                    return [int(e) for e in str(tid).split(',')]
                return int(tid)

            if ltype == 'point':
                if target_type == 'node':
                    lc.point_load(node=int(target_id), fx=float(v1), fy=float(v2))
                    if float(v3) != 0.0:
                        lc.moment(node=int(target_id), mz=float(v3))
                else:
                    lc.point_load(x=float(target_id), fx=float(v1), fy=float(v2))
                    if float(v3) != 0.0:
                        lc.moment(x=float(target_id), mz=float(v3))
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
                    lc.moment(node=int(target_id), mz=float(v1))
                else:
                    lc.moment(x=float(target_id), mz=float(v1))
            elif ltype in ['mass', 'lumped_mass']:
                m = float(v1)
                Izz = float(v2)
                # v3 parameter acts as apply_gravity flag (0/1 or False/True)
                apply_gravity = str(v3).lower() in ['true', '1', '1.0', 'yes']
                if target_type == 'node':
                    lc.lumped_mass(node=int(target_id), m=m, Izz=Izz, apply_gravity=apply_gravity)
                else:
                    lc.lumped_mass(x=float(target_id), m=m, Izz=Izz, apply_gravity=apply_gravity)
            elif ltype == 'udl':
                if target_type == 'element':
                    lc.distributed_load(element=_parse_elements(target_id), distribution='uniform', wy=float(v1), wx=float(v2))
                else:  # assumes 'range' or equivalent
                    lc.distributed_load(x_start=float(v1), x_end=float(v2), distribution='uniform', wy=float(v3), wx=float(v4))
            elif ltype == 'trap':
                if target_type == 'element':
                    lc.distributed_load(element=int(target_id), distribution='linear', wy_start=float(v1), wy_end=float(v2), wx_start=float(v3), wx_end=float(v4))
                else:
                    lc.distributed_load(x_start=float(v1), x_end=float(v2), distribution='linear', wy_start=float(v3), wy_end=float(v4), wx_start=float(v5))
            elif ltype == 'tri':
                peak_loc = 'start' if str(v2).lower() in ['0', 'start'] else 'end'
                if target_type == 'element':
                    lc.distributed_load(element=int(target_id), distribution='triangular', w_peak=float(v1), peak_loc=peak_loc)
                else:
                    # Assuming v1=x_start, v2=x_end, v3=w_peak, v4=peak_loc
                    lc.distributed_load(x_start=float(v1), x_end=float(v2), distribution='triangular', w_peak=float(v3), peak_loc=str(v4))

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

        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            # Create a deep copy of the template
            case_name = str(row_dict.get('case_name', f"Case_{len(load_cases)+1}"))
            lc = copy.deepcopy(template_lc)
            lc.name = case_name

            # Substitute placeholders
            for load in lc.loads:
                BatchProcessor._substitute_placeholders(load, row_dict)

            unresolved = BatchProcessor._get_unresolved_placeholders(lc)
            if unresolved:
                raise ValueError(f"Case '{case_name}' has unresolved placeholders: {unresolved}")

            load_cases.append(lc)

        return load_cases

    @staticmethod
    def _substitute_placeholders(load: Any, parameters: Dict[str, Any]):
        """Recursively substitute string placeholders in a Load object dynamically."""
        for attr, val in vars(load).items():
            if isinstance(val, str) and val in parameters:
                new_val = parameters[val]
                
                # Check for specific boolean parameters
                if attr in ['apply_gravity']:
                    setattr(load, attr, str(new_val).lower() in ['true', '1', '1.0', 'yes'])
                # Check for string parameters
                elif attr in ['peak_loc', 'distribution']:
                    setattr(load, attr, str(new_val))
                else:
                    # Float fallback for standard magnitude properties
                    try:
                        setattr(load, attr, float(new_val))
                    except ValueError:
                        setattr(load, attr, new_val)

    @staticmethod
    def _get_unresolved_placeholders(load_case: LoadCase) -> List[str]:
        """Find any remaining string placeholders dynamically in a LoadCase."""
        unresolved = []
        for load in load_case.loads:
            for attr, val in vars(load).items():
                # Avoid counting inherently string attributes as unresolved placeholders
                if attr in ['peak_loc', 'distribution']:
                    continue
                if isinstance(val, str):
                    unresolved.append(val)
        return unresolved
