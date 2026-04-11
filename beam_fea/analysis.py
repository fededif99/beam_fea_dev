"""
analysis.py
===========
Unified analysis engine for static, modal, and batch execution.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .loads import LoadCase, LoadCombination

class AnalysisType(Enum):
    STATIC = auto()
    MODAL = auto()
    BATCH = auto()

class AnalysisEngine:
    """
    Orchestrates the interaction between assembly, solvers, and results extraction.
    Standardizes the execution flow for all analysis types.
    """

    def __init__(self, solver):
        self.solver = solver

    def run(self, analysis_type: AnalysisType, **kwargs) -> Union[np.ndarray, Tuple, Dict]:
        """
        Entry point for all analysis execution.
        """
        if analysis_type == AnalysisType.STATIC:
            return self._run_static(**kwargs)
        elif analysis_type == AnalysisType.MODAL:
            return self._run_modal(**kwargs)
        elif analysis_type == AnalysisType.BATCH:
            return self._run_batch(**kwargs)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    def _run_static(self, load_case, bc_set) -> np.ndarray:
        """Execute a single static analysis."""
        from .post_processing import ResultsEngine
        
        # 1. Assembly (Lazy)
        if self.solver.K_global is None:
            self.solver._assemble_stiffness_matrix()

        # 2. Vector generation
        F = load_case.create_force_vector(self.solver.mesh.num_dofs, self.solver.mesh)

        # 3. Apply BCs & Solve
        K_bc, F_bc = bc_set.apply_to_system(self.solver.K_global, F)
        displacements = self.solver.static_solver.solve(K_bc, F_bc)
        reactions = self.solver.static_solver.calculate_reactions(self.solver.K_global, F)

        # 4. Store State
        self.solver.displacements = displacements
        self.solver.reactions = reactions
        self.solver.last_load_case = load_case
        self.solver.last_bc_set = bc_set
        
        # Clear result caches
        self.solver._cached_forces = None
        self.solver._cached_stresses = None

        return displacements

    def _run_modal(self, bc_set, num_modes: Optional[int] = None, 
                   load_case: Optional[Union[LoadCase, LoadCombination]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Execute modal analysis."""
        # 1. Ensure Matrices
        if self.solver.K_global is None:
            self.solver._assemble_stiffness_matrix()
        if self.solver.M_global is None:
            self.solver._assemble_mass_matrix()

        # 2. Handle Lumped Mass
        M_eff = self.solver.M_global
        if load_case is not None:
            if not load_case.has_mass_loads():
                raise ValueError("load_case provided to solve_modal does not contain any LumpedMass loads.")
            M_eff = load_case.apply_to_mass_matrix(self.solver.M_global.copy(), self.solver.mesh)

        # 3. Solve
        frequencies, mode_shapes = self.solver.modal_solver.solve(
            self.solver.K_global, M_eff, num_modes, bc_set
        )

        # 4. Store State
        self.solver.last_bc_set = bc_set
        self.solver.last_frequencies = frequencies
        self.solver.last_mode_shapes = mode_shapes
        self.solver.last_modal_participation = self.solver.modal_solver.get_modal_participation_summary(M_eff)

        return frequencies, mode_shapes

    def _run_batch(self, load_cases: List, bc_set, mode: str = 'light',
                  failure_criterion: str = 'tsai_wu') -> pd.DataFrame:
        """Execute batch static analysis across multiple load cases."""
        from .post_processing import ResultsEngine
        import copy
        
        peak_results = []
        full_results = []

        for lc in load_cases:
            # Run Static
            self._run_static(load_case=lc, bc_set=bc_set)
            
            # Extract Peaks
            peaks = ResultsEngine.get_peak_summary(self.solver, failure_criterion=failure_criterion)
            peak_results.append(peaks)
            
            if mode == 'full':
                full_results.append({
                    'name': lc.name,
                    'displacements': self.solver.displacements.copy(),
                    'reactions': self.solver.reactions.copy(),
                    'forces': copy.deepcopy(self.solver.calculate_internal_forces()),
                    'stresses': copy.deepcopy(self.solver.calculate_stresses())
                })

        # Generate Summary Table
        summary_df = ResultsEngine.create_batch_summary(self.solver, load_cases, peak_results)
        
        # Store in solver for access
        self.solver.batch_summary = summary_df
        if mode == 'full':
            self.solver.batch_results = full_results
            
        return summary_df
