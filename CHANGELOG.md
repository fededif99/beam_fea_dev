# Changelog

All notable changes to the `beam_fea` project will be documented in this file.

## [v2.19.0] - 2026-04-11

- **Feature**: Added native support for `LumpedMass` (via `mass` or `lumped_mass` identifiers) inside standard string-based structural batch configuration pipelines.
- **Optimization**: Modernized loop iterations in `BatchProcessor` reducing memory overhead via `itertuples` and NamedTuple dictionary conversion, significantly improving CSV parsing performance over massive parametric sweeps.
- **Refactor**: Replaced hardcoded 14-item property placeholder substitution in `load_from_table` with generic dynamic Python reflection (`vars(load)`), universally enabling `LoadCase` parametrization for any textual, numeric, or boolean property.

## [v2.18.0] - 2026-04-07

- **API**: Standardized entire codebase on **Safety Factor (SF)** and **Margin of Safety (MoS)**; removed all remaining references to "Failure Index" (FI) for consistency.
- **Feature**: Added `BeamSolver.verify_equilibrium()` and centralized analytical residual calculation in `ResultsEngine`, supporting all load types.
- **Physics**: Upgraded `StressEngine` to perform true **through-thickness interpolation** (linear bending, parabolic shear) for 3D stress field extraction.
- **Architecture**: Removed legacy `self.material` and `self.section` pointers from `BeamSolver`, enforcing `PropertySet` API usage.
- **Robustness**: Standardized robust section width extraction (`z_right - z_left`) and added area-consistency warnings for non-rectangular composite sections.
- **Report**: Enhanced `BeamReportGenerator` with centralized equilibrium verification and fixed SF/MoS reporting data mismatch.
- **Validation**: Updated `_validate_model` to perform per-element slenderness ($L/h$) and stability checks.

## [v2.17.0] - 2026-04-06

- **Refactor**: Re-architected `BeamSolver` into specialized engines: `AssemblyEngine`, `AnalysisEngine`, and `ResultsEngine`.
- **API**: Standardized `PropertySet.resolve()` with pre-calculation and caching for up to 100% speedup in assembly.
- **Feature**: Integrated `ResultsEngine` for unified peak extraction and `failure_criteria.py` FOS evaluation.
- **Performance**: Fully vectorized global matrix assembly using cached property arrays.
- **Report**: Enhanced `BatchReportGenerator` with automated ply-by-ply peak stress and FOS identification.

## [v2.16.0] - 2026-04-05

- **Refactor**: Optimized `ModalAnalysis` core with vectorized mass-normalization and participation calculations, achieving up to 33x speedup.
- **API**: Updated `BeamSolver.solve_modal` and `ModalAnalysis.solve` signatures to support `num_modes=None`, standardizing a default of 10 modes.
- **Performance**: Integrated `subset_by_index` in dense eigensolvers for significantly faster lower-mode extraction in large dense models.
- **Test**: Added `tests/benchmark_modal.py` for automated performance regression tracking of modal analysis engines.

## [v2.15.0] - 2026-04-04

- **Feature**: Added `LumpedMass` load type in `loads.py` for nodal mass ($m$) and rotational inertia ($I_{zz}$), supporting both static and modal analysis.
- **Physics**: Implemented automatic gravity loading for lumped masses ($F_y = -m \cdot g$) for seamless static/modal weight integration.
- **Solver**: Enhanced `solve_modal` in `solver.py` to support "effective mass" matrices via dynamic `load_case` injection.
- **Robustness**: Refactored `ModalAnalysis` core to use a high-stability partitioned dense solver for small systems ($<1000$ DOFs).
- **Performance**: Optimized sparse matrix augmentation for lumped masses using `LIL` format transformation in `loads.py`.
- **Report**: Updated `report_generator.py` to recursively detect and format `LumpedMass` data in PDF/Markdown engineering reports.

## [v2.14.0] - 2026-04-04

- **Feature**: Upgraded `FailureCriterion` methods to evaluate true pseudo-3D material stresses, resolving 2D plane-stress analytical blindspots for transverse shear ($V_y$) components.
- **API**: Expanded composite criteria (`TsaiWu`, `TsaiHill`, `MaximumStress`) to natively evaluate Interlaminar shear parameters (`tau_13`, `tau_23`) and thresholds (`S13`, `S23`).
- **Feature**: Added `_principal_stresses_3d` in `failure_criteria.py` to calculate exact analytical invariants for isotropic models (Von Mises, Tresca) via `3x3` matrix eigenvalue extraction.
- **Physics**: Updated `StressEngine` to extract internal transverse beam shear `tau_xz` and systematically rotate it into local material coordinates.
- **API**: Added `S13` and `S23` interlaminar limits to the `Ply` dataclass, supporting backward compatibility via auto-default mapping to `S`.

## [v2.13.0] - 2026-03-31

- **Feature**: New `beam_fea/failure_criteria.py` module with 7 fully vectorized failure criteria (VonMises, Tresca, MaxPrincipal, MaximumStress, TsaiHill, TsaiWu, MaximumStrain) covering metals and composites.
- **Refactor**: Updated `StressAnalysis` in `static_analysis.py` with `Union[float, np.ndarray]` type hints and a comprehensive bending stress sign convention docstring.
- **Test**: Added `tests/test_failure_criteria.py` with 20 parametric tests validating mathematical correctness and array broadcasting.
- **Example**: `ex05_composite_beam.py` now demonstrates Tsai-Wu and Max Stress failure criteria evaluation after FEA.
- **Docs**: Added `§3.11 Failure Criteria` to `README.md` with API tables and usage examples.

## [v2.12.0] - 2026-03-22

- **Refactoring (Breaking API Change)**: Made `Laminate` immutable; replaced `add_ply`/`add_stack` with a constructor-based `stack` argument.
- **Optimization**: Vectorized CLT ABD matrix and transverse shear calculations using NumPy, improving performance for thick laminates.
- **CI/CD**: Upgraded GitHub Actions to test across Python 3.10-3.12 and OS matrix (Ubuntu/Windows) with pip caching.
- **API**: Added `Laminate.from_single_material` factory method for ergonomic single-material stack-up definitions.

## [v2.11.0] - 2026-03-22

- **Refactoring (Breaking API Change)**: Completely removed deprecated legacy load methods (`uniform_load`, `trapezoidal_load`, `triangular_load`, `concentrated_moment` and the `mz` argument from `point_load`). You must now use `distributed_load()` and `moment()`.
- **Refactoring**: Completely removed the deprecated legacy aliases `UniformDistributedLoad`, `TrapezoidalDistributedLoad`, and `TriangularDistributedLoad`.
- **Bug Fix**: Fixed local coordinate (`xi`) calculation for `PointLoad` and `ConcentratedMoment` on inclined beams, ensuring accurate load projection.
- **Validation**: `point_load` and `moment` coordinates are now rigorously validated; an explicit warning is raised if loads fall outside the element mesh range.
- **Maintenance**: Migrated all template tests, visual configurations, and batch analysis parsers to the new unified Loads API.

## [v2.10.0] - 2026-03-21

- **API**: Introduced dedicated `ConcentratedMoment` class, decoupled from `PointLoad`. Moments are now a standalone load type.
- **API**: Unified all distributed load classes (`UniformDistributedLoad`, `TrapezoidalDistributedLoad`, `TriangularDistributedLoad`) into a single `DistributedLoad` class with waypoint-based profiles.
- **Feature**: New `distribution='custom'` mode — users can define arbitrary load functions evaluated via Gauss-Legendre quadrature.
- **Feature**: Isosceles triangular distributions via `peak_loc=float` (e.g., `peak_loc=500.0` for peak at x=500mm).
- **Performance**: Vectorized element overlap detection and shape function integration in the distributed load engine.
- **API**: New `LoadCase.distributed_load()` and `LoadCase.moment()` methods as the preferred API.
- **Compat**: Legacy methods (`uniform_load`, `trapezoidal_load`, `triangular_load`, `concentrated_moment`) retained as deprecated shims.

## [v2.9.1] - 2026-03-15

- **Performance**: Implemented a high-performance **vectorized assembly architecture** in `BeamSolver`.
- **Performance**: Achieved **~20x speedup** in global matrix assembly for large models (e.g., 10,000 elements reduced from 0.63s to 0.03s).
- **Architecture**: Replaced the element-wise Python loop with broadcasted NumPy operations and vectorized coordinate transformations (`matmul`, `repeat`, `tile`).
- **Optimization**: Added `PropertySet.is_uniform()` for fast-path property retrieval in homogeneous models.

## [v2.9.0] - 2026-03-15

- **Architecture**: Completely refactored the internal mesh representation to use raw NumPy arrays (`mesh.nodes` and `mesh.elements`), eliminating the `Node` and `Element` dataclasses.
- **API**: **Removed the `MeshGenerator` class entirely.** All mesh generation logic has been consolidated into static methods: `Mesh.from_path()` and `Mesh.from_arc()`.
- **Performance**: Significant performance gains in mesh creation and assembly via vectorized NumPy operations and reduced object instantiation overhead.
- **Refactor**: Updated all examples, templates, tests, and utility scripts to utilize the new consolidated `Mesh` API.

## [v2.8.0] - 2026-03-15

- **API**: Removed redundant global `z` coordinate from `Node`, `Mesh`, and `MeshGenerator` classes, standardizing the engine purely on 2D `(x, y)` global coordinates.
- **Mesh**: `Node.coordinates()` and `Mesh.get_node_coords()` now return $2$-dimensional arrays instead of $3$-dimensional arrays.
- **Mesh**: Removed `z` parameter from `Mesh.add_node()` and path-generation methods.
- **Post-Processing**: Updated numpy broadcasting dimensions internally to gracefully handle the shift to $(N, 2)$ nodal arrays without breaking existing $(N, 3)$ fallback loops.

## [v2.7.0] - 2026-03-15

- **Physics**: Replaced placeholder Timoshenko mass matrix in `UnifiedBeamElement` with the exact Friedman & Kosmatka (1993) consistent mass matrix, including rotatory inertia terms scaled by $r^2 = EI/(EA \cdot L^2)$.
- **Architecture**: Removed duplicate `UnifiedBeamElement` class definition in `element_matrices.py` — now defined exactly once.
- **Numerics**: Replaced fixed `DEFAULT_PENALTY = 1e15` in `boundary_conditions.py` with a relative penalty scaled to `max(diag(K)) * 1e10`, preventing ill-conditioning across different unit systems.
- **Composites**: Corrected `Laminate.get_sectional_stiffness` for narrow beams — coupling stiffness `ES` now uses `ABD_inv[0,3]` (compliance-based), not `B[0,0]`, correctly accounting for the free transverse-edge ($N_y=0$) constraint.
- **Tests**: Updated `test_bend_extension_coupling` threshold to reflect the physically correct (smaller) compliance-based coupling magnitude.

## [v2.6.0] - 2026-03-09

- **API**: Modified `BeamSolver.get_max_deflection()` to return a `pandas.Series` with resultant magnitude, components, and coordinates.
- **Reporting**: Fixed equilibrium check in `BeamReportGenerator` to correctly handle 2D coordinates and axial force moment contributions.
- **Environment**: Added `scripts/clean_install.py` for automated cleanup of stale metadata and fresh re-installation.
- **Documentation**: Enhanced `README.md` with advanced waypoint meshing guides and environment hygiene instructions.

## [v2.5.0] - 2026-03-10

- **Batch Analysis**: Introduced `solve_batch()` in `BeamSolver` for performing multiple analyses efficiently.
- **Batch Analysis**: Added `BatchProcessor` supporting "Load List" (Workflow 1) and "Parametric Table" (Workflow 2) from CSV files.
- **Reporting**: Added `BatchReportGenerator` for summary reports across multiple load cases, including absolute maximum "Envelope" calculations.
- **Reporting**: Overhauled `BeamReportGenerator` to use **Pandas** for all table formatting, significantly improving report layout and maintainability.
- **API**: Enhanced `Load` classes to support string placeholders for parametric studies.
- **API**: Added `export_results()` to `BeamSolver` for exporting batch analysis summaries to CSV.
- **Examples**: Added `examples/ex08_batch_analysis.py` and `examples/template_batch.py` to demonstrate the new batch capabilities.
- **Dependencies**: Added `pandas` and `tabulate` as required dependencies for enhanced data handling and markdown reporting.

## [v2.4.11] - 2026-03-07

- **Analysis**: Implemented high-fidelity **Ply-by-Ply Stress Recovery** at multiple depths (Top, Mid, Bottom) per ply.
- **Analysis**: Added transformation of stresses to **Local Material Coordinates** ($\sigma_1, \sigma_2, \tau_{12}$) for all composite plies.
- **Analysis**: Implemented multiple **Failure Criteria** for composites: Maximum Stress, Tsai-Hill, and Tsai-Wu.
- **Analysis**: Added axial and bending decomposition for longitudinal stress ($\sigma_x$).
- **Reporting**: Overhauled the ply stress summary table in Markdown reports to include local stresses and selectable failure indices.
- **API**: Enhanced `BeamSolver.generate_report` to allow user selection of the failure criterion.
- **Refactor**: Unified the element stiffness and mass matrix architecture under `UnifiedBeamElement`.
- **Refactor**: Standardized the constitutive interface across `Material` and `Laminate` via `get_sectional_stiffness`.
- **Refactor**: Unified through-thickness stress recovery logic in `StressEngine` by treating isotropic materials as single-ply laminates.
- **Cleanup**: Eliminated ad-hoc special casing for composite laminates in the global assembly and post-processing loops.
- **Docs**: Comprehensive update to `README.md` and `THEORY.md` documenting the unified architecture, advanced CLT recovery, and failure criteria.
- **Examples**: Updated all example scripts and templates (Static, Modal, Multi-Property) to reflect the latest high-fidelity composite and unified architecture capabilities.
- **Mesh**: Introduced `MeshGenerator.path_mesh` for creating general 2D angled beam meshes from a list of waypoints.
- **Mesh**: Updated `multi_span_beam` to use the unified path kernel and support relative 2D coordinate vectors `(dx, dy)`.
- **Mesh**: Added `waypoint_nodes` to the `Mesh` class to track support/segment boundaries automatically.
- **Solver**: Implemented general 2D element rotation support, enabling the analysis of angled frames and cranked beams.
- **Post-Processing**: Updated internal force and stress recovery to handle angled elements via global-to-local coordinate transformations.
- **Visualization**: Enhanced `BeamVisualizer` and `BeamReportGenerator` to correctly render angled beams, maintaining aspect ratios and using path-length axes for force diagrams.
- **Modal**: Implemented automatic mass-normalization of mode shapes.
- **Modal**: Added calculation of effective modal masses and participation factors for both X and Y directions.
- **Modal**: Enhanced reports with detailed "Modal Properties & Mass Participation" tables including cumulative mass.

## [v2.4.10] - 2026-03-07

- **Visualization**: Tightened the layout of `plot_laminate_stackup` by moving the Technical Data summary box closer to the orientation rosette.

## [v2.4.9] - 2026-03-07

- **Visualization**: Refined `plot_laminate_stackup` by removing redundant degree labels from rosette arrows (now solely identified by the legend).
- **Visualization**: Fixed cropping of the "Laminate Technical Data" box by anchoring it to the orientation sub-plot coordinate system.

## [v2.4.8] - 2026-03-07

- **Visualization**: Major overhaul of `plot_laminate_stackup` for academic and research quality.
- **Visualization**: Implemented a professional "Muted Engineering" color palette inspired by standard FEM software and scientific publications.
- **Visualization**: Fixed rosette text clashing by implementing intelligent radial label offsets and unobtrusive axis markers.
- **Visualization**: Standardized the orientation legend with fixed-size square markers to prevent layout overflow.
- **Visualization**: Redesigned the laminate summary as a clean "Technical Data" block with a minimalist aesthetic.

## [v2.4.7] - 2026-03-07

- **Visualization**: Performed an aesthetic overhaul of `plot_laminate_stackup` featuring a vivid, premium color palette for orientations.
- **Visualization**: Fixed rosette geometry using `FancyArrowPatch` for unified, perfectly aligned orientation vectors with consistent line thickness.
- **Visualization**: Polished the report layout with rounded property dashboards and high-contrast labels.

## [v2.4.6] - 2026-03-07

- **Visualization**: Implemented **Angle-based coloring** in `plot_laminate_stackup`; all plies with the same fiber orientation now share the same color for better pattern recognition.
- **Visualization**: Simplified orientation rosette to show exactly one line and one arrow per unique angle, eliminating visual clutter.
- **Visualization**: Polished layout with Ivory-colored properties box and centered orientation legend.

## [v2.4.5] - 2026-03-07

- **Visualization**: Refined `plot_laminate_stackup` to color by **Material Name** (e.g., all CFRP plies share the same color) for better structural insight in sandwich panels.
- **Visualization**: Optimized plot layout to prevent overlap between the material legend and the laminate properties summary box.
- **Visualization**: Enhanced rosettes with material-segmented stems and clear angle labels.

## [v2.4.4] - 2026-03-07

- **Visualization**: Redesigned `plot_laminate_stackup` for professional consistency; colors are now mapped to unique (Material, Angle) pairs.
- **Visualization**: Simplified rosette plot to show a single arrow per orientation with segmented colored stems for multi-material directions.
- **Reporting**: Added an explicit Material-Angle legend to composite reports for better interpretability.

## [v2.4.3] - 2026-03-07

- **Visualization**: Fixed rosette stacking in `plot_laminate_stackup`; orientation lines now use radius-offset segments to ensure all ply colors are visible when angles overlap.
- **Examples**: Polished the composite sandwich example to ensure integer-type consistency for stress evaluation grids.

## [v2.4.2] - 2026-03-07

- **Examples**: Updated `examples/ex05_composite_beam.py` to demonstrate composite sandwich panels (Aluminum core + CFRP skins).
- **Examples**: Integrated multi-section `PropertySet` assignments into the composite beam example.
- **Reporting**: Verified high-fidelity ply-by-ply stress tables and stack-up rosettes in generated reports.

## [v2.4.1] - 2026-03-07

- **Docs**: Consolidated standard operating procedures into formal `.agents/workflows/agent_sop.md` conforming to commercial FEA architecture standards
- **Maintenance**: Cleaned up the root directory by removing stray patch and markdown report files generated by previous core code updates
- **Maintenance**: Updated `.gitignore` to include Big Tech standard exclusions (`.vscode`, `.idea`, `*.patch`, `htmlcov`, etc.)
- **Maintenance**: Merged `.agent` and `.agents` directories to establish a single source of truth for repository workflows

## [v2.4.0] - 2026-03-05

- **Visualization**: Overhauled cross-section plotting to draw explicit geometry using Matplotlib patches instead of bounding boxes for all standard sections.
- **Visualization**: Implemented `plot_laminate_stackup` featuring Abaqus-inspired vertical ply stacks and HyperMesh-inspired polar orientation rosettes.
- **Visualization**: Added support for subplots in `plot_multiple_sections` to visualize models with varying cross-sections.
- **Analysis**: Implemented high-fidelity **Ply-by-Ply Stress Recovery** for composite laminates, calculating $\sigma_x, \sigma_y, \tau_{xy}$ for every layer.
- **API**: Enhanced `BoundaryConditionSet` and `LoadCase` with `.add()` methods for better object-oriented model building.
- **Reporting**: Upgraded `BeamReportGenerator` to include enhanced cross-section geometry, laminate stack-up diagrams, and ply-by-ply maximum stress tables.
- **Refactor**: Updated `SectionProperties` and `CrossSection` classes to maintain persistent naming and parent-child relationships for improved visualization.

## [v2.3.0] - 2026-03-05

- **Architecture**: Implemented the **Strategy Pattern** for internal force handling, separating pure FEA approximation (`NodalInterpolationStrategy`) from high-fidelity engineering results (`ConsistentRecoveryStrategy`).
- **Optimization**: Vectorized the internal force and stress calculation engines in the new `post_processing.py` module, achieving $O(M)$ performance scaling (where $M$ is the number of evaluation points).
- **Refactor**: Renamed `interpolate_internal_forces` to `recover_forces_consistent` and added `interpolate_forces_homogeneous` to element classes for technical clarity.
- **Reporting**: Guaranteed high-fidelity diagrams in `BeamReportGenerator` by explicitly utilizing the consistent recovery strategy.
- **Performance**: Optimized element evaluation loops by grouping points by element, significantly reducing redundant object instantiation.
- **Cleanup**: Removed dead and commented-out code in `visualizer.py` and synchronized caching logic across the solver.

## [v2.2.2] - 2026-03-05

- **Reporting**: Enhanced `BeamReportGenerator` to dynamically display longitudinal ($E_x$), transverse ($E_y$), bending ($E_b$), and shear ($G_{xy}$) moduli for composite laminates, providing better visibility into anisotropic behavior.

## [v2.2.1] - 2026-03-05

- **Physics**: Corrected effective axial modulus ($E_x$) extraction to use full compliance matrix inversion, ensuring accuracy for unbalanced and off-axis laminates
- **Theory**: Upgraded `AnisotropicBeamElement` to Timoshenko formulation, incorporating shear deformation via the $\phi$ flexibility parameter
- **Theory**: Implemented distributed load particular solutions in `AnisotropicBeamElement.interpolate_internal_forces` for statically consistent shear and moment recovery under UDLs
- **Docs**: Expanded `THEORY.md` with a comprehensive section on Classical Laminate Theory and composite beam mathematical formulations

## [v2.2.0] - 2026-03-05

- **Feature**: Full Anisotropic (Composite) Beam Support
- **Element**: Introduced `AnisotropicBeamElement` with coupled axial-bending constitutive matrices
- **Coupling**: Captures bend-extension coupling effects ($B$ matrix) in asymmetric laminates
- **API**: Updated `BeamSolver` to automatically utilize anisotropic elements for `Laminate` materials
- **Verification**: New example `examples/ex06_anisotropic_coupling.py` demonstrating coupling effects

## [v2.1.3] - 2026-03-05

- **Fix**: Resolved `flake8` CI failure by correctly handling `Material` type hints in `composites.py`
- **Feature**: Exported `Ply` and `Laminate` at the package level for improved API accessibility

## [v2.1.2] - 2026-03-05

- **Feature**: Integrated Classical Laminate Theory (CLT) via new `composites.py` module
- **Feature**: Support for ABD matrix calculation and effective property extraction (Ex, Eb) for composite beams
- **Example**: Added `examples/ex05_composite_beam.py` demonstrating composite spar flange analysis
- **Fix**: Resolved crash in `report_generator.py` when material properties are missing

## [v2.1.1] - 2026-03-03

- **Audit**: Completed comprehensive codebase audit and status update (see `STATUS_UPDATE.md`)
- **Feature**: Integrated Classical Laminate Theory (CLT) via new `composites.py` module, supporting ABD matrix calculation and effective property extraction
- **Fix**: Resolved `NameError: name 'List' is not defined` in `solver.py`
- **Fix**: Removed redundant stress calculation logic in `solver.py` to improve performance
- **Verification**: Completed `test_declarative_properties.py` by adding missing assertion for validation failure

## [v2.1.0] - 2026-03-03

- **API**: Refactored `PropertySet` into a unified **Collector** model (similar to `LoadCase`), supporting both single-assignment and multi-assignment workflows via `.add()`
- **Validation**: Introduced mandatory element-coverage validation; `BeamSolver` now raises a detailed `ValueError` if any element is missing a material or section assignment
- **UX**: Unified assignment and resolution logic into a single class, removing the internal `PropertyResolver` from the public API
- **Verification**: Expanded test suite to cover declarative property assignments, legacy compatibility, and partial overrides (101 passes)

## [v2.0.0] - 2026-03-03

- **Architecture**: Introduced `PropertySet` collector for decoupled management of materials and cross-sections, enabling advanced multi-property modeling
- **Feature**: Added `examples/template_multi_property.py` demonstrating element-level property overrides
- **Theory**: Upgraded internal force recovery to use **statically consistent superposition** (homogeneous + particular solutions), ensuring perfect accuracy for distributed loads
- **Theory**: implemented full **consistent mass matrix for Timoshenko beam elements**, including rotational inertia and shear deformation effects
- **Optimization**: Introduced **dual-path stress engine** with full NumPy vectorization for uniform beams and cached resolution for heterogeneous models
- **Fix**: Resolved 3D stress field masking bug that caused over-prediction in beams with varying depths (re-applied fix to v2.0.0 architecture)
- **Docs**: Global audit and refinement of all docstrings, comments, and manuals (README.md, THEORY.md) for professional clarity

## [v1.9.0] - 2026-03-01

- **Feature**: Refactored `BeamReportGenerator` to support dynamic, combined reporting of both static and modal analysis results in a single document
- **Feature**: Added `examples/ex04_combined_analysis.py` to demonstrate the new multi-analysis reporting capabilities
- **UX**: Refined mode shape plots to include frequency in the title and hide the undeformed shape for better clarity
- **UX**: Improved UDL span display in reports to explicitly state "Full Span" for global loads
- **Fix**: Resolved CI crash caused by missing `Agg` headless Matplotlib backend when running on GitHub Actions
- **Fix**: Fixed `F821 undefined name 'Union'` in `solver.py` and addressed build isolation dependency issues in the CI workflow

## [v1.8.0] - 2026-03-01

- **Feature**: Added comprehensive modal analysis reporting to `BeamReportGenerator`, including natural frequency tables and mode shape visualizations
- **Feature**: `BeamSolver` now persists modal results (`frequencies`, `mode_shapes`) and last-used boundary conditions to enable high-quality documented reporting
- **UX**: All examples and templates now use `get_material()` for standardized and physically consistent material retrieval from the database
- **UX**: Enabled automated report generation in `template_modal.py` to match the static analysis experience
- **Refactor**: Cleaned up verbose terminal output across all example scripts, transitioning to a report-centric user experience
- **Fix**: Resolved `plot_structure_diagram` crash when running modal analysis without a defined load case
- **Fix**: Updated `BeamVisualizer.plot_mode_shape` to support file output for automated reporting
- **Fix**: Corrected mode shape vector extraction logic (column-wise eigenvectors) to prevent displacement shape mismatch errors
- **Fix**: Resolved CI dependency conflict by pre-installing core scientific libraries before package installation

## [v1.7.5] - 2026-03-01

- **UX**: Changed default `deformation_scale` in `BeamSolver.generate_report` to `'auto'`, ensuring report plots are always legible
- **UX**: Refined `examples/template_static.py` mesh density (reduced to 50 elements) for better responsiveness and usability
- **Fix**: Resolved inconsistency where interactive `visualize()` used auto-scaling but generated reports defaulted to 1.0x

## [v1.7.4] - 2026-03-01

- **Refactor**: Completed global removal of redundant `sys.path` hacks and unused imports across `examples/`, `tests/`, and `scripts/`
- **Cleanup**: Standardized imports in `tests/test_coordinate_loads.py` by removing unused class imports
- **Maintenance**: Verified core `beam_fea/` package for import redundancy to ensure a lean development installation

## [v1.7.3] - 2026-03-01

- **Feature**: Added `_validate_model` pre-analysis check to `BeamSolver` that detects missing mesh, properties, or unstable boundary conditions before computation
- **Feature**: Implemented dynamic version synchronization between `CHANGELOG.md` and the package `__version__`
- **Fix**: Isolated versioning logic to `_version.py` to prevent CI build failures due to missing dependencies
- **UX**: Implemented automated slenderness ratio ($L/h$) check that suggests switching to Timoshenko elements for stout beams (< 10) to improve accuracy
- **UX**: Added pro-active instability warnings when axial or transverse constraints are missing, preventing singular matrix errors
- **Docs**: Completed a comprehensive audit of all comments and docstrings for clarity and technical accuracy

## [v1.7.2] - 2026-03-01

- **Optimization**: `solver.assemble_global_matrices()` was split; the mass matrix is now assembled lazily only when `solve_modal()` is called, halving element loop operations for pure static analyses
- **UX**: Visualizer defaults for SFD/BMD now scale dynamically with mesh density (`max(50, 4 * num_elements)`) rather than using a hardcoded 100 points, ensuring curves are smooth but not over-sampled

## [v1.7.1] - 2026-03-01

- **UX**: Annotation collision avoidance in structure and FBD diagrams now uses independent stacking counters (`stack_fy`, `stack_fx`, `stack_mz`) per x-coordinate, preventing labels of different force types from displacing each other
- **UX**: Vertical label offset is now anchored to global `arrow_scale` instead of per-force `length`, giving consistent spacing regardless of load magnitude differences
- **UX**: Bold `weight='bold'` added to `'Beam Deformation'` plot title in `visualizer.py` for visual consistency with other diagram titles

## [v1.7.0] - 2026-02-28

- **Feature**: Replaced hardcoded plot aesthetics with centralized `PlotStyle` dataclass in new `plot_style.py` module
- **Feature**: Intelligent axis tick formatting and `smart_units` scaling algorithm for plot dimensioning
- **UX**: Removed yellow annotation boxes in internal force diagrams, replacing them with clean dotted lines and red axis labels
- **UX**: Overhauled structure diagram drawing logic to map fixed, pinned, and roller supports to authentic mechanical engineering glyphs
- **Bugfix**: Resolved broken cross-section geometry inputs (`c_channel`, `t_beam`, `l_section`) in visual tests
- **Tests**: Added an 8-case visual rendering stress test module under `scripts/visual_test_configurations.py`

## [v1.6.2] - 2026-02-28

- **Docs**: Added explicit Requirements section to `README.md`
- **Docs**: Enhanced Installation instructions with context on editable mode

## [v1.6.1] - 2026-02-28

- **Docs**: Moved full version history to `CHANGELOG.md`
- **Docs**: Simplified `README.md` to show only the latest version
- **Build**: Updated `commit` workflow to support the new documentation structure

## [v1.6.0] - 2026-02-28

- **Build**: Added `pyproject.toml` to support standard `pip` installation and build backends
- **UX**: Installed package in editable mode (`pip install -e .`) to enable clean imports without `sys.path` hacks
- **Refactor**: Removed all manual `sys.path` manipulations from `examples/`, `scripts/`, and `tests/`

## [v1.5.0] - 2026-02-23

- **Optimization**: Linked default discretization resolution in `calculate_internal_forces` and `calculate_stresses` to `mesh.num_nodes`
- **UX**: Simplified example templates by removing hardcoded integration points, making nodal results the default experience

## [v1.4.10] - 2026-02-23

- **Bugfix**: Fixed compatibility regression in `solve_static` return value
- **Bugfix**: Resolved `NameError` in `calculate_stresses` caching logic

## [v1.4.9] - 2026-02-23

- **Refactor**: Standardized library logging by removing all `print()` side-effects from core solver methods
- **UX**: Removed the "bandage" `silent` parameter, as library methods are now silent by default
- **Cleanup**: Fixed duplicate docstrings and optimized `calculate_stresses` internal logic

## [v1.4.8] - 2026-02-23

- **Feature**: Added `get_max_internal_forces` helper to `BeamSolver` for easy extraction of peak shear and moments
- **UX**: Updated terminal output in example templates to include maximum bending moments and their locations

## [v1.4.7] - 2026-02-23

- **Optimization**: Centralized principal stress calculation ($σ_1$, $σ_2$) into `BeamSolver.calculate_stresses` using vectorized logic
- **UX**: Simplified example templates by providing pre-calculated principal stresses in results dictionary

## [v1.4.5] - 2026-02-23

- **Optimization**: Implemented result caching for internal forces and stresses to prevent redundant re-computations
- **UX**: Added `silent` mode to calculation methods to suppress console output during automated report generation

## [v1.4.3] - 2026-02-23

- **Examples**: Added simplified, modular templates for static and modal analysis (`template_static.py`, `template_modal.py`)
- **Documentation**: Added a dedicated `README.md` to the `examples/` directory for better onboarding

## [v1.4.2] - 2026-02-22

- **Minor Polish**: Updated the default date format in `BeamReportGenerator` to `DD-MM-YYYY`

## [v1.4.1] - 2026-02-22

- **Critical Fix**: Resolved a malformed ternary bug in `BeamSolver.calculate_internal_forces` spatial coherence check
- **Performance**: Eliminated redundant `sqrt` recomputations by using direct vector math for element length extraction
- **Optimization**: Streamlined memory and broadcasting overhead by precomputing the 3D stress field inverse mask

## [v1.4.0] - 2026-02-22

- **Performance**: Vectorized global matrix assembly using `np.meshgrid` (eliminates 36-iteration inner loop per element)
- **Broadcasting**: Fully vectorized `BeamSolver.calculate_stresses` lengthwise loop (nx points) using 3D NumPy broadcasting
- **Optimization**: Implemented property caching for internal force evaluations, reducing redundant element expert instantiation
- **Efficiency**: Optimized `BeamReportGenerator` by moving constant load-scaling logic out of plotting loops
- **Cleanup**: Standardized internal buffer caching in `mesh.py` for node coordinates and element start indices

## [v1.3.1] - 2026-02-22

- **Critical Fixes**: Resolved `NameError` in `CChannelSection` and `h**h` power math bug in `RectangularSection`
- **Performance**: Optimized element lookup from $O(n)$ to $O(\log n)$ using binary search in `solver.py` and `mesh.py`
- **Standardization**: Replaced magic numbers with constants (`SHEAR_CORRECTION_FACTOR`, `DEFAULT_PENALTY`)
- **API**: Expanded package-level exports to include `LSection` and all convenience helpers
- **Verification**: Passed 88/88 tests including new targeted regression suite

## [v1.3.0] - 2026-02-22

- Integrated 3D exact stress analysis field extraction ($VQ/It$ model)
- Added `BeamSolver.calculate_stresses()` for comprehensive 3D stress matrix Generation
- Upgraded `BeamReportGenerator` with stress distribution plots and Factor of Safety reporting
- Added `get_stress_profile()` to all `CrossSection` subclasses for geometric stress profiling
- Updated `BeamSolver.calculate_internal_forces()` to return axial force ($N$)
- Validated stresses against Roark's analytical formulas (< 1% error)
- Removed unused `GeometricStiffness` and `DampingMatrix` classes for 1.0.0 standardization

## [v1.2.0] - 2026-02-21

- Added coordinate-based load definitions (define loads via global $x$ positions or node/element IDs)
- Implemented `LoadCase.concentrated_moment()` and `LoadCase.triangular_load()` as dedicated API methods
- Added a dedicated `TriangularDistributedLoad` class for improved API clarity
- Standardized coordinate system documentation with ASCII diagrams
- Verified consistency across all physics and geometry modules

## [v1.1.1] - 2026-02-21

- Optimized `BeamSolver.calculate_internal_forces` (object caching, robust element lookup)
- Fixed sign convention bug in `TimoshenkoElement` force recovery
- Removed orphaned `StaticAnalysis.calculate_element_forces` method

## [v1.1.0] - 2026-02-21

- Removed `NonlinearStaticAnalysis` (Newton-Raphson) and `InfluenceLineAnalysis` (unimplemented stub) from `static_analysis.py`
- Removed `ThermalLoad` placeholder class from `loads.py`
- Removed base64 image embedding from `report_generator.py` and `visualizer.py`; reports now save PNG files to `<report_name>_images/`
- Removed `embed_images` parameter from `BeamSolver.generate_report()`
- README: refactored introduction; removed thermal load documentation

## [v1.0.0] - 2026-02-08

- Sparse CSR assembly and `spsolve` / ARPACK backend
- Internal force recovery via element shape function derivatives
- Engineering material library (20+ entries)
- Graded, curved, and multi-span mesh generation
- Markdown report generation with saved PNG plots
