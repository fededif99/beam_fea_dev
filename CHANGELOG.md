# Changelog

All notable changes to the `beam_fea` project will be documented in this file.

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
