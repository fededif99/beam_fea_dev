# Beam FEA

`beam_fea` is a Python library for linear static and modal analysis of 1D beam structures. It takes a discretised geometry (nodes, elements, cross-section, material), applies loads and boundary conditions, and returns displacements, reaction forces, internal force distributions, and natural frequencies. Element formulations and numerical implementation are documented in [THEORY.md](THEORY.md).

**Capabilities:**

- **Static analysis** — solves $K \cdot u = F$ with a sparse direct solver; supports point forces/moments, uniform and trapezoidal distributed loads (N/mm), and load combinations with scalar factors
- **Modal analysis** — extracts natural frequencies (Hz) and mode shapes via the generalised eigenvalue problem, using the ARPACK Lanczos algorithm; comprehensive reporting with natural frequency tables and mode shape plots
- **Internal force recovery** — shear force $V(x)$ and bending moment $M(x)$ recovered using statically consistent superposition of homogeneous and particular solutions, ensuring accuracy even on coarse meshes
- **Element types** — Euler-Bernoulli (slender, $L/h > 10$) and Timoshenko (shear-deformable, $L/h < 10$) via a single `element_type` argument; Timoshenko elements include full **rotational inertia** in consistent mass matrices
- **Cross-sections** — rectangular, circular, hollow, I-beam, T-beam, C-channel, box, L-angle; support for **per-element cross-section overrides** via `PropertySet`, centroid tracking, and parallel-axis offset
- **Material library** — 20+ pre-defined materials; support for **per-element material overrides** for composite or stepped beams
- **Mesh utilities** — uniform, graded, multi-span, and curved (arc) mesh generation; uniform refinement
- **Boundary conditions** — fixed, pinned, roller, elastic spring, and prescribed (non-zero) displacement constraints
- **Reporting** — Professional Markdown reports with PNG plots (deformation, SFD, BMD, Mode Shapes, Stress) saved to a `<report_name>_images/` folder

**Validation:** < 0.003% error on cantilever tip deflection, < 0.001% on fixed-fixed UDL midspan, < 0.007% on fundamental natural frequency (20-element mesh, closed-form reference).

---

## 📅 Version History

### Latest Version: v2.20.0 *(2026-08-24)*

- **Bugfix**: Resolved angled element distributed load transformation bug in `loads.py` and `post_processing.py` by applying direction cosine transformations during equivalent nodal load assembly and internal force recovery.
- **Bugfix**: Fixed dense boundary condition solver fallback in `boundary_conditions.py` to properly subtract coupling forces prior to matrix row/column zeroing.
- **Bugfix**: Corrected C-channel section centroid formula and thin-walled box section torsional constant in `cross_sections.py`.
- **Performance**: Converted `mesh.py` mesh generation to pre-allocated array assignment; removed redundant SVD condition-number check from the dense static solver; added memoization to cross-section property evaluations.
- **Architecture**: Decomposed the monolithic `report_generator.py` into modular submodules (`batch_report.py`, `diagrams.py`, streamlined `report_generator.py`); modularized `StressEngine.calculate()` into single-responsibility helpers.
- **Refactor**: Eliminated duplicated failure criteria formulations in `composites.py` by delegating to `failure_criteria.py`.
- **Docs**: Re-sequenced API Reference rendering the `Failure Criteria` section directly below `Report Generation`.

### v2.19.2 *(2026-04-14)*

- **Docs**: Comprehensive standardization on Safety Factor (SF) and Margin of Safety (MoS) across library and documentation.
- **Bugfix**: Patched `ReportGenerator` legacy element attribute access crash.
- **Feature**: Added **Angled Frame Geometry** support to the visual test suite (`config_13`), validating 2D projection in deformation results.
- **QA**: Expanded automated validation suite to 13 comprehensive end-to-end integration workflows.

- **Bugfix**: Resolved ARPACK sparse solver algorithm breakdown by extending sparse backend matrix partitioning.
- **Bugfix**: Patched `ReportGenerator` crashing while creating automated reports that contain zero external generic loads.
- **Validation**: Enormously expanded multi-factor internal QA test suite (`visual_test_configurations.py`) mapping testing domains to new configurations.

> [!NOTE]
> For the full version history, please see the [CHANGELOG.md](CHANGELOG.md).

---

## 📑 Table of Contents

1. [File Organization](#-file-organization)
2. [Getting Started](#-getting-started)
3. [Detailed API Reference](#-detailed-api-reference)
   - 3.1 [Materials](#31-materials-beam_feamaterials)
   - 3.2 [Cross-Sections](#32-cross-sections-beam_feacross_sections)
   - 3.3 [Composites & Laminates](#33-composites--laminates-beam_feacomposites)
   - 3.4 [Mesh Generation](#34-mesh-generation-beam_feamesh)
   - 3.5 [Loads](#35-loads-beam_fealoads)
   - 3.6 [Boundary Conditions](#36-boundary-conditions-beam_feaboundary_conditions)
   - 3.7 [Solver & Analysis](#37-solver--analysis-beam_feasolver)
   - 3.8 [Batch Analysis](#38-batch-analysis-beam_feabatch)
   - 3.9 [Visualization](#39-visualization-beam_feavisualizer--beam_feaplot_style)
   - 3.10 [Report Generation](#310-report-generation-beam_feareport_generator)
   - 3.11 [Failure Criteria](#311-failure-criteria-beam_feafailure_criteria)
4. [Validation & Theory](#-validation--theory)

---


## 📂 File Organization

```text
beam_fea/
├── beam_fea/                    # Core Library
│   ├── modules...               # (materials, sections, mesh, loads, etc.)
├── examples/                    # Usage Examples & Case Studies
│   ├── ex01_cantilever.py       # (Cantilever Verification)
│   ├── ex02_fixed_fixed_beam.py # (UDL Analysis)
│   ├── ex03_vibration.py       # (Modal Analysis)
│   ├── template_static.py       # (Static Analysis Template)
│   └── template_modal.py        # (Modal Analysis Template)
├── tests/                       # Unit & Logic Tests
│   ├── benchmark_modal.py       # (Modal Performance Tester)
│   ├── test_integration.py
│   └── ...
├── scripts/                     # Performance & Validation Utilities
│   ├── benchmark_performance.py
│   └── validate_accuracy.py
├── THEORY.md                    # Theoretical Foundation & Formulations
└── README.md                    # This Manual
```

---

## 🚀 Getting Started

### Requirements

`beam_fea` depends on the following core scientific libraries:

- **NumPy**: For high-performance matrix operations
- **SciPy**: For sparse solver backend and eigenvalue extraction
- **Matplotlib**: For generating analysis plots and reports
- **Pandas**: For structured result extraction and batch processing
- **Tabulate**: For formatting professional Markdown tables in reports

### Installation

To enable a professional development workflow, it is recommended to install the package in **editable mode** from the repository root:

```bash
# Clone the repository and install in development mode
pip install -e .
```

### 🧹 Environment Hygiene & Troubleshooting

If you are switching between many versions or branches and encounter versioning conflicts (e.g., `pip` reporting an old version despite being on a new branch), use the provided cleanup utility:

```bash
# Purges stale metadata and performs a fresh editable install
python scripts/clean_install.py
```

This is especially recommended for **Reviewers** to ensure their environment is perfectly synced with the current source tree.

### Quick Example: Cantilever Beam

```python
from beam_fea import BeamSolver, Mesh, get_material, LoadCase, BoundaryConditionSet, rectangular

# Setup
mesh = Mesh.from_path([(0, 0), (2000, 0)], elements_per_segment=20)
aluminum = get_material('aluminum_7075_t6')
section = rectangular(width=50, height=100)

solver = BeamSolver(mesh, aluminum, section)

# Apply boundary conditions and loads
bc = BoundaryConditionSet("Cantilever")
bc.fixed_support(0)

load = LoadCase("Tip Load")
load.point_load(node=20, fy=-10000)

# Solve
solver.solve_static(load, bc)

# Generate Report
# (Automatically generates SFD, BMD, and Deformed Shape plots for the report)
solver.generate_report("cantilever_report.md")
```

---

## 📐 Coordinate System & Conventions

The package follows a consistent global and local **Right-Handed Coordinate System (RHS)**.

### Global Coordinate System

- **X-axis**: Longitudinal axis of the beam. $x=0$ is the start of the beam (typically the left end).
- **Y-axis**: Transverse axis. Positive $y$ is upwards.
- **Z-axis**: Out-of-page axis.
- **Rotations**: Positive rotation ($\theta_z$) is Counter-Clockwise (CCW) following the right-hand rule.

```text
       ^ +Y (Transverse)
       |
       |       (Positive Rotation: CCW ↺)
       |
       O------------------> +X (Axial)
      /
     /
    L  +Z (Out-of-page)
```

### Sign Conventions

| Item | Positive Direction | Description |
| :--- | :--- | :--- |
| **Applied Loads** | $+X, +Y, +Mz$ | CCW moments are positive |
| **Deflections** | $+u, +v$ | Translation in global $x, y$ |
| **Rotations** | $+\theta$ | CCW rotation |
| **Shear Force ($V$)** | Standard Structural | Upward on the left face is positive |
| **Bending Moment ($M$)** | Standard Structural | Bottom-tension is positive ($M = -EI v''$) |

---

## 📏 Units

**This package uses a consistent unit system throughout:**

| Quantity | Unit | Symbol | Notes |
| :--------- | :----- | :------- | :----- |
| **Length** | millimeters | mm | Mesh coordinates, dimensions |
| **Force** | Newtons | N | Point loads, reactions |
| **Distributed Load** | Newtons/mm | N/mm | `wy`, `wx` in UDL |
| **Moment** | Newton-millimeters | N·mm | Applied moments, results |
| **Stress** | Megapascals | MPa | Material properties, strengths |
| **Young's Modulus** | Megapascals | MPa | `E` in Material |
| **Shear Modulus** | Megapascals | MPa | `G` in Material |
| **Density** | kg/mm³ | kg/mm³ | `rho` (typically ~1e-6 to 1e-9) |

**Example Conversions:**

```python
# Length
1 m = 1000 mm
1 inch = 25.4 mm

# Force
1 kN = 1000 N
1 lbf = 4.448 N

# Stress
1 GPa = 1000 MPa
1 ksi = 6.895 MPa

# Density (steel example)
7850 kg/m³ = 7.85e-6 kg/mm³
```

---

## 📚 Detailed API Reference

### 3.1 Materials (`beam_fea.materials`)

Defines material properties for analysis.

#### `Material` Class

```python
Material(name: str, E: float, nu: float, rho: float, yield_strength: float = None)
```

- **E**: Young's Modulus (MPa)
- **nu**: Poisson's Ratio
- **rho**: Density (kg/mm³)

#### Functions

- **`get_material(name)`**: Returns a predefined `Material` object.
  - *Options*: `'steel_a36'`, `'aluminum_7075_t6'`, `'titanium_6al_4v'`, `'carbon_fiber_p100'`, etc.
- **`list_materials()`**: Prints all available materials to console.

---

### 3.2 Cross-Sections (`beam_fea.cross_sections`)

Calculates properties like Area ($A$), Inertia ($I_y, I_z$), and Torsional Constant ($J$).

#### Section Classes

All classes calculate properties via `.properties()`.

| Class | Arguments | Description |
| :--- | :--- | :--- |
| **`RectangularSection`** | `width`, `height` | Solid rectangle |
| **`CircularSection`** | `diameter` | Solid circle |
| **`HollowCircularSection`** | `outer_diameter`, `thickness` | Pipe / Tube |
| **`IBeamSection`** | `flange_width`, `total_height`, `tw`, `tf` | I-profile |
| **`LSection`** | `leg_vert`, `leg_horiz`, `t` | Angle section |

#### Properties & Centroid Tracking

The `.properties()` method returns a `SectionProperties` object containing:

- **Centroid Tracking**: `y_centroid`, `z_centroid` (location relative to section reference)
- **Extreme Fibers**: `y_top`, `y_bottom`, `z_left`, `z_right` (distances from centroid)
- **Stress Profiles**: `get_stress_profile(y, z)` returns 2D masks, thickness $t$, and first moment $Q$
- **Standard Properties**: `A`, `Iy`, `Iz`, `J`, `Sy`, `Sz`

#### Utilities

- **`offset_section(props, offset_y, offset_z)`**: Repositions a section relative to the neutral axis using the **Parallel Axis Theorem**. Useful for modeling composite structures or eccentric beams.
| **`BoxSection`** | `width`, `height`, `thickness` | Rectangular tube |
| **`TBeamSection`** | `flange_width`, `flange_thickness`, `web_height`, `web_thickness` | T-profile |
| **`CChannelSection`** | `height`, `flange_width`, `web_thickness`, `flange_thickness` | C-profile |

#### Helper Functions

- `rectangular(w, h)` -> `SectionProperties`
- `circular(d)` -> `SectionProperties`
- `hollow_circular(d, t)` -> `SectionProperties`
- `i_beam(w, h, tw, tf)` -> `SectionProperties`
- `box(w, h, t)` -> `SectionProperties`
- `t_beam(w, tf, hw, tw)` -> `SectionProperties`
- `c_channel(h, w, tw, tf)` -> `SectionProperties`
- `l_section(lv, lh, t)` -> `SectionProperties`

---

### 3.3 Composites & Laminates (`beam_fea.composites`)

Advanced material modeling using Classical Laminate Theory (CLT) for composite structures.

#### `Ply` Class

Defines an orthotropic composite lamina including material strengths for failure analysis.

```python
from beam_fea.composites import Ply

carbon_ply = Ply(
    name="T300_Epoxy",
    E1=135000, E2=10000, nu12=0.3,
    G12=5000, G13=5000, G23=4000, # In-plane and Transverse Shear
    thickness=0.125, rho=1.6e-6,
    Xt=1500, Xc=1200, Yt=50, Yc=250, S=70, S13=50, S23=40 # Strengths (MPa)
)
```

#### `Laminate` Class

Assembles plies into a stack-up and computes the [ABD] matrix and effective engineering properties.

```python
from beam_fea.composites import Laminate

# beam_type: 'narrow' (sigma_y=0) or 'wide' (epsilon_y=0)
lam = Laminate("Wing_Spar_Flange", beam_type='narrow', stack=[
    (carbon_ply, [0, 45, -45, 90, 90, -45, 45, 0])
])

# View effective properties
props = lam.get_effective_properties()
print(props['Ex'], props['Eb'])
```

**Using Laminates in the Solver:**
To run an analysis, pass the `Laminate` directly to the `BeamSolver` as the material. The solver will automatically detect it and use the highly advanced `AnisotropicBeamElement` to capture bend-extension coupling (e.g., stretching when bent) for asymmetric layups, incorporating Timoshenko shear flexibility.

```python
solver = BeamSolver(mesh, material=lam, section=rectangular(width=25, height=lam.total_thickness))
```

---

### 3.4 Mesh Generation (`beam_fea.mesh`)

#### `Mesh` Class

Manages nodes and elements. Mesh creation is handled via high-performance static constructors:

- **`Mesh.from_path(points, elements_per_segment, grading_ratio=None)`**: Creates a general 2D mesh from a list of $(x, y)$ waypoints. Handles straight beams, piecewise paths, and multi-span layouts. **(Recommended)**
- **`Mesh.from_arc(radius, start_angle, end_angle, num_elements)`**: Meshes a circular arc for curved beam analysis.
- **`add_node(x, y)`**: Manual node addition (NumPy-backed).
- **`add_element(node1, node2)`**: Manual element connectivity.

A collector for material and cross-section properties across the beam.

```python
# Initialization (can be empty or with an initial assignment)
props = PropertySet(material=None, section=None, elements=None, name="My Properties")

# Imperative adding (Collector style)
props.add(material=steel, section=sec1)  # Applies to all (default)
props.add(material=alum, elements=[2, 3]) # Overrides specific elements
```

- **`material`**: A `Material` object.
- **`section`**: A `SectionProperties` object.
- **`elements`**: Element ID(s) this assignment applies to. Can be an `int`, `list`, `range`, or `None` (applies to all).

**Validation**: The `BeamSolver` will raise a `ValueError` during initialization if any element in the mesh is missing either a material or a section assignment.

```python
# Example: Multi-Property Beam
props = PropertySet()
props.add(material=steel, section=large_sec) # Global default
props.add(material=aluminum, elements=5)     # Local material override

solver = BeamSolver(mesh, material=props)
```

#### 3.4.1 Advanced Meshing (Waypoints & Angled Beams)

For complex structures like angled frames or multi-span beams, the `Mesh` static constructors provide automatic waypoint tracking.

**1. Point-to-Point Waypoints (`from_path`)**
Define a series of $(x, y)$ coordinates. The generator handles node merging at the "elbows" automatically.

```python
points = [(0, 0), (1000, 0), (1000, 1000)] # L-Frame
mesh = Mesh.from_path(points, elements_per_segment=[10, 10])
```

**2. Multi-Span Beams (`from_path`)**
Define multiple spans by including intermediate coordinates.

```python
points = [(0, 0), (1000, 0), (2500, 0)] # 1000mm and 1500mm spans
mesh = Mesh.from_path(points, elements_per_segment=[10, 15])
```

**3. Automatic Support Tracking**
The `mesh.waypoint_nodes` attribute stores the node IDs created at each waypoint (junction). This list is **automatically populated by all mesh generators** (including single-beam ones like `beam_mesh_1d`) and is always indexed as `[Start, Junction 1, ..., End]`.

| Index | Location | Description |
| :--- | :--- | :--- |
| `[0]` | **Start** | Start of the first beam segment. |
| `[1]` | **Junction 1** | Junction between Beam 1 and Beam 2. |
| `[i]` | **Junction i** | Junction between Beam i and Beam i+1. |
| `[-1]` | **Final End** | The absolute far end of the structure. |

```python
bc = BoundaryConditionSet("Frame Supports")

# Fixed at the start of the structure
bc.fixed_support(mesh.waypoint_nodes[0]) 

# Roller support at the first junction (between spans 1 and 2)
bc.roller_support(mesh.waypoint_nodes[1])

# Tip load at the very end of the structure
load.point_load(node=mesh.waypoint_nodes[-1], fx=5000) 
```

> [!TIP]
> **Pro Tip**: Use `mesh.waypoint_nodes[-1]` to always refer to the end of the structure, regardless of how many spans it contains!

#### `MeshRefinement` Class

- **`refine_uniform(mesh, level)`**: Subdivides every element `level` times.

---

### 3.5 Loads (`beam_fea.loads`)

#### Understanding the Load Structure

**`LoadCase`** is a **collector** for all loads that occur simultaneously in one engineering scenario (e.g., "Cruise Lift", "Engine Thrust", "Internal Pressure").

**`LoadCombination`** groups multiple `LoadCase` objects with factors for safety margin or design conditions (e.g., `1.5×Cruise + 2.0×Gust`).

---

#### `LoadCase` Class

A container for loads applied together.

```python
lc = LoadCase("Aerodynamic Lift")
lc.point_load(x=1200, fy=5000) # Concentrated force
lc.distributed_load(x_start=0, x_end=2000, distribution='uniform', wy=1.5) # Pressure distribution
```

Supports defining loads by **Node/Element IDs** or by **Global Coordinates**.

#### `LoadCase` Methods

| Method | Positional Arguments | Coordinate Arguments |
| :--- | :--- | :--- |
| **`point_load(...)`** | `node`, `fx`, `fy` | `x`, `fx`, `fy` |
| **`moment(...)`** | `node`, `mz` | `x`, `mz` |
| **`distributed_load(...)`** | `element`, `distribution`, `**kwargs` | `x_start`, `x_end`, `distribution`, `**kwargs` |
| **`lumped_mass(...)`** | `node`, `m`, `Izz`, `apply_gravity` | `x`, `m`, `Izz`, `apply_gravity` |

##### Example: Parametric-Safe Loading

```python
lc = LoadCase("Structural Loads")

# Dedicated Moment at x=500mm
lc.moment(x=500, mz=100000)

# Point load at tip (even if mesh density changes)
lc.point_load(x=2000, fy=-5000)

# Uniform Distributed Load over a specific range
lc.distributed_load(x_start=0, x_end=1000, distribution='uniform', wy=-2.0)

# Custom user-defined equation load
lc.distributed_load(x_start=0, x_end=1000, distribution='custom', load_fn=lambda x: -x**2 / 1000)
```

**Notes:**

- `wy`: Transverse load (N/mm) — causes bending
- `wx`: Axial load (N/mm) — optional, causes axial deformation
- `element` can be an `int` or `list` of element IDs

---

#### `LoadCombination` Class

Combines multiple `LoadCase` objects with design factors or safety margins.

```python
combo = LoadCombination("Ultimate Limit State (ULS)")
combo.load_case(lift_load, factor=1.5)
combo.load_case(engine_mass, factor=1.0)
```

- **`load_case(load_case, factor)`**: Adds a `LoadCase` with a scalar multiplier.

---

### 3.6 Boundary Conditions (`beam_fea.boundary_conditions`)

#### Understanding Boundary Conditions

**`BoundaryConditionSet`** is a **collector** for all supports and constraints acting on the structure.

```python
bc = BoundaryConditionSet("Simply Supported")
bc.pinned_support(0)
bc.roller_support(10)
```

---

#### `BoundaryConditionSet` Class

Manages all structural supports.

**Available Support Types:**

| Method | Constrains | Description |
| :-------- | :---------- | :------------ |
| **`fixed_support(node)`** | $u, v, \theta$ | Complete restraint (cantilever) |
| **`pinned_support(node)`** | $u, v$ | Prevents translation, allows rotation |
| **`roller_support(node, direction='y')`** | $v$ (default) | Single-direction constraint |
| **`spring_support(node, kx=0, ky=0, kr=0)`** | Elastic | Adds stiffness (N/mm or N·mm/rad) |
| **`prescribed_displacement(node, dx=0, dy=0, rotation=0)`** | Forced | Enforces non-zero displacement |

**Example:**

```python
bc = BoundaryConditionSet("Cantilever")
bc.fixed_support(0)  # Left end completely restrained

# Advanced: Prescribed displacement (e.g., thermal strain or jigging)
bc2 = BoundaryConditionSet("Prescribed Displacement")
bc2.pinned_support(0)
bc2.prescribed_displacement(10, dy=-5)  # 5mm prescribed downward deflection at node 10
```

---

### 3.7 Solver & Analysis (`beam_fea.solver`)

#### `BeamSolver` Class

The main interface for running analyses.

**Initialization:**

```python
solver = BeamSolver(mesh, material, section, element_type='euler')
```

- `element_type`: `'euler'` (default) or `'timoshenko'` (shear deformable).

**Methods:**

- **`solve_static(load_case, bc_set)`**: Runs linear static analysis. Performs automatic model validation (mesh, materials, stability, slenderness checks) before solving.
- **`solve_modal(bc_set, num_modes, load_case=None)`**: Runs eigenvalue analysis (Frequencies & Mode Shapes). If `load_case` with `LumpedMass` is provided, those masses are added to the system for the modal solve only.
- **`get_max_deflection()`**: Returns a **Pandas Series** with the node of maximum **resultant displacement** ($\sqrt{U_x^2 + U_y^2}$), including $u, v$ components and coordinates.
- **`calculate_internal_forces(num_points=100)`**: Returns dictionary of `axial_forces`, `shear_forces`, and `bending_moments`.
- **`calculate_stresses(num_x, num_y, num_z)`**: Generates 3D stress matrices (Axial, Bending, Shear, von Mises, Principal).
- **`visualize(analysis_type, **kwargs)`**: Orchestrates plots (see 3.7).
- **`generate_report(filepath, failure_criterion='tsai_wu', ...)`**: Generates professional engineering report (see 3.10).
  - *Criteria*: `'max_stress'`, `'tsai_hill'`, `'tsai_wu'`

---

### 3.8 Batch Analysis (`beam_fea.batch`)

Efficiently run multiple static load cases and generate summary reports.

#### `BatchProcessor` Class

Utilities for loading multiple load cases.

- **`load_from_list(filepath)`**: Loads unique load cases from a CSV list.
- **`load_from_table(template_lc, filepath)`**: Substitutes parameters from a CSV into a template load case.

#### `BeamSolver` Batch Methods

- **`solve_batch(load_cases, bc_set, mode='light')`**: Executes analysis on all provided cases.
  - `mode='light'`: Stores only peak results (default).
  - `mode='full'`: Stores complete results for every node in every case.
- **`generate_batch_report(filepath)`**: Generates a summary Markdown report with an **Envelope** (absolute worst-case) calculation.
- **`export_results(filepath)`**: Saves the batch summary to a CSV file.

---

### 3.9 Visualization (`beam_fea.visualizer` & `beam_fea.plot_style`)

Professional visualization suite for structural results, powered by a centralized `PlotStyle` engine.

| `analysis_type` | Method | Visualization |
| :--- | :--- | :--- |
| `'static'` | `plot_deformed_shape` | Deformed vs Undeformed beam shape |
| `'shear'` | `plot_shear_force` | Shear Force Diagram (SFD) with max-value dotted lines |
| `'moment'` | `plot_bending_moment` | Bending Moment Diagram (BMD) with max-value dotted lines |
| `'modal'` | `plot_mode_shape` | Eigenmode shapes for vibration analysis |

> [!TIP]
> **Automatic Scaling**: The visualizer calculates a `scale_factor` to ensure deflections are visible relative to the beam length (targeting ~2% of length).
> **Smart Units**: A custom `smart_units()` algorithm dynamically formats axes in optimal precision (e.g., automatically converting $15000 \text{ N}$ to $15 \text{ kN}$ or massive beam lengths to meters).

```python
# Quick Plotting via Solver
solver.visualize('shear', num_points=200)

# Overriding Global Plot Styles:
from beam_fea.plot_style import DEFAULT_STYLE
DEFAULT_STYLE.colour_primary = '#FF0000' # Change beam lines to red
```

---

### 3.10 Report Generation (`beam_fea.report_generator`)

Generates comprehensive engineering reports in Markdown format, which can be easily converted to PDF or HTML.

- **Saved Graphics**: Automatically generates PNG plots (deformation, SFD, BMD, Stress Distribution, cross-section).
- **Mathematical Precision**: Uses LaTeX math formatting for material and section properties.
- **Structural Summary**: Tabulates all nodes, elements, boundary conditions, and applied loads.
- **Result Recovery**: Lists maximum displacements, rotations, critical internal forces, and **Factor of Safety**.

```python
# Generate full report
solver.generate_report("Structural_Analysis_Report.md")
```

---

## 3.11 Failure Criteria (`beam_fea.failure_criteria`)

Unified failure analysis for both **isotropic metals** and **composite plies**. All criteria are fully vectorized — accept scalars or NumPy arrays and return an identical-shaped result dict.

### Output Dictionary

| Key | Type | Description |
| :--- | :--- | :--- |
| `'stress'` | ndarray | **Governing stress** (or dimensionless ratio/index) |
| `'SF'` | ndarray | **Safety Factor**: allowable / stress. SF ≥ 1 = safe |
| `'MoS'` | ndarray | **Margin of Safety**: SF − 1. MoS ≥ 0 = safe |
| `'passed'` | ndarray(bool) | `True` wherever SF ≥ 1 |

### Metal (Isotropic) Criteria

| Class | Formula |
| :--- | :--- |
| `VonMisesCriterion(yield_strength)` | SF = σ_y / σ_vm |
| `TrescaCriterion(yield_strength)` | SF = σ_y / (σ₁ − σ₂) |
| `MaxPrincipalStressCriterion(Ftu, Fcu)` | SF = 1 / max(σ₁/Ftu, −σ₂/Fcu) |

### Composite (Orthotropic Ply) Criteria — stresses in material axes

| Class | Formula |
| :--- | :--- |
| `MaximumStressCriterion(Xt,Xc,Yt,Yc,S,S13*,S23*)` | SF = 1 / max(..., \|τ₁₂\|/S, \|τ₁₃\|/S₁₃, \|τ₂₃\|/S₂₃) |
| `TsaiHillCriterion(Xt,Xc,Yt,Yc,S,S13*,S23*)` | SF = 1 / sqrt((σ₁/X)² − ... + (τ₁₂/S)² + (τ₁₃/S₁₃)² + (τ₂₃/S₂₃)²) |
| `TsaiWuCriterion(Xt,Xc,Yt,Yc,S,F12*,S13*,S23*)` | Full interactive tensor polynomial (results in SF) |
| `MaximumStrainCriterion(eps_Xt,eps_Xc,eps_Yt,eps_Yc,gamma_S)` | SF in strain space |

```python
from beam_fea.failure_criteria import TsaiWuCriterion, VonMisesCriterion
import numpy as np

# --- Composite ply check (vectorised over many stations) ---
crit = TsaiWuCriterion(Xt=1500, Xc=1200, Yt=50, Yc=250, S=70)
result = crit.evaluate(
    sigma_1=np.array([200.0, 800.0, 1400.0]),
    sigma_2=np.array([10.0,  20.0,  40.0]),
    tau_12=np.array([5.0,   15.0,  30.0]),
)
print(result['SF'])     # [large,   2.8..., 1.03...]
print(result['MoS'])    # [large,   1.8..., 0.03...]
print(result['passed']) # [True, True, True]

# --- Metal check ---
crit_vm = VonMisesCriterion(yield_strength=250.0)
result_vm = crit_vm.evaluate(sigma_x=180.0, sigma_y=50.0, tau_xy=30.0)
```

---

## 🔬 Validation & Theory

This codebase has been rigorously validated against classical analytical solutions.

### Validation Results Summary

| Case | Analytical (mm/Hz) | FEA Result | Error (%) |
| :--- | :--- | :--- | :--- |
| **Cantilever Tip Deflection** | 1.5875 mm | 1.5875 mm | **0.0025%** |
| **Fixed-Fixed UDL** | 0.0521 mm | 0.0521 mm | **0.0000%** |
| **Simply Supported Frequency** | 49.33 Hz | 49.33 Hz | **0.0064%** |

> See the full [Validation Report](validation_report.md) for detailed benchmarks.

### Theoretical Background

- **Euler-Bernoulli**: Standard beam theory, ignores shear. Good for $L/h > 15$.
- **Timoshenko**: Includes shear deformation and rotational inertia. Essential for stout beams.

See the **[Theoretical Manual](THEORY.md)** for matrix formulations.
