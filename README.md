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

### Latest Version: v2.0.0 *(2026-03-03)*

- **Architecture**: Unified `PropertySet` architecture for flexible multi-property modeling
- **Theory**: Statically consistent force recovery and high-fidelity Timoshenko dynamics
- **Fix**: Re-applied 3D stress masking fix for heterogeneous beam sections

> [!NOTE]
> For the full version history, please see the [CHANGELOG.md](CHANGELOG.md).

---

## 📑 Table of Contents

1. [File Organization](#-file-organization)
2. [Getting Started](#-getting-started)
3. [Detailed API Reference](#-detailed-api-reference)
   - 3.1 [Materials](#31-materials-beam_feamaterials)
   - 3.2 [Cross-Sections](#32-cross-sections-beam_feacross_sections)
   - 3.3 [Mesh Generation](#33-mesh-generation-beam_feamesh)
   - 3.4 [Loads](#34-loads-beam_fealoads)
   - 3.5 [Boundary Conditions](#35-boundary-conditions-beam_feaboundary_conditions)
   - 3.6 [Solver & Analysis](#36-solver--analysis-beam_feasolver)
   - 3.7 [Visualization](#37-visualization-beam_feavisualizer--beam_feaplot_style)
   - 3.8 [Report Generation](#38-report-generation-beam_feareport_generator)
4. [Validation & Theory](#-validation--theory)

---

## 📂 File Organization

```text
beam_fea_optimized/
├── beam_fea/                    # Core Library
│   ├── modules...               # (materials, sections, mesh, loads, etc.)
├── examples/                    # Usage Examples & Case Studies
│   ├── ex01_cantilever.py       # (Cantilever Verification)
│   ├── ex02_fixed_fixed_beam.py # (UDL Analysis)
│   ├── ex03_vibration.py       # (Modal Analysis)
│   ├── template_static.py       # (Static Analysis Template)
│   └── template_modal.py        # (Modal Analysis Template)
├── tests/                       # Unit & Logic Tests
│   ├── test_coordinate_loads.py
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

### Installation

To enable a professional development workflow, it is recommended to install the package in **editable mode** from the repository root:

```bash
# Clone the repository and install in development mode
pip install -e .
```

### Quick Example: Cantilever Beam

```python
from beam_fea import BeamSolver, MeshGenerator, get_material, LoadCase, BoundaryConditionSet, rectangular

# Setup
mesh = MeshGenerator.beam_mesh_1d(length=2000, num_elements=20)
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

### 3.3 Mesh Generation (`beam_fea.mesh`)

#### `Mesh` Class

Manages nodes and elements.

- **`add_node(x, y, z=0)`**: Adds a node, returns ID.
- **`add_element(n1, n2)`**: Connects two nodes, returns ID. Material and section assignments are managed via the `PropertySet` collector.

#### `MeshGenerator` Class

Static utilities for rapid meshing.

- **`beam_mesh_1d(length, num_elements)`**: Creates a simple straight line mesh.
- **`line_mesh(start_xy, end_xy, num_elements)`**: Meshes a line between two points.
- **`multi_span_beam(lengths, elements_per_span)`**: Creates a continuous beam mesh over multiple supports.
- **`graded_mesh(start, end, n, grading_ratio)`**: Meshes with variable element sizes (Ratio > 1 increases size).
- **`curved_beam(radius, start_ang, end_ang, n)`**: Meshes a circular arc.

#### `MeshRefinement` Class

- **`refine_uniform(mesh, level)`**: Subdivides every element `level` times.

---

### 3.4 Loads (`beam_fea.loads`)

#### Understanding the Load Structure

**`LoadCase`** is a **collector** for all loads that occur simultaneously in one engineering scenario (e.g., "Cruise Lift", "Engine Thrust", "Internal Pressure").

**`LoadCombination`** groups multiple `LoadCase` objects with factors for safety margin or design conditions (e.g., `1.5×Cruise + 2.0×Gust`).

---

#### `LoadCase` Class

A container for loads applied together.

```python
lc = LoadCase("Aerodynamic Lift")
lc.point_load(x=1200, fy=5000) # Concentrated force
lc.uniform_load(x_start=0, x_end=2000, wy=1.5) # Pressure distribution
```

Supports defining loads by **Node/Element IDs** or by **Global Coordinates**.

#### `LoadCase` Methods

| Method | Positional Arguments | Coordinate Arguments |
| :--- | :--- | :--- |
| **`point_load(...)`** | `node`, `fx`, `fy`, `mz` | `x`, `fx`, `fy`, `mz` |
| **`moment(...)`** | `node`, `mz` | `x`, `mz` |
| **`uniform_load(...)`** | `element`, `wy`, `wx` | `x_start`, `x_end`, `wy`, `wx` |
| **`trapezoidal(...)`** | `element`, `wy1`, `wy2` | `x_start`, `x_end`, `wy1`, `wy2` |
| **`triangular_load(...)`** | `element`, `w_peak`, `peak_loc` | `x_start`, `x_end`, `w_peak`, `peak_loc` |

##### Example: Parametric-Safe Loading

```python
lc = LoadCase("Structural Loads")

# Dedicated Moment at x=500mm
lc.concentrated_moment(x=500, mz=100000)

# Point load at tip (even if mesh density changes)
lc.point_load(x=2000, fy=-5000)

# Distributed load over a specific range
lc.uniform_load(x_start=0, x_end=1000, wy=-2.0)
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

### 3.5 Boundary Conditions (`beam_fea.boundary_conditions`)

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

### 3.6 Solver & Analysis (`beam_fea.solver`)

#### `BeamSolver` Class

The main interface for running analyses.

**Initialization:**

```python
solver = BeamSolver(mesh, material, section, element_type='euler')
```

- `element_type`: `'euler'` (default) or `'timoshenko'` (shear deformable).

**Methods:**

- **`solve_static(load_case, bc_set)`**: Runs linear static analysis. Performs automatic model validation (mesh, materials, stability, slenderness checks) before solving.
- **`solve_modal(bc_set, num_modes)`**: Runs eigenvalue analysis (Frequencies & Mode Shapes). Performs automatic model validation before solving.
- **`calculate_internal_forces(num_points=100)`**: Returns dictionary of `axial_forces`, `shear_forces`, and `bending_moments`.
- **`calculate_stresses(num_x, num_y, num_z)`**: Generates 3D stress matrices (Axial, Bending, Shear, von Mises, Principal).
- **`visualize(analysis_type, **kwargs)`**: Orchestrates plots (see 3.7).
- **`generate_report(filepath, ...)`**: Generates professional engineering report (see 3.8).

---

### 3.7 Visualization (`beam_fea.visualizer` & `beam_fea.plot_style`)

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

### 3.8 Report Generation (`beam_fea.report_generator`)

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
