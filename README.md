# Beam FEA

`beam_fea` is a Python library for linear static and modal analysis of 1D beam structures. It takes a discretised geometry (nodes, elements, cross-section, material), applies loads and boundary conditions, and returns displacements, reaction forces, internal force distributions, and natural frequencies. Element formulations and numerical implementation are documented in [THEORY.md](THEORY.md).

**Capabilities:**

- **Static analysis** — solves K·u = F with a sparse direct solver; supports point forces/moments, uniform and trapezoidal distributed loads (N/mm), and load combinations with scalar factors
- **Modal analysis** — extracts natural frequencies (Hz) and mode shapes via the generalised eigenvalue problem, using the ARPACK Lanczos algorithm
- **Internal force recovery** — shear force V(x) and bending moment M(x) computed analytically from element shape functions, not post-processed
- **Element types** — Euler-Bernoulli (slender, L/h > 10) and Timoshenko (shear-deformable, L/h < 10) via a single `element_type` argument
- **Cross-sections** — rectangular, circular, hollow, I-beam, T-beam, C-channel, box, L-angle; centroid tracking and parallel-axis offset
- **Material library** — 20+ pre-defined materials (structural steels, Al/Ti alloys, composites) plus user-defined entries
- **Mesh utilities** — uniform, graded, multi-span, and curved (arc) mesh generation; uniform refinement
- **Boundary conditions** — fixed, pinned, roller, elastic spring, and prescribed (non-zero) displacement constraints
- **Reporting** — Markdown report with PNG plots (deformed shape, SFD, BMD, cross-section) saved to a `<report_name>_images/` folder

**Validation:** < 0.003% error on cantilever tip deflection, < 0.001% on fixed-fixed UDL midspan, < 0.007% on fundamental natural frequency (20-element mesh, closed-form reference).

---

## 📅 Version History

### v1.1.0 *(2026-02-21)*

- Removed `NonlinearStaticAnalysis` (Newton-Raphson) and `InfluenceLineAnalysis` (unimplemented stub) from `static_analysis.py`
- Removed `ThermalLoad` placeholder class from `loads.py`
- Removed base64 image embedding from `report_generator.py` and `visualizer.py`; reports now save PNG files to `<report_name>_images/`
- Removed `embed_images` parameter from `BeamSolver.generate_report()`
- README: refactored introduction; removed thermal load documentation

### v1.0.0 *(2026-02-08)*

- Sparse CSR assembly and `spsolve` / ARPACK backend
- Internal force recovery via element shape function derivatives
- Engineering material library (20+ entries)
- Graded, curved, and multi-span mesh generation
- Markdown report generation with saved PNG plots

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
   - 3.7 [Visualization](#37-visualization-beam_feavisualizer)
   - 3.8 [Report Generation](#38-report-generation-beam_feareport_generator)
4. [Validation & Theory](#-validation--theory)

---

## 📂 File Organization

```text
beam_fea_optimized/
├── beam_fea/                    # Core Library
│   ├── modules...               # (materials, sections, mesh, etc.)
├── examples/                    # Usage Examples & Case Studies
│   ├── ex01_cantilever.py       # (Cantilever Verification)
│   ├── ex02_fixed_beam.py       # (UDL Analysis)
│   ├── ex03_vibration.py       # (Modal Analysis)
│   └── example_walkthrough.py   # (Full Tutorial)
├── tests/                       # Unit & Logic Tests
│   ├── test_sections.py
│   └── ...
├── scripts/                     # Performance & Validation Utilities
│   ├── benchmark_performance.py
│   └── validate_accuracy.py
└── README.md                    # This Manual
```

---

## 🚀 Getting Started

### Installation

```bash
pip install numpy scipy matplotlib
```

### Environment Setup

To ensure the package is correctly recognized, it is recommended to run scripts from the project root.

```powershell
# On Windows (PowerShell)
$env:PYTHONPATH="."; python examples/ex01_cantilever.py
```

### Quick Example: Cantilever Beam

```python
from beam_fea import BeamSolver, MeshGenerator, Material, LoadCase, BoundaryConditionSet
from beam_fea.cross_sections import RectangularSection

# Setup
mesh = MeshGenerator.beam_mesh_1d(2000, 20)

# Aerospace-grade aluminum
aluminum = Material('Aluminum 7075-T6', E=71700, nu=0.33, rho=2.81e-6, yield_strength=503)
section = RectangularSection(width=50, height=100).properties()

solver = BeamSolver(mesh, aluminum, section)

# Apply boundary conditions and loads
bc = BoundaryConditionSet("Cantilever")
bc.fixed_support(0)

load = LoadCase("Tip Load")
load.point_load(node=20, fy=-10000)

# Solve
solver.solve_static(load, bc)

# Visualize & Report
solver.visualize('static')
solver.visualize('shear')
solver.generate_report("cantilever_report.md")
```

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
- **Standard Properties**: `A`, `Iy`, `Iz`, `J`, `Sy`, `Sz`

#### Utilities

- **`offset_section(props, offset_y, offset_z)`**: Repositions a section relative to the neutral axis using the **Parallel Axis Theorem**. Useful for modeling composite structures or eccentric beams.
| **`BoxSection`** | `width`, `height`, `thickness` | Rectangular tube |
| **`TBeamSection`** | `flange_width`, `flange_thickness`, `web_height`, `web_thickness` | T-profile |
| **`CChannelSection`** | `height`, `flange_width`, `web_thickness`, `flange_thickness` | C-profile |

#### Helper Functions

- `rectangular(w, h)` -> `SectionProperties`
- `circular(d)` -> `SectionProperties`

---

### 3.3 Mesh Generation (`beam_fea.mesh`)

#### `Mesh` Class

Manages nodes and elements.

- **`add_node(x, y, z=0)`**: Adds a node, returns ID.
- **`add_element(n1, n2)`**: Connects two nodes, returns ID.

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

**`LoadCase`** is a **collector** for all loads that occur simultaneously in one loading scenario (e.g., "Dead Load", "Live Load", "Wind").

**`LoadCombination`** groups multiple `LoadCase` objects with factors for design code compliance (e.g., `1.2×Dead + 1.6×Live`).

---

#### `LoadCase` Class

A container for loads applied together.

```python
lc = LoadCase("Dead Load")
lc.point_load(node=5, fy=-1000)
lc.uniform_load(element=1, wy=-10)
lc.triangular_load(element=3, w_peak=-5, peak_loc='start')
```

**Available Load Types:**

| Method | Description | Parameters |
| :-------- | :----------- | :---------- |
| **`point_load(...)`** | Concentrated force/moment at a node | `node`, `fx`, `fy`, `mz` |
| **`uniform_load(...)`** | Constant distributed load (UDL) | `element`, `wy`, `wx=0` |
| **`trapezoidal_load(...)`** | Linearly varying load | `element`, `wy1`, `wy2`, `wx1=0`, `wx2=0` |
| **`triangular_load(...)`** | Triangular distribution | `element`, `w_peak`, `peak_loc='start'` |

**Notes:**

- `wy`: Transverse load (N/mm) — causes bending
- `wx`: Axial load (N/mm) — optional, causes axial deformation
- `element` can be an `int` or `list` of element IDs

---

#### `LoadCombination` Class

Combines multiple `LoadCase` objects with load factors.

```python
combo = LoadCombination("LRFD Ultimate")
combo.load_case(dead_load, factor=1.2)
combo.load_case(live_load, factor=1.6)
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

# Advanced: Settlement modeling
bc2 = BoundaryConditionSet("With Settlement")
bc2.pinned_support(0)
bc2.prescribed_displacement(10, dy=-5)  # 5mm settlement at right support
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

- **`solve_static(load_case, bc_set)`**: Runs linear static analysis.
- **`solve_modal(bc_set, num_modes)`**: Runs eigenvalue analysis (Frequencies & Mode Shapes).
- **`calculate_internal_forces(num_points=100)`**: Calculates Shear Force (V) and Bending Moment (M) on-demand along the beam length.
- **`visualize(analysis_type, **kwargs)`**: Orchestrates plots (see 3.7).
- **`generate_report(filepath, ...)`**: Generates professional engineering report (see 3.8).

---

### 3.7 Visualization (`beam_fea.visualizer`)

Professional visualization suite for structural results.

| `analysis_type` | Method | Visualization |
| :--- | :--- | :--- |
| `'static'` | `plot_deformed_shape` | Deformed vs Undeformed beam shape |
| `'shear'` | `plot_shear_force` | Shear Force Diagram (SFD) |
| `'moment'` | `plot_bending_moment` | Bending Moment Diagram (BMD) |
| `'modal'` | `plot_mode_shape` | Eigenmode shapes for vibration analysis |

> [!TIP]
> **Automatic Scaling**: The visualizer calculates a `scale_factor` to ensure deflections are visible relative to the beam length (targeting ~5% of length).

```python
# Quick Plotting via Solver
solver.visualize('shear', num_points=200)
```

---

### 3.8 Report Generation (`beam_fea.report_generator`)

Generates comprehensive engineering reports in Markdown format, which can be easily converted to PDF or HTML.

- **Saved Graphics**: Automatically generates PNG plots (deformation, SFD, BMD, cross-section) saved to a `<report_name>_images/` folder and referenced via relative links.
- **Mathematical Precision**: Uses LaTeX math formatting for material and section properties.
- **Structural Summary**: Tabulates all nodes, elements, boundary conditions, and applied loads.
- **Result Recovery**: Lists maximum displacements, rotations, and critical internal forces.

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
