---
description: Standard Operating Procedures (SOP) and best practices for agents working on beam_fea.
---

# `beam_fea` Core Engineering Standard Operating Procedures

This document outlines the strict protocols and architectural best practices for future agents and human contributors working on the `beam_fea` repository. **Adherence to these rules is mandatory** to maintain parity with commercial finite element tools like Abaqus, Nastran, HyperMesh, and Ansys.

## 1. Architecture & Codebase Design

The codebase strictly follows the classic triad of commercial finite element software. You must respect these boundaries:

- **Pre-Processing (`MeshGenerator`, `LoadCase`, `BoundaryConditionSet`)**: Responsible for discretization, material assignment, property linking, and constraint generation. Avoid doing matrix mathematics here.
- **Solver (`BeamSolver`, `element_matrices.py`)**: Responsible for assembling the global stiffness matrix $[K]$, applying boundary conditions via matrix partitioning/penalty methods, and solving $\{F\} = [K]\{d\}$.
- **Post-Processing (`post_processing.py`, `report_generator.py`)**: Responsible for internal force recovery, stress calculations, and plotting.

**Design Patterns**:

- *Strategy Pattern*: When choosing between pure FEA interpolation vs. analytically consistent recovery, use the `ForceRecoveryStrategy` paradigm to keep classes modular.
- *Vectorization First*: Avoid Python `for` loops wherever geometrically possible. Construct your 3D stress fields and local coordinate mappings using `numpy.meshgrid` and array broadcasting to achieve $O(1)$ loop performance.

## 2. Unit Systems & Consistency

Like Abaqus and Nastran, `beam_fea` relies on the user to provide a consistent unit system. However, for all core development physics, assume the standard **N-mm-t-s (MPa) System**:

- Length: **mm**
- Force: **N**
- Stress/Modulus: **MPa** ($N/mm^2$)
- Density: **Tonne / $mm^3$** ($1.0 \times 10^{-9}$ for water)
- Mass: **Tonne** ($1000$ kg)

## 3. Onboarding & Task Execution

Before starting any task:

- **Always Work on Main**: Keep in sync with the `main` branch.
- **Review Theoretical Background**: Check `THEORY.md` if making changes to finite element formulations (Euler Vs Timoshenko flexural/shear matrices).
- **Run the Examples**: Inspect the `examples/` directory to see how customers are expected to interact with the API.

## 4. Verification and Validation (V&V)

Do not conflate verification (solving the equations right) with validation (solving the right equations).

- **Verification**: Ensure your finite element formulation matches analytical closed-form solutions (e.g. $v = \frac{FL^3}{3EIz}$ for a cantilever).
- **Testing Standard**: Add a new test case to `tests/` before committing new physics. Use `pytest`.
- **Warning on Matrix Singularity**: Always verify rigid body modes are constrained. Unconstrained models yield structurally singular matrices. Check constraints appropriately.

## 5. Commit & Versioning Workflow (The `@[/commit]` workflow)

All agent commits must follow this strict sequence:

1. **Changelog**: Append the fix to `CHANGELOG.md` under the appropriate semantic version tag.
2. **README API**: If a public method signature changes, update `README.md` immediately.
3. **Run Tests**: `python -m pytest tests/ -v`. Do not commit failing code.
4. **Git Operations**: Stage (`git add`), commit (`git commit -m "vX.Y.Z: ..."`), and push.

## 6. PR & Review Strategy

If generating a PR:

- State the objective (e.g. *Implementing nonlinear geometric effects*).
- Highlight the **Physics impacted**.
- Detail backward compatible fallbacks.
- Ensure all docstrings are thorough and follow NumPy standard formatting.

---
*Created by Antigravity (Google DeepMind) in adherence to Tier-1 Structural Analysis standards.*
