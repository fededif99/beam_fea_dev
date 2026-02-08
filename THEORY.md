# Theoretical Manual and Reference

This document provides the theoretical foundation for the **Beam FEA** package, tailored for aerospace, mechanical, and naval engineering applications. It covers governing physics, finite element discretization, and numerical implementation details.

---

## 1. Governing Equations

The package support two distinct beam theories to handle structures ranging from slender wing spars to stout turbine shafts.

### 1.1 Euler-Bernoulli Beam Theory

Standard theory for slender structures (Length/Depth > 15). It assumes sections remain plane and strictly normal to the centroidal axis.

**Application:**

- Fuselage stringers and longerons
- High-aspect-ratio wing spars (uav/glider)
- Stabilizer components

**Implementation Details:**

- **Shape Functions**: Cubic Hermite polynomials are used for transverse displacement ($v$) and rotation ($\theta$), while linear shape functions handle axial ($u$) deformation.
- **Stiffness Matrix**: Derived using the Principle of Virtual Work. The bending-shear coupling terms ($12EI/L^3$) represent the restored forces from curvature, neglecting axial-bending interaction (linear theory).

**Governing Equation:**
$$ \frac{d^2}{dx^2} \left( EI \frac{d^2 w}{dx^2} \right) = q(x) $$

### 1.2 Timoshenko Beam Theory

Essential for short, deep components or high-frequency vibrations where shear deformation cannot be ignored.

**Application:**

- Gas turbine main shafts
- Engine mounting brackets
- Heavy machinery fixtures and frames

**Implementation Details:**

- **Shear Flexibility**: Introduces the non-dimensional parameter $\Phi = \frac{12EI}{G A_s L^2}$, where $A_s = \kappa A$.
- **Shear Correction ($\kappa$)**: Accounts for the non-uniform shear stress distribution.
  - Rectangular: $5/6$
  - Circular: $9/10$
  - I-Beam/Thin-walled: $\approx 0.5$ (shear primarily carried by web)
- **Locking Prevention**: The stiffness matrix is formulated using the exact solution to the Timoshenko differential equations, preventing "shear locking" in the slender limit ($\Phi \to 0$).

---

## 2. Finite Element Formulation

The system uses a 2D beam element with **3 Degrees of Freedom (DOFs)** per node, aligned with standard aeronautical structural analysis:

1. **$u$**: Axial (longitudinal) displacement
2. **$v$**: Transverse (vertical) displacement
3. **$\theta$**: Rotation in the $x$-$y$ plane

### 2.1 Coordinate Systems & Axis Conventions

- **Global System $(X, Y)$**: User-defined coordinate space.
- **Local System $(x, y)$**: Element-aligned system where $x$ runs from Node 1 to Node 2.
- **Cross-Section System $(y, z)$**:
  - $x$ is the longitudinal axis (out of page).
  - $y$ is the vertical axis (height).
  - $z$ is the lateral axis (width).
  - Moments of Inertia: $I_y$ (Inertia for bending about the $z$-axis, resisting $y$-deflection).

### 2.2 Local Stiffness & Mass Matrices

- **Stiffness ($k_{local}$)**: Uses exact formulation for both EB and Timoshenko elements.
- **Mass ($m_{local}$)**: Implements the **Consistent Mass Matrix**. This captures the inertia of the beam volume far more accurately than "Lumped Mass" methods, which is critical for predicting high-order vibration modes in rotating machinery.

---

## 3. Global Assembly & Constraints

### 3.1 Direct Stiffness Assembly

Global matrices are assembled using sparse triplet formats (`coo_matrix`) to minimize memory overhead when analyzing large assemblies (e.g., full fuselage mockups).

### 3.2 Boundary Conditions (Fixtures)

The package treats supports as **Kinematic Constraints**:

- **Fixed Fixtures**: All DOFs ($u, v, \theta$) set to zero or a prescribed value.
- **Pin Joints**: Displacement ($u, v$) constrained; rotation ($\theta$) remains free.
- **Elastic Foundations**: Modeled via `SpringSupport` classes adding to the diagonal of $K$.

---

## 4. Numerical Solvers

- **Statics**: LU-decomposition optimized for sparse CSR matrices (`spsolve`).
- **Dynamics**: Lanczos algorithm (`eigsh`) to extract the fundamental natural frequencies and mode shapes, avoiding the expensive computation of high-frequency noise.

---

## 5. Validation References

Verified against industry-standard benchmarks:

1. **Roark's Formulas for Stress and Strain**: Analytical deflection of aerospace-grade sections.
2. **Timoshenko, Vibration Problems in Engineering**: Exact solutions for shaft critical speeds.
