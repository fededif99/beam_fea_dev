# Theoretical Manual and Reference

This document provides the theoretical foundation for the **Beam FEA** package, tailored for aerospace, mechanical, and naval engineering applications. It covers governing physics, finite element discretization, and numerical implementation details.

---

## 1. Governing Equations

The package support two distinct beam theories to handle structures ranging from slender wing spars to stout turbine shafts.

### 1.1 Euler-Bernoulli Beam Theory

Standard theory for slender structures (Length/Depth > 15). It assumes sections remain plane and strictly normal to the centroidal axis. In the unified model, this is recovered by setting transverse shear flexibility $\phi = 0$.

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
- **Mass ($m_{local}$)**: Implements the **Consistent Mass Matrix**. This captures the inertia of the beam volume far more accurately than "Lumped Mass" methods.
  - **Euler-Bernoulli**: Standard 420-denominator matrix.
  - **Timoshenko**: High-fidelity formulation including both translational inertia and **rotational inertia** (effect of cross-section depth), adjusted by the shear flexibility parameter $\Phi$.

---

## 3. Global Assembly & Constraints

### 3.1 Direct Stiffness Assembly

Global matrices are assembled using sparse triplet formats (`coo_matrix`) for efficient memory management during large-scale structural analysis.

### 3.2 Boundary Conditions (Fixtures)

The package treats supports as **Kinematic Constraints**:

- **Fixed Fixtures**: All DOFs ($u, v, \theta$) set to zero or a prescribed value.
- **Pin Joints**: Displacement ($u, v$) constrained; rotation ($\theta$) remains free.
- **Elastic Foundations**: Modeled via `SpringSupport` classes adding to the diagonal of $K$.

---

## 4. Post-Processing & Result Recovery

### 4.1 Statically Consistent Force Recovery

For elements with distributed loads, simple interpolation of nodal displacements (homogeneous solution) is insufficient. The package implements **Internal Force Recovery** by superimposing:

1. **Homogeneous Solution**: Internal forces derived from the element's nodal displacement vector $u$ ($F_h = k_{local} \cdot u$).
2. **Particular Solution**: The contribution of intra-element distributed loads, derived via analytical integration of the load profile along the element.

This ensures that shear force $V(x)$ and bending moment $M(x)$ diagrams are accurate even with a single element per span.

### 4.2 Numerical Solvers

- **Statics**: LU-decomposition utilizing sparse CSR matrices (`spsolve`).
- **Dynamics**: Lanczos algorithm (`eigsh`) to extract the fundamental natural frequencies and mode shapes, avoiding the expensive computation of high-frequency noise.

---

## 5. Validation References

Verified against industry-standard benchmarks:

1. **Roark's Formulas for Stress and Strain**: Analytical deflection of aerospace-grade sections.
2. **Timoshenko, Vibration Problems in Engineering**: Exact solutions for shaft critical speeds.

---

## 3. Classical Laminate Theory (CLT) & Composite Beams

The package supports composite beams modelled via **Classical Laminate Theory** (CLT), following Jones (1999) and Reddy (2003).

### 3.1 Ply-Level Reduced Stiffness

For an orthotropic ply in plane stress, the reduced stiffness matrix $[Q]$ is (Jones §2.4):

$$Q_{11} = \frac{E_1}{1 - \nu_{12}\nu_{21}}, \quad Q_{22} = \frac{E_2}{1 - \nu_{12}\nu_{21}}, \quad Q_{12} = \frac{\nu_{12} E_2}{1-\nu_{12}\nu_{21}}, \quad Q_{66} = G_{12}$$

where $\nu_{21} = \nu_{12} E_2 / E_1$. For a ply at angle $\theta$ the transformed matrix $[\bar{Q}]$ is computed via standard trigonometric transformation (Jones §2.9).

### 3.2 ABD Matrix (Jones §4.2)

For a laminate with $n$ plies spanning $z \in [-t/2, t/2]$:

$$A_{ij} = \sum_k \bar{Q}_{ij}^{(k)}(z_k - z_{k-1})$$
$$B_{ij} = \frac{1}{2}\sum_k \bar{Q}_{ij}^{(k)}(z_k^2 - z_{k-1}^2)$$
$$D_{ij} = \frac{1}{3}\sum_k \bar{Q}_{ij}^{(k)}(z_k^3 - z_{k-1}^3)$$

- $[A]$: extensional stiffness (N/mm)
- $[B]$: bend-extension coupling stiffness (N) — zero for symmetric laminates
- $[D]$: bending stiffness (N·mm)

The constitutive relation is: $\{N, M\} = [ABD]\{\varepsilon^0, \kappa\}$.

### 3.3 Effective Engineering Properties (MIT OCW 16.20)

All effective moduli are derived from the **compliance matrix** $[a] = [A]^{-1}$ to correctly handle off-axis and unbalanced laminates:

$$E_x = \frac{1}{a_{11} t}, \quad E_y = \frac{1}{a_{22} t}, \quad G_{xy} = \frac{1}{a_{66} t}, \quad \nu_{xy} = -\frac{a_{12}}{a_{11}}$$

> **Note**: The simplified formula $E_x = (A_{11}A_{22} - A_{12}^2)/(A_{22}\cdot t)$ is only valid for symmetric+balanced laminates where $A_{16} = A_{26} = 0$.

The effective bending modulus uses the **full ABD inverse**:

$$E_b = \frac{12}{[ABD]^{-1}_{[3,3]} \cdot t^3}$$

This captures B-D coupling for asymmetric laminates.

### 3.4 Narrow vs. Wide Beam Assumptions

The choice of effective stiffness parameters depends on the beam's cross-sectional aspect ratio:

- **Wide Beam** (Plate-like, $\varepsilon_y = 0$): Stiffness terms $A_{11}$ and $D_{11}$ are used directly.
- **Narrow Beam** (1D Beam-like, $\sigma_y = 0$): Compliance terms $a_{11}$ and $d_{11}$ (from $[ABD]^{-1}$) are used to account for the zero-stress boundary condition on the free edges.

### 3.5 Unified Stiffness Architecture

All element matrices are derived from a unified coupled Timoshenko formulation:

$$EA = \text{Axial Stiffness (N)}, \quad ES = \text{Coupling (N·mm)}, \quad EI = \text{Bending (N·mm²)}, \quad GA_s = \text{Shear (N)}$$

Constitutive law: $\{N, M, V\} = \text{diag}([A_{coupled}], GA_s) \{\varepsilon_0, \kappa, \gamma\}$.

The element stiffness matrix $k_{local}$ is assembled from:

1. **Axial sub-matrix**: $\frac{EA}{L}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$
2. **Timoshenko bending sub-matrix**: Standard $\phi$-corrected form with $\phi = 12EI/(GA_s L^2)$.
3. **Coupling kernel**: $\frac{ES}{L}(u_2-u_1)(\theta_2-\theta_1)$ accounting for $B_{11}$ effects.

### 3.6 Transverse Shear Stress Recovery ($\tau_{xz}$)

Out-of-plane shear stress is recovered by integrating the 3D equilibrium equation layer-wise through the thickness $z$:

$$\frac{\partial \sigma_x}{\partial x} + \frac{\partial \tau_{xz}}{\partial z} = 0 \implies \tau_{xz}(z) = -\int_{-t/2}^{z} \frac{\partial \sigma_x}{\partial x} dz$$

where $\frac{\partial \sigma_x}{\partial x}$ is derived from the rate of change of mid-plane strains $\{\frac{\partial \varepsilon^0}{\partial x}, \frac{\partial \kappa}{\partial x}\}$ which are related to the axial load rate $dN/dx$ and shear force $V = dM/dx$.

### 3.7 Composite Failure Criteria

Calculated in local material coordinates $(1, 2, 12)$:

1. **Maximum Stress**: $FI = \max\left(\frac{\sigma_1}{X}, \frac{\sigma_2}{Y}, \frac{|\tau_{12}|}{S}\right)$
2. **Tsai-Hill**: $FI = \left(\frac{\sigma_1}{X}\right)^2 - \frac{\sigma_1\sigma_2}{X^2} + \left(\frac{\sigma_2}{Y}\right)^2 + \left(\frac{\tau_{12}}{S}\right)^2$
3. **Tsai-Wu**: $FI = F_1\sigma_1 + F_{11}\sigma_1^2 + F_2\sigma_2 + F_{22}\sigma_2^2 + F_{66}\tau_{12}^2 + 2F_{12}\sigma_1\sigma_2$

### 3.6 References

1. R. M. Jones, *Mechanics of Composite Materials*, 2nd ed., 1999.
2. J. N. Reddy, *Mechanics of Laminated Composite Plates and Shells*, 2003.
3. L. P. Kollár & G. S. Springer, *Mechanics of Composite Structural Elements*, 2003.
4. MIT OCW 16.20 Structural Mechanics — CLT lecture notes.
