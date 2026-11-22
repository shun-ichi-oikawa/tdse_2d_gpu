# tdse_2d_gpu

A 2D Time-Dependent Schrödinger Equation (TDSE) GPU solver for charged-particle cyclotron motion under arbitrary electromagnetic perturbations.

This repository provides the CUDA/C/Fortran source code accompanying the research paper:
> **S. Oikawa**, *"Generalization of Quantum-Mechanical Equation Structures for Cyclotron Motion under Arbitrary Perturbations"* (submitted).

---

## Overview

`tdse_2d_gpu` numerically integrates the two-dimensional time-dependent Schrödinger equation for a charged particle (e.g., a proton) in a uniform magnetic field $\mathbf{B} = (0, 0, B_z)$ subject to arbitrary electrostatic or electromagnetic perturbations. The solver is designed to investigate long-term wave-packet dynamics, expectation-value trajectories, variance evolution, and quantum-classical correspondence over multiple cyclotron periods.

### Key Numerical & Physical Features
* **Spatial Discretization & Domain**: Finite-difference method (FDM) on a 2D Cartesian grid $[-L_x, L_x] \times [-L_y, L_y]$ (default $L_x = L_y = 16$, grid spacing $h_x = h_y = 0.02$, corresponding to $1601 \times 1601$ grid points).
* **Time Integration**: Implicit Crank–Nicolson-type scheme solved iteratively via Successive Over-Relaxation (SOR) on NVIDIA GPUs (double precision `FP64`).
* **Boundary Conditions**: Inhomogeneous fixed Dirichlet boundary conditions. At $\tau = 0$, the interior domain is initialized with a standard Gaussian wave packet, while the boundary values are determined via a natural boundary condition ($\partial^2 \psi / \partial n^2 = 0$, implemented as linear extrapolation $\psi_0 = 2\psi_1 - \psi_2$ from the two adjacent interior grid points) and held fixed for $\tau > 0$.
* **Gauge & Perturbation Flexibility**:
  * Vector potential formulated in the Landau gauge $\mathbf{A} = (-B_z y, 0, 0)$ (with optional magnetic field gradient parameter `LB_INV`).
  * Supports weak uniform electric fields (default), polynomial/sinusoidal electric fields, and arbitrary spatial distributions of point/Gaussian field charges evaluated via Double Exponential numerical integration (`DEHINT.for`).
* **Checkpointing & Restart**: Periodically saves the full complex wave function (`*.bin`) and initial reference energy (`EN0.bin`) in binary format for seamless checkpoint/restart or interruption handling.

---

## Repository Structure

```text
tdse_2d_gpu/          <-- Root of the GitHub repository (corresponds to the working directory `sch`)
├── Makefile
├── sch.cu
├── sch.inp
├── EXPECT.cu
├── FDM.cu
├── PHI_SOR.cu
├── potential.cu
├── write.cu
├── constant.cu
├── func.cu
├── timer.c
├── restart.c
├── DEHINT.for
├── cudacomplex.h
├── LICENSE
├── README.md
├── OUT/              <-- Created automatically at runtime (or tracked via `.gitkeep`)
├── out/
├── SAVE/
├── save/
└── PLT/
    ├── multi3.plt
    ├── multi3_ufm.plt
    ├── multi3_rev.plt
    ├── toEPS_multi3.plt
    ├── bb8.plt
    ├── bb8_initial.plt
    ├── bb8_j.plt
    └── potential.plt
```

---

## Requirements

* **GPU**: NVIDIA GPU with Compute Capability `sm_61` or higher (tested on RTX 3070 `sm_86`; preconfigured for `sm_61`, `sm_70` [Volta/TITAN V], `sm_75`, and `sm_86`).
* **CUDA Toolkit**: CUDA 11.x or compatible (`nvcc`, `cudart`, `cublas`).
* **Compilers**: `gcc` / `g++` and `gfortran`.
* **Visualization (Optional)**: `gnuplot` (for scripts in `PLT/`).

---

## Building the Executables

The unified `Makefile` compiles three distinct binaries in a single command without directory conflicts:

```bash
make all
```

| Binary | Compiler Macro | Physics Configuration | Output File | Density Dir | Checkpoint Dir |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sch`** | *(none)* | General electrostatic/magnetic perturbation (default: weak uniform $E_y$) | `000.dat` | `OUT/` | `SAVE/` |
| **`sch_fp`** | `-DFieldParticle` | Uniform magnetic field + arbitrary Gaussian field charges (`DEHINT.for`) | `000.dat` | `OUT/` | `SAVE/` |
| **`sch_ufm`** | `-D_UFM_` | Unperturbed uniform magnetic field reference baseline | `000.ufm` | `out/` | `save/` |

To clean compiled binaries and object files:
```bash
make clean
```

---

## Running Simulations

1. **Create the output directories** before running the solver:
   ```bash
   mkdir -p OUT SAVE out save
   ```
2. **Execute the desired binary** from the repository root:
   ```bash
   ./sch        # General perturbation run
   ./sch_ufm    # Unperturbed uniform magnetic field reference run
   ./sch_fp     # Field-particle perturbation run
   ```
   *(Note: Both `sch` and `sch_fp` write to `000.dat`, `OUT/`, and `SAVE/`, whereas `sch_ufm` writes to `000.ufm`, `out/`, and `save/`. Therefore, `sch_ufm` can be executed concurrently with either `sch` or `sch_fp` without file collisions, while `sch` and `sch_fp` should not be run simultaneously in the same directory.)*

3. **Graceful Interruption**:
   To stop a running job cleanly after the next density output step (`N_rec_rho`), create an empty file named `interruption` inside `SAVE/` (for `sch` and `sch_fp`) or `save/` (for `sch_ufm`):
   ```bash
   touch SAVE/interruption   # For sch / sch_fp
   touch save/interruption   # For sch_ufm
   ```

---

## Configuration & Reproducing Paper Results

### 1. Input Parameters (`sch.inp`)
In `sch.inp`, variables ending with an underscore (`_`) denote reference physical quantities in SI units (initialized to CODATA values), while variables without an underscore are dimensionless:

* `N_rec = 100`: Time-step interval for writing expectation values (`000.dat` / `000.ufm`).
* `N_rec_rho = 300`: Time-step interval for writing 2D probability densities (`OUT/*.rho` / `out/*.rho`).
* `N_step = 100000`: Number of time steps per $5$ cyclotron periods ($\Delta \tau = 5 \times 2\pi / N_{\text{step}}$).
* `hx = 2e-02`, `hy = 2e-02`: Spatial grid spacing.
* `Lx = 16`, `Ly = 16`: Half-width of the spatial domain ($-16 \le x, y \le 16$).
* `omg = 1.01`: SOR relaxation factor.
* `v0x = 0`, `v0y = 1`, `x0 = -1`, `y0 = 0`: Initial classical guiding-center/velocity parameters.
* `Bz = 1`, `m = 1`, `q = 1`: Dimensionless magnetic field, particle mass, and charge.
* `v0_ = 10` [m/s], `Bz_ = 10` [T], `mp_ = 1.67262192e-27` [kg], `e_ = 1.60217663e-19` [C], `h_bar_ = 1.05457182e-34` [J·s]: Reference SI scales.

### 2. Reproducing Specific Cases from the Paper
* **Dimensionless Planck Constant $\hbar^{\ast} \approx 0.362$:** Set `h_bar_ = 0.6324e-34` (with `v0_ = 10`, `Bz_ = 10`, `mp_ = 1.67262192e-27`, `e_ = 1.60217663e-19`) in `sch.inp`.
* **Quadratic Electric Field Perturbation ($E_y \propto y^2$)**:
  In `potential.cu` (lines 63–64), replace the linear potential energy with the cubic potential energy:
  ```c
  REAL y2 = y * y;
  return - q * Ey * y * y2 / 3 * LE4_INV;
  ```
* **50.25 Cyclotron Periods Long-Term Runs**:
  In `sch.cu` (around line 138), change the cycle multiplier `(5 / 5)` to `(50 / 5)`:
  ```c
  double N_t_s = N_step * m / (q * Bz) * 105 / 100 * (50 / 5);
  ```

---

## Output Data Format (`000.dat` / `000.ufm`)

Time-series expectation values and variances are recorded in **19 columns** (with standard output printed every 20 steps):

| Col | Header | Description | Col | Header | Description |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | `time` | Dimensionless time $\tau = k \Delta \tau$ | **11** | `var_mv` | Total kinetic momentum variance |
| **2** | `<x>` | Expectation value $\langle x \rangle$ | **12** | `E` | Total energy expectation value $\langle H \rangle$ |
| **3** | `<y>` | Expectation value $\langle y \rangle$ | **13** | `E-E0` | Energy drift $\langle H \rangle - E_0$ |
| **4** | `var_x` | Position variance $\sigma_x^2$ | **14** | `<Px>` | Canonical momentum $\langle P_x \rangle$ |
| **5** | `var_y` | Position variance $\sigma_y^2$ | **15** | `<Py>` | Canonical momentum $\langle P_y \rangle$ |
| **6** | `var_r` | Total spatial variance $\sigma_x^2 + \sigma_y^2$ | **16** | `var_Px` | Canonical momentum variance $\sigma_{P_x}^2$ |
| **7** | `<mv_x>` | Kinetic momentum $\langle m v_x \rangle$ | **17** | `var_Py` | Canonical momentum variance $\sigma_{P_y}^2$ |
| **8** | `<mv_y>` | Kinetic momentum $\langle m v_y \rangle$ | **18** | `var_P` | Total canonical momentum variance |
| **9** | `var_mvx` | Kinetic momentum variance $\sigma_{m v_x}^2$ | **19** | `err` | Normalization error $\iint \lvert \psi \rvert^2 \thinspace dx \thinspace dy - 1$ | | | |
| **10** | `var_mvy` | Kinetic momentum variance $\sigma_{m v_y}^2$ | | | |

---

## Visualization (`PLT/`)

All scripts inside `PLT/` are designed to be executed from within the `PLT/` directory (`cd PLT && gnuplot <script>.plt`), referencing `../000.dat`, `../000.ufm`, and `../sch.inp`:

* `multi3.plt` / `multi3_ufm.plt`: $2 \times 2$ multi-panel real-time monitoring plots for `000.dat` and `000.ufm`.
* `multi3_rev.plt`: $2 \times 3$ multi-panel dimensionless diagnostic plot.
* `toEPS_multi3.plt`: Publication-quality EPS figure generator.
* `bb8.plt`, `bb8_initial.plt`, `bb8_j.plt`: Wave-packet probability density and current visualizations.
* `potential.plt`: Generates animated GIF (`real8.gif`) of wave-packet motion over the potential landscape.

---

## Related Publications

This work builds upon a series of studies on quantum-classical correspondence and wave-packet dynamics in magnetic fields:

1. **S. Oikawa**, *"Generalization of Quantum-Mechanical Equation Structures for Cyclotron Motion under Arbitrary Perturbations"* (submitted to Journal of Mathematical Physics, Manuscript ID: JMP26-AR-03467).
2. **S. Oikawa**, T. Imai, and E. Ueda, *"Quantum-Mechanical Wave-Packet Dynamics of a Charged Particle in a Uniform Magnetic Field"*, *Plasma and Fusion Research* **6**, 2401083 (2011). DOI: `10.1585/pfr.6.2401083`
3. R. Ueda, **S. Oikawa**, and T. Imai, *"Numerical Analysis of Quantum-Mechanical Wave-Packet Dynamics in a Uniform Magnetic Field"*, *Plasma and Fusion Research* **6**, 2401087 (2011). DOI: `10.1585/pfr.6.2401087`
4. **S. Oikawa**, R. Ueda, and T. Imai, *"Quantum-Mechanical Effects on Cyclotron Motion in a Uniform Magnetic Field"*, *Plasma and Fusion Research* **7**, 2401053 (2012). DOI: `10.1585/pfr.7.2401053`
5. P.K. Chan, **S. Oikawa**, and R. Ueda, *"Wave-Packet Dynamics of a Charged Particle in Uniform Magnetic and Electric Fields"*, *Plasma and Fusion Research* **7**, 2401060 (2012). DOI: `10.1585/pfr.7.2401060`
6. **S. Oikawa**, P.K. Chan, and R. Ueda, *"Quantum-Mechanical Behavior of a Charged Particle in a Magnetic Field with a Gradient"*, *Plasma and Fusion Research* **8**, 2401048 (2013). DOI: `10.1585/pfr.8.2401048`
7. P.K. Chan, **S. Oikawa**, and W. Kosaka, *"Quantum-Mechanical Wave-Packet Dynamics in Non-Uniform Electric Fields"*, *Plasma and Fusion Research* **11**, 2401031 (2016). DOI: `10.1585/pfr.11.2401031`
8. P.K. Chan, **S. Oikawa**, and W. Kosaka, *"Long-Term Evolution of Quantum Wave Packets in Crossed Electromagnetic Fields"*, *Plasma and Fusion Research* **11**, 2401092 (2016). DOI: `10.1585/pfr.11.2401092`
9. W. Kosaka, **S. Oikawa**, and P.K. Chan, *"Two-Dimensional Quantum Wave-Packet Simulation of Charged Particles under Electrostatic Perturbations"*, *Plasma and Fusion Research* **11**, 2401098 (2016). DOI: `10.1585/pfr.11.2401098`
10. P.K. Chan, **S. Oikawa**, and W. Kosaka, *"Quantum-Classical Correspondence for Charged-Particle Motion in Inhomogeneous Electromagnetic Fields"*, *Plasma and Fusion Research* **12**, 2401017 (2017). DOI: `10.1585/pfr.12.2401017`

---

## Author & Citation

* **Author**: Shun-ichi Oikawa (及川 俊一)  
  Faculty of Engineering, Hokkaido University, Sapporo 060-8628, Japan  
  Contact: `lopia312oikawa@gmail.com`
  ORCID: [0009-0001-5603-2787](https://orcid.org)
  Zenodo Archive (v1.0.0): DOI 10.5281/zenodo.24416581

If you use this code in your research, please cite the accompanying paper and the Zenodo archive of this repository.

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** — see the `LICENSE` file for details.
