# bf2d — 2D Axisymmetric Blast-Furnace Solver

Physics-grade CFD solver for ironmaking blast furnaces in axisymmetric (r-z)
coordinates. Couples gas pressure-velocity (Darcy-Forchheimer Poisson),
coupled gas-solid energy, mass-fraction species transport, solid descent
and 12-reaction kinetics, with a cohesive-zone tracker and raceway
boundary model.

Designed as the physics core for MECON's blast-furnace digital twin. Extends
the existing 1D Streamlit solver to a full 2D field simulation.

Plant validation references:
- Fu, Chen, Zhao, D'Alessio, Ferron, Zhou (2014) "CFD modeling of multiphase
  reacting flow in blast furnace shaft with layered burden", *Applied Thermal
  Engineering* 66, 298-308.
- Zhou, Tang, Wang, Fu, Okosun, Silaen, Wu (2016) "Comprehensive Numerical
  Modeling of the Blast Furnace Ironmaking Process", *JOM* 68(5), 1353-1362.
- Adema, Yang, Boom (2009) "Coupled DEM-CFD Modelling of the Ironmaking Blast
  Furnace", *7th Int. Conf. on CFD in the Minerals and Process Industries*, CSIRO.
- Moukalled, Mangani & Darwish (2016) *The Finite Volume Method in
  Computational Fluid Dynamics*, Springer (798 pp).

---

## Physics

| Equation | Discretisation | Notes |
|---|---|---|
| Gas pressure-velocity (Darcy-Forchheimer) | FV Poisson on pressure with K(\|u\|) iteration | Replaces SIMPLE/PISO; correct for porous media |
| Gas continuity | Built into the Poisson RHS | Mass sources from reactions + tuyere |
| Coupled gas + solid energy | Block-implicit 2N×2N sparse | Ranz-Marshall h_v, radiation, wall Robin BC |
| 6 gas species (N₂, O₂, CO, CO₂, H₂, H₂O) | Segregated upwind FV on **mass fractions** | Σ Y = 1 by conservation, no renormalisation |
| 6 solid phases (Fe₂O₃, Fe₃O₄, FeO, Fe, coke, gangue) | Upwind FV, kinematic descent | Mass fractions |
| 12 reactions | Arrhenius, volumetric, Yagi-style availability | CO + H₂ reduction, Boudouard, WGS, direct reduction |
| Cohesive zone | Softening 1373 K → melting 1673 K with porosity collapse | Coke-slit permeability preserved |
| Raceway | Mesh-independent tuyere area from physical bore diameter | A_phys = n_tuyeres × π × d²/4 |
| Gas exit | Defined uptake ring at top by `gas_exit_r_min/max` | Plant-realistic 4-offtake geometry |

### Why Darcy-Forchheimer instead of SIMPLE

For packed-bed flow at BF velocities (~0.5-2 m/s), Ergun friction dominates
inertia by ~1000×. The full Navier-Stokes momentum equation collapses to:

```
u = -K(|u|, ρ, ε, dp) · ∇p              [Darcy-Forchheimer]
1/K = 150 μ (1-ε)² / (ε³ dp²)           [Ergun viscous]
    + 1.75 ρ |u| (1-ε) / (ε³ dp)        [Forchheimer inertial]
```

Substituting into continuity gives a single elliptic Poisson equation:

```
∇·(ρ K ∇p) = -S_mass
```

This bypasses the pressure-velocity coupling pathology that gives SIMPLE
solvers spurious recirculation in narrow-inlet/narrow-outlet packed beds.
This is the formulation used by Fu (2014) and Zhou (2016) for plant-scale
BF shaft modeling.

### Mesh

Axisymmetric staggered (Patankar convention):
- Scalars at cell centres `(i, j)`
- `u_r` on east faces `(i+1/2, j)`
- `u_z` on north faces `(i, j+1/2)`

Four-zone wall profile (hearth → bosh → belly → stack) built from
user-supplied radii and band heights. Cells outside the wall are masked
inactive across all solvers.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required.

## Run

```bash
# Default industrial case
python run_simulation.py

# Fast verification run
python run_simulation.py --nr 20 --nz 80 --max-iter 200 --output out/small

# Full plant-scale resolution
python run_simulation.py --nr 30 --nz 120 --max-iter 500 --output out/full
```

## Programmatic use

```python
from bf2d.parameters import BFParameters2D
from bf2d.simulation import BlastFurnaceSimulation

p = BFParameters2D(
    nr=30, nz=120,
    max_outer_iter=500, dt=0.5,
    momentum_relaxation=0.4, pressure_relaxation=0.3,
    energy_relaxation=0.3, species_relaxation=0.5,
    # Tuyere: 32 tuyeres of 130mm bore
    n_tuyeres=32, tuyere_diameter=0.13,
    # Gas exit: uptake ring at top
    gas_exit_r_min=1.5, gas_exit_r_max=3.5,
)
sim = BlastFurnaceSimulation(p)
sim.run()
```

## Current capabilities (verified)

On 15×40 mesh, 100 iterations, ~30s wall-time:

- ✓ Mass conservation: 100% (134.3 kg/s in = 134.3 kg/s out)
- ✓ Flame T = 2273 K (plant RAFT range 2100-2400 K)
- ✓ Pressure 2.5 bar at top exit (plant range)
- ✓ No recirculation; u_z monotonically positive
- ✓ Σ Y_k = 1 to 4 decimal places everywhere (mass-fraction conservation)
- ✓ Mesh-independent tuyere area
- ◐ Top gas composition: needs kinetics calibration (see PHASE_HISTORY.md)

## Project structure

```
bf_2d_solver_phase1/
├── bf2d/                       # Physics package
│   ├── parameters.py           # Frozen dataclass with validation
│   ├── mesh.py                 # 4-zone wall profile, staggered grid
│   ├── properties.py           # Gas density, viscosity, Cp, Ergun
│   ├── kinetics.py             # 12 reactions with Yagi availability
│   ├── cohesive_zone.py        # Softening/melting tracker
│   ├── raceway.py              # Mesh-independent tuyere geometry
│   ├── solver_gas_df.py        # ★ Production: Darcy-Forchheimer
│   ├── solver_gas.py           # Legacy: SIMPLE/SIMPLEC (kept for ref)
│   ├── solver_pressure.py      # Reference: clean Darcy-Forchheimer
│   ├── solver_energy.py        # Block-coupled gas+solid
│   ├── solver_species.py       # Mass-fraction species transport
│   ├── linear_solver.py        # GMRES+AMG with spsolve fallback
│   ├── residuals.py            # Per-equation residual tracking
│   ├── simulation.py           # Outer-iteration orchestrator
│   ├── postprocess.py          # Plots and KPI extraction
│   └── logging_config.py       # Structured JSON logs
├── tests/                      # Unit + integration tests
├── run_simulation.py           # CLI entry point
├── diag_run.py                 # Diagnostic harness
├── pyproject.toml
├── requirements.txt
├── README.md                   # This file
└── PHASE_HISTORY.md            # Complete development log
```
