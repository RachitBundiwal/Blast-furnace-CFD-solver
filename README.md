# bf2d — 2D Axisymmetric Blast-Furnace Solver

Physics-grade CFD solver for ironmaking blast furnaces in axisymmetric (r-z)
coordinates. Couples gas pressure-velocity (Darcy-Forchheimer Poisson),
coupled gas-solid energy, mass-fraction species transport, kinematic solid
descent, and a 13-reaction kinetics network, with a cohesive-zone tracker
and a 0D raceway combustion model. See [Physics](#physics) below for the
full governing-equation writeup.

Designed as the physics core for MECON's blast-furnace digital twin. Extends
the existing 1D Streamlit solver to a full 2D field simulation.

Plant validation references:
- Fu, Chen, Zhao, D'Alessio, Ferron, Zhou (2014) "CFD modeling of multiphase
  reacting flow in blast furnace shaft with layered burden", *Applied Thermal
  Engineering* 66, 298-308.
- Zhou, Tang, Wang, Fu, Okosun, Silaen, Wu (2016) "Comprehensive Numerical
  Modeling of the Blast Furnace Ironmaking Process", *JOM* 68(5), 1353-1362.
- Austin, Nogami & Yagi (1997) "A Mathematical Model for Blast Furnace
  Reaction Analysis Based on the Four Fluid Model", *ISIJ International*
  37(8), 748-755. (Wall Robin-BC constants; reaction rate forms.)
- Adema, Yang, Boom (2009) "Coupled DEM-CFD Modelling of the Ironmaking Blast
  Furnace", *7th Int. Conf. on CFD in the Minerals and Process Industries*, CSIRO.
- Moukalled, Mangani & Darwish (2016) *The Finite Volume Method in
  Computational Fluid Dynamics*, Springer (798 pp).

---

## Physics

The solver treats the furnace as two interpenetrating continua — an
ascending reacting gas and a descending reacting solid burden — coupled by
mass, momentum and heat exchange through the packed bed, on a 2D
axisymmetric `(r, z)` domain. This is the standard Eulerian-Eulerian
"multi-fluid" formulation used in the plant-scale shaft models this
project validates against (Fu 2014; Zhou 2016; Austin, Nogami & Yagi
1997). Nothing here resolves individual particles or the turbulent
raceway cavity directly — those are closed with correlations (Ergun,
Ranz-Marshall) and a 0D combustion sub-model, exactly as in the
literature models.

| Equation | Discretisation | Notes |
|---|---|---|
| Gas pressure-velocity (Darcy-Forchheimer) | FV Poisson on pressure with K(\|u\|) iteration | Replaces SIMPLE/PISO; correct for porous media |
| Gas continuity | Built into the Poisson RHS | Mass sources from reactions + tuyere |
| Coupled gas + solid energy | Block-implicit 2N×2N sparse, solved jointly every iteration | Ranz-Marshall h_v with NTU bound, radiation, wall Robin BC |
| 6 gas species (N₂, O₂, CO, CO₂, H₂, H₂O) | Segregated upwind FV on **mass fractions** | Σ Y = 1 by conservation, no renormalisation |
| 7 solid phases (Fe₂O₃, Fe₃O₄, FeO, Fe, coke, gangue, flux) | Upwind FV, kinematic descent | Partial loadings on the carrier stream, not forced to sum to 1 |
| 13 reactions (12 core + calcination) | Arrhenius, volumetric, equilibrium-limited + Yagi availability capping | CO/H₂ reduction ladder, Boudouard, water-gas, WGS, direct reduction, flux calcination |
| Cohesive zone | Softening 1373 K → melting 1673 K with porosity collapse | Coke-slit permeability preserved |
| Raceway | 0D adiabatic combustion CSTR + mesh-independent tuyere area + cavity porosity | A_phys = n_tuyeres × π × d²/4 |
| Radial burden distribution | Center-working charge weighting + relative-radius descent | Drives the inverted-V cohesive zone (Fu §2.4/§3.1) |
| Gas exit | Defined uptake ring at top by `gas_exit_r_min/max` | Plant-realistic 4-offtake geometry |

### 1. Gas mass & momentum — Darcy-Forchheimer pressure Poisson

For packed-bed flow at BF velocities (~0.5-2 m/s), Ergun friction dominates
inertia by ~1000×, so the full Navier-Stokes momentum equation collapses to
an algebraic (inertialess) Darcy-Forchheimer law:

```
u = -K(|u|, ρ, ε, dp)⁻¹ · ∇p − ρ g / K       [Darcy-Forchheimer + gravity]
1/K = 150 μ (1-ε)² / (ε³ dp²)                [Ergun viscous term]
    + 1.75 ρ |u| (1-ε) / (ε³ dp)             [Forchheimer inertial term]
```

`ε` is local porosity and `dp` the local Sauter-mean particle diameter —
both fields, since they collapse inside the cohesive zone and change with
the radial burden composition. Substituting the momentum closure into the
continuity equation gives a single elliptic Poisson equation for pressure:

```
∇·(ρ K ∇p) = S_mass + ∇·(ρ² g / K)
```

`S_mass` is the net volumetric gas-mass source from reactions (reduction
strips oxygen from the burden into the gas; gasification and calcination
add carbon and CO₂). Because `K` depends on the very velocity it produces,
each outer iteration re-solves the Poisson equation for a few inner
sub-iterations, updating `K` from the previous velocity estimate
(Picard/fixed-point iteration on the nonlinear Ergun term) before moving
on. This elliptic formulation is what Fu (2014) and Zhou (2016) use for
plant-scale BF shaft models — it bypasses the pressure-velocity coupling
pathology that gives naive SIMPLE/PISO solvers spurious recirculation in
narrow-inlet, narrow-outlet packed beds.

**Boundary conditions:** the tuyere is a prescribed mass-flux Dirichlet
face (`G_tuyere = ṁ_blast / A_phys`, not a volumetric source — see §7);
the top uptake ring (`gas_exit_r_min..r_max`) is a pressure Dirichlet at
`top_pressure`, with cells outside the ring sealed (no-flux, representing
the charging-system seal); the wall and axis are no-flux.

### 2. Coupled gas-solid energy

Two temperature fields, `T_g` and `T_s`, are solved as one block-coupled
`2N×2N` sparse linear system every outer iteration (not sequentially) —
this is what lets the raceway's ~2000 K combustion heat couple correctly
to the cold incoming burden without operator-splitting lag. Each cell's
energy balance has convection (upwind, face-interpolated `ρ·cp` to handle
the large density contrast between raceway and shaft gas), conduction
(bed effective conductivity, a simplified Zehner-Schlünder parallel
model `k_eff = (1-ε)k_solid + ε·k_gas`), interphase exchange, and
reaction heat sourced to whichever phase actually carries the reacting
species (combustion/gasification heat the solid; indirect-reduction heat
the gas — enthalpies below).

**Gas-solid heat exchange** uses the Ranz-Marshall correlation,
`Nu = 2 + 0.6 Re^0.5 Pr^(1/3)`, converted to a volumetric coefficient via
the bed's specific interfacial area `a_sf = 6(1-ε)/dp`, plus a Rosseland
radiation term. The raw coefficient can reach ~10⁵ W/(m³ K) in narrow
raceway-band cells, which would equilibrate the gas to the solid
temperature within a single cell — unphysical, since the real limit is
how much enthalpy the gas can carry across a cell in one residence time.
The coefficient is therefore **NTU-bounded**:
`h_eff = ṁ·cp/V · (1 − e⁻ᴺᵀᵁ)`, which reduces to the raw correlation at
low NTU and saturates at the mass-flow limit at high NTU. Without this
bound the "thermal reserve zone" plant data shows (`T_g` holding near
1100 K through much of the shaft, Fu 2014 Fig. 9) collapses, and
indirect-reduction reactions cut off prematurely as `T_g` crashes to
`T_s`.

**Wall boundary condition** is a Robin BC, `q = h_wall (T − T_wall)`,
split 50/50 between the gas and solid energy equations at wall-adjacent
cells. The constants (`h_wall = 20 W/(m²K)`, `T_wall = 298 K`) are the
published values from Austin, Nogami & Yagi (1997, ISIJ 37(8) p.751) —
not tuned.

**Reaction enthalpies** (J/mol of the limiting reactant, at 298 K):

| Reaction | ΔH [J/mol] | Sense |
|---|---:|---|
| R1  (C + O₂ → CO₂) | −3.94×10⁵ | exothermic |
| R1b (C + ½O₂ → CO) | −1.11×10⁵ | exothermic |
| R2  (3Fe₂O₃ + CO → 2Fe₃O₄ + CO₂) | −1.36×10⁴ | exothermic |
| R3  (Fe₃O₄ + CO → 3FeO + CO₂) | +3.79×10⁴ | endothermic |
| R4  (FeO + CO → Fe + CO₂) | −1.65×10⁴ | exothermic |
| R2H/R3H/R4H (H₂ analogues) | +5.0×10³ / +6.30×10⁴ / +2.50×10⁴ | endothermic |
| R5  (C + CO₂ → 2CO, Boudouard) | +1.72×10⁵ | strongly endothermic |
| R6  (C + H₂O → CO + H₂) | +1.31×10⁵ | endothermic |
| R7  (CO + H₂O ⇌ CO₂ + H₂, WGS) | −4.10×10⁴ | exothermic (forward) |
| R9  (CaCO₃ → CaO + CO₂) | +1.78×10⁵ | strongly endothermic |
| R_DR (FeO + C → Fe + CO) | +1.52×10⁵ | endothermic |

### 3. Species transport — conservative mass-fraction form

The six gas species are transported in **mass-fraction** form,
`∂(ε ρ Y_k)/∂t + ∇·(ρ u Y_k) = S_k`, rather than mole fraction. With
mass-balanced kinetics (Σ S_k = 0 by construction — every reaction's
source terms are stoichiometrically closed) this form guarantees
Σ Y_k = 1 at every cell from the discretization alone, with **no
renormalization** — an earlier mole-fraction implementation needed an
ad-hoc "drift to N₂" correction that silently broke conservation
whenever the inert diluent was locally depleted (e.g. near the cohesive
zone where H₂ production is strong). The linear system's diagonal
carries the cell's true outgoing convective flux (strict conservative
upwind — see PHASE_HISTORY Phase 10 for why a source-term-only diagonal
is insufficient once reactions are adding/removing gas mass in the
cell). The tuyere inlet is applied as a face-flux boundary term (not a
Dirichlet row override), so gas already circulating through a raceway
cell isn't destroyed and re-created at blast composition each iteration.

### 4. Solid-phase transport — kinematic descent

Granular burden mechanics (DEM-scale particle motion) is out of scope;
instead the solid descends **kinematically** at a velocity back-solved
from mass continuity, `u_s(r,z) = ṁ_charge / (ρ_s,bulk · A_active(z))`,
scaled by a radial factor `Ks(r) = ks_center + (1 − ks_center)·r/R_wall`
(Fu 2014 §3.1: solids near the tuyeres descend faster than the axis
because raceway coke consumption pulls the wall column down), normalized
so total mass flow through every horizontal plane is conserved exactly.

Because the furnace wall tapers (hearth → bosh → belly → stack), a
column descending at fixed radius `r` would run off the profile at every
taper. The solid transport instead advects at fixed **relative** radius
`x = r/R_wall(z)` — each receiving cell's inflow is built from the
overlap of its relative-radius bin with the donor cells one row above,
with overlap weights summing to 1 per donor (conservative by
construction). This is what lets a radially-charged burden (§8) keep its
ore/coke profile intact as it descends through the tapering geometry,
instead of columns at the widening walls running out of a donor.

Solid fields are carried as **partial loadings** on the fixed carrier
stream, not forced to sum to 1: when reduction strips oxygen from the
burden into the gas (or gasification/calcination removes carbon), the
solid stream genuinely loses mass, and `Σf < 1` in that cell correctly
represents a lighter-loaded stream — it is not redistributed into an
inert phase or renormalized away. Properties that need a true mass
fraction (heat capacity, particle-size blending) normalize by `Σf` at
the point of use.

### 5. Reaction kinetics

Twelve gas-solid/solid-solid reactions plus flux calcination, evaluated
every outer iteration from the current `T_g`, `T_s`, gas composition and
solid-phase loadings:

| # | Reaction | Type | Rate-limited by |
|---|---|---|---|
| R1  | C + O₂ → CO₂ | raceway full combustion | Arrhenius, O₂/coke availability |
| R1b | C + ½O₂ → CO | raceway partial combustion | Arrhenius, O₂/coke availability |
| R2  | 3Fe₂O₃ + CO → 2Fe₃O₄ + CO₂ | indirect (hematite) | Arrhenius (irreversible; K_eq ≫ 1) |
| R3  | Fe₃O₄ + CO → 3FeO + CO₂ | indirect (magnetite) | Arrhenius × equilibrium driving force |
| R4  | FeO + CO → Fe + CO₂ | indirect (wüstite) | Arrhenius × equilibrium driving force |
| R2H/R3H/R4H | same, with H₂ → H₂O | indirect, H₂ route | as above, H₂-based equilibria |
| R5  | C + CO₂ → 2CO | Boudouard / solution loss | Arrhenius × Boudouard equilibrium |
| R6  | C + H₂O → CO + H₂ | water-gas | Arrhenius × equilibrium (derived from R5+R7) |
| R7  | CO + H₂O ⇌ CO₂ + H₂ | water-gas shift | Arrhenius, kinetically frozen below 1100 K |
| R_DR | FeO + C → Fe + CO | direct reduction | solid-kinetic branch + molten-drainage branch |
| R9  | CaCO₃ → CaO + CO₂ | flux calcination | Arrhenius, onset 1050 K, flux availability |

All rates are Arrhenius, `k = A₀ exp(−E_A / RT)`, using `T_g` for
gas-driven indirect reduction and `T_s` for solid/coke-driven reactions
(standard Yagi-type convention). Several reactions are **not**
irreversible — they carry an explicit thermodynamic driving force
`(C_reactant − C_product/K_eq)`, clamped to zero, so the reaction
self-quenches at equilibrium instead of running until a reactant is
exhausted:

- **R3, R4, R3H, R4H** (indirect reduction): equilibrium constants fit
  to the Baur-Glaessner/Chaudron Fe-O-C phase boundaries (magnetite/
  wüstite and wüstite/iron), e.g. `log₁₀ K_R3 = 2.17 − 1834/T`. An
  earlier fit here was off by ~2× on both R3 (too restrictive — it
  drove R3 to zero throughout the plant's actual thermal-reserve-zone
  gas composition, starving the reduction ladder's middle step) and R4
  (too permissive — it let indirect reduction run where thermodynamics
  forbid it). Both were re-derived from the published phase boundaries
  (PHASE_HISTORY Phase 9).
- **R5** (Boudouard): `log₁₀ K_p = 9.14 − 8969/T` [atm] (standard
  JANAF-derived fit).
- **R6** (water-gas): equilibrium derived as
  `K_R6 = K_R5 · K_R7` (Hess's law on the reaction network), with its
  pre-exponential anchored to the measured coke gasification rate ratio
  `k_H₂O/k_CO₂ ≈ 3.5`.
- **R7** (WGS): `ln K_eq = −4.33 + 4577/T`; kinetically frozen below
  1100 K (homogeneous, uncatalyzed WGS does not reach equilibrium in the
  cool upper shaft at BF gas residence times — without the freeze,
  reverse WGS strips the H₂ that plant data shows surviving to the top).

**R_DR** (direct reduction) has two branches: a solid-state kinetic term
active above 1273 K, and a **liquid-drainage branch** — once the burden
is fully molten (`T_s` above the liquidus), wüstite drips through the
coke bed and is reduced on the ~600 s liquid-residence timescale
(Zhou 2016 reaction R11, `FeO(l) + C(s) → Fe(l) + CO`), so molten iron
oxide is not simply deleted from the mass ledger but genuinely converted
to hot metal.

**Gating logic** enforces the plant-observed vertical ordering of the
reduction sequence (Fu 2014; Austin, Nogami & Yagi 1997) with smooth
`tanh` transitions rather than hard cutoffs:
- a **solid-phase-existence factor** shuts off indirect reduction as the
  ore passes from softening (1373 K) to melting (1673 K) — above the
  liquidus, reduction proceeds only by direct coke contact (R_DR), not
  gas percolation;
- a **cohesive-zone gas-solid contact gate** shuts off indirect
  reduction once `T_s` exceeds the softening point, since fused ore
  layers are no longer permeable to the reducing gas;
- **height gates** (`feo_form_height_frac`, `feo_reduce_height_frac`)
  confine wüstite *formation* to the lower ~2/3 of the furnace and
  wüstite *reduction to iron* to the lower ~1/3, matching the observed
  plant reduction-ladder ordering.

Every rate is additionally capped by **local reactant availability**
(Yagi-style: a reaction cannot consume a reactant faster than
`concentration / dt` makes it available within one solver step — solid
and gas phases use their own respective timesteps) and by a generous
physical ceiling (60 mol/(m³ s)) as a numerical safety net.

### 6. Cohesive zone (softening → melting band)

The cohesive zone (CZ) is the band where ore has softened but not yet
melted, `T_soft = 1373 K < T_s < T_melt = 1673 K`. Inside this band,
porosity collapses toward `cz_porosity_min` (fused ore becomes nearly
impermeable) **except** where local coke mass fraction is high — coke
layers keep their base porosity, preserving the "coke-slit" gas
channels a real CZ relies on for permeability. Above the liquidus, the
same collapsed porosity persists into the dripping zone (molten
material trickling through the coke skeleton) but is excluded from the
CZ *shape* diagnostics, which use the softening-to-liquidus band only
(matching how Fu 2014 Fig. 5 defines the zone). `cz_shape_metrics()`
extracts a per-radius CZ band and an inversion metric — the axis apex
height above the lowest band foot — used both as a validation KPI and
for the `cohesive_zone.png` diagnostic plot.

### 7. Raceway / tuyere boundary model

The raceway (the turbulent combustion cavity in front of each tuyere)
is not resolved directly — its ms-scale eddy-limited combustion rates
are far higher than a porous-bed continuum can represent. Instead,
following Fu (2014) and Zhou (2016), it is closed with a **0D adiabatic
CSTR**: blast air (with O₂ enrichment and humidification), PCI coal, and
coke carbon react completely inside the raceway, and an enthalpy balance
solves for the exit ("bosh gas") composition and temperature — typically
~38% CO, ~50% N₂, T ≈ 2100-2400 K, matching Fu Table 5. This composition
and temperature are then imposed as the tuyere's momentum/species/energy
boundary condition.

The **physical tuyere area** (`A_phys = n_tuyeres · π · d²/4`) is
computed independently of mesh resolution and mapped onto the nearest
discrete wall faces, so the inlet velocity stays in the physically
correct 150–250 m/s range regardless of grid refinement.

The tuyere band is also treated as a **cavity**: the raceway is a real
void blown ~1 m into the coke bed (`tuyere_penetration_fraction`), not a
solid packed-bed cell. Overriding local porosity/particle-size to cavity
values (`raceway_cavity_eps`) in that band lets the blast distribute
over the penetration annulus instead of jetting straight up the
adjacent wall cell — without this, the wall column overheats and the
cohesive-zone crown flattens (see PHASE_HISTORY Phase 12).

### 8. Radial burden distribution (the inverted-V driver)

A healthy blast furnace shows an **inverted-V** cohesive zone — apex
high on the axis, feet low at the walls — because plant charging is
"center-working": more coke (higher permeability) is charged near the
axis and more ore near the wall, channeling hot reducing gas up the
center (Fu 2014 §2.4; Zhou 2016 Fig. 2). This is reproduced by two
coupled mechanisms, both driven by a single radial weight field
`w(r,z) = clip(w₀ + (w₁−w₀)(r/R_wall)^q, 0, 1.6)` with
`w₀ = 1 − bias`, `w₁ = 1 + bias` (area-mean renormalized to 1 per row so
the column-average charge is exactly preserved):

1. **Radial charge composition** — the stockline inlet itself is
   center-working: ore-family phases (Fe₂O₃, gangue) are scaled by
   `w(r)` at the top row, coke absorbing the complement.
2. **Radial descent factor** `Ks(r)` (§4) — the wall column descends
   faster, giving the ore-rich wall material less residence time to
   heat relative to the coke-rich, slower-descending axis column.

Together these channel gas preferentially up the center and cool the
wall column, producing the inverted-V silhouette self-consistently from
charge distribution and flow physics rather than from an imposed shape.
An opt-in **layered-burden** mode (`layered_burden=True`,
`bf2d/burden_layers.py`) additionally resolves alternating horizontal
ore/coke bands (Fu's "cell-type" technique) with the same radial
weighting applied to the layer structure.

### Mesh

Axisymmetric staggered (Patankar convention):
- Scalars at cell centres `(i, j)`
- `u_r` on east faces `(i+1/2, j)`
- `u_z` on north faces `(i, j+1/2)`

Four-zone wall profile (hearth → bosh → belly → stack) built from
user-supplied radii and band heights. Cells outside the wall are masked
inactive across all solvers.

### Solution procedure (one outer iteration)

1. Update gas/solid properties (density, viscosity, cp, effective
   particle diameter) from the current fields.
2. Detect the cohesive zone; blend porosity/particle-diameter fields
   (bounded update, ≤30% change per iteration); apply the radial-burden
   perturbation and raceway-cavity override.
3. Evaluate all 13 reaction rates; under-relax against the previous
   iteration's rates (damps a limit-cycle the strongly-coupled CO/CO₂
   pairs can otherwise fall into on fine meshes).
4. Solve the coupled Darcy-Forchheimer pressure-velocity system
   (inner Picard loop on the Ergun coefficient).
5. Compute the kinematic solid descent velocity field.
6. Solve gas species transport (mass-fraction form).
7. Solve solid-phase composition transport (relative-radius descent).
8. Solve the coupled gas+solid energy system.
9. Assemble per-equation residuals; adapt relaxation factors; check
   convergence (all residuals below tolerance **and** the solid-phase
   inventory steady for 5 consecutive iterations — residual-only
   convergence can false-positive while the burden composition is still
   mid-flush down the column).

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
