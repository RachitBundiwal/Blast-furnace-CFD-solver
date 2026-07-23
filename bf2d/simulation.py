"""
Main orchestrator for the Phase 1 solver.

Phase 1 changes vs. Phase 0
---------------------------
* Coupled (u_r, u_z, P) block momentum solve every outer iteration via
  ``solver_gas.solve_coupled_momentum`` (replaces broken PISO predictor +
  pressure-correction sequence).
* Tuyere injection is a **prescribed face inlet** in momentum, energy and
  species solvers (no volumetric source term).
* Per-equation L2 residuals reported every iteration; convergence requires
  ALL residuals below tolerance.
* Adaptive relaxation: residuals dropping monotonically -> raise omega up to
  0.9.  Stalled or rising -> halve omega and reset.
* Cohesive-zone porosity update is bounded so the bed permeability cannot
  collapse in a single iteration (Phase 0 used a 50/50 blend with no bound).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .cohesive_zone import detect_cohesive_zone
from .burden_layers import (
    DP_COKE, DP_ORE, EPS_COKE, EPS_ORE,
    apply_radial_ore_bias, layered_bed_properties, ore_fraction_field,
    radial_ore_weight,
)
from .constants import GAS_SPECIES, MW
from .kinetics import (
    ReactionRates, compute_rates, assemble_species_sources,
    assemble_solid_sources,
)
from .logging_config import configure_logging
from .mesh import Mesh2D, build_mesh
from .parameters import BFParameters2D
from .properties import (
    gas_density, gas_viscosity, gas_cp_mass, solid_cp_mass,
    effective_particle_diameter, gas_mw_mix, normalize_mole_fractions,
)
from .raceway import build_raceway, Raceway
from .residuals import (
    ResidualSnapshot, ResidualHistory, ReferenceScales, l2_normalised,
)
from .solver_energy import solve_energy
from .solver_gas_df import solve_coupled_momentum, MomentumSolution
from .solver_species import (
    solve_species, update_solid_phases, compute_solid_descent_velocity,
)


# ---------------------------------------------------------------------------
@dataclass
class SimulationFields:
    """Mutable field container carried between iterations."""
    T_g: np.ndarray
    T_s: np.ndarray
    P: np.ndarray
    u_r: np.ndarray
    u_z: np.ndarray
    x_gas: Dict[str, np.ndarray]
    f_solid: Dict[str, np.ndarray]
    eps: np.ndarray
    dp_eff: np.ndarray
    rho_g: np.ndarray
    cohesive_mask: np.ndarray


# ---------------------------------------------------------------------------

def burden_inlet_composition(p) -> dict:
    """Charge composition (mass fractions) derived from rate parameters.

    Single source of truth: uses coke/ore/flux rates so the inlet mix is
    always consistent with the specified operating point.  Ore is treated
    as 90 % Fe2O3 + 10 % gangue by mass (Fe balance for Fu 2014:
    1501 kg ore/tHM -> ~945 kg Fe/tHM => ~1352 kg Fe2O3/tHM)."""
    ore, coke, flux = p.ore_rate_kg_s, p.coke_rate_kg_s, p.flux_rate_kg_s
    total = max(ore + coke + flux, 1e-9)
    return {
        "Fe2O3":  0.90 * ore / total,
        "Fe3O4":  0.0,
        "FeO":    0.0,
        "Fe":     0.0,
        "C_coke": coke / total,
        "gangue": 0.10 * ore / total,
        "flux":   flux / total,
    }

class BlastFurnaceSimulation:
    """Top-level driver for the Phase 1 coupled solver."""

    def __init__(
        self,
        params: BFParameters2D,
        *,
        log_level: str = "INFO",
        debug: bool = False,
    ) -> None:
        self.p = params
        self.log = configure_logging(log_level, debug=debug)
        self.mesh: Mesh2D = build_mesh(params)
        self._ore_frac = (
            ore_fraction_field(self.mesh, params.ore_layer_thickness_m,
                               params.coke_layer_thickness_m)
            if params.layered_burden else None)
        # Radial burden distribution (inverted-V CZ driver, Fu 2014 §2.4).
        # Layered path: multiply the layer ore-fraction map by w(r) with
        # column-average preservation.  Homogeneous path: static eps/dp
        # perturbation fields built from the SAME w(r) and the Fu Table 6
        # ore/coke endmember properties, so the inverted V does not
        # require layered_burden=True.  Secure default (bias = 0) leaves
        # both paths exactly as before.
        self._d_eps_radial: Optional[np.ndarray] = None
        self._d_dp_radial: Optional[np.ndarray] = None
        self._inlet_radial: Optional[Dict[str, np.ndarray]] = None
        if params.burden_radial_ore_bias > 0.0:
            w_r = radial_ore_weight(self.mesh, params)
            if self._ore_frac is not None:
                self._ore_frac = apply_radial_ore_bias(
                    self._ore_frac, w_r, self.mesh)
            else:
                # Mean ore volume fraction from the charged layer
                # thicknesses (Fu 2014 plant layers); the perturbation
                # delta_f = f_mean * (w - 1) has zero area-weighted mean
                # per row, so column-average bed properties are preserved.
                f_mean = params.ore_layer_thickness_m / (
                    params.ore_layer_thickness_m
                    + params.coke_layer_thickness_m)
                d_f = f_mean * (w_r - 1.0)
                self._d_eps_radial = d_f * (EPS_ORE - EPS_COKE)
                self._d_dp_radial = d_f * (DP_ORE - DP_COKE)
            # Radial CHARGE composition (Fu 2014 Sect. 2.4): the
            # stockline inlet itself is center-working -- ore phases
            # scaled by w(r) at the top row, coke absorbing the
            # complement, area-weighted mean pinned to the plant charge
            # exactly.  With the axial descent each column preserves its
            # charged composition, so the ore-rich wall column soaks
            # raceway-jet heat (feet drop at the wall) and its finer
            # particles raise wall flow resistance (gas pushed
            # centerward) -- the genuine inverted-V mechanism.
            self._inlet_radial = self._radial_inlet_composition(w_r)
        self.raceway: Raceway = build_raceway(self.mesh, params)
        self.fields: SimulationFields = self._initial_fields()
        self.residuals = ResidualHistory(stall_window=5, stall_factor=0.95)

        # Build masks for downstream use.
        # Species at raceway cells: Dirichlet to blast composition.
        # Temperature at raceway cells: NOT Dirichlet -- the energy equation
        # must be free to compute the post-combustion (flame) temperature.
        # Plant reality: blast enters at ~1473 K but reaches ~2200 K within
        # the raceway cavity due to coke combustion.  Pinning the cell to
        # 1473 K would suppress the flame heat the combustion creates.
        self._raceway_mask_cells = self._build_raceway_cell_mask()
        self._raceway_T_field = np.full(
            (self.mesh.nr, self.mesh.nz), self.fields.T_g[0, 0],
        )
        self._raceway_T_field[self._raceway_mask_cells] = params.blast_temperature
        # Pin gas T at raceway-adjacent cells to the bosh-gas T (= raceway
        # flame T after combustion).  This is the canonical BF-shaft model
        # boundary condition (Fu 2014, Zhou 2016): the shaft model does not
        # resolve the raceway combustion explicitly, so we set the cell
        # immediately above the tuyere band to the raceway-exit T (~2273 K)
        # and let the gas cool through the column via reaction endotherms,
        # heat transfer to burden, and wall losses.
        #
        # In an earlier iteration of this code, blast_temperature was set
        # to the actual hot-blast inlet T (1473 K) and we deliberately did
        # NOT pin -- because pinning at 1473 K would have suppressed the
        # combustion heat that real blast carries.  Now blast_temperature
        # is the post-combustion bosh-gas T, so pinning is the correct BC.
        self._energy_T_dirichlet_mask = self._raceway_mask_cells.copy()

        # Gas-exit mask: cells in the top row that are part of the uptake
        # ring [gas_exit_r_min, gas_exit_r_max].  Cells outside this ring
        # at the top are treated as sealed (zero u_z, no pressure pin, no
        # composition outflow).  This represents the uptake offtake ducts.
        self._gas_exit_mask = self._build_gas_exit_mask()

        # Reference scales for residual normalisation
        self.refs = self._compute_reference_scales()

        # Adaptive relaxation state (one factor for coupled momentum,
        # separate factors for energy / species so we can tune independently)
        self._omega_mom = params.momentum_relaxation
        self._omega_energy = params.energy_relaxation
        self._omega_species = params.species_relaxation
        self._omega_solid = params.solid_relaxation
        self._omega_lo = 0.10
        self._omega_hi = 0.90
        # Previous-iteration reaction rates for source under-relaxation
        self._rates_prev: Optional[ReactionRates] = None

        self.log.info(
            "simulation_initialised",
            extra={
                "nr": self.mesh.nr, "nz": self.mesh.nz,
                "active_cells": int(self.mesh.active.sum()),
                "tuyere_faces": len(self.raceway.face_indices),
                "v_tuyere_m_s": round(self.raceway.v_tuyere, 1),
                "blast_kg_s": round(self.raceway.total_blast_kg_s, 2),
            },
        )

    # ---- Initial conditions ------------------------------------------------
    def _initial_fields(self) -> SimulationFields:
        p = self.p
        nr, nz = self.mesh.nr, self.mesh.nz
        shape = (nr, nz)

        z = self.mesh.z_c
        T_g = np.tile(
            p.blast_temperature - (p.blast_temperature - p.burden_inlet_temperature)
            * (z / max(z[-1], 1.0)),
            (nr, 1),
        )
        T_s = np.tile(
            p.burden_inlet_temperature + (0.8 * p.blast_temperature - p.burden_inlet_temperature)
            * (1.0 - z / max(z[-1], 1.0)),
            (nr, 1),
        )

        P = np.full(shape, p.top_pressure + 5.0e4)
        P[:, -1] = p.top_pressure

        # Initial gas: bosh-gas-like (CO-rich, CO2-lean).  The previous
        # 13% CO2 initial fill was harmless while the Boudouard reaction
        # was inert, but with plant-scale R5 kinetics a hot column full
        # of CO2 ignites a large artificial solution-loss transient
        # (hundreds of kg/s of phantom gas mass) before the fields
        # relax.  Initial conditions are a solver choice, not physics;
        # steady state is unchanged.
        x_gas = {
            "N2":  np.full(shape, 0.55),
            "O2":  np.full(shape, 0.005),
            "CO":  np.full(shape, 0.36),
            "CO2": np.full(shape, 0.05),
            "H2":  np.full(shape, 0.03),
            "H2O": np.full(shape, 0.005),
        }
        x_gas = normalize_mole_fractions(x_gas)

        inlet = burden_inlet_composition(p)
        f_solid = {ph: np.full(shape, inlet[ph]) for ph in inlet}

        eps = np.full(shape, p.void_fraction)
        dp_eff = np.full(shape, 0.5 * (p.coke_diameter + p.ore_diameter))
        rho_g = gas_density(P, T_g, x_gas)

        # Initial guess for u_z: uniform upward at the value that would
        # match the inlet mass flow if it were spread across the column
        # cross-section.  This prevents the SIMPLE solver from getting
        # locked into a stagnant fixed point at u_z = 0.
        m_dot_blast = self.raceway.total_blast_kg_s
        rho_avg = float(rho_g[self.mesh.active].mean())
        A_avg = float(self.mesh.area_z[:, nz // 2][self.mesh.active[:, nz // 2]].sum())
        u_z_guess = m_dot_blast / max(rho_avg * A_avg, 1.0)

        u_z_init = np.zeros((nr, nz + 1))
        for j in range(nz + 1):
            for i in range(nr):
                # Apply only to active cells (not solid wall outside the
                # furnace profile)
                if 0 <= j < nz and self.mesh.active[i, j]:
                    u_z_init[i, j] = u_z_guess
                elif j == nz and self.mesh.active[i, nz - 1]:
                    u_z_init[i, j] = u_z_guess
        u_z_init[:, 0] = 0.0   # closed bottom

        return SimulationFields(
            T_g=T_g, T_s=T_s, P=P,
            u_r=np.zeros((nr + 1, nz)),
            u_z=u_z_init,
            x_gas=x_gas, f_solid=f_solid,
            eps=eps, dp_eff=dp_eff, rho_g=rho_g,
            cohesive_mask=np.zeros(shape, dtype=bool),
        )

    def _radial_inlet_composition(self, w_r: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-cell top-row charge composition from the radial ore weight.

        Ore-family inlet fractions (Fe2O3, gangue) scale with w(r) at
        the stockline; flux is charged uniformly; coke takes the
        complement.  w is capped so wall coke stays positive, then
        fixed-point renormalized so the area-weighted inlet equals
        ``burden_inlet_composition`` exactly (global charge balance
        untouched).
        """
        base = burden_inlet_composition(self.p)
        j = self.mesh.nz - 1
        act = self.mesh.active[:, j]
        ring = np.pi * (self.mesh.r_f[1:] ** 2 - self.mesh.r_f[:-1] ** 2)
        w_top = w_r[:, j].copy()
        ore_mean = base["Fe2O3"] + base["gangue"]
        # Cap: ore_i <= 95% of (1 - flux) so coke never vanishes at the
        # wall.  The charge is ~74% ore by mass, so the cap binds at
        # moderate bias; renormalize the capped weight back to
        # area-weighted mean 1 (fixed point, monotone).
        w_max = 0.95 * (1.0 - base["flux"]) / max(ore_mean, 1e-9)
        for _ in range(8):
            w_top = np.minimum(w_top, w_max)
            mean = float((w_top[act] * ring[act]).sum()
                         / max(ring[act].sum(), 1e-12))
            if abs(mean - 1.0) < 1e-6:
                break
            w_top = w_top / max(mean, 1e-12)
        w_top = np.minimum(w_top, w_max)
        inlet: Dict[str, np.ndarray] = {}
        nr = self.mesh.nr
        for ph, val in base.items():
            inlet[ph] = np.full(nr, val)
        inlet["Fe2O3"] = base["Fe2O3"] * w_top
        inlet["gangue"] = base["gangue"] * w_top
        inlet["C_coke"] = np.clip(
            1.0 - inlet["Fe2O3"] - inlet["gangue"] - base["flux"],
            0.0, 1.0)
        return inlet

    def _build_raceway_cell_mask(self) -> np.ndarray:
        """Cells immediately adjacent to a tuyere face -- used as Dirichlet
        cells for gas T and species (the gas in those cells is the blast)."""
        mask = np.zeros((self.mesh.nr, self.mesh.nz), dtype=bool)
        for (i_face, j) in self.raceway.face_indices:
            i_cell = i_face - 1   # cell west of the wall face
            if 0 <= i_cell < self.mesh.nr and self.mesh.active[i_cell, j]:
                mask[i_cell, j] = True
        return mask

    def _build_gas_exit_mask(self) -> np.ndarray:
        """Top-row cells that lie in the uptake ring.

        Returns a (nr, nz) boolean array, all False except possibly the
        top row.  Cells in the top row are True iff their radial centre
        r_c lies within [gas_exit_r_min, gas_exit_r_max].
        """
        mask = np.zeros((self.mesh.nr, self.mesh.nz), dtype=bool)
        if self.mesh.nz == 0:
            return mask
        j = self.mesh.nz - 1
        r_min = self.p.gas_exit_r_min
        r_max = self.p.gas_exit_r_max
        for i in range(self.mesh.nr):
            if not self.mesh.active[i, j]:
                continue
            r = self.mesh.r_c[i]
            if r_min <= r <= r_max:
                mask[i, j] = True
        return mask

    def _compute_reference_scales(self) -> ReferenceScales:
        p = self.p
        rw = self.raceway
        # Active-cell volume sum
        V_total = float(np.sum(self.mesh.vol[self.mesh.active]))
        # Use mid-stack cp as reference
        cp_mid = float(gas_cp_mass(
            np.array([0.5 * (p.blast_temperature + p.burden_inlet_temperature)]),
            {sp: np.array([rw.x_gas_blast[sp]]) for sp in rw.x_gas_blast},
        )[0])
        return ReferenceScales(
            rho_ref=rw.rho_blast,
            u_ref=max(rw.v_tuyere, 0.5),
            cp_ref=cp_mid,
            T_ref=p.blast_temperature,
            P_ref=p.top_pressure,
            L_ref=p.furnace_height,
            V_total=V_total,
            rho_s_ref=1800.0 * (1.0 - p.void_fraction),
        )

    # ---- Main loop ---------------------------------------------------------
    def run(self) -> Dict:
        p = self.p
        prev_means: Dict[str, float] | None = None
        solid_steady_count = 0
        for it in range(p.max_outer_iter):
            try:
                snap = self._one_outer_iteration(it)
            except Exception as exc:
                self.log.error(
                    "outer_iteration_failed",
                    extra={"iteration": it, "error_type": type(exc).__name__},
                )
                break

            self.residuals.add(snap)

            if it % p.log_every == 0:
                self._log_iteration(snap)

            # NOTE: adaptive relaxation is deliberately NOT applied here.
            # The _adapt_relaxation rule is a one-way trap: any residual
            # rise cuts every omega 30% (recovery only 5%/iter), so early
            # transients drive all omegas to the 0.10 floor.  At the
            # floor, per-iteration updates are so small that the solve
            # residuals fall below tolerance while the fields are frozen
            # mid-transient -- run() then declares convergence with ZERO
            # iron produced (false convergence observed at the Fu 2014
            # operating point: "converged" at it=162, omegas 0.11,
            # Fe_mean 0.0001% vs 15% for the same 200 iterations at fixed
            # omegas).  All validated physics (validate_plant.py, the
            # diagnostic scripts) runs at the fixed relaxation factors
            # from BFParameters2D, which is therefore what run() uses.

            # Solid-inventory steadiness: the per-equation solve residuals
            # cannot see the SLOW plant-scale transient of the burden
            # composition flushing down the column (~140 outer iterations
            # at dt_solid = 200 s).  Without this guard, run() declared
            # convergence at it ~ 2*nz with the solid column mid-flush
            # (Fe_mean 9.7% and still rising vs 15% at the true plateau).
            # Require every solid-phase MEAN mass fraction to change by
            # < 1e-5 per iteration for 5 consecutive iterations.
            active = self.mesh.active
            means = {ph: float(v[active].mean())
                     for ph, v in self.fields.f_solid.items()}
            if prev_means is not None:
                d_phase = max(abs(means[ph] - prev_means[ph])
                              for ph in means)
                solid_steady_count = (solid_steady_count + 1
                                      if d_phase < 1.0e-5 else 0)
            prev_means = means

            # More robust convergence check than just "below tolerance for one
            # iteration".  Early iterations have small SIMPLE-corrections and
            # small RHS values, which can spuriously satisfy the tolerance
            # before the field is anywhere near steady state.  Require:
            #   (a) a minimum iteration count (mesh-scaled — gas needs at
            #       least nz iterations to traverse the column),
            #   (b) all residuals below tolerance for 5 consecutive iters,
            #   (c) solid inventory steady for 5 consecutive iters.
            min_iter = max(20, 2 * self.mesh.nz)
            if (it > min_iter and snap.all_below(p.convergence_rms)
                    and solid_steady_count >= 5):
                # Confirm: last 5 snapshots all below tolerance
                recent = self.residuals._snaps[-5:]
                if len(recent) == 5 and all(s.all_below(p.convergence_rms)
                                            for s in recent):
                    self.log.info("converged",
                                  extra={"iteration": it,
                                         "worst_residual": snap.max()})
                    break

        return self._package_results()

    # ---- One outer iteration ----------------------------------------------
    def _one_outer_iteration(self, it: int) -> ResidualSnapshot:
        mesh = self.mesh
        p = self.p
        f = self.fields

        # ---- Properties update -------------------------------------------
        f.rho_g = gas_density(f.P, f.T_g, f.x_gas)
        mu_g = gas_viscosity(f.T_g)
        cp_g = gas_cp_mass(f.T_g, f.x_gas)
        cp_s = solid_cp_mass(f.T_s, f.f_solid)
        mw_mix = gas_mw_mix(f.x_gas)
        dp_base = effective_particle_diameter(
            f.f_solid, p.coke_diameter, p.ore_diameter,
        )

        # Cohesive zone (bounded update: max 30% change per iteration).
        # Homogeneous path with radial burden bias: anchor the CZ
        # detection to the STATIC radially-biased base fields (not the
        # evolving f.eps) — otherwise the additive perturbation would be
        # re-applied through the relaxed update every iteration and
        # integrate without bound.
        if (self._d_eps_radial is not None
                and not p.layered_burden):
            eps_base_in = np.clip(
                p.void_fraction + self._d_eps_radial, 0.05, 0.85)
            dp_base_in = np.clip(
                dp_base + self._d_dp_radial, 4.0e-3, None)
        else:
            eps_base_in = f.eps
            dp_base_in = dp_base
        cz_mask, eps_cz, dp_cz = detect_cohesive_zone(
            f.T_s, f.f_solid, eps_base_in, dp_base_in, p, mesh.active,
        )
        if p.layered_burden and self._ore_frac is not None:
            # Fu 2014 layered treatment: porosity / particle size follow
            # the static layer structure, with ore-layer fusion inside the
            # CZ (coke-slit gas channelling).  Same 30 % bounded update as
            # the homogeneous path for iteration stability.
            eps_lay, dp_lay = layered_bed_properties(self._ore_frac,
                                                     cz_mask, p)
            f.eps = 0.7 * f.eps + 0.3 * eps_lay
            f.dp_eff = 0.7 * f.dp_eff + 0.3 * dp_lay
        else:
            f.eps = 0.7 * f.eps + 0.3 * eps_cz
            f.dp_eff = 0.7 * f.dp_eff + 0.3 * dp_cz
        f.cohesive_mask = cz_mask

        # Raceway cavity: the tuyere band is a physical void blown into
        # the bed (penetration ~1 m).  Overriding eps/dp there AFTER the
        # CZ blend lets the blast distribute over the penetration
        # annulus instead of jetting up the last wall cell (which
        # over-heated the wall column and flattened the CZ crown).
        if p.raceway_cavity_eps > 0.0:
            rc = mesh.is_tuyere
            f.eps[rc] = p.raceway_cavity_eps
            f.dp_eff[rc] = p.coke_diameter

        rho_s_bulk = np.full_like(f.T_s, 1800.0) * (1.0 - f.eps)

        # ---- Reaction kinetics -------------------------------------------
        # Normalised height field z/H (0 at tuyere base, 1 at stockline)
        # broadcast to (nr, nz).  Drives the reduction-sequence height gates.
        z_norm = np.broadcast_to(
            (mesh.z_c / max(p.furnace_height, 1e-6))[None, :],
            (mesh.nr, mesh.nz),
        )
        # Effective solid-integration step this iteration (must match the
        # dt_override passed to update_solid_phases below -- the solid
        # availability caps inside compute_rates are per THIS step).
        dt_solid_eff = 200.0 if it > 10 else p.dt
        rates = compute_rates(
            f.T_g, f.T_s, f.P, f.rho_g, mw_mix, f.x_gas, f.f_solid,
            f.eps, mesh.active,
            z_norm=z_norm,
            feo_form_height_frac=p.feo_form_height_frac,
            feo_reduce_height_frac=p.feo_reduce_height_frac,
            dt_solid=dt_solid_eff,
        )
        # Reaction-source under-relaxation (standard segregated-solver
        # source damping, Patankar Ch. 7).  The strongly coupled
        # CO/CO2 pairs (R4 makes CO2 where R5 consumes it at hot coke)
        # can flip-flop cell-by-cell between outer iterations on fine
        # meshes -- each field looks bounded, but the time-averaged
        # source the species solver integrates exceeds what the flow
        # exports, and the top-gas composition drifts secularly
        # (observed at 20x120 beyond ~300 iterations: N2 55 -> 14%).
        # Blending the RATES between outer iterations damps the limit
        # cycle; at a true steady state prev == new, so the converged
        # physics is unchanged.
        # Soft start: treat the pre-first-iteration rates as zero so the
        # sources ramp in over the first ~10 iterations (1 - 0.7^n)
        # instead of spiking into the un-settled early flow field.
        lam_rate = 0.3
        if self._rates_prev is None:
            self._rates_prev = ReactionRates(**{
                k: np.zeros_like(getattr(rates, k))
                for k in rates.__dataclass_fields__})
        rates = ReactionRates(**{
            k: (1.0 - lam_rate) * getattr(self._rates_prev, k)
               + lam_rate * getattr(rates, k)
            for k in rates.__dataclass_fields__})
        self._rates_prev = rates
        species_src_mol = assemble_species_sources(rates)
        solid_src = assemble_solid_sources(rates)

        # ---- Liquid drainage below the cohesive zone ----------------------
        # Once burden softens (T_s > softening), Fe3O4 reduces to FeO and
        # FeO becomes liquid wustite/iron that drains by gravity to the
        # hearth.  In a continuum model this is represented as a SINK term
        # on the solid-phase mass equation: liquid leaves the cell faster
        # than the bulk descent rate.
        #
        # Without this drain, Fe3O4 and FeO accumulate in the lower half
        # of the furnace because the bulk descent velocity is much slower
        # than physical liquid drainage.  The result is that R3/R4 (which
        # depend on f_mag, f_wus) fire artificially in the tuyere zone
        # where T_g is high, consuming all the CO and producing CO2 at the
        # exit instead of leaving plant-realistic CO ~24%.
        #
        # Reference: Fu 2014 §2.1 assumption 4: "The layer volume maintains
        # constant during burden descending [above the CZ]" -- below the
        # CZ, liquid metal & slag flow with much higher velocity, and the
        # solid-phase continuum tracks only the coke skeleton.
        # Zhou 2016 R11 explicitly models "C + FeO(l) -> Fe(l) + CO" as a
        # liquid-phase reaction below the CZ.
        #
        # Implementation: above the softening temperature, drain rate is
        # proportional to the local liquid mass:
        #   dm_liq / dt = -k_drain * rho_solid * f_phase
        # with k_drain ~ 1/tau_drain, tau_drain ~ residence time of liquid
        # in the bosh (typical 10-30 minutes for metal, 1-2 hours for slag,
        # we use 600 s as a representative timescale).
        sigma = np.clip(
            (f.T_s - p.softening_temperature)
            / max(p.melting_temperature - p.softening_temperature, 1.0),
            0.0, 1.0,
        )
        tau_drain = 600.0   # s, characteristic liquid residence time
        # Drain Fe3O4 progressively as sigma -> 1.  FeO is NOT drained
        # here any more (Phase 9): molten wustite dripping through the
        # coke bed is REDUCED on the same timescale (Zhou 2016 R11,
        # C + FeO(l) -> Fe(l) + CO) -- that pathway now lives inside
        # kinetics.compute_rates as the liquid branch of R_DR, so it
        # consumes coke, produces CO, and books the Fe, instead of
        # deleting iron units from the ledger.
        for ph in ("Fe3O4",):
            if ph in solid_src:
                drain = -sigma * rho_s_bulk * f.f_solid[ph] / tau_drain
                solid_src[ph] = solid_src[ph] + drain
        # Note: we do not drain Fe (already metallic) the same way, since
        # in a continuum the Fe phase tracks both solid sponge iron and
        # carburized hot metal until the CZ; below CZ, hot metal flows
        # downward as liquid, which we represent by leaving Fe to be
        # advected normally (the "kinematic descent" approximation).
        # We similarly do not drain coke (drains via R5/R6/R_DR consumption).

        # Net mass source from reactions [kg/(m^3 s)]
        species_src_mass = np.zeros_like(f.T_g)
        for sp, S in species_src_mol.items():
            species_src_mass += S * MW[sp]

        # Transient source limiter for the PRESSURE system only.  Cold-
        # start initial fields hold the whole column's flux/coke/CO2
        # inventory at reactive temperatures, so the first iterations
        # can flash hundreds of kg/s of phantom gas (steady-state net
        # source is ~15-20% of blast).  Feeding that flash into the
        # pressure Poisson inflates the bed pressure several-fold before
        # the fields relax.  Rescale the source seen by momentum to a
        # physical bound; species/energy keep the full (rate-relaxed)
        # sources, and at steady state the limiter is inactive.
        m_src_tot = float(
            (species_src_mass * mesh.vol)[mesh.active].sum())
        m_src_cap = 0.5 * self.raceway.total_blast_kg_s
        if m_src_tot > m_src_cap:
            species_src_mass = species_src_mass * (m_src_cap / m_src_tot)

        # ---- Coupled momentum block solve --------------------------------
        mom_sol: MomentumSolution = solve_coupled_momentum(
            u_r_old=f.u_r, u_z_old=f.u_z, P_old=f.P,
            rho=f.rho_g, mu=mu_g, eps=f.eps, dp_eff=f.dp_eff,
            mass_source=species_src_mass,
            mesh=mesh, p=p, raceway=self.raceway,
            relaxation=self._omega_mom,
            gas_exit_mask=self._gas_exit_mask,
        )
        f.u_r, f.u_z, f.P = mom_sol.u_r, mom_sol.u_z, mom_sol.P

        # ---- Solid descent (kinematic) -----------------------------------
        u_s_descent = compute_solid_descent_velocity(mesh, p, rho_s_bulk)

        # ---- Species transport -------------------------------------------
        f.x_gas, sp_residuals = solve_species(
            x_gas=f.x_gas, rho_g=f.rho_g, mw_mix=mw_mix,
            u_r=f.u_r, u_z=f.u_z, eps=f.eps,
            sources_mol=species_src_mol,
            raceway_mask_gas=self._raceway_mask_cells,
            raceway_composition=self.raceway.x_gas_blast,
            mesh=mesh, p=p,
            relaxation=self._omega_species,
        )

        # ---- Solid phase update ------------------------------------------
        burden_inlet = (self._inlet_radial
                        if self._inlet_radial is not None
                        else burden_inlet_composition(p))
        f.f_solid, sol_residuals = update_solid_phases(
            f_solid=f.f_solid, rho_s_bulk=rho_s_bulk,
            solid_sources=solid_src, u_s_descent=u_s_descent,
            mesh=mesh, p=p,
            burden_inlet_composition=burden_inlet,
            relaxation=self._omega_solid,
            # Solid-phase continuation: at dt=0.5s the burden moves ~5e-4 m
            # per iteration, so thousands of iterations would be needed for
            # fresh burden to reach the reduction zones.  Once flow is
            # converged, ramp dt_solid up so each outer iter advances the
            # solid composition by a meaningful amount of plant time.
            # dt_solid = 200 s gives ~0.2 m of descent per iteration, so
            # the column flushes through in ~140 iters (close to plant
            # residence time of 5 hours / 140 iters = ~130 s/iter).
            dt_override=dt_solid_eff,
        )

        # ---- Coupled energy ----------------------------------------------
        # Energy continuation: keep dt small early when flow is unstable,
        # ramp up to large dt for steady-state pursuit once continuity
        # and momentum residuals are below 1e-3.  This is critical because
        # at small dt the time-derivative term anchors T to its previous
        # value, blocking the heating that should come from raceway
        # combustion.
        flow_stable = (
            mom_sol.residual_continuity < 1e-3 and
            mom_sol.residual_r_mom < 1e-3 and
            mom_sol.residual_z_mom < 1e-3
        )
        if flow_stable and it > 5:
            # Once flow is converged, use a very large effective dt to
            # drive energy to steady state quickly.  At dt_energy=500s
            # the temporal-derivative anchor is ~1000x weaker than at
            # dt=0.5, so the energy equation converges to its true
            # steady-state in ~10-30 outer iterations rather than
            # thousands.
            dt_energy = max(p.dt, 500.0)
        else:
            dt_energy = p.dt              # standard pseudo-transient

        T_g_new, T_s_new, r_T_g, r_T_s = solve_energy(
            T_g=f.T_g, T_s=f.T_s,
            u_r=f.u_r, u_z=f.u_z,
            rho_g=f.rho_g, rho_s_bulk=rho_s_bulk,
            cp_g=cp_g, cp_s=cp_s,
            eps=f.eps, dp_eff=f.dp_eff,
            solid_descent_velocity=u_s_descent,
            rates=rates, mesh=mesh, p=p,
            raceway_T_gas=self._raceway_T_field,
            raceway_mask_gas=self._energy_T_dirichlet_mask,  # raceway cells: T = T_blast
            relaxation=self._omega_energy,
            dt_energy=dt_energy,
            tuyere_cell_mask=self._raceway_mask_cells,
            T_tuyere_inlet=p.blast_temperature,
        )
        f.T_g, f.T_s = T_g_new, T_s_new

        # ---- Build residual snapshot -------------------------------------
        snap = ResidualSnapshot(
            iteration=it,
            continuity=l2_normalised(
                np.array([mom_sol.residual_continuity]),
                self.refs.continuity_scale(),
            ),
            r_momentum=l2_normalised(
                np.array([mom_sol.residual_r_mom]),
                self.refs.momentum_scale(),
            ),
            z_momentum=l2_normalised(
                np.array([mom_sol.residual_z_mom]),
                self.refs.momentum_scale(),
            ),
            gas_energy=l2_normalised(
                np.array([r_T_g]), self.refs.gas_energy_scale(),
            ),
            solid_energy=l2_normalised(
                np.array([r_T_s]), self.refs.solid_energy_scale(),
            ),
            species_max=max(
                l2_normalised(np.array([v]), self.refs.species_scale())
                for v in sp_residuals.values()
            ),
            solid_max=max(
                l2_normalised(np.array([v]), self.refs.solid_species_scale())
                for v in sol_residuals.values()
            ),
            species_detail={
                k: l2_normalised(np.array([v]), self.refs.species_scale())
                for k, v in sp_residuals.items()
            },
            solid_detail={
                k: l2_normalised(np.array([v]), self.refs.solid_species_scale())
                for k, v in sol_residuals.items()
            },
        )
        return snap

    # ---- Adaptive relaxation ----------------------------------------------
    def _adapt_relaxation(self) -> None:
        """Bump omega up when residuals fall, down when they rise.

        Rule (per iteration, once we have 5+ snapshots):
          ratio = current_worst / previous_worst
          ratio < 0.99   -> residuals falling, raise omega 5% (capped at 0.9)
          ratio > 1.01   -> residuals rising, drop omega 30% (floor at 0.05)
          otherwise stalled -> hold
        """
        if len(self.residuals) < 5:
            return
        snaps = self.residuals._snaps      # internal but ok
        cur = snaps[-1].max()
        prev = snaps[-2].max()
        if prev < 1e-30:
            return
        ratio = cur / prev
        if ratio > 1.01:
            # Residual rising: drop omega aggressively
            self._omega_mom    = max(self._omega_lo, self._omega_mom    * 0.7)
            self._omega_energy = max(self._omega_lo, self._omega_energy * 0.7)
            self._omega_species = max(self._omega_lo, self._omega_species * 0.7)
        elif ratio < 0.99:
            # Residual falling cleanly: raise omega gently
            self._omega_mom    = min(self._omega_hi, self._omega_mom    * 1.05)
            self._omega_energy = min(self._omega_hi, self._omega_energy * 1.05)
            self._omega_species = min(self._omega_hi, self._omega_species * 1.05)
        # else: hold

    # ---- Diagnostics -------------------------------------------------------
    def _log_iteration(self, snap: ResidualSnapshot) -> None:
        f = self.fields
        active = self.mesh.active
        Tmax = float(np.max(f.T_g[active]))
        Tmin = float(np.min(f.T_g[active]))
        Fe = float(np.mean(f.f_solid["Fe"][active])) * 100
        cz_frac = (float(self.fields.cohesive_mask.sum())
                   / max(int(active.sum()), 1))
        self.log.info(
            "iteration",
            extra={
                "it": snap.iteration,
                "res_worst":   round(snap.max(), 6),
                "res_cont":    round(snap.continuity, 6),
                "res_r_mom":   round(snap.r_momentum, 6),
                "res_z_mom":   round(snap.z_momentum, 6),
                "res_T_g":     round(snap.gas_energy, 6),
                "res_T_s":     round(snap.solid_energy, 6),
                "res_species": round(snap.species_max, 6),
                "res_solid":   round(snap.solid_max, 6),
                "T_g_max_K":   round(Tmax, 1),
                "T_g_min_K":   round(Tmin, 1),
                "Fe_mean_%":   round(Fe, 2),
                "cz_frac":     round(cz_frac, 3),
                "omega_mom":   round(self._omega_mom, 3),
            },
        )

    # ---- Results -----------------------------------------------------------
    def _package_results(self) -> Dict:
        f = self.fields
        mesh = self.mesh
        return {
            "r_c": mesh.r_c, "z_c": mesh.z_c,
            "active": mesh.active,
            "T_g": f.T_g, "T_s": f.T_s, "P": f.P,
            "u_r": f.u_r, "u_z": f.u_z,
            "x_gas": {sp: v.copy() for sp, v in f.x_gas.items()},
            "f_solid": {ph: v.copy() for ph, v in f.f_solid.items()},
            "eps": f.eps, "dp_eff": f.dp_eff,
            "rho_g": f.rho_g,
            "cohesive_mask": f.cohesive_mask,
            "raceway_T_flame": self.raceway.T_blast,
            "history": self.residuals.as_list(),
            "n_iterations": len(self.residuals),
            "final_residuals": (self.residuals.latest().to_dict()
                                if self.residuals.latest() else {}),
        }
