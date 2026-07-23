"""
Operating-parameter container for the 2D axisymmetric blast-furnace solver.

All fields are validated on construction.  A single ``BFParameters2D``
instance is the only configuration object that downstream modules accept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

from .constants import (
    T_MIN_PHYSICAL, T_MAX_PHYSICAL, P_MIN_PHYSICAL, P_MAX_PHYSICAL,
)


class ParameterValidationError(ValueError):
    """Raised when input parameters violate physical or numerical constraints."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ParameterValidationError(msg)


@dataclass(frozen=True)
class BFParameters2D:
    """Immutable operating-condition container.

    All temperatures in Kelvin, flow rates in SI unless stated.
    The dataclass is frozen to guarantee reproducibility: pass a new instance
    to restart a case rather than mutating state.
    """

    # ----- Geometry (axisymmetric r-z) --------------------------------------
    furnace_height: float = 28.0     # m tuyere to stock line
    hearth_radius: float = 4.5       # m at tuyere level
    bosh_radius: float = 5.6         # m widest section
    belly_radius: float = 5.4        # m above bosh
    stack_top_radius: float = 4.2    # m at throat
    hearth_height: float = 3.5       # m hearth band
    bosh_height: float = 3.5         # m bosh band
    belly_height: float = 2.5        # m belly
    # remaining height = stack

    # ----- Mesh --------------------------------------------------------------
    nr: int = 40                     # radial cells
    nz: int = 120                    # axial cells
    mesh_stretch_r: float = 1.0      # >1 clusters near wall
    mesh_stretch_z: float = 1.05     # clusters near tuyere

    # ----- Blast (tuyere inlet = raceway exit conditions) -------------------
    # The "blast" boundary in this model represents the gas LEAVING the
    # raceway (post-PCI/coke combustion), not the actual hot blast entering
    # the tuyeres.  Fu et al. (2014) Table 5 reports 2300 K as the
    # "averaging bosh gas temperature" — i.e., the gas T at the cell
    # immediately above the raceway, after coke combustion.  This is the
    # correct value to use here because the model does not resolve the
    # raceway combustion explicitly (blast composition has zero O2).
    #
    # Setting blast_temperature = 1473 K (the actual hot-blast T) would
    # leave the model missing ~700 K of combustion sensible heat, as it
    # only sees the post-raceway gas without the heat-release that gets
    # the gas to that high T in the real furnace.
    # ----- Plant operating point (Fu 2014, Table 5) --------------------------
    # ALL burden/blast rates below are anchored to this productivity so the
    # per-tHM ratios match the validation plant:
    #   productivity 4993 tHM/day  ->  P = 0.05779 tHM/s
    #   ore   1501 kg/tHM -> 86.7 kg/s      coke 411 kg/tHM -> 23.8 kg/s
    #   flux   112 kg/tHM ->  6.5 kg/s      PCI   67 kg/tHM (per-tHM param)
    #   blast 1050 Nm3/tHM -> 3640 Nm3/min  (typical PCI operation
    #                                        900-1150 Nm3/tHM; Fu does not
    #                                        publish blast rate directly)
    # The previous defaults (6500 Nm3/min with 38 kg/s ore) implied
    # 4279 Nm3/tHM — 4x oversized blast per tonne — which structurally
    # forced top-gas N2 high and CO2/H2 low regardless of kinetics.
    productivity_tHM_day: float = 4993.0
    blast_rate_Nm3_min: float = 3640.0
    blast_temperature: float = 2273.15   # K (2000 degC = bosh gas T per Fu 2014)
    oxygen_enrichment: float = 0.04      # fractional (beyond 0.21) — kept
                                          # for downstream use; blast composition
                                          # itself is post-combustion (no O2)
    # ----- Tuyere injectants (hydrogen-enrichment envelope) ------------------
    # Volume fractions of the DRY blast replaced by injected gas.  Secure
    # defaults: 0.0 (plain hot blast + PCI).  Validated bounds keep the
    # raceway oxygen ledger physical.  NG is treated as CH4 undergoing
    # partial oxidation in the raceway (CH4 + 1/2 O2 -> CO + 2 H2), which
    # consumes blast O2; H2 injection passes through to the bosh gas.
    blast_h2_vol_frac: float = 0.0       # injected H2, fraction of dry blast
    blast_ng_vol_frac: float = 0.0       # injected natural gas (as CH4)
    blast_moisture_gpm3: float = 25.0    # g H2O per Nm3 (humidified blast;
                                          # plant range 15-30 g/Nm3).  Each
                                          # mol of blast moisture -> 1 mol H2
                                          # in bosh gas via C + H2O -> CO + H2.
    pci_rate_kg_tHM: float = 67.0        # pulverized-coal injection
                                          # (Fu 2014 plant value, 4993 MTHM/d)
    # Fraction of blast moisture that leaves the raceway UNREACTED as
    # H2O vapour (the 0D raceway balance previously converted ALL
    # moisture to CO + H2 via R6, which starved the shaft's in-situ
    # C + H2O -> CO + H2 and let reverse water-gas-shift strip H2 in the
    # hot zone -- diagnosed root cause of the top-gas H2 deficit).
    # Plant anchor: bosh gas retains a few % H2O; Zhou et al. 2016 (JOM
    # 68(5)) show raceway H2O still reacting with coke as the gas
    # travels up the bosh (their Fig. 5 discussion).  Validated [0, 1];
    # 0.0 reproduces the pre-Phase-9 all-reacted behaviour.
    raceway_h2o_slip: float = 0.35
    # Hydrogen content of charged coke (ultimate analysis; typical
    # metallurgical coke 0.4-0.7 wt% H).  Devolatilized on heating and
    # exits in the top gas; without it the plant top-gas H ledger cannot
    # close (moisture + PCI H alone < plant top-gas H2 6.3% + H2O 1%).
    # Released into the bosh gas in the raceway 0D balance (location
    # approximation: H2 is conserved once formed).  Set 0 to disable.
    coke_h_mass_frac: float = 0.005
    n_tuyeres: int = 32
    tuyere_elevation: float = 0.5        # m above grid base
    tuyere_penetration_fraction: float = 0.22   # fraction of hearth radius
    tuyere_diameter: float = 0.13        # m, individual tuyere bore diameter
                                          # (used to compute physical inlet
                                          # area independent of mesh resolution)

    # ----- Top (burden charge) — Fu 2014 per-tHM rates at 4993 tHM/day -------
    coke_rate_kg_s: float = 23.75        # 411 kg/tHM (Fu Table 7 coke rate)
    ore_rate_kg_s: float = 86.7          # 1501 kg/tHM (Fu Table 5)
    flux_rate_kg_s: float = 6.47         # 112 kg/tHM (Fu Table 5); carbonate
                                          # flux (CaCO3/MgCO3) that calcines
                                          # in-shaft via R9 -> MeO + CO2
    burden_inlet_temperature: float = 323.15   # K
    top_pressure: float = 2.5e5          # Pa (2.5 bar abs)

    # ----- Gas exit (uptake offtakes) ---------------------------------------
    # Real BF: gas exits through 4 uptake ducts at the throat top, total
    # exit area ~6-12 m^2 (NOT the entire stockline cross-section, which is
    # sealed by the burden charging system).  In the axisymmetric solver we
    # represent this as an annular ring of cells at the top (j = nz-1) with
    # radii in [gas_exit_r_min, gas_exit_r_max].  Cells outside this ring at
    # the top row are treated as sealed (no-flux).
    gas_exit_r_min: float = 0.0          # m, inner radius of the exit ring
    gas_exit_r_max: float = 4.2          # m, outer radius of the exit ring
                                          # (default = stack_top_radius =
                                          # full top, kept for backwards compat)

    # ----- Reduction-sequence height gating ---------------------------------
    # The iron reduction proceeds stepwise DOWN the furnace:
    #   Fe2O3 -> Fe3O4 -> FeO -> Fe
    # and each step occupies a distinct vertical zone (Fu et al. 2014,
    # Fig. 7 & 9; Austin, Nogami & Yagi 1997).  Because a coupled 2D
    # continuum model can smear these zones vertically (the temperature
    # field alone does not enforce the crisp ordering seen in plant data),
    # we add explicit height gates on the two reactions that plant data
    # shows are confined to the LOWER furnace:
    #
    #   * FeO FORMATION  (R3, R3H: Fe3O4 -> FeO) begins only once the
    #     burden has descended past `feo_form_height_frac` of the total
    #     height (measured from the TOP).  Above that, only Fe2O3 -> Fe3O4
    #     occurs.  Fu 2014: "wustite starts to generate from 800 K".
    #
    #   * FeO REDUCTION  (R4, R4H, R_DR: FeO -> Fe) occurs only in the
    #     lower `feo_reduce_height_frac` of the furnace, near/below the
    #     cohesive zone.  Fu 2014: "[FeO] is not reduced until near the CZ.
    #     Below the CZ, almost all the wustite has been converted to iron".
    #
    # Fractions are measured as z/H where z=0 is the tuyere base (bottom)
    # and z=H is the stockline (top).  A gate value of 1/3 means "only in
    # the bottom third".  Set to 1.0 to disable a gate (reaction allowed
    # at any height, recovering the pure temperature-gated behaviour).
    feo_form_height_frac: float = 0.67   # FeO forms only below this z/H
                                          # (i.e. after descending through
                                          #  the top third) -- default 2/3
    feo_reduce_height_frac: float = 0.33 # FeO->Fe only below this z/H
                                          # (bottom third, near/below the
                                          #  cohesive zone). Fu 2014: wustite
                                          #  "not reduced until near the CZ".

    # ----- Layered burden (Fu 2014 cell-type technique) -----------------------
    # Opt-in: alternating horizontal ore/coke layers with per-cell porosity
    # and particle size, including the CZ ore-fusion / coke-slit mechanism.
    # Defaults off (homogeneous bed) until validated against Nagarnar
    # charging-matrix data.  Layer thicknesses: Fu 2014 plant range 0.2-0.8 m.
    layered_burden: bool = False
    ore_layer_thickness_m: float = 0.35
    coke_layer_thickness_m: float = 0.45

    # ----- Radial burden distribution (inverted-V cohesive zone) -------------
    # Plant charging places more ore near the WALL and more coke at the
    # CENTER (center-working practice), which channels hot gas up the axis
    # and produces the inverted-V cohesive zone (Fu 2014 Sect. 2.4 burden
    # distribution model; Zhou 2016 Fig. 2).  The local ore fraction is
    # weighted radially:
    #     w(r) = clip(w0 + (w1 - w0) * (r / R_wall(z))^q, 0, 1.6)
    # with w0 = 1 - bias (axis), w1 = 1 + bias (wall), then renormalized by
    # its area-weighted mean per row so the COLUMN-AVERAGE ore fraction (and
    # hence the global mass balance / burden_inlet_composition) is preserved.
    # Secure default 0.0 = radially uniform = pre-Task-1 behaviour.
    # Sensible plant range 0.2-0.5.
    burden_radial_ore_bias: float = 0.0
    burden_radial_ore_exponent: float = 2.0   # q in w(r); shape of the ramp

    # ----- Radial solid-descent factor (Fu 2014 Sect. 3.1) -------------------
    # Solids descend faster near the wall where raceway coke consumption
    # pulls the burden down; Fu uses a relative descent-speed factor varying
    # linearly from 0.8 at the center to 1.0 at the wall:
    #     Ks(r) = ks_center + (1 - ks_center) * r / R_wall(z)
    # normalized by its area-weighted mean per z-plane so the total mass
    # flow through every plane is conserved.  ks_center = 1.0 reproduces
    # radially uniform plug descent.  Default 0.8 per Fu 2014 Sect. 3.1.
    ks_center: float = 0.8

    # ----- Bed / packing ----------------------------------------------------
    coke_diameter: float = 0.045         # m
    ore_diameter: float = 0.018          # m
    void_fraction: float = 0.42
    ore_to_coke_volume_ratio: float = 4.5
    ore_fe_content_mass: float = 0.62    # mass fraction Fe in burden ore
    wall_heat_loss_W_m2: float = 9000.0  # DEPRECATED (kept for backward
                                          # compat); the wall BC now uses
                                          # the explicit Robin parameters
                                          # below.
    # ----- Wall Robin BC (Austin, Nogami & Yagi 1997, ISIJ 37(8) p.751:
    # q_wall = h_wall (T_ave - T_wall) with h_wall = 20 W/m2K, T_wall =
    # 298 K, flux split 50/50 between gas and solid).  The previous
    # implicit derivation (9000/800 = 11.25 W/m2K to 373 K) under-cooled
    # the bosh-corner by ~2x, which let the corner rise past softening
    # and broke the inverted-V silhouette (wall shoulders).
    wall_htc_W_m2K: float = 20.0
    wall_coolant_T_K: float = 298.15
    # ----- Raceway cavity porosity ------------------------------------------
    # The raceway is a physical CAVITY blown into the coke bed, extending
    # ~tuyere_penetration_fraction x hearth_radius inward (~1 m; plant
    # raceway depth 1-1.5 m).  Injecting the blast through wall faces
    # into a packed-bed cell makes the jet hug the wall and overheat the
    # wall column (flattening the cohesive-zone crown).  Treating the
    # raceway band (mesh.is_tuyere region) as a high-void cavity lets the
    # inflow distribute over the penetration annulus before entering the
    # bed.  Set to 0.0 to disable (packed bed everywhere, legacy).
    raceway_cavity_eps: float = 0.80

    # ----- Cohesive zone model ---------------------------------------------
    softening_temperature: float = 1373.15   # K (1100 degC)
    melting_temperature: float = 1673.15     # K (1400 degC)
    cz_porosity_min: float = 0.08
    cz_porosity_coke_window: float = 0.40    # coke slits stay porous

    # ----- Numerics (PISO + outer Picard) ----------------------------------
    dt: float = 0.25                      # s (pseudo-time / transient step)
    max_outer_iter: int = 2000
    max_piso_corrections: int = 2
    momentum_relaxation: float = 0.55
    pressure_relaxation: float = 0.35
    energy_relaxation: float = 0.55
    species_relaxation: float = 0.70
    solid_relaxation: float = 0.40
    convergence_rms: float = 5.0e-5
    steady_state_target: bool = True      # march pseudo-time to steady state

    # ----- Diagnostics ------------------------------------------------------
    log_every: int = 25
    save_every: int = 200
    output_dir: str = "output"
    case_name: str = "bf2d_run"

    # ----- Derived bookkeeping ---------------------------------------------
    _validated: bool = field(default=True, init=False, repr=False)

    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Reject NaN / Inf on every numeric field (fuzzing defence)
        for f_name in self.__dataclass_fields__:
            value = getattr(self, f_name)
            if isinstance(value, float):
                _check(math.isfinite(value),
                       f"{f_name} must be finite (got {value!r})")
        # Geometry sanity
        _check(self.furnace_height > 5.0, "furnace_height must be > 5 m")
        _check(self.nr >= 8 and self.nz >= 20,
               "mesh too coarse: need nr>=8, nz>=20")
        _check(self.nr <= 500 and self.nz <= 2000,
               "mesh too fine: guard against accidental runaway allocation")
        _check(0.0 < self.void_fraction < 0.9, "void_fraction must be in (0,0.9)")
        _check(self.bosh_radius >= self.hearth_radius,
               "bosh_radius must be >= hearth_radius")
        _check(self.bosh_radius >= self.stack_top_radius,
               "bosh_radius must be >= stack_top_radius")

        # Operating conditions
        _check(T_MIN_PHYSICAL <= self.blast_temperature <= T_MAX_PHYSICAL,
               f"blast_temperature out of physical range "
               f"[{T_MIN_PHYSICAL},{T_MAX_PHYSICAL}]")
        _check(0.0 <= self.oxygen_enrichment <= 0.5,
               "oxygen_enrichment must be in [0,0.5]")
        _check(0.0 <= self.blast_h2_vol_frac <= 0.30,
               "blast_h2_vol_frac must be in [0,0.30]")
        _check(0.0 <= self.blast_ng_vol_frac <= 0.30,
               "blast_ng_vol_frac must be in [0,0.30]")
        _check(self.blast_h2_vol_frac + self.blast_ng_vol_frac <= 0.35,
               "combined H2+NG injection must be <= 0.35 of blast")
        _check(self.productivity_tHM_day > 0.0,
               "productivity_tHM_day must be positive")
        _check(0.0 <= self.raceway_h2o_slip <= 1.0,
               "raceway_h2o_slip must be in [0, 1]")
        _check(0.0 <= self.coke_h_mass_frac <= 0.02,
               "coke_h_mass_frac must be in [0, 0.02] "
               "(typical met coke 0.004-0.007)")
        _check(0.05 <= self.ore_layer_thickness_m <= 2.0,
               "ore_layer_thickness_m must be in [0.05, 2.0]")
        _check(0.05 <= self.coke_layer_thickness_m <= 2.0,
               "coke_layer_thickness_m must be in [0.05, 2.0]")
        _check(0.0 <= self.burden_radial_ore_bias <= 0.6,
               "burden_radial_ore_bias must be in [0, 0.6] "
               "(0 = radially uniform; plant range 0.2-0.5)")
        _check(0.5 <= self.burden_radial_ore_exponent <= 5.0,
               "burden_radial_ore_exponent must be in [0.5, 5.0]")
        _check(0.5 <= self.ks_center <= 1.0,
               "ks_center must be in [0.5, 1.0] "
               "(1.0 = uniform plug descent; Fu 2014 uses 0.8)")
        _check(0.0 <= self.wall_htc_W_m2K <= 200.0,
               "wall_htc_W_m2K must be in [0, 200] "
               "(Austin/Nogami/Yagi 1997 use 20)")
        _check(250.0 <= self.wall_coolant_T_K <= 600.0,
               "wall_coolant_T_K must be in [250, 600] K")
        _check(self.raceway_cavity_eps == 0.0
               or 0.5 <= self.raceway_cavity_eps <= 0.95,
               "raceway_cavity_eps must be 0.0 (off) or in [0.5, 0.95]")
        _check(self.blast_rate_Nm3_min > 0.0, "blast_rate must be positive")
        _check(P_MIN_PHYSICAL <= self.top_pressure <= P_MAX_PHYSICAL,
               "top_pressure out of physical range")
        _check(self.coke_rate_kg_s > 0 and self.ore_rate_kg_s > 0,
               "feed rates must be positive")
        _check(0.0 < self.tuyere_penetration_fraction < 1.0,
               "tuyere_penetration_fraction must be in (0,1)")
        _check(1 <= self.n_tuyeres <= 80, "n_tuyeres implausible")
        _check(0.05 <= self.tuyere_diameter <= 0.5,
               "tuyere_diameter must be 0.05-0.5 m (typical 0.10-0.20 m)")
        _check(0.0 < self.feo_form_height_frac <= 1.0,
               "feo_form_height_frac must be in (0,1]")
        _check(0.0 < self.feo_reduce_height_frac <= 1.0,
               "feo_reduce_height_frac must be in (0,1]")
        _check(self.feo_reduce_height_frac <= self.feo_form_height_frac,
               "feo_reduce_height_frac must be <= feo_form_height_frac "
               "(FeO must form higher up than where it is reduced)")

        # Numerics
        _check(self.dt > 0.0, "dt must be positive")
        _check(1 <= self.max_piso_corrections <= 5,
               "max_piso_corrections must be in [1,5]")
        for rel in (self.momentum_relaxation, self.pressure_relaxation,
                    self.energy_relaxation, self.species_relaxation,
                    self.solid_relaxation):
            _check(0.05 <= rel <= 1.0, "relaxation factor out of range [0.05,1]")
        _check(self.convergence_rms > 0, "convergence_rms must be positive")

    # -----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation (°C alongside K for readability)."""
        d = asdict(self)
        d["blast_temperature_C"] = self.blast_temperature - 273.15
        d["burden_inlet_temperature_C"] = self.burden_inlet_temperature - 273.15
        d["softening_temperature_C"] = self.softening_temperature - 273.15
        d["melting_temperature_C"] = self.melting_temperature - 273.15
        return d
