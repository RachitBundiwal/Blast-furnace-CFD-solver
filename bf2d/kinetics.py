"""
Arrhenius reaction kinetics for the blast-furnace solver.

Reactions modelled
------------------
R1  : C(s) + O2(g)    -> CO2(g)                     (raceway combustion)
R1b : C(s) + 1/2 O2   -> CO(g)                      (partial combustion)
R2  : 3 Fe2O3 + CO    -> 2 Fe3O4 + CO2             (indirect, hematite)
R3  : Fe3O4 + CO      -> 3 FeO + CO2               (indirect, magnetite)
R4  : FeO + CO        -> Fe + CO2                  (indirect, wustite)
R2H : 3 Fe2O3 + H2    -> 2 Fe3O4 + H2O             (H2 reduction)
R3H : Fe3O4 + H2      -> 3 FeO + H2O
R4H : FeO + H2        -> Fe + H2O
R5  : C + CO2         -> 2 CO                      (Boudouard / solution loss)
R6  : C + H2O         -> CO + H2                   (water-gas)
R7  : CO + H2O        <-> CO2 + H2                 (water-gas shift)
R_DR: FeO + C         -> Fe + CO                   (direct reduction, >1273 K)

All rate expressions return volumetric molar rates  r_i  [mol / (m^3 s)]
and can be assembled into gas and solid species source terms by
``assemble_sources``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .constants import GAS_SPECIES, MW, R_GAS, EPS_MOLE
from .parameters import BFParameters2D


# ---------------------------------------------------------------------------
@dataclass
class ReactionRates:
    """Volumetric reaction rates stored as ``(nr, nz)`` arrays."""
    R1: np.ndarray
    R1b: np.ndarray
    R2: np.ndarray
    R3: np.ndarray
    R4: np.ndarray
    R2H: np.ndarray
    R3H: np.ndarray
    R4H: np.ndarray
    R5: np.ndarray
    R6: np.ndarray
    R7: np.ndarray
    R_DR: np.ndarray
    R9: np.ndarray   # flux calcination CaCO3 -> CaO + CO2


# Activation energies E_A [J/mol] -- consensus literature values,
# adjusted (R6, R7) for plant-validation match against Fu et al. 2014
# top-gas composition (CO 24%, H2 6%, CO2 22%, H2O 1%) on a 4993 MTHM/d
# furnace.
E_A = {
    "R1":   4.52e4,
    "R1b":  4.20e4,
    "R2":   3.30e4,    # phase 5: kept; cold-zone shutoff via explicit T cutoff in rate
    "R3":   7.00e4,
    "R4":   7.00e4,
    "R2H":  3.80e4,
    "R3H":  8.00e4,
    "R4H":  8.00e4,
    "R5":   1.90e5,
    "R6":   1.60e5,    # was 1.9e5 -- lowered to give plant-realistic
                       #   H2 generation (~200 mol/s) from blast moisture
    "R7":   7.00e4,    # original literature value (Yagi-Austin)
    "R_DR": 1.40e5,
    "R9":   1.87e5,    # calcite calcination (lit. E_A 170-190 kJ/mol)
}

# Pre-exponential factors (units such that r = k * c * (phase fraction))
# Calibrated for plant 4993 MTHM/d operation (Fu 2014 case 1)
#
# Phase 4 calibration (this session):
# After fixing the C_k = ρ x / MW_species bug (was MW_mix^-1 not MW_species^-1),
# the Arrhenius rates are physically scaled.  But Yagi-style A0 values from
# the literature are calibrated against EFFECTIVE volumetric rates that
# include the (3/r_p) × ε_solid factor.  Our cell-volume rates already
# include the porosity/area through the cell volume, so the bare A0 values
# need to be scaled down by ~50× to match plant rates of 1-10 mol/(m^3 s)
# in the indirect-reduction zone.
#
# Reference rates (Fu 2014, Yagi 1993):
#   R3 typical peak: 5-8 mol/(m^3 s)  at T ≈ 1100 K
#   R4 typical peak: 3-5 mol/(m^3 s)  at T ≈ 1000 K
#   R5 (Boudouard): 0.5-2 mol/(m^3 s) at T_s > 1200 K
#   R6 (water-gas): 0.1-0.5 mol/(m^3 s) at T_s > 1100 K
A0 = {
    "R1":   2.50e3,    # tuyere combustion -- nominal value
    "R1b":  5.00e3,
    "R2":   2.00e1,    # phase 5 calib: lower A0 + explicit T_g > 1073K cutoff
                       # plant peak 5-8 mol/(m^3 s) (Fu 2014)
    "R3":   1.00e3,    # phase 9: restored to the pre-phase-5 value.  The
                       # /2 halving was calibrated at the operating point
                       # with 4x-oversized blast per tHM (same argument
                       # as the phase-7 R_DR restoration).  R3 remains
                       # equilibrium-limited + availability-capped +
                       # feo_form-gated.  Plant peak 5-8 mol/(m^3 s).
    "R4":   5.00e3,    # phase 9: restored pre-phase-5 value (see R3).
                       # plant peak 3-5 mol/(m^3 s)
    "R2H":  5.00e0,    # phase 4 -- already calibrated
    "R3H":  1.00e2,    # phase 4
    "R4H":  5.00e2,    # phase 4
    "R5":   4.00e6,    # phase 9: re-anchored WITH the Boudouard
                       # equilibrium driving force now in the rate (see
                       # compute_rates).  k5(1673 K) = 4e6*exp(-13.66)
                       # ~ 4.7 -> peak R5 ~ 1.4 mol/(m^3 s) at bosh
                       # conditions = the documented plant peak band
                       # (0.5-2).  The old 3e4 gave k ~ 0.035 (rate
                       # ~ 0.01) because it had to double as a fake
                       # equilibrium limiter for the mid-shaft.
    "R6":   1.60e6,    # phase 9: k6(1673 K) ~ 3.5 x k5(1673 K), the
                       # measured coke k_H2O/k_CO2 ratio (~3.5,
                       # PMC8153985); equilibrium-limited like R5.
    "R7":   2.00e0,    # phase 6 calib: was 1.0e1, /5.
                       # Homogeneous WGS is slow without a catalyst; in the
                       # BF gas residence time (~s) it only partially
                       # equilibrates.  A high rate makes reverse WGS
                       # (CO2+H2->CO+H2O) consume nearly all H2 in the hot
                       # zone, driving top H2 to ~2% vs plant 6.3%.  Plant
                       # H2 is consumed mainly by FeO+H2 (R4H), not WGS.
    "R_DR": 5.00e6,    # restored to literature value (phase 7).
                       # The earlier /10 reduction was calibrated at an
                       # operating point with 4x-oversized blast per tHM;
                       # at the Fu-consistent point the plant direct-
                       # reduction degree (35-45 %, RIST diagram) requires
                       # wustite reaching the CZ to be consumed by
                       # FeO + C -> Fe + CO.  R_DR remains availability-
                       # capped (cannot exceed local FeO / coke supply).
    "R9":   5.00e8,    # calcination: onset ~1050 K, availability-capped
                       # (flux cannot calcine faster than it is charged)
                       # plant peak 1-3 mol/(m^3 s) at CZ
}


# ---------------------------------------------------------------------------
def _arrhenius(A: float, Ea: float, T: np.ndarray) -> np.ndarray:
    """Standard Arrhenius rate constant."""
    return A * np.exp(-Ea / (R_GAS * np.maximum(T, 250.0)))


def _gas_concentration(rho: np.ndarray, mw_mix: np.ndarray,
                       x_sp: np.ndarray, mw_sp: float) -> np.ndarray:
    """Molar concentration of gas species  [mol/m^3]."""
    return np.maximum(rho * x_sp / mw_mix * (mw_mix / mw_sp), 0.0)
    # Equivalent simpler form: rho_gas * y_mass_sp / mw_sp; this form
    # also works given x_sp is mole-fraction.


def compute_rates(
    T_g: np.ndarray,
    T_s: np.ndarray,
    P: np.ndarray,
    rho_g: np.ndarray,
    mw_mix: np.ndarray,
    x_gas: Dict[str, np.ndarray],
    f_solid: Dict[str, np.ndarray],
    eps: np.ndarray,
    active: np.ndarray,
    z_norm: np.ndarray | None = None,
    feo_form_height_frac: float = 1.0,
    feo_reduce_height_frac: float = 1.0,
    dt_solid: float = 0.5,
) -> ReactionRates:
    """Evaluate all reaction rates at every active cell.

    Kinetic driving force uses the gas temperature for indirect-reduction
    reactions and solid temperature for direct reduction / Boudouard, as is
    standard in Yagi-type models.  A tiny regulariser prevents spurious
    amplification at cold start-up.

    Height gating (reduction sequence)
    ----------------------------------
    ``z_norm`` is the normalised height field z/H (0 at the tuyere base,
    1 at the stockline).  When supplied, two spatial gates enforce the
    plant-observed vertical ordering of the reduction sequence
    (Fu et al. 2014; Austin, Nogami & Yagi 1997):

      * FeO FORMATION (R3, R3H) is suppressed above ``feo_form_height_frac``
        -- i.e. wustite only starts to form once burden has descended past
        the top (1 - frac) of the furnace.
      * FeO REDUCTION (R4, R4H, R_DR) is suppressed above
        ``feo_reduce_height_frac`` -- i.e. wustite is reduced to iron only
        in the lower part of the furnace, near/below the cohesive zone.

    Both default to 1.0 (no gating), preserving the pure temperature-gated
    behaviour when ``z_norm`` is not provided.
    """
    shape = T_g.shape
    zero = np.zeros(shape, dtype=float)

    # --- Concentrations ---------------------------------------------------
    # x_gas stores mole fractions (the simulation orchestrator passes
    # the post-solve mole-fraction field).  Correct molar concentration
    # for ideal gas:
    #
    #     C_k = (P / RT) × x_k = ρ_total × x_k / MW_mix
    #
    # NOT  ρ_total × x_k / MW_species  (that formula is for mass-fraction
    # input, where Y_k = x_k × MW_k / MW_mix).
    #
    # The previous formula `ρ × x / MW_species` over-estimated H2 concentration
    # by MW_mix/MW_H2 ≈ 14×, driving R2H/R3H/R4H rates to runaway and
    # creating a feedback loop with R6 (water-gas) that ate all N2.
    mw_mix_safe = np.maximum(mw_mix, 1.0e-3)
    n_total = rho_g / mw_mix_safe   # mol/m^3 of gas mixture
    C_O2  = n_total * np.maximum(x_gas["O2"],  0.0)
    C_CO  = n_total * np.maximum(x_gas["CO"],  0.0)
    C_CO2 = n_total * np.maximum(x_gas["CO2"], 0.0)
    C_H2  = n_total * np.maximum(x_gas["H2"],  0.0)
    C_H2O = n_total * np.maximum(x_gas["H2O"], 0.0)

    # --- Phase fractions (guard against negatives) ------------------------
    f_hem = np.maximum(f_solid["Fe2O3"], 0.0)
    f_mag = np.maximum(f_solid["Fe3O4"], 0.0)
    f_wus = np.maximum(f_solid["FeO"],   0.0)
    f_coke = np.maximum(f_solid["C_coke"], 0.0)
    f_flux = np.maximum(f_solid.get("flux", np.zeros_like(f_coke)), 0.0)

    # ----------------------------------------------------------------------
    # Solid-phase availability factor.
    #
    # In real BF, the indirect-reduction reactions R2/R3/R4 (and their H2
    # counterparts) require *solid* iron oxides.  Once the burden softens
    # and melts (T_s > T_melt ~ 1673 K), the iron-oxide phases become
    # liquid and the reaction kinetics change drastically -- they react
    # via DIRECT reduction (R_DR: FeO(l) + C(s) -> Fe(l) + CO) instead of
    # indirect reduction.
    #
    # Without this factor, R3/R4 fire artificially in the tuyere band
    # (T_s > 2000 K) where Fe3O4 has been advected in by descent.  This
    # consumes all the freshly-made CO from R1/R1b, producing a top gas
    # of pure CO2 (no CO) -- inconsistent with plant values (CO ~24%).
    #
    # Reference: Fu 2014 §2.1; Zhou 2016 R11.
    T_soft = 1473.0   # K (typical iron-ore softening start)
    T_melt = 1673.0   # K (typical liquidus)
    sigma = np.clip((T_s - T_soft) / max(T_melt - T_soft, 1.0), 0.0, 1.0)
    solid_factor = 1.0 - sigma  # 1 below softening, 0 above melting
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Height gates for the reduction sequence (Fu et al. 2014; Austin 1997).
    #
    # z_norm is z/H in [0,1] with 0 at the tuyere base and 1 at the
    # stockline.  Two smooth gates enforce the plant-observed vertical
    # ordering:
    #
    #   feo_form_gate   : 1 in the lower part (z_norm < feo_form_frac),
    #                     0 in the upper part.  Applied to R3 / R3H so that
    #                     wustite only STARTS to form once the burden has
    #                     descended below feo_form_frac of the height.
    #
    #   feo_reduce_gate : 1 in the lowest part (z_norm < feo_reduce_frac),
    #                     0 above.  Applied to R4 / R4H / R_DR so that
    #                     wustite is REDUCED to iron only near/below the
    #                     cohesive zone.
    #
    # We use a smooth tanh transition (width ~5% of height) rather than a
    # hard step, to avoid a discontinuous source term that would upset the
    # segregated iteration.  When z_norm is None the gates are unity
    # everywhere (feature disabled).
    # ----------------------------------------------------------------------
    if z_norm is not None:
        width = 0.03  # transition half-width in z/H units (sharper gate
                      # reduces FeO->Fe leakage above the cohesive zone;
                      # at frac + 2*width the gate is < 2%)
        # gate = 1 where z_norm < frac, ramping to 0 above.
        feo_form_gate = 0.5 * (1.0 - np.tanh((z_norm - feo_form_height_frac) / width))
        feo_reduce_gate = 0.5 * (1.0 - np.tanh((z_norm - feo_reduce_height_frac) / width))
    else:
        feo_form_gate = np.ones_like(T_g)
        feo_reduce_gate = np.ones_like(T_g)

    # --- Cohesive-zone gas-solid contact cutoff for INDIRECT reduction ---
    # Once the ore softens (T_s > ~1373 K, Fu 2014 CZ upper boundary),
    # the ore layers fuse and become virtually impermeable: the reducing
    # gas can no longer percolate through the ore, so gas-solid indirect
    # reduction (R2-R4, R2H-R4H) shuts down.  Reduction of the remaining
    # (now molten) FeO proceeds by DIRECT contact with coke:
    #     R_DR:  FeO(l) + C(s) -> Fe(l) + CO
    # (Zhou 2016 reaction R11).  This switch-over is what sets the plant
    # direct-reduction degree (~35-45 %) and hence the top-gas CO/CO2
    # split -- without it the model over-runs indirect reduction and
    # over-utilises CO (eta_CO ~ 70 % vs plant ~48 %).
    T_soften = 1373.0
    cz_contact_gate = 0.5 * (1.0 - np.tanh((T_s - T_soften) / 40.0))
    # ----------------------------------------------------------------------

    # --- R1 + R1b : coke combustion (raceway) -----------------------------
    k1  = _arrhenius(A0["R1"],  E_A["R1"],  T_s)
    k1b = _arrhenius(A0["R1b"], E_A["R1b"], T_s)
    R1  = k1  * C_O2 * f_coke ** 0.5
    R1b = k1b * C_O2 * f_coke ** 0.5
    # split oxygen between full and partial combustion by temperature:
    # higher T favours partial (CO) -- simple bridging factor
    split_co = np.clip((T_s - 1200.0) / 800.0, 0.0, 0.9)
    R1  *= (1.0 - split_co)
    R1b *= split_co

    # --- R2 : 3 Fe2O3 + CO -> 2 Fe3O4 + CO2 -------------------------------
    # Equilibrium ratio (CO2/CO)_eq for hematite reduction is very large
    # in BF range (>100), so R2 is essentially irreversible.  No
    # equilibrium correction needed.
    k2 = _arrhenius(A0["R2"], E_A["R2"], T_g)
    R2 = k2 * C_CO * f_hem * (T_g > 573) * solid_factor
    R2 = R2 * cz_contact_gate

    # --- R3 : Fe3O4 + CO -> 3 FeO + CO2 -----------------------------------
    # Equilibrium-limited with driving force (C_CO - C_CO2 / K_eq).
    # Phase 9 AUDIT FIX: the previous fit (ln K = 1.10 - 1620/T, i.e.
    # K(1200 K) ~ 0.8) contradicts the Baur-Glaessner / Chaudron
    # equilibrium: the magnetite/wustite boundary sits at ~20-35% CO in
    # (CO+CO2), i.e. K = pCO2/pCO ~ 2-5 across 900-1400 K.  With the old
    # constant, any gas with CO2/CO > ~0.8 (the entire thermal-reserve
    # zone, where plant magnetite reduction actually happens) drove R3
    # to ZERO -- this was the structural starvation of the reduction
    # ladder's middle step (Fe production stuck near ~50% of plant).
    # Adopted fit: log10 K = 2.17 - 1834/T (Muchi-lineage BF models,
    # consistent with Baur-Glaessner: K(900) ~ 1.3, K(1200) ~ 4.4).
    k3 = _arrhenius(A0["R3"], E_A["R3"], T_g)
    Keq_R3 = 10.0 ** (2.17 - 1834.0 / np.maximum(T_g, 400.0))
    driving_R3 = np.maximum(C_CO - C_CO2 / np.maximum(Keq_R3, 1e-6), 0.0)
    R3 = k3 * driving_R3 * f_mag * (T_g > 843) * solid_factor * feo_form_gate
    R3 = R3 * cz_contact_gate
    # phase 9: onset 1073 -> 843 K.  The Chaudron (Fe-O-C) diagram puts
    # the wustite stability boundary at 570 degC = 843 K -- below that
    # magnetite reduces directly to Fe and NO wustite can form, above it
    # Fe3O4 + CO -> FeO is thermodynamically open (Biswas, "Principles
    # of Blast Furnace Ironmaking"; Fu 2014: "wustite starts to generate
    # from 800 K").  The old 1073 K gate sat ABOVE the plant thermal-
    # reserve-zone T (~830-1000 K), so R3 was structurally starved: the
    # burden could never form wustite where the plant does.  The
    # equilibrium driving term + feo_form_gate keep the plant vertical
    # ordering.

    # --- R4 : FeO + CO -> Fe + CO2 ----------------------------------------
    # Wustite reduction has K_eq ~ 0.3-0.5; equilibrium-limited.
    # Phase 9 AUDIT FIX: the previous fit (ln K = 910/T - 1.07,
    # K(1000) ~ 0.85) is ~2x too PERMISSIVE vs the Baur-Glaessner
    # wustite/iron boundary (~72% CO at 1000 K -> K = 28/72 ~ 0.39;
    # ~70% at 1200 K -> ~0.43).  The over-permissive constant let
    # indirect wustite reduction run where plant thermodynamics forbid
    # it, starving the direct-reduction pathway of FeO and pushing the
    # top-gas CO2/CO split high.  Adopted fit through the BG anchor
    # points (900 K, 0.31) and (1200 K, 0.43):
    #     ln K_eq_R4 = 0.09 - 1115/T
    k4 = _arrhenius(A0["R4"], E_A["R4"], T_g)
    Keq_R4 = np.exp(0.09 - 1115.0 / np.maximum(T_g, 400.0))
    driving_R4 = np.maximum(C_CO - C_CO2 / np.maximum(Keq_R4, 1e-6), 0.0)
    R4 = k4 * driving_R4 * f_wus * (T_g > 973) * solid_factor * feo_reduce_gate
    R4 = R4 * cz_contact_gate
    # feo_reduce_gate: wustite reduced to iron only near/below the
    # cohesive zone in the lower furnace (Fu 2014).

    # --- H2 counterparts ---------------------------------------------------
    # Same equilibrium logic for H2 reductions (driving force = C_H2 -
    # C_H2O / K_eq).  Equilibrium constants from Austin 1997:
    #   ln K_eq_R3H = -3577/T + 3.74
    #   ln K_eq_R4H = -827/T  + 0.755
    # K_eq_R2H is very large (>>1), no limiting needed.
    k2H = _arrhenius(A0["R2H"], E_A["R2H"], T_g)
    k3H = _arrhenius(A0["R3H"], E_A["R3H"], T_g)
    k4H = _arrhenius(A0["R4H"], E_A["R4H"], T_g)
    Keq_R3H = np.exp(-3577.0 / np.maximum(T_g, 400.0) + 3.74)
    # Phase 9 AUDIT FIX (same Baur-Glaessner audit as Keq_R4): previous
    # fit gave K_R4H(1000) ~ 0.93 vs the H2 wustite/iron boundary
    # ~72% H2 -> K ~ 0.39.  Fit through (1000 K, 0.39), (1200 K, 0.67):
    Keq_R4H = np.exp(2.30 - 3240.0 / np.maximum(T_g, 400.0))
    R2H = k2H * C_H2 * f_hem * (T_g > 573) * solid_factor
    R2H = R2H * cz_contact_gate
    driving_R3H = np.maximum(C_H2 - C_H2O / np.maximum(Keq_R3H, 1e-6), 0.0)
    R3H = k3H * driving_R3H * f_mag * (T_g > 873) * solid_factor * feo_form_gate
    R3H = R3H * cz_contact_gate
    driving_R4H = np.maximum(C_H2 - C_H2O / np.maximum(Keq_R4H, 1e-6), 0.0)
    R4H = k4H * driving_R4H * f_wus * (T_g > 973) * solid_factor * feo_reduce_gate
    R4H = R4H * cz_contact_gate

    # --- R5 : Boudouard  C + CO2 -> 2 CO   (strongly endothermic) ---------
    # Phase 9 structural fix: the previous form had NO equilibrium term,
    # so its A0 had been crushed ~100x below the paper's own target peak
    # (0.5-2 mol/m^3 s) to stop mid-shaft over-firing -- which killed the
    # bosh-zone solution loss the plant carbon balance requires
    # (~80-120 kg C/tHM; handoff/RIST).  The correct self-limiting
    # structure is the Boudouard equilibrium driving force:
    #     C + CO2 <-> 2 CO,   log10 Kp = 9.14 - 8969/T   [Kp in atm]
    # (standard JANAF-derived fit; e.g. "Gasification Reactions of
    # Metallurgical Coke", IntechOpen ch. 40416, Eq. 6).  In concentration
    # form Kc = Kp*P_atm/(R T), driving = C_CO2 - C_CO^2/Kc: the reaction
    # now quenches thermodynamically in the thermal-reserve zone
    # (plant: CO/(CO+CO2) ~ 0.9 mid-furnace, ~1.0 at the bottom) instead
    # of being suppressed by a non-physical A0.  Rate form (proportional
    # to coke mass and CO2) per Austin, Nogami & Yagi 1997 ISIJ 37(8)
    # Table 2 rxn 8 (their ref: Kobayashi & Omori 1977).
    # A0 anchor: k5(1673 K) ~ 4.7 -> peak R5 ~ 1.4 mol/(m^3 s) at bosh
    # conditions (C_CO2 ~ 1.5 mol/m^3, f_coke ~ 0.2), inside the
    # documented plant peak band; E_A = 190 kJ/mol is within the
    # measured 200-260 kJ/mol family for met coke in CO2 (DAEM study,
    # ACS Omega 2021, PMC8153985).
    k5 = _arrhenius(A0["R5"], E_A["R5"], T_s)
    Kp_boud = 10.0 ** (9.14 - 8969.0 / np.maximum(T_s, 400.0))   # [atm]
    Kc_boud = Kp_boud * 101325.0 / (R_GAS * np.maximum(T_s, 400.0))
    driving_R5 = np.maximum(C_CO2 - C_CO ** 2 / np.maximum(Kc_boud, 1e-10),
                            0.0)
    R5 = k5 * driving_R5 * f_coke * (T_s > 1123)

    # --- R6 : Water-gas  C + H2O -> CO + H2 ------------------------------
    # Same structure.  Equilibrium derived from constants already in this
    # module:  (C + H2O <-> CO + H2) = (C + CO2 <-> 2CO) + (CO + H2O <->
    # CO2 + H2), so ln K_WG = ln K_Boudouard + ln K_WGS with the WGS fit
    # used by R7 below (ln K_WGS = -4.33 + 4577/T).  A0 anchor:
    # k6(1673 K) ~ 3.5 x k5(1673 K) -- the measured coke-gasification
    # rate ratio k_H2O/k_CO2 ~ 3.5 (PMC8153985).  This is what lets the
    # raceway H2O slip (Task 2) actually convert to H2 at hot coke
    # in-shaft rather than exiting as vapour.
    k6 = _arrhenius(A0["R6"], E_A["R6"], T_s)
    ln_K_wg = (np.log(np.maximum(Kc_boud, 1e-10))
               + (-4.33 + 4577.0 / np.maximum(T_s, 400.0)))
    Kc_wg = np.exp(np.clip(ln_K_wg, -50.0, 50.0))
    driving_R6 = np.maximum(
        C_H2O - C_CO * C_H2 / np.maximum(Kc_wg, 1e-10), 0.0)
    R6 = k6 * driving_R6 * f_coke * (T_s > 1073)

    # --- R7 : WGS equilibrium approach  CO + H2O <-> CO2 + H2 ------------
    # The water-gas shift is fast only where it is catalysed by the hot
    # burden/ash and where T is high.  In the cool upper shaft (T_g below
    # ~1100 K) the homogeneous WGS is kinetically frozen, so the reaction
    # does NOT reach equilibrium there.  Without this freeze, R7 runs in
    # reverse (CO2 + H2 -> CO + H2O) throughout the cool upper column and
    # consumes the H2, driving top-gas H2 well below the plant value
    # (~6.3%).  Plant data (Fu 2014) shows H2 surviving to the top, which
    # is only reproduced when WGS is frozen in the thermal-reserve zone.
    # Reference: Austin, Nogami & Yagi (1997); Fu et al. (2014) Fig. 6(c).
    k7 = _arrhenius(A0["R7"], E_A["R7"], T_g)
    # Equilibrium constant (fit: ln Keq = -4.33 + 4577/T)
    Keq = np.exp(-4.33 + 4577.0 / np.maximum(T_g, 400.0))
    fwd = C_CO * C_H2O
    rev = C_CO2 * C_H2 / np.maximum(Keq, 1e-10)
    wgs_active = (T_g > 1100.0)   # kinetic freeze below 1100 K.
    # Homogeneous (uncatalysed) WGS is slow in the cool upper shaft;
    # the reaction only approaches equilibrium in the hot lower shaft.
    # Without this freeze, reverse WGS runs through the thermal-reserve
    # zone and strips H2 that plant data shows surviving to the top
    # (Fu 2014 Fig 6c).
    R7 = k7 * (fwd - rev) * wgs_active

    # --- Direct reduction  FeO + C -> Fe + CO  (cohesive zone) -----------
    # Two branches (both Zhou 2016 R11 territory):
    #  (a) solid-state kinetic branch (unchanged);
    #  (b) LIQUID branch: once the burden softens/melts (sigma_melt > 0),
    #      molten wustite drips through the coke bed and is reduced by
    #      direct contact on the drainage timescale tau ~ 600 s.  Plant
    #      hot metal arrives ~94.5% Fe -- essentially ALL FeO reaching
    #      the melting front is reduced before the hearth, so FeO must
    #      not leave the books unreduced.  (Pre-Phase-9 the orchestrator
    #      DELETED draining FeO as a pure mass sink, which is why the
    #      direct-reduction degree and bosh CO regeneration both ran
    #      low: the drain was eating R_DR's feedstock.)
    #      Rate = sigma_melt * rho_bulk * f_wus / (tau_drain * MW_FeO).
    k_DR = _arrhenius(A0["R_DR"], E_A["R_DR"], T_s)
    sigma_melt = np.clip((T_s - 1373.0) / 300.0, 0.0, 1.0)
    tau_drain = 600.0   # s, liquid residence timescale (matches the
                        # Fe3O4 drain in simulation.py)
    rho_bulk = 1800.0 * (1.0 - np.clip(eps, 0.0, 0.95))
    R_DR_liq = sigma_melt * rho_bulk * f_wus / (tau_drain * MW["FeO"])
    R_DR = (k_DR * f_wus * f_coke * (T_s > 1273) + R_DR_liq
            ) * feo_reduce_gate

    # --- R9 : Flux calcination  CaCO3 -> CaO + CO2 -----------------------
    # Endothermic decomposition of carbonate flux (limestone/dolomite),
    # Fu 2014 Table 2 reaction 9.  Onset ~1050 K at BF shaft CO2 partial
    # pressures (equilibrium decomposition T rises with pCO2; the simple
    # threshold captures the plant behaviour without a pCO2 submodel).
    # Rate is availability-capped below, so A0 only needs to be "fast
    # enough" above onset.
    k9 = _arrhenius(A0["R9"], E_A["R9"], T_s)
    R9 = k9 * f_flux * (T_s > 1050.0)
    # feo_reduce_gate: direct reduction is a cohesive-zone / lower-furnace
    # phenomenon (Fu 2014: below the CZ wustite converts to iron).

    # ---------------------------------------------------------------------
    # Availability limiting (Yagi-style cap)
    # ---------------------------------------------------------------------
    # No reaction can consume reactants faster than they are locally
    # available within one time step.  This is the standard treatment
    # in BF models (Yagi 1993; Austin et al. 1997) and is essential for
    # numerical stability: without it, fast Arrhenius rates at raceway
    # temperatures produce non-physical mass sources that overwhelm
    # the continuity equation.
    #
    # For each reaction we cap by the most-limiting reactant.  The cap
    # is C_reactant / dt where dt is the simulation pseudo-timestep, so
    # at most one reactant is fully consumed per step.
    # ---------------------------------------------------------------------
    # SOLID caps use the dt the SOLID solver actually integrates with
    # (``dt_solid``, up to 200 s under the solid-continuation scheme).
    # Pre-Phase-9 this was hard-coded to 0.5 s while the solid update
    # marched at 200 s -- a 400x per-step inventory overdraw that let
    # the reduction ladder overshoot mass fractions past 1.0, where the
    # clip + gangue-floor redistribution FABRICATED solid mass and fed
    # a perpetual CO source (top-gas N2 dilution on fine meshes).
    dt_cap_solid = max(dt_solid, 1e-6)
    # GAS caps keep the gas-transport timescale (0.5 s).
    dt_cap = 0.5

    # Solid mass-fraction availability:  f / dt  [1/s]
    # Coke availability for combustion / Boudouard / water-gas / DR:
    avail_coke = f_coke / dt_cap_solid
    # Iron-phase availability:
    avail_hem = f_hem / dt_cap_solid
    avail_flux = f_flux * 1800.0 / MW["CaCO3"] / dt_cap_solid
    avail_mag = f_mag / dt_cap_solid
    avail_wus = f_wus / dt_cap_solid

    # Gas-species concentration availability  [mol/(m^3 s)]:
    avail_O2  = C_O2  / dt_cap
    avail_CO  = C_CO  / dt_cap
    avail_CO2 = C_CO2 / dt_cap
    avail_H2  = C_H2  / dt_cap
    avail_H2O = C_H2O / dt_cap

    # R1: C(s) + O2 -> CO2     (1 mol O2 + 1 mol C consumed per mol R1)
    R1  = np.minimum(R1,  np.minimum(avail_O2, avail_coke * 1800.0 / MW["C"]))
    # R1b: C + 0.5 O2 -> CO    (0.5 mol O2 + 1 mol C per mol R1b)
    R1b = np.minimum(R1b, np.minimum(2.0 * avail_O2, avail_coke * 1800.0 / MW["C"]))
    # R2: 3 Fe2O3 + CO -> 2 Fe3O4 + CO2  (1/3 mol Fe2O3 + 1 mol CO per mol R2 if r.R2 is per-mol-CO)
    R2  = np.minimum(R2,  np.minimum(avail_CO, avail_hem * 1800.0 / MW["Fe2O3"] / 3.0))
    R3  = np.minimum(R3,  np.minimum(avail_CO, avail_mag * 1800.0 / MW["Fe3O4"]))
    R4  = np.minimum(R4,  np.minimum(avail_CO, avail_wus * 1800.0 / MW["FeO"]))
    R2H = np.minimum(R2H, np.minimum(avail_H2, avail_hem * 1800.0 / MW["Fe2O3"] / 3.0))
    R9  = np.minimum(R9, avail_flux)
    R3H = np.minimum(R3H, np.minimum(avail_H2, avail_mag * 1800.0 / MW["Fe3O4"]))
    R4H = np.minimum(R4H, np.minimum(avail_H2, avail_wus * 1800.0 / MW["FeO"]))
    R5  = np.minimum(R5,  np.minimum(avail_CO2, avail_coke * 1800.0 / MW["C"]))
    R6  = np.minimum(R6,  np.minimum(avail_H2O, avail_coke * 1800.0 / MW["C"]))
    # R7 is WGS-equilibrium-approach -- can be signed; clip both directions
    R7  = np.clip(R7, -np.minimum(avail_CO2, avail_H2),
                       np.minimum(avail_CO,  avail_H2O))
    R_DR = np.minimum(R_DR, np.minimum(
        avail_wus * 1800.0 / MW["FeO"],
        avail_coke * 1800.0 / MW["C"],
    ))

    # All rates are non-negative now (R7 may be signed)
    R1  = np.maximum(R1, 0.0)
    R1b = np.maximum(R1b, 0.0)
    R2  = np.maximum(R2, 0.0)
    R3  = np.maximum(R3, 0.0)
    R4  = np.maximum(R4, 0.0)
    R2H = np.maximum(R2H, 0.0)
    R3H = np.maximum(R3H, 0.0)
    R4H = np.maximum(R4H, 0.0)
    R5  = np.maximum(R5, 0.0)
    R6  = np.maximum(R6, 0.0)
    R_DR = np.maximum(R_DR, 0.0)

    # ---------------------------------------------------------------------
    # Hard physical clamp -- safety-net upper bound on any single rate.
    # Plant-realistic peak rates for BF kinetics are ~5-30 mol/(m^3 s).
    # Combustion can reach 50.  The availability cap above SHOULD be
    # sufficient, but if a cell has unphysical concentrations (a known
    # symptom of mole-fraction conservation drift), local rates can still
    # explode.  This clamp prevents bogus rates from corrupting the energy
    # and species sources.  It is a temporary mitigation -- the proper fix
    # is mass-fraction (Y) transport which is naturally conservative.
    # ---------------------------------------------------------------------
    RATE_MAX = 60.0   # mol/(m^3 s), generous physical bound
    R1   = np.minimum(R1,   RATE_MAX)
    R1b  = np.minimum(R1b,  RATE_MAX)
    R2   = np.minimum(R2,   RATE_MAX)
    R3   = np.minimum(R3,   RATE_MAX)
    R4   = np.minimum(R4,   RATE_MAX)
    R2H  = np.minimum(R2H,  RATE_MAX)
    R3H  = np.minimum(R3H,  RATE_MAX)
    R4H  = np.minimum(R4H,  RATE_MAX)
    R5   = np.minimum(R5,   RATE_MAX)
    R6   = np.minimum(R6,   RATE_MAX)
    R_DR = np.minimum(R_DR, RATE_MAX)
    R9   = np.minimum(np.maximum(R9, 0.0), RATE_MAX)
    R7   = np.clip(R7, -RATE_MAX, RATE_MAX)

    # Mask to active cells
    def _mask(x: np.ndarray) -> np.ndarray:
        return np.where(active, x, zero)

    return ReactionRates(
        R1=_mask(R1), R1b=_mask(R1b), R2=_mask(R2), R3=_mask(R3), R4=_mask(R4),
        R2H=_mask(R2H), R3H=_mask(R3H), R4H=_mask(R4H),
        R5=_mask(R5), R6=_mask(R6), R7=_mask(R7), R_DR=_mask(R_DR),
        R9=_mask(R9),
    )


# ---------------------------------------------------------------------------
def assemble_species_sources(r: ReactionRates) -> Dict[str, np.ndarray]:
    """Per-species volumetric molar production  S  [mol / (m^3 s)].

    Sign convention: positive means the species is **generated** locally.
    """
    S = {sp: np.zeros_like(r.R1) for sp in GAS_SPECIES}

    # Oxygen consumed in combustion (R1 full burn -> 1 mol O2 / mol C ; R1b half)
    S["O2"]  -= r.R1 + 0.5 * r.R1b
    # CO generated / consumed
    S["CO"]  += r.R1b                  # partial combustion
    S["CO"]  += 2.0 * r.R5             # Boudouard
    S["CO"]  += r.R6                   # water-gas
    S["CO"]  += r.R_DR                 # direct reduction
    S["CO"]  -= (r.R2 + r.R3 + r.R4)   # indirect reduction consumes CO
    S["CO"]  -= r.R7                   # WGS forward consumes CO
    # CO2
    S["CO2"] += r.R1                   # full combustion
    S["CO2"] += r.R2 + r.R3 + r.R4
    S["CO2"] += r.R7
    S["CO2"] -= r.R5
    S["CO2"] += r.R9                   # flux calcination releases CO2
    # H2
    S["H2"]  += r.R6                   # water-gas
    S["H2"]  += r.R7                   # WGS forward
    S["H2"]  -= (r.R2H + r.R3H + r.R4H)
    # H2O
    S["H2O"] += r.R2H + r.R3H + r.R4H
    S["H2O"] -= r.R6
    S["H2O"] -= r.R7
    # N2 inert
    # (nothing)
    return S


def assemble_solid_sources(r: ReactionRates) -> Dict[str, np.ndarray]:
    """Per-phase volumetric mass source terms [kg / (m^3 s)].

    Positive means the solid phase gains mass locally.
    Stoichiometry (mol consumed / produced per unit reaction rate):

        R1, R1b: consumes 1 mol C              per mol O2(full) / mol O2(half*2)
        R2:  consumes 3 Fe2O3, produces 2 Fe3O4
        R3:  consumes   Fe3O4, produces 3 FeO
        R4:  consumes   FeO,   produces   Fe
        H2 analogues:  same iron stoichiometry
        R5:  consumes 1 mol C
        R6:  consumes 1 mol C
        R_DR: consumes 1 mol C and 1 mol FeO, produces 1 mol Fe
    """
    S = {
        "Fe2O3":  -3.0 * MW["Fe2O3"] * (r.R2 + r.R2H),
        "Fe3O4":  (2.0 * MW["Fe3O4"] * (r.R2 + r.R2H)
                   - MW["Fe3O4"] * (r.R3 + r.R3H)),
        "FeO":    (3.0 * MW["FeO"] * (r.R3 + r.R3H)
                   - MW["FeO"] * (r.R4 + r.R4H)
                   - MW["FeO"] * r.R_DR),
        "Fe":     MW["Fe"] * (r.R4 + r.R4H + r.R_DR),
        "C_coke": -MW["C"] * (r.R1 + r.R1b + r.R5 + r.R6 + r.R_DR),
        # NOTE: R5 (Boudouard, C + CO2 -> 2CO) consumes ONE carbon per mol
        # reaction.  A previous version double-counted (2.0 * R5), which
        # over-consumed coke in the solution-loss zone by 2x and would
        # have corrupted the coke-rate KPI.
        "gangue": (MW["CaCO3"] - MW["CO2"]) * r.R9,   # CaO residue joins inert pool
        "flux":   -MW["CaCO3"] * r.R9,
    }
    return S
