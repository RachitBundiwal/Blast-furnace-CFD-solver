"""
Tuyere / raceway boundary as a *prescribed inlet* on staggered faces.

Phase 1 note
------------
This module treats the tuyere as a velocity + temperature + species Dirichlet
boundary on the radial faces that line up with the tuyere elevation band, at
the outer (wall-side) column of active cells.  The gas momentum solver
assigns the corresponding Dirichlet rows directly, rather than using a
volumetric source term as Phase 0 did.

Phase 2 will replace this module with a two-fluid raceway sub-domain
governing coke combustion and particle motion inside the blow-through cavity.
The public interface (``Raceway`` exposing ``face_indices`` etc.) is stable
and Phase 2 can be swapped in without touching any other solver code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .constants import GAS_SPECIES, MW, MOL_VOL_NTP, R_GAS
from .mesh import Mesh2D
from .parameters import BFParameters2D


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Raceway:
    """Immutable description of the tuyere boundary.

    Attributes
    ----------
    face_indices
        List of ``(i_face, j)`` tuples identifying r-face indices on which
        the inlet BC is enforced.
    v_tuyere
        Reference inward blast velocity [m/s] computed at blast-density
        conditions.  Used for diagnostics; the actual applied velocity is
        derived from G_tuyere / rho_face_local.
    G_tuyere
        Mass flux at the tuyere face [kg/(m^2 s)].  This is the *physical*
        boundary specification: mass per unit area per second.  The actual
        face velocity is computed as -G_tuyere / rho_face every iteration,
        keeping the *total mass flow* invariant under local density changes.
    T_blast
        Blast-gas inlet temperature [K].
    x_gas_blast
        Mole-fraction composition of the injected gas.
    total_blast_kg_s
        Total mass flow across all tuyere faces [kg/s].
    rho_blast
        Reference density at blast conditions [kg/m^3].
    """

    face_indices: List[Tuple[int, int]]
    v_tuyere: float
    G_tuyere: float          # kg/(m^2 s) -- THE boundary specification
    T_blast: float
    x_gas_blast: Dict[str, float]
    total_blast_kg_s: float
    rho_blast: float
    shape_mask: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ------- Helpers ---------------------------------------------------------
    def is_tuyere_face(self, i_face: int, j: int) -> bool:
        return (i_face, j) in set(self.face_indices)


# ---------------------------------------------------------------------------
def _raceway_exit_temperature(
    p: BFParameters2D,
    x_bosh: Dict[str, float],
) -> float:
    """Compute the post-combustion (bosh gas) temperature via 0D enthalpy balance.

    The raceway is modelled as a 0D adiabatic CSTR.  Hot blast at T_blast
    enters; combustion with coke C releases heat; bosh gas exits at
    T_bosh.

    Enthalpy balance (per Nm^3 dry blast):
        H_in  = n_O2 cp_O2 (T_blast - 298) + n_N2 cp_N2 (T_blast - 298)
              + n_H2O cp_H2O (T_blast - 298)
              + n_C cp_C (T_solid - 298)              [coke sensible]
        Q_rxn = n_O2_partial * ΔH_R1b + n_O2_full * ΔH_R1
              + n_H2O * ΔH_R6                          [combustion heat]
        H_out = Σ n_k cp_k (T_bosh - 298)              [bosh gas]
        H_in + Q_rxn = H_out
    Solve for T_bosh.

    Plant calibration: Fu 2014 Table 5 reports T_bosh = 2300 K for hot
    blast at 1473 K (1200°C) with O2 enrichment 4% and moisture 15 g/Nm^3.
    Our 0D model should give T_bosh in the 2100-2400 K range to match.
    """
    # Reaction enthalpies [J/mol], at 298 K reference:
    DH_R1  = -393.5e3   # C + O2 -> CO2     (full combustion)
    DH_R1b = -110.5e3   # C + 1/2 O2 -> CO  (partial combustion)
    DH_R6  = +131.0e3   # C + H2O -> CO + H2 (water-gas, endothermic)

    # cp values [J/(mol·K)], approximate constants in 1500-2500 K range
    cp = {
        "N2":  33.0,
        "O2":  35.0,
        "CO":  33.0,
        "CO2": 50.0,
        "H2":  29.5,
        "H2O": 41.0,
    }
    cp_C = 22.0  # graphite cp [J/(mol·K)]

    # Per Nm^3 dry blast (excluding moisture)
    x_o2_dry = 0.21 + p.oxygen_enrichment
    x_n2_dry = 1.0 - x_o2_dry
    n_h2o_per_Nm3 = p.blast_moisture_gpm3 / (MW["H2O"] * 1000.0)
    n_dry_per_Nm3 = 1.0 / MOL_VOL_NTP

    n_o2  = x_o2_dry * n_dry_per_Nm3
    n_n2  = x_n2_dry * n_dry_per_Nm3
    n_h2o = n_h2o_per_Nm3
    T_blast = p.blast_temperature

    # Combustion split
    f_partial = float(np.clip((T_blast - 1200.0) / 800.0, 0.0, 0.95))
    n_o2_partial = n_o2 * f_partial
    n_o2_full    = n_o2 * (1.0 - f_partial)

    # Carbon consumed (mol C per Nm^3 blast)
    n_C_R1b = n_o2_partial * 1.0     # 1 mol C per mol O2 in R1b (C + 1/2 O2 -> CO; need 2 R1b per O2)
    # Wait: R1b is "C + 1/2 O2 -> CO".  So 2 mol C per mol O2 in R1b.
    n_C_R1b = n_o2_partial * 2.0
    n_C_R1  = n_o2_full * 1.0        # 1 mol C per mol O2 in R1
    # Only the REACTED moisture consumes carbon / absorbs R6 heat; the
    # slipped fraction transits as vapour (see raceway_h2o_slip).
    n_h2o_reacted = n_h2o * (1.0 - p.raceway_h2o_slip)
    n_h2o_slip = n_h2o * p.raceway_h2o_slip
    n_C_R6  = n_h2o_reacted * 1.0    # 1 mol C per mol reacted H2O in R6
    n_C_total = n_C_R1b + n_C_R1 + n_C_R6

    # Reaction heat released (positive = released)
    # R1b: 2 C + O2 -> 2 CO, ΔH per O2 = 2*(-110.5) = -221 kJ/mol O2
    Q_R1b = -DH_R1b * 2.0 * n_o2_partial
    Q_R1  = -DH_R1  * 1.0 * n_o2_full
    Q_R6  = -DH_R6  * 1.0 * n_h2o_reacted   # R6 endothermic -> negative
    Q_rxn = Q_R1b + Q_R1 + Q_R6

    # Sensible enthalpy in (per Nm^3 blast)
    T_ref = 298.0
    H_blast_in = (n_o2 * cp["O2"] + n_n2 * cp["N2"] + n_h2o * cp["H2O"]) * (T_blast - T_ref)
    # Coke enters at solid temperature near the raceway -- use 1700 K
    # as plant-typical solid temperature there
    T_solid_raceway = 1700.0
    H_coke_in = n_C_total * cp_C * (T_solid_raceway - T_ref)
    H_in = H_blast_in + H_coke_in

    # Bosh gas composition (mol per Nm^3 blast)
    n_co  = 2.0 * n_o2_partial + n_h2o_reacted
    n_co2 = n_o2_full
    n_h2  = n_h2o_reacted
    n_n2_out = n_n2

    # Bosh-gas total cp [J/K per Nm^3 blast] (slipped H2O exits as vapour)
    sum_n_cp = (n_n2_out * cp["N2"] + n_co * cp["CO"] +
                n_co2 * cp["CO2"] + n_h2 * cp["H2"] +
                n_h2o_slip * cp["H2O"])

    # Solve H_in + Q_rxn = sum_n_cp * (T_bosh - T_ref)
    T_bosh = T_ref + (H_in + Q_rxn) / max(sum_n_cp, 1e-9)

    # Plant range: 2100-2400 K.  Cap to avoid nonsense if inputs are extreme.
    return float(np.clip(T_bosh, 1800.0, 2500.0))


def _blast_composition(p: BFParameters2D) -> Tuple[Dict[str, float], float]:
    """Composition of gas LEAVING the raceway (not the raw blast).

    Returns ``(mole_fractions, exit_moles_per_Nm3_dry_blast)`` -- the
    second value is required to convert the blast volumetric rate into
    the raceway-EXIT molar/mass flow (exit moles exceed the blast-air
    moles by ~35%: combustion carbon, PCI and coke H add moles).

    Plant practice
    --------------
    In an actual blast furnace, the raceway is a turbulent void zone
    where blast O2 is fully consumed by coke combustion within ~1-2 ms
    residence time.  Eddy-breakup-limited rates reach ~1000-3000
    mol/(m³ s), far higher than what Arrhenius kinetics give in a
    porous-bed cell (~50 mol/(m³ s)).

    A 2D axisymmetric continuum model cannot resolve the raceway
    turbulence.  The standard approach (Fu et al. 2014; Zhou et al.
    2016) is to compute the *raceway exit* gas composition by an
    external 0D combustion balance and inject THAT at the tuyere face:

        Inputs to raceway: O2, N2, H2O (humid blast), C (coke), [PCI]
        Reactions in raceway (assumed complete):
            C + 1/2 O2 -> CO       (R1b, partial)
            C + O2     -> CO2      (R1,  full)
            C + H2O    -> CO + H2  (R6,  water-gas)
        At T_blast > 2000 K, R1b dominates over R1 by Boudouard
        equilibrium (CO/CO2 >> 1).
        H2O is consumed quickly via R6 producing CO + H2.

    Output: high-temperature reducing gas, ~38% CO, ~50% N2, ~3% H2,
    ~5% CO2 (typical bosh gas, Fu 2014 Table 5: 2300K).

    Plant validation: Fu 2014 cite bosh gas T = 2300 K at the raceway
    exit; Zhou 2016 confirm ~36-40% CO from O2-enriched blast.
    """
    # Inputs per Nm^3 dry blast
    x_o2_dry = 0.21 + p.oxygen_enrichment
    x_n2_dry = 1.0 - x_o2_dry
    n_h2o_per_Nm3 = p.blast_moisture_gpm3 / (MW["H2O"] * 1000.0)
    n_dry_per_Nm3 = 1.0 / MOL_VOL_NTP

    n_o2 = x_o2_dry * n_dry_per_Nm3
    n_n2 = x_n2_dry * n_dry_per_Nm3
    n_h2o = n_h2o_per_Nm3

    # Raceway combustion balance.
    # All O2 consumed.  Split between full and partial combustion based
    # on temperature: at T_blast > 2000 K, partial combustion dominates
    # because CO is the equilibrium product (Boudouard reverse).
    #
    # Plant validation (Fu 2014 Table 5): bosh gas ~38% CO, ~5% CO2 at
    # T = 2300 K -> CO/CO2 ~ 7-8 -> effective f_partial ~ 0.85.
    # (f_partial = 1.0 is unphysical: Boudouard equilibrium leaves
    # 5-10% CO2 even at raceway T.)
    T_b = p.blast_temperature
    f_partial = float(np.clip((T_b - 1200.0) / 1100.0, 0.0, 0.85))

    # ---------------------------------------------------------------
    # Tuyere gas injectants (hydrogen-enrichment envelope).
    #
    # NG injection (as CH4) undergoes partial oxidation in the raceway,
    #     CH4 + 1/2 O2 -> CO + 2 H2,
    # which CONSUMES blast O2 before it can burn coke/PCI carbon, so
    # the O2 ledger is debited FIRST.  (Zhou 2016 uses a 28-step NG
    # mechanism; this single-step partial oxidation is the standard
    # reduced form for shaft models and preserves element balances.)
    #
    # H2 injection passes through the O2-starved, carbon-rich raceway
    # into the bosh gas (net raceway-exit effect: added H2).
    # ---------------------------------------------------------------
    n_h2_inj = p.blast_h2_vol_frac * n_dry_per_Nm3
    n_ng_inj = p.blast_ng_vol_frac * n_dry_per_Nm3
    o2_for_ng = min(0.5 * n_ng_inj, n_o2)      # cannot exceed available O2
    n_o2_for_carbon = n_o2 - o2_for_ng

    n_o2_partial = n_o2_for_carbon * f_partial          # -> 2 CO per O2
    n_o2_full    = n_o2_for_carbon * (1.0 - f_partial)  # -> 1 CO2 per O2

    # CO + CO2 produced from O2 + coke combustion
    n_co  = 2.0 * n_o2_partial
    n_co2 = 1.0 * n_o2_full

    # NG partial-oxidation products
    n_co += n_ng_inj
    n_h2  = 2.0 * n_ng_inj + n_h2_inj

    # H2O reacts with coke (R6: C + H2O -> CO + H2) during its ~ms
    # transit of the raceway cavity; the fraction `raceway_h2o_slip`
    # survives unreacted into the bosh gas and is converted IN-SHAFT by
    # R6 at hot coke, where the produced H2 survives the rising, cooling
    # gas (Zhou 2016 Fig. 5: raceway H2O continues reacting with coke up
    # the bosh; plant bosh gas retains a few % H2O).  Element balance is
    # exact for any slip: each slipped H2O carries its own 2 H + 1 O.
    n_h2o_reacted = n_h2o * (1.0 - p.raceway_h2o_slip)
    n_co += n_h2o_reacted
    n_h2 += n_h2o_reacted
    n_h2o_remaining = n_h2o * p.raceway_h2o_slip

    # ---------------------------------------------------------------
    # PCI (pulverized coal injection) contribution.
    #
    # Plant practice (Fu 2014: 67 kg PCI per tHM): coal is injected
    # through the tuyeres into the raceway.  Its carbon and hydrogen
    # end up as additional CO and H2 in the bosh gas, diluting N2.
    # This is the main reason plant bosh gas is ~50% N2 rather than
    # the ~59% an O2-only balance predicts.
    #
    # Balance per Nm^3 blast: PCI mass / Nm^3 blast is approximated by a
    # plant-representative ratio (productivity/blast-rate).  PCI coal ~
    # 85% C, 5% H by mass.  In the O2-limited raceway, gas-phase carbon
    # ends up as CO; only a fraction of C and H reaches the gas promptly
    # (the rest stays as char / is consumed downstream).
    # ---------------------------------------------------------------
    # Exact scaling: PCI mass flow = pci_rate [kg/tHM] x productivity [tHM/s];
    # divide by blast volumetric flow [Nm3/s] to get kg PCI per Nm3 blast.
    # (The earlier rough 1.5e-3 factor was calibrated against a blast rate
    # inconsistent with plant per-tHM ratios and over-injected PCI ~2.8x.)
    productivity_tHM_s = p.productivity_tHM_day / 86400.0
    blast_Nm3_s = p.blast_rate_Nm3_min / 60.0
    pci_kg_per_Nm3 = (p.pci_rate_kg_tHM * productivity_tHM_s
                      / max(blast_Nm3_s, 1e-9))
    pci_C_mol = 0.85 * pci_kg_per_Nm3 / 0.012      # mol C per Nm3
    pci_H_mol = 0.05 * pci_kg_per_Nm3 / 0.001      # mol H atoms per Nm3
    f_pci_C_to_gas = 0.7   # fraction of PCI carbon reaching gas as CO
    f_pci_H_to_gas = 0.95  # PCI hydrogen volatilising to H2: coal
                           # devolatilization is nearly complete at
                           # raceway T (>2000 K), so almost all volatile
                           # H is released.  Plant top-gas H2 (~6.3%,
                           # Fu 2014) requires near-full release.
    n_co += f_pci_C_to_gas * pci_C_mol
    n_h2 += 0.5 * f_pci_H_to_gas * pci_H_mol       # 2H -> H2

    # ---------------------------------------------------------------
    # Coke volatile hydrogen (ultimate-analysis H, typically 0.4-0.7
    # wt% of met coke).  It devolatilizes as the coke heats in the
    # shaft and leaves in the top gas; the plant top-gas hydrogen
    # ledger (H2 6.3% + H2O ~1%, Fu Table 7) cannot close from blast
    # moisture + PCI H alone.  Released here as H2 per Nm^3 blast in
    # proportion to the charged coke rate (release-location
    # approximation; once formed, H2 is elementally conserved).
    # ---------------------------------------------------------------
    coke_kg_per_Nm3 = p.coke_rate_kg_s / max(blast_Nm3_s, 1e-9)
    n_h2 += 0.5 * coke_kg_per_Nm3 * p.coke_h_mass_frac / 0.001008

    # Total moles at raceway exit per Nm^3 of dry blast.  This is
    # substantially MORE than the blast-air moles (~44.6/Nm^3): coke
    # carbon burned to CO/CO2, PCI volatiles, water-gas H2 and coke H
    # all add moles (typically n_exit ~ 60/Nm^3, x1.35).
    n_total = n_n2 + n_co + n_co2 + n_h2 + n_h2o_remaining
    x = {
        "N2":  n_n2  / n_total,
        "O2":  0.0,                   # fully consumed in raceway
        "CO":  n_co  / n_total,
        "CO2": n_co2 / n_total,
        "H2":  n_h2  / n_total,
        "H2O": n_h2o_remaining / n_total,
    }
    return x, n_total


def _blast_density(p: BFParameters2D, x_gas: Dict[str, float]) -> float:
    """Ideal-gas blast density at tuyere conditions [kg/m^3]."""
    mw_mix = sum(x_gas[sp] * MW[sp] for sp in GAS_SPECIES)
    # Tuyere is at the bottom; pressure ~ top + shaft dP (~0.5 bar typical)
    P_tuyere = p.top_pressure + 0.5e5
    return P_tuyere * mw_mix / (R_GAS * p.blast_temperature)


# ---------------------------------------------------------------------------
def build_raceway(mesh: Mesh2D, p: BFParameters2D) -> Raceway:
    """Identify tuyere inlet faces and compute prescribed BC values.

    Mesh-independent area handling
    ------------------------------
    The total tuyere inlet area is a *physical* quantity:

        A_phys = n_tuyeres * pi * (tuyere_diameter / 2)^2

    Plant typical: 32 tuyeres of 130 mm diameter → A_phys ≈ 0.42 m^2.
    Mass flux at the tuyere is then:

        G_tuyere = total_blast_kg_s / A_phys     [kg/(m^2 s)]

    and the inlet velocity is:

        v_tuyere = G_tuyere / rho_blast          [m/s]

    On a discrete axisymmetric mesh, the physical area is approximated by
    selecting axial faces in the tuyere band whose accumulated area best
    matches A_phys.  We accept the face area a single discrete face
    contributes (annular ring face area = 2 pi r_wall dz) and pick the
    faces closest to the tuyere centre elevation, stopping when the
    accumulated area exceeds A_phys.  The actual G is then recomputed
    from the *accepted* discrete area so total mass flow is conserved.

    This keeps the inlet velocity in the physically correct ~150-250 m/s
    range regardless of mesh resolution.  Previously, on fine meshes the
    "tuyere band" axial extent shrank with dz, giving few faces and
    artificially high (or arbitrarily clipped) v_tuyere.
    """
    x_blast, n_exit_per_Nm3 = _blast_composition(p)
    rho_blast = _blast_density(p, x_blast)

    # Raceway-EXIT mass flow: exit moles x exit mean MW.  Phase 10 FIX:
    # this used to be (dry-blast-air moles) x (exit MW) -- a mixed-basis
    # product that undercounted the injected stream by ~25% (the exit
    # stream carries the burned coke carbon, PCI and coke hydrogen on
    # top of the blast air; n_exit ~ 1.35x the air moles).  Found by the
    # element ledger: N2/H2 tuyere inflow ran well below the 0D balance,
    # which starved the furnace of gas per tonne of burden and pushed
    # top-gas N2 low / CO2 high.
    mol_exit_s = (p.blast_rate_Nm3_min / 60.0) * n_exit_per_Nm3
    mw_mix = sum(x_blast[sp] * MW[sp] for sp in GAS_SPECIES)
    total_blast = mol_exit_s * mw_mix

    # ------------------------------------------------------------------
    # Physical tuyere area (mesh-independent)
    # ------------------------------------------------------------------
    A_phys = p.n_tuyeres * np.pi * (p.tuyere_diameter / 2.0) ** 2

    # ------------------------------------------------------------------
    # Candidate faces: any axial position where the tuyere band spans
    # AND there is at least one active cell that touches the wall.  We
    # rank candidates by distance from p.tuyere_elevation and pick the
    # closest few until accumulated area >= A_phys.
    # ------------------------------------------------------------------
    candidates: List[Tuple[float, int, int, float]] = []
    # (distance, i_face, j, face_area)
    for j in range(mesh.nz):
        active_i = np.where(mesh.active[:, j])[0]
        if active_i.size == 0:
            continue
        i_outer = int(active_i.max())
        i_face = i_outer + 1
        if i_face > mesh.nr:
            continue
        z_face = mesh.z_c[j]
        dist = abs(z_face - p.tuyere_elevation)
        face_area = float(mesh.area_r[i_face, j])
        candidates.append((dist, i_face, j, face_area))

    if not candidates:
        raise RuntimeError("No candidate tuyere faces; check geometry/mesh")

    candidates.sort(key=lambda t: t[0])

    # Accumulate faces until we cover A_phys (or run out)
    face_indices: List[Tuple[int, int]] = []
    total_area = 0.0
    for (_dist, i_face, j, face_area) in candidates:
        if total_area >= A_phys:
            break
        face_indices.append((i_face, j))
        total_area += face_area

    # Edge case: even one face exceeds A_phys (very fine mesh).  Keep at
    # least one face so the BC is non-empty.
    if not face_indices:
        _, i_face, j, face_area = candidates[0]
        face_indices = [(i_face, j)]
        total_area = face_area

    if total_area < 1e-9:
        raise RuntimeError("Tuyere area is zero; check geometry/mesh")

    # ------------------------------------------------------------------
    # Compute G and v from the *accepted* discrete area so that the
    # total mass flow is exactly preserved.  v_tuyere is reported for
    # diagnostics; the inlet condition is enforced as G = m_dot / A.
    # ------------------------------------------------------------------
    G_tuyere = total_blast / total_area
    v_tuyere = G_tuyere / rho_blast

    # Sanity check: v should be in plant range 100-300 m/s.  If outside,
    # warn but don't clip -- clipping causes mass-conservation errors.
    # The user can adjust tuyere_diameter or n_tuyeres if needed.
    # (No print here -- diagnostic via Raceway.v_tuyere field.)

    # Build a shape mask for visualisation (cells adjacent to a tuyere face)
    shape_mask = np.zeros((mesh.nr, mesh.nz), dtype=float)
    for (i_face, j) in face_indices:
        i_cell = i_face - 1
        if 0 <= i_cell < mesh.nr and mesh.active[i_cell, j]:
            shape_mask[i_cell, j] = 1.0

    # Compute raceway exit (bosh) temperature from 0D enthalpy balance.
    # The blast enters at T_blast (~1473 K) but the gas LEAVING the raceway
    # is at bosh T (~2100-2400 K, plant) due to combustion heat release.
    # This is the temperature that should be used as the boundary
    # condition for the gas energy equation in the shaft model.
    T_bosh = _raceway_exit_temperature(p, x_blast)

    return Raceway(
        face_indices=face_indices,
        v_tuyere=v_tuyere,
        G_tuyere=G_tuyere,
        T_blast=T_bosh,             # post-combustion (bosh) T, not raw blast
        x_gas_blast=x_blast,        # post-combustion composition
        total_blast_kg_s=total_blast,
        rho_blast=rho_blast,
        shape_mask=shape_mask,
    )
