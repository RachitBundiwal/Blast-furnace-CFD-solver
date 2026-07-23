"""
Gas-species transport + solid-phase advection (Phase 1, mass-fraction form).

Architecture
------------
This module solves for **mass fractions Y_k = m_k / m_total** internally,
because the conservative form

    ∂(ε ρ Y_k)/∂t  +  ∇·(ρ u Y_k)  =  S_k_mass        [kg/(m³ s)]

is naturally well-posed under upwind FV: with mass-balanced kinetics
(Σ S_k_mass = 0), the discrete equation guarantees Σ Y_k = 1 at every
cell at every iteration with no renormalisation required.

The earlier mole-fraction (x) implementation required a "drift to N2"
correction to enforce Σ x_k = 1, which broke conservation when the
inert diluent was locally depleted (a real failure mode in cells where
H2 production was strong, near the cohesive zone).

Public interface stays mole-fraction-based for backward compatibility.
The conversion x ↔ Y happens inside solve_species:

    Y_k = (x_k MW_k) / Σ(x_j MW_j)
    x_k = (Y_k / MW_k) / Σ(Y_j / MW_j)

References
----------
- Patankar (1980) §5.4: scalar transport with FV upwind preserves
  positivity and boundedness for any conserved scalar.
- Versteeg & Malalasekera (2007) §5.6: conservative form is essential
  for multi-species reacting flow; non-conservative formulations cannot
  enforce Σ Y_k = 1 through the discretization alone.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sps

from .constants import GAS_SPECIES, MW, EPS_MOLE, SOLID_PHASES
from .linear_solver import solve_linear_system
from .mesh import Mesh2D
from .parameters import BFParameters2D


def _idx(i: int, j: int, nr: int) -> int:
    return i + nr * j


def _face_avg_r(f: np.ndarray, i_face: int, j: int, nr: int) -> float:
    if i_face == 0:
        return f[0, j]
    if i_face == nr:
        return f[nr - 1, j]
    return 0.5 * (f[i_face - 1, j] + f[i_face, j])


def _face_avg_z(f: np.ndarray, i: int, j_face: int, nz: int) -> float:
    if j_face == 0:
        return f[i, 0]
    if j_face == nz:
        return f[i, nz - 1]
    return 0.5 * (f[i, j_face - 1] + f[i, j_face])


# ---------------------------------------------------------------------------
def _x_to_Y(x_gas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Mole fractions -> mass fractions, cell-wise."""
    mw_total = sum(x_gas[sp] * MW[sp] for sp in GAS_SPECIES)
    mw_total = np.maximum(mw_total, 1e-9)
    return {sp: x_gas[sp] * MW[sp] / mw_total for sp in GAS_SPECIES}


def _Y_to_x(Y_gas: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Mass fractions -> mole fractions, cell-wise."""
    n_total = sum(Y_gas[sp] / MW[sp] for sp in GAS_SPECIES)
    n_total = np.maximum(n_total, 1e-9)
    return {sp: (Y_gas[sp] / MW[sp]) / n_total for sp in GAS_SPECIES}


def _composition_x_to_Y(x_dict: Dict[str, float]) -> Dict[str, float]:
    """Scalar mole-fraction composition -> scalar mass-fraction composition."""
    mw_total = sum(x_dict.get(sp, 0.0) * MW[sp] for sp in GAS_SPECIES)
    mw_total = max(mw_total, 1e-9)
    return {sp: x_dict.get(sp, 0.0) * MW[sp] / mw_total for sp in GAS_SPECIES}


# ---------------------------------------------------------------------------
def solve_species(
    x_gas: Dict[str, np.ndarray],
    rho_g: np.ndarray,
    mw_mix: np.ndarray,
    u_r: np.ndarray,
    u_z: np.ndarray,
    eps: np.ndarray,
    sources_mol: Dict[str, np.ndarray],
    raceway_mask_gas: np.ndarray,
    raceway_composition: Dict[str, float],
    mesh: Mesh2D,
    p: BFParameters2D,
    relaxation: float = 0.7,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Advance the six gas species using conservative mass-fraction transport.

    Inputs and outputs are in **mole fractions** for backward compatibility
    with the rest of the solver; internally the equation is solved in
    mass fractions Y, which is the physically conservative form.

    Parameters
    ----------
    x_gas : dict
        Current cell-centre mole fractions per species.
    sources_mol : dict
        Per-species volumetric production in mol/(m³·s).  Converted to
        kg/(m³·s) inside this function via S_mass = S_mol × MW.
    raceway_composition : dict
        Tuyere inlet *mole-fraction* composition.

    Returns
    -------
    new_x_gas : dict
        Updated mole fractions (Σ = 1 by construction).
    residuals : dict
        Per-species L2 residual of the linear solve.
    """
    nr, nz = mesh.nr, mesh.nz
    n = nr * nz

    # ------------------------------------------------------------------
    # Convert x -> Y for transport.  Sources convert mol -> kg.
    # Inlet composition converts x -> Y as a scalar dictionary.
    # ------------------------------------------------------------------
    Y_gas = _x_to_Y(x_gas)
    sources_mass = {sp: sources_mol[sp] * MW[sp] for sp in GAS_SPECIES}
    Y_blast = _composition_x_to_Y(raceway_composition)

    new_Y: Dict[str, np.ndarray] = {}
    residuals: Dict[str, float] = {}

    for sp in GAS_SPECIES:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        b = np.zeros(n)

        def add(r: int, c: int, v: float) -> None:
            rows.append(r); cols.append(c); vals.append(v)

        for j in range(nz):
            for i in range(nr):
                k = _idx(i, j, nr)
                if not mesh.active[i, j]:
                    add(k, k, 1.0); b[k] = Y_gas[sp][i, j]
                    continue

                V = mesh.vol[i, j]

                # Face-interpolated densities (kg/m^3)
                rho_n = _face_avg_z(rho_g, i, j + 1, nz)
                rho_s = _face_avg_z(rho_g, i, j,     nz)
                rho_e = _face_avg_r(rho_g, i + 1, j, nr)
                rho_w = _face_avg_r(rho_g, i,     j, nr)

                A_n = mesh.area_z[i, j + 1]
                A_s = mesh.area_z[i, j]
                A_e = mesh.area_r[i + 1, j]
                A_w = mesh.area_r[i, j]

                # Face mass fluxes [kg/s]
                F_n = rho_n * u_z[i, j + 1] * A_n
                F_s = rho_s * u_z[i, j]     * A_s
                F_e = rho_e * u_r[i + 1, j] * A_e
                F_w = rho_w * u_r[i, j]     * A_w

                # Upwind coefficients [kg/s]: incoming fluxes
                aN = max(-F_n, 0.0); aS = max(F_s, 0.0)
                aE = max(-F_e, 0.0); aW = max(F_w, 0.0)
                # Outgoing fluxes (the ACTUAL flux field, sign-resolved)
                out_flux = (max(F_n, 0.0) + max(-F_s, 0.0)
                            + max(F_e, 0.0) + max(-F_w, 0.0))

                # Time term [kg/s]
                a_t = eps[i, j] * rho_g[i, j] * V / p.dt

                # Diagonal = ACTUAL total outgoing mass flux + time term
                # (strict conservative upwind: what leaves this cell
                # arrives at its neighbour; a species can only exit the
                # domain through a real boundary flux).  Phase 9 used
                # a_P = sum(inflows) [assumes zero net source]; a first
                # fix added +S_total*V [assumes the flux field satisfies
                # continuity with the reaction sources].  On fine meshes
                # the Darcy flux divergence deviates from S_total enough
                # that inert N2 was destroyed at ~35% of its inflow
                # (element-ledger measurement).  Using the sign-resolved
                # outgoing fluxes makes per-species transport exactly
                # conservative w.r.t. the flux field; the Sigma Y = 1
                # residual then equals the local continuity error only
                # (handled by the existing >1% safety net, which should
                # no longer trigger systematically).
                # The top-row outflow (F_n > 0 at j = nz-1) is included
                # in out_flux automatically -- no special-casing needed;
                # gas exports at the cell's own composition.
                a_P = out_flux + a_t
                a_P_rel = a_P / relaxation
                add(k, k, a_P_rel)

                if j < nz - 1 and mesh.active[i, j + 1]:
                    add(k, _idx(i, j + 1, nr), -aN)
                if j > 0 and mesh.active[i, j - 1]:
                    add(k, _idx(i, j - 1, nr), -aS)
                if i < nr - 1 and mesh.active[i + 1, j]:
                    add(k, _idx(i + 1, j, nr), -aE)
                if i > 0 and mesh.active[i - 1, j]:
                    add(k, _idx(i - 1, j, nr), -aW)

                # RHS: time term carries old Y; mass source S_mass × V is
                # in kg/s and contributes directly.  Patankar deferred-
                # correction relaxation term is added.
                b[k] = (a_t * Y_gas[sp][i, j]
                        + sources_mass[sp][i, j] * V
                        + (1.0 - relaxation) / relaxation * a_P * Y_gas[sp][i, j])

                # Conservative tuyere inlet (Phase 10): the blast enters
                # through the raceway cell's wall-side face, so its
                # inflow coefficient (aE from the enforced face u_r) is
                # already in a_P but its donor cell is outside the
                # active domain.  Supply the donor composition on the
                # RHS -- the same face-flux pattern the energy solver
                # uses for the inlet enthalpy.  The previous Dirichlet
                # row replacement (Y pinned to blast in raceway cells)
                # was non-conservative: any gas circulating INTO those
                # cells was destroyed and re-created at blast
                # composition, which was the last elemental-H sink in
                # the shaft (~70 mol/s H2-equivalent).
                if raceway_mask_gas[i, j]:
                    b[k] += aE * Y_blast.get(sp, 0.0)

        A = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        x0 = Y_gas[sp].ravel(order="F")
        y, _ = solve_linear_system(A, b, x0=x0)
        # Clip negatives -- upwind on conservative transport keeps Y >= 0
        # for any positive inflow, but tiny negative drift from solver
        # tolerance is possible.
        new_Y[sp] = np.clip(y.reshape((nz, nr)).T, 0.0, 1.0)
        residuals[sp] = float(np.linalg.norm(A @ y - b))

    # ------------------------------------------------------------------
    # Conservation diagnostic: Σ Y_k should be 1.0 to within solver
    # tolerance.  We do NOT renormalise -- if the kinetics is mass-balanced
    # the discrete transport preserves Σ Y_k exactly.  Drift here would
    # indicate a real bug (e.g. an unbalanced reaction stoichiometry).
    # We tolerate drift up to 1% before reporting.
    # ------------------------------------------------------------------
    total_Y = sum(new_Y[sp] for sp in GAS_SPECIES)
    drift = np.abs(total_Y - 1.0)
    max_drift = float(drift[mesh.active].max()) if mesh.active.any() else 0.0
    if max_drift > 0.01:
        # Pragmatic safety net: if drift is >1% in any cell, the kinetics
        # source must have a small mass-balance error.  Renormalise gently
        # to keep the simulation stable, but log it via the "mass_balance"
        # residual.  This should be near zero in a healthy run.
        for sp in GAS_SPECIES:
            new_Y[sp] = new_Y[sp] / np.maximum(total_Y, 1e-9)
    residuals["mass_balance"] = max_drift

    # ------------------------------------------------------------------
    # Convert Y -> x for return (rest of solver still uses x).
    # ------------------------------------------------------------------
    new_x = _Y_to_x(new_Y)

    return new_x, residuals


# ---------------------------------------------------------------------------
def update_solid_phases(
    f_solid: Dict[str, np.ndarray],
    rho_s_bulk: np.ndarray,
    solid_sources: Dict[str, np.ndarray],
    u_s_descent: np.ndarray,
    mesh: Mesh2D,
    p: BFParameters2D,
    burden_inlet_composition: Dict[str, float],
    relaxation: float = 0.5,
    dt_override: float | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Advect + react solid mass-fraction fields.

    Parameters
    ----------
    dt_override : if not None, use this dt for the solid-phase time term
                  instead of p.dt.  This is essential for reaching steady
                  state quickly: the burden descent velocity is ~1e-3 m/s,
                  so with p.dt=0.5s the solid only moves 5e-4 m per outer
                  iteration -- thousands of iterations would be needed for
                  the burden to traverse the column.  Setting dt_override
                  to ~200s makes the solid time term comparable to the
                  advection term, so each outer iteration advances the
                  solid composition by a "real" 200 seconds of plant time.
                  This is the same continuation strategy used for energy.
    """
    nr, nz = mesh.nr, mesh.nz
    n = nr * nz
    new_f: Dict[str, np.ndarray] = {}
    residuals: Dict[str, float] = {}

    dt_use = dt_override if dt_override is not None else p.dt

    # ------------------------------------------------------------------
    # Kinematic radial-spreading descent (re-landed from Phase 11).
    #
    # The burden descends at constant RELATIVE radius x = r / R_wall(z)
    # (standard kinematic burden-descent assumption; Fu 2014 Sect. 3.1
    # defines Ks on r/R).  A purely axial upwind breaks at every profile
    # widening: outer columns have no donor and starve, inner columns
    # would need f > 1 to carry the charge at the plug velocity -- the
    # clip then destroyed 15-40% of every phase (solid-ledger
    # measurement, incl. 100% of produced Fe3O4/FeO -> Fe production
    # stuck at ~53% of plant -> coke_rate KPI inflated to ~600).  Each
    # receiver cell takes the row-above outflux of the donors
    # overlapping its RELATIVE-radius bin; overlap weights sum to 1 per
    # donor, so the operator is conservative by construction and
    # preserves radial burden profiles in x (the Fu Sect. 2.4 radial
    # charge composition survives the descent).
    # ------------------------------------------------------------------
    r_f = mesh.r_f
    donors: list[list[list[tuple[int, float]]]] = [[] for _ in range(nz)]
    for j in range(nz - 1):
        R_lo = max(float(mesh.r_wall[j]), 1e-9)       # receiver row
        R_hi = max(float(mesh.r_wall[j + 1]), 1e-9)   # donor row above
        row_map: list[list[tuple[int, float]]] = [[] for _ in range(nr)]
        for i in range(nr):                            # receiver bins
            if not mesh.active[i, j]:
                continue
            x_lo = min(r_f[i] / R_lo, 1.0)
            x_hi = min(r_f[i + 1] / R_lo, 1.0)
            for i2 in range(nr):                       # donor bins
                if not mesh.active[i2, j + 1]:
                    continue
                d_lo = min(r_f[i2] / R_hi, 1.0)
                d_hi = min(r_f[i2 + 1] / R_hi, 1.0)
                ov = min(x_hi, d_hi) - max(x_lo, d_lo)
                if ov > 1e-12 and (d_hi - d_lo) > 1e-12:
                    row_map[i].append((i2, ov / (d_hi - d_lo)))
        donors[j] = row_map

    for ph, f_field in f_solid.items():
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        b = np.zeros(n)

        def add(r: int, c: int, v: float) -> None:
            rows.append(r); cols.append(c); vals.append(v)

        S = solid_sources.get(ph, np.zeros_like(f_field))

        for j in range(nz):
            for i in range(nr):
                k = _idx(i, j, nr)
                if not mesh.active[i, j]:
                    add(k, k, 1.0); b[k] = f_field[i, j]
                    continue

                V = mesh.vol[i, j]
                A_s = mesh.area_z[i, j]
                rho_sol_s = _face_avg_z(rho_s_bulk, i, j,     nz)

                u_sol = -abs(u_s_descent[i, j])   # downward negative
                F_s = rho_sol_s * u_sol * A_s

                a_t = rho_s_bulk[i, j] * V / dt_use
                # Diagonal = ACTUAL outgoing solid flux + time term
                # (strict conservative upwind): downward flow exits
                # through the south face, |F_s|; it arrives in the row
                # below via the relative-radius transfer map.
                out_flux_s = max(-F_s, 0.0)
                a_P = out_flux_s + a_t
                a_P_rel = a_P / relaxation
                add(k, k, a_P_rel)

                # Inflow: donor cells of the row above at the same
                # relative radius, each contributing its own south-face
                # outflux weighted by the bin overlap.
                if j < nz - 1:
                    for (i2, w) in donors[j][i]:
                        rho_d = _face_avg_z(rho_s_bulk, i2, j + 1, nz)
                        F_d = rho_d * abs(u_s_descent[i2, j + 1]) \
                            * mesh.area_z[i2, j + 1]
                        if F_d > 0.0:
                            add(k, _idx(i2, j + 1, nr), -w * F_d)

                # Discrete equation balance:
                #   rho_s V (f_new - f_old)/dt + advection = S_phase * V
                # On the LHS we put a_P f_new (with rho V/dt as time term).
                # On the RHS we put a_t f_old + S_phase * V.
                # Units: a_t has [kg/s], a_t*f has [kg/s]; S [kg/(m^3 s)] * V
                # has [kg/s]. Both consistent.
                b[k] = (a_t * f_field[i, j]
                        + S[i, j] * V
                        + (1.0 - relaxation) / relaxation * a_P * f_field[i, j])

                if j == nz - 1:
                    # Replace row entirely with Dirichlet: f = burden_inlet
                    # (we add the diagonal contribution after the inner loop
                    # finalises the row).  To avoid cumulative additions
                    # we use a tracking flag.
                    pass

        # Apply Dirichlet on top row (burden inlet) by overwriting rows.
        # The inlet composition may be RADIAL (per-cell arrays, Fu 2014
        # Sect. 2.4 charging distribution: more ore at the wall, more
        # coke at the axis) or a scalar plant-average dict.  The
        # relative-radius descent map preserves the charged radial
        # profile (in x = r/R) all the way down the column.
        A = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tolil()
        inlet_ph = burden_inlet_composition.get(ph, 0.0)
        for i in range(nr):
            if mesh.active[i, nz - 1]:
                k = _idx(i, nz - 1, nr)
                A.rows[k] = [k]
                A.data[k] = [1.0]
                b[k] = (float(inlet_ph[i])
                        if isinstance(inlet_ph, np.ndarray) else inlet_ph)
        A = A.tocsr()
        x0 = f_field.ravel(order="F")
        x, _ = solve_linear_system(A, b, x0=x0)
        new_f[ph] = np.clip(x.reshape((nz, nr)).T, 0.0, 1.2)
        residuals[ph] = float(np.linalg.norm(A @ x - b))

    # NO drift-to-gangue redistribution and NO renormalisation (re-landed
    # from Phase 11).  The solid fields are PARTIAL LOADINGS relative to
    # the fixed kinematic carrier stream (rho_s_bulk * u_s): when the
    # burden loses mass to the gas (reduction O-removal, gasification,
    # calcination), Sigma f drops below 1 -- that IS the physics (the
    # stream shrinks).  The old redistribution back-filled every kilogram
    # lost to gas with phantom GANGUE (solid ledger: 75 kg/s gangue
    # exported vs 8.7 charged) and the overshoot guard destroyed
    # intermediate oxides wholesale.  Properties that need TRUE mass
    # fractions (cp, particle-size blending) normalise by Sigma f at the
    # point of use.
    return new_f, residuals


# ---------------------------------------------------------------------------
def compute_solid_descent_velocity(
    mesh: Mesh2D, p: BFParameters2D, rho_s_bulk: np.ndarray,
) -> np.ndarray:
    """Kinematic axial descent velocity [m/s].

    Radial profile: Fu 2014 Sect. 3.1 relative descent-speed factor
    ``Ks(r) = ks_center + (1 - ks_center) * r / R_wall(z)`` (solids
    descend faster near the wall where raceway coke consumption pulls
    the burden down; slower central descent = more residence heating =
    one driver of the inverted-V cohesive zone).  Ks is normalized by
    its area-weighted mean per z-plane so the total solid mass flow
    through every plane is conserved.  ``ks_center = 1.0`` reproduces
    the radially uniform plug descent.
    """
    u_s = np.zeros((mesh.nr, mesh.nz))
    mdot = p.coke_rate_kg_s + p.ore_rate_kg_s + p.flux_rate_kg_s
    ks_c = p.ks_center
    ring_area = np.pi * (mesh.r_f[1:] ** 2 - mesh.r_f[:-1] ** 2)  # (nr,)
    for j in range(mesh.nz):
        act = mesh.active[:, j]
        active_area = float(ring_area[act].sum())
        if active_area < 1e-6:
            continue
        r_local = max(float(mesh.r_wall[j]), 1e-9)
        ks = ks_c + (1.0 - ks_c) * np.clip(mesh.r_c / r_local, 0.0, 1.0)
        ks_mean = float((ks[act] * ring_area[act]).sum() / active_area)
        ks_norm = ks / max(ks_mean, 1e-12)
        for i in range(mesh.nr):
            if act[i]:
                u_s[i, j] = (ks_norm[i] * mdot
                             / (max(rho_s_bulk[i, j], 100.0) * active_area))
    return np.clip(u_s, 1e-6, 5.0e-3)
