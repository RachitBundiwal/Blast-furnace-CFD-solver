"""
Coupled gas-solid energy solver (Phase 1).

Changes from Phase 0
--------------------
* Convective fluxes use **face-interpolated density and Cp** (not cell-centre),
  restoring strict conservation across the gas density contrast
  between raceway (~0.8 kg/m^3) and shaft (~3 kg/m^3).
* **Radial convection is included** on both gas and solid phases; Phase 0
  dropped it, which let the raceway heat fail to spread laterally.
* **Wall heat loss is a Robin BC on the diagonal**:
  ``q = h_wall (T - T_coolant)`` with ``h_wall = wall_heat_loss_W_m2 / (T_ref - T_cool)``.
  Phase 0 pushed it into the RHS as a constant sink, which could drive wall
  cells below coolant temperature.
* Returns the **solve residual** ``||A x - b||_2`` so the orchestrator can
  report it as a per-equation convergence metric.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

from .constants import DELTA_H_RXN
from .kinetics import ReactionRates
from .linear_solver import solve_linear_system
from .mesh import Mesh2D
from .parameters import BFParameters2D
from .properties import ranz_marshall_h, bed_thermal_conductivity


# ---------------------------------------------------------------------------
def _idx_gas(i: int, j: int, nr: int) -> int:
    return i + nr * j


def _idx_solid(i: int, j: int, nr: int, nz: int) -> int:
    return nr * nz + i + nr * j


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
def solve_energy(
    T_g: np.ndarray,
    T_s: np.ndarray,
    u_r: np.ndarray,
    u_z: np.ndarray,
    rho_g: np.ndarray,
    rho_s_bulk: np.ndarray,
    cp_g: np.ndarray,
    cp_s: np.ndarray,
    eps: np.ndarray,
    dp_eff: np.ndarray,
    solid_descent_velocity: np.ndarray,
    rates: ReactionRates,
    mesh: Mesh2D,
    p: BFParameters2D,
    raceway_T_gas: np.ndarray,       # (nr, nz) Dirichlet T near tuyere faces
    raceway_mask_gas: np.ndarray,    # (nr, nz) bool where to force T_gas = T_blast
    relaxation: float = 0.7,
    dt_energy: float | None = None,  # if None, use p.dt
    tuyere_cell_mask: np.ndarray | None = None,    # (nr,nz) cells with tuyere east face
    T_tuyere_inlet: float | None = None,           # K, blast temperature
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Solve block-coupled (T_g, T_s) system.

    Parameters
    ----------
    dt_energy : optional float
        Effective pseudo-timestep for the energy equation.  Pass a value
        larger than p.dt (or float('inf')) to give the energy equation
        more freedom per outer iteration once the flow is stable.  This
        is the standard "energy continuation" trick in pseudo-transient
        SIMPLE solvers: tight dt for momentum/species stability, large
        dt for energy steady-state.

    Returns
    -------
    T_g_new, T_s_new, residual_gas, residual_solid
    """
    nr, nz = mesh.nr, mesh.nz
    ntot = 2 * nr * nz
    dt_eff = dt_energy if dt_energy is not None else p.dt

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(ntot)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    # Volumetric coupling
    u_z_c = 0.5 * (u_z[:, :-1] + u_z[:, 1:])
    u_r_c = 0.5 * (u_r[:-1, :] + u_r[1:, :])
    u_mag = np.sqrt(u_z_c ** 2 + u_r_c ** 2)
    h_v = ranz_marshall_h(T_g, T_s, rho_g, u_mag, dp_eff, eps)

    # ------------------------------------------------------------------
    # NTU-bounded gas-solid heat transfer
    # ------------------------------------------------------------------
    # The Ranz-Marshall correlation gives volumetric h_v [W/m^3/K] that
    # in narrow cells produces NTU = h_v * V / (m_dot_gas * cp_gas) >> 1,
    # which causes the gas to equilibrate instantly with the solid at
    # every axial level.  This is incorrect physics: the gas residence
    # time per cell is dz / u_z, and during that time the maximum heat
    # transfer is m_dot * cp_gas * (T_g - T_s).  A larger h_v cannot
    # transfer more heat than this hard limit.
    #
    # The standard fix is to cap h_v_eff so that h_v_eff * V <=
    # min(NTU_max, 1) * m_dot * cp.  Equivalently, the effectiveness
    # 1 - exp(-NTU) saturates at 1 for high NTU, so we use:
    #
    #     h_v_eff = (m_dot * cp / V) * (1 - exp(-NTU))
    #
    # which equals h_v at low NTU and saturates at m_dot*cp/V at high NTU.
    #
    # This is critical at the tuyere band where u_z is high and cells
    # are small -- without this cap, the blast equilibrates with cold
    # burden in the first cell and the flame temperature collapses to
    # the solid temperature (~1100 K instead of plant RAFT ~2100-2400 K).
    #
    # Reference: Wakao & Kaguei (1982) "Heat and Mass Transfer in Packed
    # Beds", Sect. 4.5 -- the effectiveness formulation appears in any
    # heat-exchanger textbook (e.g. Incropera Ch. 11).
    # ------------------------------------------------------------------
    # NTU-bounded gas-solid heat transfer (uses pre-computed cp_g passed in)
    # Use the FULL velocity magnitude and the cell's effective flow length,
    # not just the axial component / axial cell height.  At the tuyere
    # band, flow is dominantly radial (u_r ~ 50-150 m/s, u_z ~ 0.5 m/s),
    # so axial-only NTU bounds dramatically over-cap heat transfer.
    #
    # Residence-time formulation:
    #   τ_cell = V / Q_flow_through_cell
    #   m_dot_through_cell = rho * u_mag * A_perp
    # where A_perp is the cross-section perpendicular to u.  For axial-
    # dominant flow A_perp = area_z; for radial-dominant flow A_perp =
    # area_r.  Approximation: use V/τ_cell = m_dot * cp where
    #   m_dot = rho * u_mag * A_perp
    # On a finite cell V/τ ≈ V * u_mag / L_perp  with L_perp = V/A_perp.
    # So h_v_max = rho * u_mag * cp / L_perp  with L_perp the flow-aligned
    # cell length.
    #
    # Practical approximation: project u onto the dominant cell axis and
    # use that axis' length.
    h_v_max = np.zeros_like(h_v)
    for j in range(nz):
        for i in range(nr):
            if not mesh.active[i, j]:
                continue
            ur = abs(u_r_c[i, j])
            uz = abs(u_z_c[i, j])
            # Length scale aligned with dominant flow direction
            if ur > uz:
                L_flow = mesh.dr[i] if hasattr(mesh, 'dr') and i < len(mesh.dr) else 0.3
            else:
                L_flow = mesh.dz[j] if hasattr(mesh, 'dz') and j < len(mesh.dz) else 0.5
            u_dom = max(ur, uz, 0.05)
            # h_v_max = rho*u*cp / L_flow  (residence-time-based bound)
            h_v_max[i, j] = rho_g[i, j] * u_dom * cp_g[i, j] / max(L_flow, 0.01)

    # NTU-bounded effective coefficient
    # NTU = h_v / h_v_max gives effective NTU per cell-residence-time
    # h_v_eff = h_v_max * (1 - exp(-NTU))
    # which equals h_v for NTU<<1 and saturates at h_v_max for NTU>>1.
    NTU_local = h_v / np.maximum(h_v_max, 1e-3)
    h_v = h_v_max * (1.0 - np.exp(-np.minimum(NTU_local, 30.0)))

    k_eff_s = bed_thermal_conductivity(T_s, eps)
    k_g = 0.0241 * np.ones_like(T_g)   # simple placeholder; conduction is small

    # Reaction heat (positive = heat into phase)
    Q_gas = (
        -DELTA_H_RXN["R2_Fe2O3_CO"] * rates.R2
        - DELTA_H_RXN["R3_Fe3O4_CO"] * rates.R3
        - DELTA_H_RXN["R4_FeO_CO"]   * rates.R4
        - DELTA_H_RXN["R2H_Fe2O3_H2"] * rates.R2H
        - DELTA_H_RXN["R3H_Fe3O4_H2"] * rates.R3H
        - DELTA_H_RXN["R4H_FeO_H2"]   * rates.R4H
        - DELTA_H_RXN["R7_WGS"] * rates.R7
    )
    Q_solid = (
        -DELTA_H_RXN["R1_C_O2"]     * rates.R1
        - DELTA_H_RXN["R1b_C_halfO2"] * rates.R1b
        - DELTA_H_RXN["R5_Boudouard"] * rates.R5
        - DELTA_H_RXN["R6_WaterGas"]  * rates.R6
        - DELTA_H_RXN["R_DR"]         * rates.R_DR
        - DELTA_H_RXN["R9_calcination"] * rates.R9   # calcination heat sink
    )

    # Wall Robin BC: q = h_wall (T - T_wall), flux split 50/50 between
    # the gas and solid phases.  Constants are the published values from
    # Austin, Nogami & Yagi 1997 (ISIJ 37(8) p.751): h_wall = 20 W/m2K,
    # T_wall = 298 K, alpha_wall = 0.5.  (Previous implicit derivation
    # 9000/800 = 11.25 W/m2K to 373 K under-cooled the bosh corner ~2x.)
    T_wall_cool = p.wall_coolant_T_K
    h_wall = p.wall_htc_W_m2K

    for j in range(nz):
        for i in range(nr):
            kg = _idx_gas(i, j, nr)
            ks = _idx_solid(i, j, nr, nz)

            if not mesh.active[i, j]:
                add(kg, kg, 1.0); b[kg] = T_g[i, j]
                add(ks, ks, 1.0); b[ks] = T_s[i, j]
                continue

            V = mesh.vol[i, j]
            dz = mesh.dz[j]
            dr = mesh.dr[i]

            # Face-averaged rho*cp
            rho_cp_n = _face_avg_z(rho_g * cp_g, i, j + 1, nz)
            rho_cp_s = _face_avg_z(rho_g * cp_g, i, j,     nz)
            rho_cp_e = _face_avg_r(rho_g * cp_g, i + 1, j, nr)
            rho_cp_w = _face_avg_r(rho_g * cp_g, i,     j, nr)

            A_n = mesh.area_z[i, j + 1]
            A_s = mesh.area_z[i, j]
            A_e = mesh.area_r[i + 1, j]
            A_w = mesh.area_r[i, j]

            # Neighbour existence (diffusive conductances are per-face and
            # must vanish on boundary faces: a conductance left on the
            # diagonal without a matching off-diagonal entry is a phantom
            # Robin sink anchored to T = 0 K.  Before this fix every
            # boundary cell -- axis, wall, bottom row -- leaked heat that
            # way; at the wall the phantom sink was ~6x the intended stave
            # Robin loss and pulled toward 0 K instead of coolant T).
            has_n = j < nz - 1 and mesh.active[i, j + 1]
            has_s = j > 0 and mesh.active[i, j - 1]
            has_e = i < nr - 1 and mesh.active[i + 1, j]
            has_w = i > 0 and mesh.active[i - 1, j]

            # --- Gas block -----------------------------------------------
            F_n = rho_cp_n * u_z[i, j + 1] * A_n
            F_s = rho_cp_s * u_z[i, j]     * A_s
            F_e = rho_cp_e * u_r[i + 1, j] * A_e
            F_w = rho_cp_w * u_r[i, j]     * A_w
            aN = max(-F_n, 0.0); aS = max(F_s, 0.0)
            aE = max(-F_e, 0.0); aW = max(F_w, 0.0)

            D_n = k_g[i, j] * A_n / max(dz, 1e-3) if has_n else 0.0
            D_s = k_g[i, j] * A_s / max(dz, 1e-3) if has_s else 0.0
            D_e = k_g[i, j] * A_e / max(dr, 1e-3) if has_e else 0.0
            D_w = k_g[i, j] * A_w / max(dr, 1e-3) if has_w else 0.0

            a_t = eps[i, j] * rho_g[i, j] * cp_g[i, j] * V / dt_eff
            a_hv = h_v[i, j] * V

            # ------------------------------------------------------------
            # Top-row outflow BC for gas:
            #   ∂T/∂z = 0 at the top means the diffusive north-flux is 0
            #   (has_n is False there, so D_n = 0).  The convective north
            #   flux F_n * T_local IS still real (gas leaves the domain
            #   carrying its enthalpy out), but in upwind notation
            #   aN = max(-F_n, 0) represents the contribution from the
            #   NORTH neighbour back to the cell, which doesn't exist for
            #   outflow.  So we zero aN for top-row gas cells.
            # The S, E, W faces and time term remain unchanged.
            # ------------------------------------------------------------
            is_top_gas = (j == nz - 1)
            aN_eff = 0.0 if is_top_gas else aN

            a_P_g = (aN_eff + aS + aE + aW
                     + D_n + D_s + D_e + D_w + a_t + a_hv)

            # Robin wall BC on gas
            if mesh.is_wall[i, j]:
                A_wall = 2.0 * np.pi * mesh.r_wall[j] * dz
                a_wall = h_wall * A_wall * 0.5   # half to gas
                a_P_g += a_wall
                b[kg] += a_wall * T_wall_cool

            a_P_rel_g = a_P_g / relaxation
            add(kg, kg, a_P_rel_g)

            if has_s:
                add(kg, _idx_gas(i, j - 1, nr), -(aS + D_s))
            if has_n:
                add(kg, _idx_gas(i, j + 1, nr), -(aN + D_n))
            if has_w:
                add(kg, _idx_gas(i - 1, j, nr), -(aW + D_w))
            if has_e:
                add(kg, _idx_gas(i + 1, j, nr), -(aE + D_e))

            # Convective inlet on a tuyere east face: the east neighbour
            # cell does not exist in the active domain, so the blast
            # enthalpy aE * T_blast is added as an explicit RHS source.
            # Without this term, the inlet only delivers mass but no heat.
            if (tuyere_cell_mask is not None
                    and T_tuyere_inlet is not None
                    and tuyere_cell_mask[i, j]):
                b[kg] += aE * T_tuyere_inlet

            add(kg, ks, -a_hv)

            b[kg] += (Q_gas[i, j] * V
                      + a_t * T_g[i, j]
                      + (1.0 - relaxation) / relaxation * a_P_g * T_g[i, j])

            # Raceway Dirichlet override on gas
            if raceway_mask_gas[i, j]:
                # Dirichlet via row replacement is done after assembly
                pass

            # --- Solid block ---------------------------------------------
            # Solid moves purely axial, downward (u < 0 in our sign convention)
            u_sol = -abs(solid_descent_velocity[i, j])
            rho_cp_s_sol = rho_s_bulk[i, j] * cp_s[i, j]
            # Upwind on axial faces; solid comes from above (j+1)
            F_sol_n = rho_cp_s_sol * u_sol * A_n
            F_sol_s = rho_cp_s_sol * u_sol * A_s
            aN_s = max(-F_sol_n, 0.0)   # into cell from north (from above)
            aS_s = max(F_sol_s, 0.0)    # leaving south

            D_n_s = k_eff_s[i, j] * A_n / max(dz, 1e-3) if has_n else 0.0
            D_s_s = k_eff_s[i, j] * A_s / max(dz, 1e-3) if has_s else 0.0
            D_e_s = k_eff_s[i, j] * A_e / max(dr, 1e-3) if has_e else 0.0
            D_w_s = k_eff_s[i, j] * A_w / max(dr, 1e-3) if has_w else 0.0

            a_t_s = (1.0 - eps[i, j]) * rho_s_bulk[i, j] * cp_s[i, j] * V / dt_eff
            a_P_s = (aN_s + aS_s + D_n_s + D_s_s + D_e_s + D_w_s
                     + a_t_s + a_hv)

            if mesh.is_wall[i, j]:
                A_wall = 2.0 * np.pi * mesh.r_wall[j] * dz
                a_wall = h_wall * A_wall * 0.5
                a_P_s += a_wall
                b[ks] += a_wall * T_wall_cool

            a_P_rel_s = a_P_s / relaxation
            add(ks, ks, a_P_rel_s)

            if has_n:
                add(ks, _idx_solid(i, j + 1, nr, nz), -(aN_s + D_n_s))
            if has_s:
                add(ks, _idx_solid(i, j - 1, nr, nz), -(aS_s + D_s_s))
            if has_w:
                add(ks, _idx_solid(i - 1, j, nr, nz), -D_w_s)
            if has_e:
                add(ks, _idx_solid(i + 1, j, nr, nz), -D_e_s)

            add(ks, kg, -a_hv)

            b[ks] += (Q_solid[i, j] * V
                      + a_t_s * T_s[i, j]
                      + (1.0 - relaxation) / relaxation * a_P_s * T_s[i, j])

            # Top burden inlet Dirichlet handled after assembly via row replacement

    A = sp.coo_matrix((vals, (rows, cols)), shape=(ntot, ntot)).tolil()
    # Apply Dirichlet rows by replacement (avoids huge-diagonal residual artefact)
    for j in range(nz):
        for i in range(nr):
            if not mesh.active[i, j]:
                continue
            kg = _idx_gas(i, j, nr)
            ks = _idx_solid(i, j, nr, nz)
            # Raceway gas-T Dirichlet
            if raceway_mask_gas[i, j]:
                A.rows[kg] = [kg]
                A.data[kg] = [1.0]
                b[kg] = raceway_T_gas[i, j]
            # Top burden T_s Dirichlet
            if j == nz - 1:
                A.rows[ks] = [ks]
                A.data[ks] = [1.0]
                b[ks] = p.burden_inlet_temperature
    A = A.tocsr()

    x0 = np.concatenate([T_g.ravel(order="F"), T_s.ravel(order="F")])
    x, stats = solve_linear_system(A, b, x0=x0)

    T_g_new = x[:nr * nz].reshape((nz, nr)).T
    T_s_new = x[nr * nz:].reshape((nz, nr)).T

    T_g_new = np.clip(T_g_new, 280.0, 2600.0)
    T_s_new = np.clip(T_s_new, 280.0, 2300.0)

    # Per-equation L2 residuals
    res_vec = A @ x - b
    r_gas = float(np.linalg.norm(res_vec[:nr * nz]))
    r_solid = float(np.linalg.norm(res_vec[nr * nz:]))

    return T_g_new, T_s_new, r_gas, r_solid
