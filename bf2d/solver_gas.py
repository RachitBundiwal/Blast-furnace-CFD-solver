"""
Gas momentum + pressure coupling via SIMPLE-C with two-pass assembly.

Algorithm (per outer iteration)
-------------------------------
Pass 1: Assemble momentum rows for u_r and u_z.  Solve the momentum block
        with the **previous** pressure field as an explicit source.  Store
        the diagonal coefficient a_P at every face -- this is the
        Rhie-Chow / SIMPLE coupling factor.

Pass 2: Derive a pressure-Poisson equation from continuity by substituting
        the momentum-equation expression for u_face into div(rho u) = S:

            u_face = (H_face - A_face dP/dx_face) / a_P_face

        This gives, after rearrangement,

            div( (rho A^2 / a_P) grad P ) = div( rho H / a_P ) - S_mass

        which is a proper Laplacian on P with diagonal dominance, solved
        on cell centres with the standard 5-point stencil in axisymmetric
        coordinates.

Pass 3: Correct face velocities using the new pressure field:

            u_face = u*_face - (A_face / a_P_face) (dP/dx)_face

This is the standard SIMPLE scheme as documented in Patankar (1980),
Ch.6, and Versteeg-Malalasekera (2007), Ch.6.5.  It is *not* the broken
"axial-only predictor" of Phase 0 nor the singular fully-coupled block
of the first Phase 1 attempt -- it is the algorithm that every working
incompressible flow solver in production use is based on.

Boundary conditions
-------------------
* Axis (i_face = 0):    u_r = 0
* Wall faces:           u_r = 0 (no penetration)
* Tuyere wall faces:    u_r = -v_tuyere (Dirichlet inward)
* Bottom (j_face = 0):  u_z = 0 (closed hearth bottom)
* Top (j_face = nz):    Neumann zero-gradient on u_z (outflow)
* Pressure top axis:    Dirichlet P = top_pressure (reference pin)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

from .constants import G_ACC
from .linear_solver import LinearSolveStats, solve_linear_system
from .mesh import Mesh2D
from .parameters import BFParameters2D
from .properties import ergun_pressure_drop_coeff
from .raceway import Raceway


# ---------------------------------------------------------------------------
# Index helpers (all blocks now decoupled into 3 separate solves)
# ---------------------------------------------------------------------------
def _idx_P(i: int, j: int, nr: int) -> int:
    return i + nr * j

def _idx_uR(i_face: int, j: int, nr: int) -> int:
    return i_face + (nr + 1) * j

def _idx_uZ(i: int, j_face: int, nr: int) -> int:
    return i + nr * j_face


@dataclass
class MomentumSolution:
    u_r: np.ndarray
    u_z: np.ndarray
    P:   np.ndarray
    residual_continuity: float
    residual_r_mom: float
    residual_z_mom: float
    linear_stats: LinearSolveStats


# ===========================================================================
# Helpers
# ===========================================================================
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


# ===========================================================================
# Pass 1: Momentum solves (radial + axial), storing a_P diagonals
# ===========================================================================
def _assemble_radial_momentum(
    u_r_old: np.ndarray, u_z_old: np.ndarray, P_old: np.ndarray,
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray, dp_eff: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    relaxation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sps.csr_matrix, np.ndarray]:
    """Assemble & solve the radial momentum equation.

    Returns
    -------
    u_r_new, a_P_uR (diagonal coeff per face), H_uR (RHS minus pressure term),
    A_uR (sparse matrix used), b_uR.
    """
    nr, nz = mesh.nr, mesh.nz
    n_uR = (nr + 1) * nz
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n_uR)
    a_P_diag = np.ones(n_uR)            # store for pressure equation
    sum_aF   = np.zeros(n_uR)           # Σ|aF| for SIMPLEC velocity correction
    H_explicit = np.zeros(n_uR)         # H term (b without pressure-gradient
                                        #         contribution and without u^old time term)

    tuyere_faces = set(raceway.face_indices)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    for j in range(nz):
        for i_face in range(nr + 1):
            row = _idx_uR(i_face, j, nr)

            # Boundary faces
            if i_face == 0:
                add(row, row, 1.0); b[row] = 0.0
                a_P_diag[row] = 1.0
                continue
            i_west = i_face - 1
            i_east = i_face
            west_active = (i_west < nr) and mesh.active[i_west, j]
            east_active = (i_east < nr) and mesh.active[i_east, j]
            if not (west_active or east_active):
                add(row, row, 1.0); b[row] = 0.0
                a_P_diag[row] = 1.0
                continue
            if i_face == nr or (not east_active):
                if (i_face, j) in tuyere_faces:
                    # Mass-flux inlet: prescribe -G_tuyere/rho_face_local
                    # so that *mass flow* is conserved as local density
                    # changes (e.g. when raceway gets hot and dense).
                    # Without this, a velocity-pinned BC + variable density
                    # produces non-physical mass injection rates.
                    rho_face = (rho[i_west, j] if west_active
                                else rho[i_east, j] if east_active else 1.0)
                    u_dirichlet = -raceway.G_tuyere / max(rho_face, 1e-3)
                    add(row, row, 1.0); b[row] = u_dirichlet
                else:
                    add(row, row, 1.0); b[row] = 0.0
                a_P_diag[row] = 1.0
                continue

            # Interior r-face
            r_f = mesh.r_f[i_face]
            dr_west = mesh.dr[i_west]
            dr_east = mesh.dr[i_east]
            dz = mesh.dz[j]
            dr_face = 0.5 * (dr_west + dr_east)
            V_face = 2.0 * np.pi * r_f * dr_face * dz

            rho_f = 0.5 * (rho[i_west, j] + rho[i_east, j])
            mu_f  = 0.5 * (mu[i_west, j]  + mu[i_east, j])
            eps_f = 0.5 * (eps[i_west, j] + eps[i_east, j])
            dp_f  = 0.5 * (dp_eff[i_west, j] + dp_eff[i_east, j])

            u_r_here = u_r_old[i_face, j]
            u_z_here = 0.25 * (
                u_z_old[i_west, j] + u_z_old[i_west, j + 1] +
                u_z_old[i_east, j] + u_z_old[i_east, j + 1]
            )
            u_mag = np.sqrt(u_r_here**2 + u_z_here**2) + 1e-8

            K_e = ergun_pressure_drop_coeff(
                np.array([rho_f]), np.array([mu_f]),
                np.array([u_mag]), np.array([eps_f]),
                np.array([dp_f]),
            )[0]
            a_fric = K_e * V_face

            # Convection (first-order upwind)
            u_r_west = 0.5 * (u_r_old[i_face - 1, j] + u_r_here)
            u_r_east = (0.5 * (u_r_here + u_r_old[i_face + 1, j])
                        if (i_face + 1) <= nr else u_r_here)
            F_w = rho_f * u_r_west * 2.0 * np.pi * mesh.r_c[i_west] * dz
            F_e = rho_f * u_r_east * 2.0 * np.pi * mesh.r_c[i_east] * dz
            aW = max(F_w, 0.0); aE = max(-F_e, 0.0)

            u_z_n = 0.5 * (u_z_old[i_west, j + 1] + u_z_old[i_east, j + 1])
            u_z_s = 0.5 * (u_z_old[i_west, j]     + u_z_old[i_east, j])
            A_zring = np.pi * (mesh.r_f[i_east]**2 - mesh.r_c[i_west]**2)
            A_zring = max(A_zring, 1e-9)
            F_n = rho_f * u_z_n * A_zring
            F_s = rho_f * u_z_s * A_zring
            aN = max(-F_n, 0.0); aS = max(F_s, 0.0)

            D_r = mu_f * V_face / max(dr_face**2, 1e-6)
            D_z = mu_f * V_face / max(dz**2, 1e-6)
            a_t = eps_f * rho_f * V_face / p.dt

            a_P = a_fric + aW + aE + aN + aS + 2.0*D_r + 2.0*D_z + a_t
            a_P_rel = a_P / relaxation
            a_P_diag[row] = a_P_rel
            sum_aF[row]   = aW + aE + aN + aS + 2.0*D_r + 2.0*D_z
            add(row, row, a_P_rel)

            if i_face - 1 >= 0:
                add(row, _idx_uR(i_face - 1, j, nr), -(aW + D_r))
            if i_face + 1 <= nr:
                add(row, _idx_uR(i_face + 1, j, nr), -(aE + D_r))
            if j > 0:
                add(row, _idx_uR(i_face, j - 1, nr), -(aS + D_z))
            if j < nz - 1:
                add(row, _idx_uR(i_face, j + 1, nr), -(aN + D_z))

            # Pressure-gradient explicit source (kept on RHS)
            A_face = 2.0 * np.pi * r_f * dz
            P_west = P_old[i_west, j] if west_active else P_old[i_east, j]
            P_east = P_old[i_east, j] if east_active else P_old[i_west, j]
            b_press = -eps_f * (P_east - P_west) * A_face

            b_total = (b_press
                       + a_t * u_r_here
                       + (1.0 - relaxation) / relaxation * a_P * u_r_here)
            b[row] = b_total

            # H explicit (everything except pressure gradient and the
            # under-relaxation residual remainder); used for pressure eq.
            H_explicit[row] = (a_t * u_r_here
                               + (1.0 - relaxation) / relaxation * a_P * u_r_here)

    A = sps.coo_matrix((vals, (rows, cols)), shape=(n_uR, n_uR)).tocsr()
    x0 = u_r_old.ravel(order="F")
    u_r_flat, _ = solve_linear_system(A, b, x0=x0)
    u_r_new = u_r_flat.reshape((nz, nr + 1)).T

    return u_r_new, a_P_diag, H_explicit, A, b, sum_aF


def _assemble_axial_momentum(
    u_r_old: np.ndarray, u_z_old: np.ndarray, P_old: np.ndarray,
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray, dp_eff: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D,
    relaxation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sps.csr_matrix, np.ndarray]:
    """Assemble & solve the axial momentum equation."""
    nr, nz = mesh.nr, mesh.nz
    n_uZ = nr * (nz + 1)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n_uZ)
    a_P_diag = np.ones(n_uZ)
    sum_aF   = np.zeros(n_uZ)         # Σ|aF| for SIMPLEC
    H_explicit = np.zeros(n_uZ)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    for j_face in range(nz + 1):
        for i in range(nr):
            row = _idx_uZ(i, j_face, nr)

            j_south = j_face - 1
            j_north = j_face
            south_active = (j_south >= 0) and mesh.active[i, j_south]
            north_active = (j_north < nz) and mesh.active[i, j_north]

            if not (south_active or north_active):
                add(row, row, 1.0); b[row] = 0.0
                a_P_diag[row] = 1.0
                continue

            if j_face == 0:
                add(row, row, 1.0); b[row] = 0.0
                a_P_diag[row] = 1.0
                continue
            if j_face == nz:
                if south_active:
                    add(row, row, 1.0)
                    add(row, _idx_uZ(i, nz - 1, nr), -1.0)
                    b[row] = 0.0
                else:
                    add(row, row, 1.0); b[row] = 0.0
                a_P_diag[row] = 1.0
                continue

            # Interior z-face
            dz_face = 0.5 * (mesh.dz[j_south] + mesh.dz[j_north])
            dr = mesh.dr[i]
            A_face = np.pi * (mesh.r_f[i+1]**2 - mesh.r_f[i]**2)
            V_face = A_face * dz_face

            rho_f = 0.5 * (rho[i, j_south] + rho[i, j_north])
            mu_f  = 0.5 * (mu[i, j_south]  + mu[i, j_north])
            eps_f = 0.5 * (eps[i, j_south] + eps[i, j_north])
            dp_f  = 0.5 * (dp_eff[i, j_south] + dp_eff[i, j_north])

            u_z_here = u_z_old[i, j_face]
            u_r_here = 0.25 * (
                u_r_old[i, j_south] + u_r_old[i+1, j_south] +
                u_r_old[i, j_north] + u_r_old[i+1, j_north]
            )
            u_mag = np.sqrt(u_r_here**2 + u_z_here**2) + 1e-8

            K_e = ergun_pressure_drop_coeff(
                np.array([rho_f]), np.array([mu_f]),
                np.array([u_mag]), np.array([eps_f]),
                np.array([dp_f]),
            )[0]
            a_fric = K_e * V_face

            # Convection in z
            u_z_s_face = (0.5 * (u_z_old[i, j_face - 1] + u_z_here)
                          if j_face > 0 else u_z_here)
            u_z_n_face = (0.5 * (u_z_here + u_z_old[i, j_face + 1])
                          if j_face < nz else u_z_here)
            F_s = rho_f * u_z_s_face * A_face
            F_n = rho_f * u_z_n_face * A_face
            aS = max(F_s, 0.0); aN = max(-F_n, 0.0)

            # Convection in r
            u_r_w = 0.5 * (u_r_old[i, j_south] + u_r_old[i, j_north])
            u_r_e = 0.5 * (u_r_old[i+1, j_south] + u_r_old[i+1, j_north])
            A_rw = 2.0 * np.pi * mesh.r_f[i] * dz_face
            A_re = 2.0 * np.pi * mesh.r_f[i+1] * dz_face
            F_w = rho_f * u_r_w * A_rw
            F_e = rho_f * u_r_e * A_re
            aW = max(F_w, 0.0); aE = max(-F_e, 0.0)

            D_r = mu_f * V_face / max(dr**2, 1e-6)
            D_z = mu_f * V_face / max(dz_face**2, 1e-6)
            a_t = eps_f * rho_f * V_face / p.dt

            a_P = a_fric + aS + aN + aW + aE + 2.0*D_r + 2.0*D_z + a_t
            a_P_rel = a_P / relaxation
            a_P_diag[row] = a_P_rel
            sum_aF[row]   = aS + aN + aW + aE + 2.0*D_r + 2.0*D_z
            add(row, row, a_P_rel)

            if j_face - 1 >= 0:
                add(row, _idx_uZ(i, j_face - 1, nr), -(aS + D_z))
            if j_face + 1 <= nz:
                add(row, _idx_uZ(i, j_face + 1, nr), -(aN + D_z))
            if i > 0:
                add(row, _idx_uZ(i - 1, j_face, nr), -(aW + D_r))
            if i < nr - 1:
                add(row, _idx_uZ(i + 1, j_face, nr), -(aE + D_r))

            # Pressure gradient explicit source
            P_south = P_old[i, j_south] if south_active else P_old[i, j_north]
            P_north = P_old[i, j_north] if north_active else P_old[i, j_south]
            b_press = -eps_f * (P_north - P_south) * A_face

            # Gravity (acts downward, reduces upward acceleration)
            b_grav = -eps_f * rho_f * G_ACC * V_face

            H_explicit[row] = (a_t * u_z_here + b_grav
                               + (1.0 - relaxation) / relaxation * a_P * u_z_here)
            b[row] = b_press + H_explicit[row]

    A = sps.coo_matrix((vals, (rows, cols)), shape=(n_uZ, n_uZ)).tocsr()
    x0 = u_z_old.ravel(order="F")
    u_z_flat, _ = solve_linear_system(A, b, x0=x0)
    u_z_new = u_z_flat.reshape((nz + 1, nr)).T

    return u_z_new, a_P_diag, H_explicit, A, b, sum_aF


# ===========================================================================
# Pass 2: Pressure-correction Poisson equation (SIMPLE)
# ===========================================================================
def _solve_pressure_correction(
    u_r_star: np.ndarray, u_z_star: np.ndarray, P_old: np.ndarray,
    a_P_uR: np.ndarray, a_P_uZ: np.ndarray,
    rho: np.ndarray, mass_source: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    gas_exit_mask: np.ndarray | None = None,
    sum_aF_uR: np.ndarray | None = None,
    sum_aF_uZ: np.ndarray | None = None,
) -> tuple[np.ndarray, sps.csr_matrix, np.ndarray]:
    """Solve  div( (rho A^2 / a_P_eff) grad P') = div(rho u*) - S_mass.

    Returns the pressure correction P' (zero at reference cell).

    SIMPLEC variant
    ---------------
    When `sum_aF_uR` and `sum_aF_uZ` are supplied (the sum of off-diagonal
    coefficients per face), the velocity-correction denominator becomes
    `a_P - Σa_F` instead of `a_P` (Moukalled, Mangani & Darwish 2016
    Eq. 15.172 and Sect. 15.7.1).  In packed-bed flow this reduces to
    `a_fric + a_t` (Ergun friction + time-term), which is the physically
    correct Darcy-Forchheimer permeability.  Standard SIMPLE uses `a_P`
    which over-estimates the resistance to pressure correction by 1-2
    orders of magnitude in porous media, leading to the slow / pathological
    convergence we observed (recirculation zones, mass-conservation drift).
    """
    nr, nz = mesh.nr, mesh.nz
    n_P = nr * nz
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n_P)

    tuyere_faces = set(raceway.face_indices)

    # ------------------------------------------------------------------
    # SIMPLEC effective denominators (Moukalled Eq. 15.172).
    #
    # The momentum solver stored a_P_rel = a_P_unrelaxed / ω in
    # a_P_uR / a_P_uZ.  SIMPLEC replaces the pressure-correction
    # denominator by (a_P_unrelaxed - Σa_F).  For a packed bed this
    # equals (a_fric + a_t), which is physically the correct
    # Darcy-Forchheimer permeability.
    #
    # ω is recovered from the parameters object via p.momentum_relaxation
    # (this is the value used by the momentum solver for both u_r and u_z
    # in the present staggered-grid design).
    # ------------------------------------------------------------------
    omega = max(p.momentum_relaxation, 1e-3)
    if sum_aF_uR is not None:
        aP_un_uR = a_P_uR * omega
        aP_eff_uR = np.maximum(aP_un_uR - sum_aF_uR, 0.05 * aP_un_uR)
        # Preserve Dirichlet rows (where a_P_uR == 1.0)
        aP_eff_uR = np.where(a_P_uR < 1.5, a_P_uR, aP_eff_uR)
    else:
        aP_eff_uR = a_P_uR

    if sum_aF_uZ is not None:
        aP_un_uZ = a_P_uZ * omega
        aP_eff_uZ = np.maximum(aP_un_uZ - sum_aF_uZ, 0.05 * aP_un_uZ)
        aP_eff_uZ = np.where(a_P_uZ < 1.5, a_P_uZ, aP_eff_uZ)
    else:
        aP_eff_uZ = a_P_uZ

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    # Top row gas-exit Dirichlet: pin P_prime = 0 only on cells that lie
    # in the uptake ring (gas_exit_mask).  Top cells outside the ring are
    # treated as sealed (no-flux) -- they just see their normal pressure
    # equation with the north face having zero d_n (handled below by the
    # face-area logic).
    # If no mask is provided, fall back to "all top cells are exit cells".

    for j in range(nz):
        for i in range(nr):
            row = _idx_P(i, j, nr)

            if not mesh.active[i, j]:
                add(row, row, 1.0); b[row] = 0.0
                continue
            # Top row: pin at exit cells only
            if j == nz - 1:
                is_exit = (gas_exit_mask is None) or gas_exit_mask[i, j]
                if is_exit:
                    add(row, row, 1.0); b[row] = 0.0
                    continue
                # Sealed top cell: normal Poisson assembly, but the north
                # face has no neighbour and no Dirichlet pin.  d_n will
                # naturally be 0 because the u_z[:, nz] = 0 there.

            A_w = mesh.area_r[i, j]
            A_e = mesh.area_r[i+1, j]
            A_s = mesh.area_z[i, j]
            A_n = mesh.area_z[i, j+1]

            rho_w = _face_avg_r(rho, i,     j, nr)
            rho_e = _face_avg_r(rho, i+1,   j, nr)
            rho_s = _face_avg_z(rho, i, j,     nz)
            rho_n = _face_avg_z(rho, i, j+1,   nz)

            # Coupling coefficients d_face = rho A^2 / a_P_eff
            #
            # SIMPLEC (Moukalled Eq. 15.172 / Sect. 15.7.1) uses a_P - Σa_F
            # as the velocity-correction denominator.  In packed-bed flow
            # this reduces to a_fric + a_t (Darcy-Forchheimer permeability),
            # which is physically correct.  Standard SIMPLE uses a_P, which
            # is 100-1000× too stiff at typical BF velocities and produces
            # the slow / pathological convergence we previously observed.

            # West face (i, j) on u_r grid -- skipped if Dirichlet (axis or wall)
            d_w = 0.0
            if i > 0:
                aP_face = aP_eff_uR[_idx_uR(i, j, nr)]
                if aP_face > 1.5:   # not a Dirichlet placeholder (1.0)
                    d_w = rho_w * A_w * A_w / aP_face

            # East face (i+1, j) -- skipped if wall (incl. tuyere)
            d_e = 0.0
            if i + 1 <= nr:
                is_tuyere = (i + 1, j) in tuyere_faces
                is_wall = (i + 1 == nr) or (i + 1 < nr and not mesh.active[i + 1, j])
                if not is_wall and not is_tuyere:
                    aP_face = aP_eff_uR[_idx_uR(i + 1, j, nr)]
                    if aP_face > 1.5:
                        d_e = rho_e * A_e * A_e / aP_face

            # South face (i, j) on u_z grid -- skipped at j_face=0 (closed bottom)
            d_s = 0.0
            if j > 0:
                aP_face = aP_eff_uZ[_idx_uZ(i, j, nr)]
                if aP_face > 1.5:
                    d_s = rho_s * A_s * A_s / aP_face

            # North face (i, j+1) -- exists at top too (Neumann outflow)
            d_n = 0.0
            if j + 1 <= nz:
                aP_face = aP_eff_uZ[_idx_uZ(i, j + 1, nr)]
                if aP_face > 1.5:
                    d_n = rho_n * A_n * A_n / aP_face

            a_P = d_w + d_e + d_s + d_n
            if a_P < 1e-20:
                add(row, row, 1.0); b[row] = 0.0
                continue

            # ------------------------------------------------------------
            # Pressure-correction diagonal augmentation.
            #
            # With SIMPLEC (used in this solver), the pressure-correction
            # equation does not need under-relaxation in the canonical
            # incompressible-flow case (Moukalled, Mangani & Darwish 2016
            # §15.7.1, p623: "Due to a better estimate in SIMPLEC ... the
            # relaxation of pressure becomes unnecessary").
            #
            # However, in packed-bed flow with strong density variation
            # (cohesive-zone porosity collapse, raceway temperature jump),
            # the d_face coefficients can still vary by 100x-1000x between
            # adjacent cells.  A small augmentation (alpha = 0.7-0.9)
            # protects against ill-conditioned cells without slowing the
            # bulk convergence.
            #
            # Patankar (1980) Eq. 6.7-3: a_P_aug = a_P / alpha,
            #                            b_aug   = b + (a_P_aug - a_P) * P'_old
            # We resolve P' from scratch each outer iteration so P'_old = 0
            # and only the diagonal grows.
            # ------------------------------------------------------------
            alpha_p_relax = 0.3
            a_P_aug = a_P / alpha_p_relax
            add(row, row, a_P_aug)
            if i > 0 and mesh.active[i-1, j] and d_w > 0:
                add(row, _idx_P(i-1, j, nr), -d_w)
            if i < nr - 1 and mesh.active[i+1, j] and d_e > 0:
                add(row, _idx_P(i+1, j, nr), -d_e)
            if j > 0 and mesh.active[i, j-1] and d_s > 0:
                add(row, _idx_P(i, j-1, nr), -d_s)
            if j < nz - 1 and mesh.active[i, j+1] and d_n > 0:
                add(row, _idx_P(i, j+1, nr), -d_n)

            # RHS: -div(rho u*) + S_mass*V
            # Tuyere faces: the east face velocity is hard-pinned by the
            # Dirichlet BC (-v_tuyere), so its flux is KNOWN and contributes
            # as an explicit mass source rather than a SIMPLE-corrected term.
            flux_w = rho_w * u_r_star[i, j] * A_w
            flux_e = rho_e * u_r_star[i+1, j] * A_e
            flux_s = rho_s * u_z_star[i, j] * A_s
            flux_n = rho_n * u_z_star[i, j+1] * A_n
            div_rhou_star = (flux_e - flux_w) + (flux_n - flux_s)
            V = mesh.vol[i, j]
            b[row] = -div_rhou_star + mass_source[i, j] * V

    A = sps.coo_matrix((vals, (rows, cols)), shape=(n_P, n_P)).tocsr()
    x, stats = solve_linear_system(A, b, x0=np.zeros(n_P))
    P_prime = x.reshape((nz, nr)).T

    return P_prime, A, b


# ===========================================================================
# Pass 3: Velocity correction
# ===========================================================================
def _correct_velocities(
    u_r_star: np.ndarray, u_z_star: np.ndarray, P_prime: np.ndarray,
    a_P_uR: np.ndarray, a_P_uZ: np.ndarray, rho: np.ndarray,
    mesh: Mesh2D, raceway: Raceway,
    gas_exit_mask: np.ndarray | None = None,
    sum_aF_uR: np.ndarray | None = None,
    sum_aF_uZ: np.ndarray | None = None,
    omega: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply  u' = -(A_face/a_P_eff) (P_east' - P_west')  to face velocities.

    SIMPLEC variant: a_P_eff = a_P - Σa_F (Moukalled Eq. 15.172).  Falls
    back to standard SIMPLE if sum_aF_* arrays are not supplied.
    """
    # Build effective denominators (must match what _solve_pressure_correction
    # used, otherwise the correction step is inconsistent with the pressure
    # equation that produced P_prime).
    if sum_aF_uR is not None:
        aP_un_uR = a_P_uR * omega
        aP_eff_uR = np.maximum(aP_un_uR - sum_aF_uR, 0.05 * aP_un_uR)
        aP_eff_uR = np.where(a_P_uR < 1.5, a_P_uR, aP_eff_uR)
    else:
        aP_eff_uR = a_P_uR

    if sum_aF_uZ is not None:
        aP_un_uZ = a_P_uZ * omega
        aP_eff_uZ = np.maximum(aP_un_uZ - sum_aF_uZ, 0.05 * aP_un_uZ)
        aP_eff_uZ = np.where(a_P_uZ < 1.5, a_P_uZ, aP_eff_uZ)
    else:
        aP_eff_uZ = a_P_uZ

    nr, nz = mesh.nr, mesh.nz
    u_r_new = u_r_star.copy()
    u_z_new = u_z_star.copy()

    # Radial corrections
    for j in range(nz):
        for i_face in range(1, nr):       # axis face skipped (Dirichlet)
            i_west = i_face - 1; i_east = i_face
            if not (mesh.active[i_west, j] and mesh.active[i_east, j]):
                continue
            aP = aP_eff_uR[_idx_uR(i_face, j, nr)]
            if aP < 1.5:
                continue
            A_face = mesh.area_r[i_face, j]
            dP = P_prime[i_east, j] - P_prime[i_west, j]
            u_r_new[i_face, j] = u_r_star[i_face, j] - A_face / aP * dP

    # Axial corrections
    for j_face in range(1, nz):           # bottom skipped, top is Neumann
        for i in range(nr):
            j_south = j_face - 1; j_north = j_face
            if not (mesh.active[i, j_south] and mesh.active[i, j_north]):
                continue
            aP = aP_eff_uZ[_idx_uZ(i, j_face, nr)]
            if aP < 1.5:
                continue
            A_face = mesh.area_z[i, j_face]
            dP = P_prime[i, j_north] - P_prime[i, j_south]
            u_z_new[i, j_face] = u_z_star[i, j_face] - A_face / aP * dP

    # Re-apply Dirichlet: axis u_r=0, walls u_r=0, tuyere mass-flux inlets,
    # hearth bottom u_z=0
    u_r_new[0, :] = 0.0
    u_r_new[nr, :] = 0.0
    # NOTE: tuyere face velocity is set in the predictor with current
    # density; we don't override here because the predictor's value is
    # already correct.  Only re-impose hard-wall faces:
    u_z_new[:, 0] = 0.0

    # ----------------------------------------------------------------
    # Top face outflow BC, following Moukalled §15.6.1.3 (Eq. 15.140-15.142)
    # plus §15.6.2.3 (Eq. 15.169) for global mass-flow correction.
    #
    # Step 1: extrapolate u_z at the top face (zero-gradient).
    # Step 2: apply a single uniform scaling to all outlet face velocities
    #         so total outlet mass = total inlet mass.
    #
    # The scaling enforces global mass conservation across the SIMPLEC
    # iteration loop.  Local divergence near the outlet is corrected by
    # the next iteration's pressure correction.  This is the canonical
    # treatment for pressure-specified outlets (Versteeg & Malalasekera
    # 2007 Sect. 9.7; Moukalled et al. 2016 Eq. 15.169).
    # ----------------------------------------------------------------
    if nz > 0:
        if gas_exit_mask is None:
            top_exit = mesh.active[:, nz - 1]
        else:
            top_exit = gas_exit_mask[:, nz - 1] & mesh.active[:, nz - 1]

        # Step 1: zero-gradient extrapolation at top face
        for i in range(nr):
            if top_exit[i]:
                u_z_new[i, nz] = max(u_z_new[i, nz - 1], 0.0)
            else:
                u_z_new[i, nz] = 0.0

        # Step 2: global mass-flow scaling (Moukalled Eq. 15.169)
        m_dot_in = 0.0
        for i_face, j in raceway.face_indices:
            rho_t = rho[nr - 1, j] if i_face == nr else rho[i_face, j]
            A_t = mesh.area_r[i_face, j]
            m_dot_in += rho_t * abs(u_r_new[i_face, j]) * A_t

        m_dot_out = 0.0
        for i in range(nr):
            if top_exit[i]:
                m_dot_out += rho[i, nz - 1] * u_z_new[i, nz] * mesh.area_z[i, nz]

        if m_dot_in > 1e-9:
            if m_dot_out > 0.01 * m_dot_in:
                # Scale extrapolated velocities to match global mass balance
                scale = m_dot_in / m_dot_out
                for i in range(nr):
                    if top_exit[i]:
                        u_z_new[i, nz] *= scale
            else:
                # Bootstrap: extrapolation produced near-zero outflow.
                # Use uniform velocity that satisfies global mass.
                total_rhoA = sum(rho[i, nz - 1] * mesh.area_z[i, nz]
                                 for i in range(nr) if top_exit[i])
                if total_rhoA > 1e-9:
                    u_z_uniform = m_dot_in / total_rhoA
                    for i in range(nr):
                        if top_exit[i]:
                            u_z_new[i, nz] = u_z_uniform

    return u_r_new, u_z_new


# ===========================================================================
# Public entry point
# ===========================================================================
def solve_coupled_momentum(
    u_r_old: np.ndarray, u_z_old: np.ndarray, P_old: np.ndarray,
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray, dp_eff: np.ndarray,
    mass_source: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    relaxation: float = 0.5,
    gas_exit_mask: np.ndarray | None = None,
) -> MomentumSolution:
    """One full SIMPLE outer step (predictor + pressure correction + velocity update).

    Uses SIMPLEC (Moukalled, Mangani & Darwish 2016 §15.7.1) for the
    velocity-correction denominator, which gives Darcy-Forchheimer
    permeability in the porous-bed limit.  Standard SIMPLE produces
    pathological recirculation in narrow-inlet/narrow-outlet packed-bed
    geometries; SIMPLEC eliminates this by using a_P - Σa_F instead of
    a_P in the velocity correction.
    """
    # Pass 1: momentum predictors (return Σa_F per face for SIMPLEC)
    u_r_star, a_P_uR, _, A_uR, b_uR, sum_aF_uR = _assemble_radial_momentum(
        u_r_old, u_z_old, P_old, rho, mu, eps, dp_eff, mesh, p, raceway, relaxation,
    )
    u_z_star, a_P_uZ, _, A_uZ, b_uZ, sum_aF_uZ = _assemble_axial_momentum(
        u_r_old, u_z_old, P_old, rho, mu, eps, dp_eff, mesh, p, relaxation,
    )

    # Pass 2: pressure correction (SIMPLEC)
    P_prime, A_P, b_P = _solve_pressure_correction(
        u_r_star, u_z_star, P_old, a_P_uR, a_P_uZ, rho, mass_source,
        mesh, p, raceway, gas_exit_mask=gas_exit_mask,
        sum_aF_uR=sum_aF_uR, sum_aF_uZ=sum_aF_uZ,
    )

    # Pass 3: velocity correction (SIMPLEC -- must use same denominator
    # as the pressure equation, otherwise the corrected velocities won't
    # satisfy continuity).
    u_r_new, u_z_new = _correct_velocities(
        u_r_star, u_z_star, P_prime, a_P_uR, a_P_uZ, rho, mesh, raceway,
        gas_exit_mask=gas_exit_mask,
        sum_aF_uR=sum_aF_uR, sum_aF_uZ=sum_aF_uZ,
        omega=relaxation,
    )

    # ----------------------------------------------------------------
    # Pressure update with SIMPLEC.
    #
    # Moukalled §15.7.1 (p623): "Due to a better estimate in SIMPLEC...
    # the relaxation of pressure becomes unnecessary".  In practice we
    # use alpha_P = 0.3 (Patankar-recommended for SIMPLE).  Bumping to
    # 0.7+ destabilises packed-bed flow with strong density variation.
    # ----------------------------------------------------------------
    alpha_P = 0.3
    P_new = P_old + alpha_P * P_prime
    P_new[~mesh.active] = P_old[~mesh.active]
    P_new = np.clip(P_new, 2.0e4, 1.5e6)
    # Re-impose top pressure Dirichlet (gas outlet) at exit cells only
    if gas_exit_mask is not None:
        for i in range(mesh.nr):
            if gas_exit_mask[i, mesh.nz - 1] and mesh.active[i, mesh.nz - 1]:
                P_new[i, mesh.nz - 1] = p.top_pressure
    else:
        for i in range(mesh.nr):
            if mesh.active[i, mesh.nz - 1]:
                P_new[i, mesh.nz - 1] = p.top_pressure

    # ----------------------------------------------------------------
    # Residuals
    # ----------------------------------------------------------------
    P_old_norm = max(float(np.linalg.norm(P_old.ravel())), 1.0)
    P_prime_norm = float(np.linalg.norm(P_prime.ravel()))
    u_r_corr_norm = float(np.linalg.norm((u_r_new - u_r_star).ravel()))
    u_z_corr_norm = float(np.linalg.norm((u_z_new - u_z_star).ravel()))
    u_r_norm = max(float(np.linalg.norm(u_r_new.ravel())), 1.0)
    u_z_norm = max(float(np.linalg.norm(u_z_new.ravel())), 1.0)

    res_r = u_r_corr_norm / u_r_norm
    res_z = u_z_corr_norm / u_z_norm
    res_P = P_prime_norm / P_old_norm

    return MomentumSolution(
        u_r=u_r_new, u_z=u_z_new, P=P_new,
        residual_r_mom=res_r, residual_z_mom=res_z,
        residual_continuity=res_P,
        linear_stats=LinearSolveStats("simple_three_pass", 0, 0.0, True),
    )
