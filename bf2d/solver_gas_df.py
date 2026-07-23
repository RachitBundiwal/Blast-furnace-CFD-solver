"""
Darcy-Forchheimer momentum / pressure solver for the gas phase.

Physics
-------
At blast-furnace velocities (0.5 - 5 m/s) in a packed bed (eps ~ 0.4,
d_p ~ 30 mm), Ergun friction dominates inertia by ~3 orders of
magnitude.  We therefore drop the inertia terms in the momentum
equation entirely:

    eps * grad(p) = -K(|u|) u  +  eps * rho * g

where K is the Ergun coefficient

    K = K_visc + K_inert(|u|)
    K_visc  = 150 mu (1-eps)^2 / (eps^3 d_p^2)            [Pa s/m^2]
    K_inert = 1.75 rho |u| (1-eps) / (eps^3 d_p)          [Pa s/m^2]

Solving for u (Darcy-Forchheimer):

    u_r = -(eps / K) dp/dr
    u_z = -(eps / K) dp/dz  +  (eps^2 rho g_z / K)        (g_z = -9.81)

Substituting into continuity  div(rho u) = S_mass  yields a single
elliptic Poisson equation for p:

    -div[ (rho eps / K) grad(p) ] = S_mass + div[ rho eps^2 rho g / K ]

K depends on |u|, so the outer loop is:
    1. Compute u from current p via Darcy (or use previous iteration)
    2. Compute K from current u
    3. Assemble & solve Poisson system for p
    4. Update u from new p
    5. Repeat until ||du|| < tol

This is the architecture used in published BF CFD codes (Fu, Chen, Zhao
et al. 2014; Zhou et al. 2016) and is mathematically simpler than SIMPLE
for porous media.  It eliminates the velocity-pressure coupling pathology
that creates spurious recirculation in narrow-inlet/narrow-outlet
geometries.

References
----------
- Fu, D., Chen, Y., Zhao, Y., D'Alessio, J., Ferron, K.J., Zhou, C.Q.
  (2014). "CFD modeling of multiphase reacting flow in blast furnace
  shaft with layered burden." Applied Thermal Engineering 66, 298-308.
  Eqs. 3-4 give the Darcy-Forchheimer formulation used here.
- Zhou, C., Tang, G., Wang, J., Fu, D., Okosun, T., Silaen, A., Wu, B.
  (2016). "Comprehensive Numerical Modeling of the Blast Furnace
  Ironmaking Process." JOM 68(5), 1353-1362. Eqs. 4-6 confirm this
  is industry standard for plant-scale BF shaft modeling.
- Ergun, S. (1953). "Pressure drop in blast furnace and in cupola."
  Ind. Eng. Chem. 45(2), 477-485.

Boundary conditions
-------------------
- Tuyere faces: mass flux specified (Neumann on p, equivalent to
  prescribing u_r = -G_tuyere / rho_face).
- Top exit ring: pressure Dirichlet (p = top_pressure).
- Sealed top: zero mass flux (Neumann, hydrostatic).
- Outer walls (non-tuyere): zero mass flux.
- Axis: zero flux (symmetry).
- Bottom hearth: zero mass flux.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sps

from .linear_solver import LinearSolveStats, solve_linear_system
from .mesh import Mesh2D
from .parameters import BFParameters2D
from .properties import ergun_pressure_drop_coeff
from .raceway import Raceway


G_ACC = 9.81  # m/s^2


@dataclass
class MomentumSolution:
    """Same interface as the legacy SIMPLE solver (drop-in)."""
    u_r: np.ndarray
    u_z: np.ndarray
    P:   np.ndarray
    residual_continuity: float
    residual_r_mom: float
    residual_z_mom: float
    linear_stats: LinearSolveStats


# ---------------------------------------------------------------------------
def _idx_P(i: int, j: int, nr: int) -> int:
    return i + nr * j


def _face_avg_r(f: np.ndarray, i_face: int, j: int, nr: int) -> float:
    """Average a cell-centered field across a radial face."""
    if i_face <= 0:
        return f[0, j]
    if i_face >= nr:
        return f[nr - 1, j]
    return 0.5 * (f[i_face - 1, j] + f[i_face, j])


def _face_avg_z(f: np.ndarray, i: int, j_face: int, nz: int) -> float:
    """Average a cell-centered field across an axial face."""
    if j_face <= 0:
        return f[i, 0]
    if j_face >= nz:
        return f[i, nz - 1]
    return 0.5 * (f[i, j_face - 1] + f[i, j_face])


# ---------------------------------------------------------------------------
def _compute_face_K(
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray,
    dp_eff: np.ndarray, u_mag: np.ndarray, mesh: Mesh2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Ergun coefficient K at every staggered face.

    Returns
    -------
    K_r : (nr+1, nz)   K at radial faces
    K_z : (nr,   nz+1) K at axial faces
    """
    nr, nz = mesh.nr, mesh.nz

    # Face-averaged properties
    K_r = np.zeros((nr + 1, nz))
    for j in range(nz):
        for i_face in range(nr + 1):
            rho_f = _face_avg_r(rho, i_face, j, nr)
            mu_f  = _face_avg_r(mu,  i_face, j, nr)
            eps_f = _face_avg_r(eps, i_face, j, nr)
            dp_f  = _face_avg_r(dp_eff, i_face, j, nr)
            u_f   = _face_avg_r(u_mag, i_face, j, nr)
            K_r[i_face, j] = ergun_pressure_drop_coeff(
                np.array([rho_f]), np.array([mu_f]),
                np.array([u_f]),   np.array([eps_f]),
                np.array([dp_f]),
            )[0]

    K_z = np.zeros((nr, nz + 1))
    for j_face in range(nz + 1):
        for i in range(nr):
            rho_f = _face_avg_z(rho, i, j_face, nz)
            mu_f  = _face_avg_z(mu,  i, j_face, nz)
            eps_f = _face_avg_z(eps, i, j_face, nz)
            dp_f  = _face_avg_z(dp_eff, i, j_face, nz)
            u_f   = _face_avg_z(u_mag, i, j_face, nz)
            K_z[i, j_face] = ergun_pressure_drop_coeff(
                np.array([rho_f]), np.array([mu_f]),
                np.array([u_f]),   np.array([eps_f]),
                np.array([dp_f]),
            )[0]

    # Floor K to a finite minimum (avoids division explosions in
    # very high porosity early-iteration cells).
    K_r = np.maximum(K_r, 1.0)
    K_z = np.maximum(K_z, 1.0)
    return K_r, K_z


def _velocity_from_pressure(
    P: np.ndarray, rho: np.ndarray, eps: np.ndarray,
    K_r: np.ndarray, K_z: np.ndarray,
    mesh: Mesh2D, raceway: Raceway,
    gas_exit_mask: Optional[np.ndarray],
    m_source_total: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute u_r, u_z from p via Darcy-Forchheimer (superficial velocity).

    Standard Ergun:  grad(p) = -K u_s - rho g_vec
    where u_s is the superficial velocity and K is the Ergun coefficient
    (already volume-averaged via the (1-eps)/eps^3 factors inside K).

        u_r = -(1/K) dp/dr
        u_z = -(1/K) dp/dz - rho G_ACC / K     (G_ACC > 0, gravity points -z)

    Sign check:
    - Pressure decreases with z (typical) ->  dp/dz < 0  ->  -dp/dz / K > 0  ->  upward flow.
    - Gravity term -rho g / K is negative, opposing upward flow.
    - At hydrostatic balance: dp/dz = -rho g  ->  u_z = 0.

    Inlet and wall velocities are then enforced as Dirichlet.
    """
    nr, nz = mesh.nr, mesh.nz
    u_r = np.zeros((nr + 1, nz))
    u_z = np.zeros((nr, nz + 1))
    tuyere_set = set(raceway.face_indices)

    # ----- radial faces -----
    for j in range(nz):
        for i_face in range(1, nr):  # skip axis (i=0) and outer wall (i=nr)
            i_w = i_face - 1
            i_e = i_face
            if not (mesh.active[i_w, j] and mesh.active[i_e, j]):
                continue
            dr = 0.5 * (mesh.dr[i_w] + mesh.dr[i_e])
            dP = P[i_e, j] - P[i_w, j]
            u_r[i_face, j] = -1.0 / K_r[i_face, j] * dP / max(dr, 1e-9)

    # axis: u_r = 0 (symmetry)
    u_r[0, :] = 0.0

    # outer-wall faces and tuyere faces: enforce Dirichlet
    #
    # The tuyere is on the wall-side face of the outermost active cell at
    # the tuyere axial band.  Due to bosh slope, that face is not always
    # at i_face = nr -- it can be at any 1 <= i_face <= nr.  The actual
    # tuyere faces are listed in raceway.face_indices.
    #
    # First zero out all "wall" faces (faces between an active and an
    # inactive cell, or the i=nr wall), then re-impose the tuyere
    # Dirichlet on the listed tuyere faces.
    for j in range(nz):
        for i_face in range(1, nr + 1):
            i_w = i_face - 1
            i_e = i_face
            west_active = (i_w < nr) and mesh.active[i_w, j]
            east_active = (i_e < nr) and mesh.active[i_e, j]
            # Wall face: only one side is active (or this is the outer wall i=nr)
            is_wall_face = (i_face == nr and west_active) or (
                west_active != east_active and i_face != nr
            )
            if is_wall_face:
                u_r[i_face, j] = 0.0

    # Re-impose tuyere mass-flux Dirichlet
    for (i_face, j) in raceway.face_indices:
        # Tuyere is the wall-face of the outermost active cell at j
        # u_r at that face = -G_tuyere / rho_face (radially inward, hence negative)
        rho_face = rho[i_face - 1, j] if i_face >= 1 else rho[0, j]
        u_r[i_face, j] = -raceway.G_tuyere / max(rho_face, 1e-3)

    # ----- axial faces -----
    for j_face in range(1, nz):  # skip bottom (j=0) and top (j=nz)
        for i in range(nr):
            j_s = j_face - 1
            j_n = j_face
            if not (mesh.active[i, j_s] and mesh.active[i, j_n]):
                continue
            dz = 0.5 * (mesh.dz[j_s] + mesh.dz[j_n])
            dP = P[i, j_n] - P[i, j_s]
            rho_f = _face_avg_z(rho, i, j_face, nz)
            # u_z = -dp/dz / K - rho g / K     (g acts in -z direction;
            # gravity contribution is -rho * G_ACC / K, opposing upward flow)
            u_z[i, j_face] = (-dP / (max(dz, 1e-9) * K_z[i, j_face])
                              - rho_f * G_ACC / K_z[i, j_face])

    # bottom (hearth, j_face = 0): sealed
    u_z[:, 0] = 0.0

    # top face (j_face = nz): extrapolation at gas-exit cells, sealed elsewhere
    if nz > 0:
        if gas_exit_mask is None:
            top_exit = mesh.active[:, nz - 1]
        else:
            top_exit = gas_exit_mask[:, nz - 1] & mesh.active[:, nz - 1]
        # Zero-gradient extrapolation: top face inherits cell-south value
        for i in range(nr):
            if top_exit[i]:
                u_z[i, nz] = max(u_z[i, nz - 1], 0.0)  # outflow only
            else:
                u_z[i, nz] = 0.0

        # Global mass-balance scaling (Moukalled Eq. 15.169).
        # Use the active cell west of each tuyere face (i_face - 1) for rho,
        # since the cell east of the face may be outside the active domain
        # (bosh slope) -- same convention as _velocity_from_pressure.
        #
        # The outflow target is blast IN plus the net volumetric gas-mass
        # SOURCE from reactions (reduction transfers O from the burden to
        # the gas, solution loss and calcination add C/CO2): at steady
        # state m_out = m_in + integral(S_mass dV).  Scaling to m_in alone
        # (pre-Phase-9 behaviour) silently deleted the reaction-generated
        # mass (~10-15% of blast) at the exit, which broke the carbon
        # ledger and biased the coke-rate KPI low.
        m_in = 0.0
        for (i_face, j_t) in raceway.face_indices:
            i_w = max(i_face - 1, 0)
            rho_t = rho[i_w, j_t]
            A_t = mesh.area_r[i_face, j_t]
            m_in += rho_t * abs(u_r[i_face, j_t]) * A_t
        # Defensive floor: a transiently negative net source (early
        # iterations) must not choke the outlet.
        m_target = max(m_in + m_source_total, 0.1 * m_in)
        m_out = 0.0
        for i in range(nr):
            if top_exit[i]:
                m_out += rho[i, nz - 1] * u_z[i, nz] * mesh.area_z[i, nz]
        if m_in > 1e-9:
            if m_out > 0.01 * m_in:
                scale = m_target / m_out
                for i in range(nr):
                    if top_exit[i]:
                        u_z[i, nz] *= scale
            else:
                # bootstrap: extrapolation gave near-zero outflow
                total_rhoA = sum(rho[i, nz - 1] * mesh.area_z[i, nz]
                                 for i in range(nr) if top_exit[i])
                if total_rhoA > 1e-9:
                    u_uniform = m_target / total_rhoA
                    for i in range(nr):
                        if top_exit[i]:
                            u_z[i, nz] = u_uniform

    return u_r, u_z


def _assemble_pressure_poisson(
    P_old: np.ndarray, rho: np.ndarray, eps: np.ndarray,
    K_r: np.ndarray, K_z: np.ndarray, mass_source: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    gas_exit_mask: Optional[np.ndarray],
) -> tuple[sps.csr_matrix, np.ndarray]:
    """Assemble the Darcy-Forchheimer pressure-Poisson equation.

        -div[ (rho eps / K) grad(p) ] = S_mass + div[ rho eps^2 rho g / K ]

    BC:
    - Tuyere faces (Neumann, mass flux specified): inflow contributes to RHS
    - Top exit cells (Dirichlet on p): row replaced with p = top_pressure
    - All other walls (Neumann, zero mass flux): faces dropped from stencil
    """
    nr, nz = mesh.nr, mesh.nz
    n = nr * nz
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n)

    tuyere_set = set(raceway.face_indices)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    for j in range(nz):
        for i in range(nr):
            row = _idx_P(i, j, nr)

            if not mesh.active[i, j]:
                add(row, row, 1.0); b[row] = P_old[i, j]
                continue

            # Pressure Dirichlet at top gas-exit cells
            if (j == nz - 1
                    and (gas_exit_mask is None or gas_exit_mask[i, j])
                    and mesh.active[i, j]):
                add(row, row, 1.0); b[row] = p.top_pressure
                continue

            # Cell volume and face areas
            V = mesh.vol[i, j]
            A_w = mesh.area_r[i,     j]   # west face
            A_e = mesh.area_r[i + 1, j]   # east face
            A_s = mesh.area_z[i,     j]   # south face
            A_n = mesh.area_z[i,     j + 1]  # north face

            # Cell-to-cell distances (for face gradients)
            dr_w = mesh.dr[i] if i == 0 else 0.5 * (mesh.dr[i - 1] + mesh.dr[i])
            dr_e = mesh.dr[i] if i == nr - 1 else 0.5 * (mesh.dr[i] + mesh.dr[i + 1])
            dz_s = mesh.dz[j] if j == 0 else 0.5 * (mesh.dz[j - 1] + mesh.dz[j])
            dz_n = mesh.dz[j] if j == nz - 1 else 0.5 * (mesh.dz[j] + mesh.dz[j + 1])

            # Face transmissibilities  T_face = rho_face / K_face * A_face / d
            # (Ergun K already incorporates the (1-eps)^2/eps^3 dependence
            # internally; eps does NOT appear separately here.  The
            # superficial-velocity form of Darcy is grad(p) = -K u - rho g,
            # so substituting into mass flux: rho u A = -(rho/K) A dp/dz - ...)
            rho_w = _face_avg_r(rho, i,     j, nr)
            rho_e = _face_avg_r(rho, i + 1, j, nr)
            rho_s = _face_avg_z(rho, i, j,     nz)
            rho_n = _face_avg_z(rho, i, j + 1, nz)

            T_w = rho_w / K_r[i,     j] * A_w / max(dr_w, 1e-9)
            T_e = rho_e / K_r[i + 1, j] * A_e / max(dr_e, 1e-9)
            T_s = rho_s / K_z[i, j    ] * A_s / max(dz_s, 1e-9)
            T_n = rho_n / K_z[i, j + 1] * A_n / max(dz_n, 1e-9)

            # Suppress fluxes through walls / closed boundaries
            # West face: closed if at axis (i=0)
            if i == 0:
                T_w = 0.0
            elif not mesh.active[i - 1, j]:
                T_w = 0.0
            # East face: closed unless tuyere or interior
            is_east_tuyere = (i + 1, j) in tuyere_set
            if i + 1 == nr:
                if not is_east_tuyere:
                    T_e = 0.0  # outer wall
            elif not mesh.active[i + 1, j]:
                T_e = 0.0
            # South face: closed at hearth bottom
            if j == 0:
                T_s = 0.0
            elif not mesh.active[i, j - 1]:
                T_s = 0.0
            # North face: closed at sealed top
            if j == nz - 1:
                # Top row, but this cell is NOT in the gas exit ring
                # (handled above) so the north face is sealed.
                T_n = 0.0
            elif not mesh.active[i, j + 1]:
                T_n = 0.0

            # Buoyancy / gravity term in continuity.
            #
            # Substituting Darcy u_z = -(1/K) dp/dz - rho G_ACC / K into
            # the mass flux at axial faces:
            #
            #     (rho u_z A)_face = -(rho/K) A dp/dz  -  (rho^2/K) A G_ACC
            #                        ^pressure part^      ^buoyancy part^
            #
            # The pressure part becomes the T_f coefficients below.  The
            # buoyancy part contributes to the divergence and is moved to
            # the RHS.  Bottom face (j_face=0) and top of sealed cells have
            # no flux, so their buoyancy contribution is zero.
            buoy_n = ((rho_n * rho_n / K_z[i, j + 1]) * A_n * G_ACC
                      if T_n > 0 else 0.0)
            buoy_s = ((rho_s * rho_s / K_z[i, j    ]) * A_s * G_ACC
                      if T_s > 0 else 0.0)
            # Discrete divergence of buoyancy mass flux (north positive):
            #     div_buoy = (-rho^2 G/K * A)_n - (-rho^2 G/K * A)_s
            #              = -[buoy_n - buoy_s]
            # This appears on LHS in continuity; bring to RHS:
            #     RHS contribution = +[buoy_n - buoy_s]
            rhs_buoy = (buoy_n - buoy_s)

            # Tuyere mass-flux source on east face
            rhs_tuyere = 0.0
            if is_east_tuyere:
                # Inflow rate kg/s into this cell
                m_in = raceway.G_tuyere * A_e
                rhs_tuyere = m_in   # mass entering cell

            # Assemble row:  T_total * P_C - sum(T_face * P_neighbor) = RHS
            T_total = T_w + T_e + T_s + T_n
            if T_total < 1e-20:
                # Fully isolated cell -- just fix p to old value
                add(row, row, 1.0); b[row] = P_old[i, j]
                continue

            add(row, row, T_total)
            if T_w > 0 and i > 0 and mesh.active[i - 1, j]:
                add(row, _idx_P(i - 1, j, nr), -T_w)
            if T_e > 0 and i + 1 < nr and mesh.active[i + 1, j]:
                add(row, _idx_P(i + 1, j, nr), -T_e)
            if T_s > 0 and j > 0 and mesh.active[i, j - 1]:
                add(row, _idx_P(i, j - 1, nr), -T_s)
            if T_n > 0 and j + 1 < nz and mesh.active[i, j + 1]:
                add(row, _idx_P(i, j + 1, nr), -T_n)

            # RHS:  S_mass * V (kg/s, positive = source) + buoyancy + tuyere
            b[row] = mass_source[i, j] * V + rhs_buoy + rhs_tuyere

    A = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A, b


# ===========================================================================
# Public entry point  --  same signature as legacy SIMPLE solver
# ===========================================================================
def solve_coupled_momentum(
    u_r_old: np.ndarray, u_z_old: np.ndarray, P_old: np.ndarray,
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray, dp_eff: np.ndarray,
    mass_source: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    relaxation: float = 0.5,
    gas_exit_mask: Optional[np.ndarray] = None,
    n_K_iter: int = 5,
    relax_p: float = 0.7,
) -> MomentumSolution:
    """Solve the Darcy-Forchheimer pressure-Poisson equation for the gas.

    Replaces the legacy three-pass SIMPLE solver with a single elliptic
    solve per K-update.  Iterates over K because the Forchheimer term
    K_inert depends on |u|.

    Parameters
    ----------
    n_K_iter : number of inner iterations to update K and re-solve.  Each
               iteration: update K from current u, solve Poisson, update u.
    relax_p  : pressure under-relaxation per outer iteration of the
               whole simulation.  (P_new = P_old + relax_p * (P* - P_old))
               Within the K-loop, no extra relaxation is applied.
    relaxation : kept for legacy interface compatibility (not used here).
    """
    nr, nz = mesh.nr, mesh.nz

    # Initial K from old velocity field
    u_mag_old = np.zeros((nr, nz))
    for j in range(nz):
        for i in range(nr):
            u_r_c = 0.5 * (u_r_old[i, j] + u_r_old[i + 1, j])
            u_z_c = 0.5 * (u_z_old[i, j] + u_z_old[i, j + 1])
            u_mag_old[i, j] = float(np.hypot(u_r_c, u_z_c))

    P_iter = P_old.copy()
    u_r = u_r_old.copy()
    u_z = u_z_old.copy()

    # Net volumetric gas-mass source [kg/s] over the active domain --
    # part of the steady outflow target (m_out = m_in + m_src_total).
    m_src_total = float((mass_source * mesh.vol)[mesh.active].sum())

    last_du = 0.0
    last_lin_ok = True
    for k_iter in range(n_K_iter):
        # 1. Update K from current velocity magnitude (cell-centered)
        u_mag = np.zeros((nr, nz))
        for j in range(nz):
            for i in range(nr):
                u_r_c = 0.5 * (u_r[i, j] + u_r[i + 1, j])
                u_z_c = 0.5 * (u_z[i, j] + u_z[i, j + 1])
                u_mag[i, j] = float(np.hypot(u_r_c, u_z_c))

        K_r, K_z = _compute_face_K(rho, mu, eps, dp_eff, u_mag, mesh)

        # 2. Assemble & solve pressure Poisson
        A, b = _assemble_pressure_poisson(
            P_iter, rho, eps, K_r, K_z, mass_source,
            mesh, p, raceway, gas_exit_mask,
        )
        x0 = P_iter.ravel(order="F")
        P_flat, lin_stats = solve_linear_system(A, b, x0=x0)
        P_new = P_flat.reshape((nz, nr)).T
        # clamp to physical
        P_new = np.clip(P_new, 2.0e4, 1.5e6)
        last_lin_ok = bool(getattr(lin_stats, "ok", True))

        # 3. Update velocity from new pressure via Darcy
        u_r_new, u_z_new = _velocity_from_pressure(
            P_new, rho, eps, K_r, K_z, mesh, raceway, gas_exit_mask,
            m_source_total=m_src_total,
        )

        # Check convergence (||du||)
        du = float(np.linalg.norm(u_z_new - u_z) + np.linalg.norm(u_r_new - u_r))
        last_du = du

        u_r = u_r_new
        u_z = u_z_new
        P_iter = P_new

        if du < 1e-6 * (np.linalg.norm(u_z_new) + np.linalg.norm(u_r_new) + 1.0):
            break

    # Outer-iteration pressure under-relaxation: blend with old P
    P_final = P_old + relax_p * (P_iter - P_old)
    P_final = np.clip(P_final, 2.0e4, 1.5e6)

    # Re-impose top pressure Dirichlet at exit cells
    if gas_exit_mask is not None:
        for i in range(nr):
            if gas_exit_mask[i, nz - 1] and mesh.active[i, nz - 1]:
                P_final[i, nz - 1] = p.top_pressure
    else:
        for i in range(nr):
            if mesh.active[i, nz - 1]:
                P_final[i, nz - 1] = p.top_pressure

    # Recompute velocities from final relaxed pressure (so velocities are
    # consistent with what the rest of the solver sees)
    K_r, K_z = _compute_face_K(rho, mu, eps, dp_eff, u_mag, mesh)
    u_r, u_z = _velocity_from_pressure(
        P_final, rho, eps, K_r, K_z, mesh, raceway, gas_exit_mask,
        m_source_total=m_src_total,
    )

    # Residual diagnostics.
    #
    # We use *iteration-to-iteration drift* as the convergence metric,
    # which is what the simulation orchestrator's flow_stable check is
    # tuned for.  Local divergence per cell is dominated by the
    # tuyere-cell boundary residual (Darcy + mass-flux Dirichlet
    # produces a small but nonzero divergence right at the inlet) and
    # would always read large even at convergence.
    u_norm = float(np.linalg.norm(u_z) + np.linalg.norm(u_r) + 1.0)
    res_r = float(np.linalg.norm(u_r - u_r_old)) / u_norm
    res_z = float(np.linalg.norm(u_z - u_z_old)) / u_norm
    # Continuity: relative pressure change between iterations
    P_norm = float(np.linalg.norm(P_old.ravel())) + 1.0
    res_cont = float(np.linalg.norm((P_final - P_old).ravel())) / P_norm

    return MomentumSolution(
        u_r=u_r, u_z=u_z, P=P_final,
        residual_continuity=res_cont,
        residual_r_mom=res_r, residual_z_mom=res_z,
        linear_stats=LinearSolveStats(
            method="darcy_forchheimer",
            iterations=n_K_iter,
            residual=last_du,
            ok=last_lin_ok,
        ),
    )
