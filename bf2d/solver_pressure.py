"""
Darcy-Forchheimer pressure-velocity solver for packed-bed gas flow.

Architecture (Fu et al. 2014 / Zhou et al. 2016)
--------------------------------------------------
Replaces the SIMPLE three-pass momentum/pressure solver with a single
elliptic Poisson equation for pressure, derived from the Darcy-
Forchheimer momentum closure.

Physics
-------
For packed-bed flow at BF velocities (~0.5–2 m/s), Ergun friction
dominates inertia by ~1000×.  The full Navier-Stokes momentum equation
collapses to the algebraic relation:

    u = -K(|u|, ρ, ε, dp) · ∇p                            [Darcy-Forchheimer]

where K is the Ergun-Forchheimer permeability:

    1/K = (150 μ (1-ε)²) / (ε³ dp²)  +  (1.75 ρ |u| (1-ε)) / (ε³ dp)
        ≡ K_friction(|u|, ρ, ε, dp)                       [Ergun friction]

So K_g = 1/K_friction  has dimensions [m²/(Pa·s)] = [m·s/kg].

Substituting into the continuity equation  ∂(ε ρ)/∂t + ∇·(ρ u) = S_mass:

    ∇·(ρ K ∇p) = -S_mass + ∂(ε ρ)/∂t                      [Pressure-Poisson]

This is solved iteratively because K depends on |u|:
    1. Guess u (from previous outer iteration)
    2. Compute K(|u|)
    3. Solve  ∇·(ρ K ∇p) = -S_mass  for p
    4. Update  u = -K ∇p
    5. Repeat until |Δu| / |u| < tol  (typically 5-10 inner iterations)

Why this is correct for packed beds
-----------------------------------
At ε = 0.42, dp = 0.03 m, u = 1 m/s, ρ = 1 kg/m³:
    Inertial term  ρ(u·∇)u ~ ρu²/L ~ 0.04 Pa/m
    Ergun friction        ~ 1000  Pa/m
The inertia is ~25,000× smaller than friction.  Dropping it is exact
to that precision.

This is the formulation used in published BF CFD codes:
- Fu, Chen, Zhao, D'Alessio, Ferron, Zhou (2014) "CFD modeling of
  multiphase reacting flow in blast furnace shaft with layered burden",
  Applied Thermal Engineering 66, 298-308.  Eq. 3 in their Table 1.
- Zhou, Tang, Wang, Fu, Okosun, Silaen, Wu (2016) "Comprehensive
  Numerical Modeling of the Blast Furnace Ironmaking Process", JOM
  68(5), 1353-1362.  Section "Blast Furnace Shaft modeling" Eq. 14.

Boundary conditions
-------------------
- Tuyere inlet:  Neumann on p, with ∂p/∂n set so that local flux equals
                 the prescribed mass flux G_tuyere.  Equivalent to:
                 (ρ K ∂p/∂n) = -G_tuyere on the tuyere face.
- Top exit:      Dirichlet  p = p_top  on gas-exit cells in the uptake ring.
- Sealed top:    Neumann  ∂p/∂n = 0  on top cells outside the exit ring
                 (no mass flow through stockline).
- Walls:         Neumann  ∂p/∂n = 0  on lateral walls.
- Axis:          Neumann  ∂p/∂n = 0  (axisymmetric centerline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

from .linear_solver import LinearSolveStats, solve_linear_system
from .mesh import Mesh2D
from .parameters import BFParameters2D
from .raceway import Raceway


# ---------------------------------------------------------------------------
@dataclass
class PressureSolution:
    """Result of one Darcy-Forchheimer pressure-velocity solve."""
    u_r: np.ndarray            # (nr+1, nz)  radial face velocity
    u_z: np.ndarray            # (nr,   nz+1) axial face velocity
    P:   np.ndarray            # (nr,   nz)  cell-centre pressure
    residual_continuity: float # max divergence per cell, kg/s
    residual_velocity:   float # ||Δu|| / ||u||
    inner_iterations:    int   # K-update iterations until converged
    linear_stats: LinearSolveStats


# ---------------------------------------------------------------------------
def _idx_P(i: int, j: int, nr: int) -> int:
    return i + nr * j


def _ergun_K_friction(rho: np.ndarray, mu: np.ndarray,
                      u_mag: np.ndarray, eps: np.ndarray,
                      dp: np.ndarray) -> np.ndarray:
    """Ergun friction coefficient K_friction such that dP/dL = -K_friction · u.

    Returns K_friction with units Pa·s/m².  The Darcy-Forchheimer
    permeability is K = 1 / K_friction.
    """
    eps_safe = np.clip(eps, 0.02, 0.95)
    dp_safe  = np.maximum(dp, 5.0e-3)
    term1 = 150.0 * mu * (1.0 - eps_safe) ** 2 / (eps_safe ** 3 * dp_safe ** 2)
    term2 = 1.75 * rho * np.abs(u_mag) * (1.0 - eps_safe) / (eps_safe ** 3 * dp_safe)
    return term1 + term2


def _face_avg_r(f: np.ndarray, i_face: int, j: int, nr: int) -> float:
    if i_face <= 0:
        return f[0, j]
    if i_face >= nr:
        return f[nr - 1, j]
    return 0.5 * (f[i_face - 1, j] + f[i_face, j])


def _face_avg_z(f: np.ndarray, i: int, j_face: int, nz: int) -> float:
    if j_face <= 0:
        return f[i, 0]
    if j_face >= nz:
        return f[i, nz - 1]
    return 0.5 * (f[i, j_face - 1] + f[i, j_face])


# ---------------------------------------------------------------------------
def _compute_face_permeabilities(
    u_r: np.ndarray, u_z: np.ndarray,
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray, dp: np.ndarray,
    mesh: Mesh2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Darcy-Forchheimer permeability K = 1/K_friction at every face.

    Returns
    -------
    K_r : (nr+1, nz) array, permeability at radial faces
    K_z : (nr, nz+1) array, permeability at axial faces

    K is computed using Ergun closure with the LOCAL face velocity
    magnitude.  This is the Forchheimer (quadratic) correction; in
    pure-Darcy limit (low Re) the second term vanishes.
    """
    nr, nz = mesh.nr, mesh.nz

    # Cell-centre velocity magnitude (for Forchheimer term at the cells).
    # We compute |u| at cell centres first then average to faces; this is
    # cheaper than computing |u| at faces directly and gives ~identical
    # results for smooth fields.
    u_r_c = np.zeros_like(rho)
    u_z_c = np.zeros_like(rho)
    u_r_c[:, :] = 0.5 * (u_r[:nr, :] + u_r[1:, :])
    u_z_c[:, :] = 0.5 * (u_z[:, :nz] + u_z[:, 1:])
    u_mag_c = np.sqrt(u_r_c**2 + u_z_c**2)

    # Cell-centre Ergun friction
    Kf_c = _ergun_K_friction(rho, mu, u_mag_c, eps, dp)
    # Permeability at cell centres
    K_c = 1.0 / np.maximum(Kf_c, 1e-12)

    # Average to faces (harmonic mean for resistance-in-series — physically
    # correct for permeability, since 1/K = sum of resistances).
    K_r = np.zeros((nr + 1, nz))
    K_z = np.zeros((nr, nz + 1))

    # Radial faces: harmonic average between cell west and cell east
    for j in range(nz):
        for i_face in range(nr + 1):
            if i_face == 0:
                K_r[i_face, j] = K_c[0, j]
            elif i_face == nr:
                K_r[i_face, j] = K_c[nr - 1, j]
            else:
                K_w = K_c[i_face - 1, j]
                K_e = K_c[i_face, j]
                # Harmonic mean
                K_r[i_face, j] = 2.0 * K_w * K_e / max(K_w + K_e, 1e-30)

    # Axial faces: harmonic average between cell south and cell north
    for j_face in range(nz + 1):
        for i in range(nr):
            if j_face == 0:
                K_z[i, j_face] = K_c[i, 0]
            elif j_face == nz:
                K_z[i, j_face] = K_c[i, nz - 1]
            else:
                K_s = K_c[i, j_face - 1]
                K_n = K_c[i, j_face]
                K_z[i, j_face] = 2.0 * K_s * K_n / max(K_s + K_n, 1e-30)

    return K_r, K_z


# ---------------------------------------------------------------------------
def _assemble_pressure_poisson(
    K_r: np.ndarray, K_z: np.ndarray,
    rho: np.ndarray, mass_source: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    gas_exit_mask: np.ndarray | None,
) -> tuple[sps.csr_matrix, np.ndarray]:
    """Assemble the linear system A·p = b for the pressure-Poisson equation.

    Discrete form on staggered grid:

        Σ_face  (ρ_face K_face A_face / d_face) (p_F - p_C)  =  -S_mass · V_C

    where the sum is over the four faces of cell C, ρ_face is the face
    density, K_face is the face permeability, A_face is the face area,
    d_face is the centre-to-centre distance to neighbour F.

    Boundary contributions:
    - Tuyere face:     adds  (m_dot_tuyere)  to the RHS  (mass injection)
    - Top exit cell:   row replaced with Dirichlet  p = p_top
    - Sealed face:     contribution dropped (zero flux Neumann)
    - Wall face:       contribution dropped
    - Axis face:       contribution dropped (already handled by area=0)
    """
    nr, nz = mesh.nr, mesh.nz
    n = nr * nz
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n)

    tuyere_faces = set(raceway.face_indices)

    def add(r: int, c: int, v: float) -> None:
        rows.append(r); cols.append(c); vals.append(v)

    for j in range(nz):
        for i in range(nr):
            row = _idx_P(i, j, nr)

            if not mesh.active[i, j]:
                add(row, row, 1.0); b[row] = 0.0
                continue

            # Top-row gas-exit cells: Dirichlet p = p_top
            if j == nz - 1:
                if gas_exit_mask is None or gas_exit_mask[i, j]:
                    add(row, row, 1.0)
                    b[row] = p.top_pressure
                    continue
                # Sealed top cell: assemble normally; the north face
                # contribution is dropped below (zero-flux Neumann).

            V = mesh.vol[i, j]
            a_diag = 0.0

            # ---- West face (i, j) ----
            if i > 0 and mesh.active[i - 1, j]:
                # Interior radial face: harmonic K, area, distance
                rho_w = _face_avg_r(rho, i, j, nr)
                K_w = K_r[i, j]
                A_w = mesh.area_r[i, j]
                d_w = mesh.r_c[i] - mesh.r_c[i - 1]
                if d_w > 1e-9:
                    coeff_w = rho_w * K_w * A_w / d_w
                    a_diag += coeff_w
                    add(row, _idx_P(i - 1, j, nr), -coeff_w)
            # i == 0: axis face, zero flux by symmetry, no contribution.
            # i > 0 but neighbour inactive: wall, zero flux.

            # ---- East face (i+1, j) ----
            face_e = (i + 1, j)
            is_tuyere_e = face_e in tuyere_faces
            if is_tuyere_e:
                # Tuyere face: prescribed inlet mass flux G_tuyere injects
                # into this cell.  Add to RHS as a source term.
                # G_tuyere is in kg/(m² s); flux = G * A_face.
                A_face = mesh.area_r[i + 1, j]
                m_dot_in = raceway.G_tuyere * A_face
                # Sign convention: positive mass into the cell
                b[row] += m_dot_in
            elif i + 1 < nr and mesh.active[i + 1, j]:
                # Interior face
                rho_e = _face_avg_r(rho, i + 1, j, nr)
                K_e = K_r[i + 1, j]
                A_e = mesh.area_r[i + 1, j]
                d_e = mesh.r_c[i + 1] - mesh.r_c[i]
                if d_e > 1e-9:
                    coeff_e = rho_e * K_e * A_e / d_e
                    a_diag += coeff_e
                    add(row, _idx_P(i + 1, j, nr), -coeff_e)
            # else: wall (i+1==nr) or inactive neighbour, zero flux.

            # ---- South face (i, j) ----
            if j > 0 and mesh.active[i, j - 1]:
                rho_s = _face_avg_z(rho, i, j, nz)
                K_s = K_z[i, j]
                A_s = mesh.area_z[i, j]
                d_s = mesh.z_c[j] - mesh.z_c[j - 1]
                if d_s > 1e-9:
                    coeff_s = rho_s * K_s * A_s / d_s
                    a_diag += coeff_s
                    add(row, _idx_P(i, j - 1, nr), -coeff_s)
            # j == 0: hearth bottom, sealed (zero flux).

            # ---- North face (i, j+1) ----
            if j + 1 < nz and mesh.active[i, j + 1]:
                # Interior axial face
                rho_n = _face_avg_z(rho, i, j + 1, nz)
                K_n = K_z[i, j + 1]
                A_n = mesh.area_z[i, j + 1]
                d_n = mesh.z_c[j + 1] - mesh.z_c[j]
                if d_n > 1e-9:
                    coeff_n = rho_n * K_n * A_n / d_n
                    a_diag += coeff_n
                    add(row, _idx_P(i, j + 1, nr), -coeff_n)
            elif j + 1 == nz and mesh.active[i, j + 1 - 1]:
                # j+1 == nz is the row above-top boundary.  If we're a
                # non-exit top cell, the north neighbour at (i, nz) is
                # the boundary, treated as zero-flux Neumann -> nothing
                # to add.  If we ARE an exit cell, we already returned
                # above with Dirichlet.
                pass
            # else: wall or inactive neighbour, zero flux.

            # ---- Diagonal ----
            if a_diag < 1e-30:
                # Disconnected cell (e.g. surrounded by walls): pin to
                # whatever the previous pressure was.  Shouldn't happen
                # for a healthy mesh but is a safety net.
                add(row, row, 1.0)
                b[row] = p.top_pressure
            else:
                add(row, row, a_diag)
                # RHS: mass source from kinetics in kg/(m³ s) times volume.
                # Sign: positive S_mass adds mass to the gas, which
                # appears as an effective sink for the divergence eq:
                #     ∇·(ρ K ∇p) = - S_mass
                # so positive S_mass goes to the RHS as +S_mass · V (we
                # are assembling the *negative* of div(ρu) on the RHS
                # because ρu = -ρ K ∇p).
                b[row] += -mass_source[i, j] * V

    A = sps.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A, b


# ---------------------------------------------------------------------------
def _compute_velocities_from_pressure(
    P: np.ndarray, K_r: np.ndarray, K_z: np.ndarray,
    mesh: Mesh2D, raceway: Raceway, gas_exit_mask: np.ndarray | None,
    p: BFParameters2D, rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply  u = -K ∇p  at every face to get the velocity field.

    Boundary faces:
    - Tuyere face:   u_r set from prescribed G_tuyere / ρ_face
    - Hearth bottom: u_z = 0 (sealed)
    - Top exit:      u_z computed from the interior gradient (Dirichlet on p)
    - Sealed top:    u_z = 0
    - Walls:         u = 0
    - Axis:          u_r = 0 (symmetry)
    """
    nr, nz = mesh.nr, mesh.nz
    u_r = np.zeros((nr + 1, nz))
    u_z = np.zeros((nr, nz + 1))
    tuyere_faces = set(raceway.face_indices)

    # Radial faces
    for j in range(nz):
        for i_face in range(nr + 1):
            if i_face == 0:
                u_r[i_face, j] = 0.0  # axis
                continue

            face_id = (i_face, j)
            if face_id in tuyere_faces:
                # Prescribed mass-flux inlet -- velocity from local density
                rho_f = _face_avg_r(rho, i_face, j, nr)
                u_r[i_face, j] = -raceway.G_tuyere / max(rho_f, 1e-3)
                continue

            if i_face == nr:
                u_r[i_face, j] = 0.0  # outer wall, no flow
                continue

            i_west, i_east = i_face - 1, i_face
            if not (mesh.active[i_west, j] and mesh.active[i_east, j]):
                u_r[i_face, j] = 0.0
                continue

            # Interior radial face: u = -K (p_east - p_west) / d
            d = mesh.r_c[i_east] - mesh.r_c[i_west]
            if d < 1e-9:
                u_r[i_face, j] = 0.0
            else:
                u_r[i_face, j] = -K_r[i_face, j] * (P[i_east, j] - P[i_west, j]) / d

    # Axial faces
    for j_face in range(nz + 1):
        for i in range(nr):
            if j_face == 0:
                u_z[i, j_face] = 0.0  # hearth, sealed
                continue

            if j_face == nz:
                # Top boundary face
                if not mesh.active[i, nz - 1]:
                    u_z[i, j_face] = 0.0
                    continue
                is_exit = (gas_exit_mask is None) or gas_exit_mask[i, nz - 1]
                if not is_exit:
                    u_z[i, j_face] = 0.0  # sealed top
                    continue
                # Exit cell: extrapolate from interior gradient.
                # Top boundary p is Dirichlet at p.top_pressure.  The
                # half-cell distance to the boundary is dz[nz-1]/2, so:
                #   u_z = -K (p_top - p_C) / (dz/2)
                d = 0.5 * mesh.dz[nz - 1]
                if d < 1e-9:
                    u_z[i, j_face] = 0.0
                else:
                    u_z[i, j_face] = (
                        -K_z[i, j_face] * (p.top_pressure - P[i, nz - 1]) / d
                    )
                # Outflow only (clip negative -> 0; in healthy convergence
                # this never fires)
                if u_z[i, j_face] < 0.0:
                    u_z[i, j_face] = 0.0
                continue

            j_south, j_north = j_face - 1, j_face
            if not (mesh.active[i, j_south] and mesh.active[i, j_north]):
                u_z[i, j_face] = 0.0
                continue

            d = mesh.z_c[j_north] - mesh.z_c[j_south]
            if d < 1e-9:
                u_z[i, j_face] = 0.0
            else:
                u_z[i, j_face] = -K_z[i, j_face] * (P[i, j_north] - P[i, j_south]) / d

    return u_r, u_z


# ---------------------------------------------------------------------------
def solve_darcy_forchheimer(
    u_r_old: np.ndarray, u_z_old: np.ndarray, P_old: np.ndarray,
    rho: np.ndarray, mu: np.ndarray, eps: np.ndarray, dp_eff: np.ndarray,
    mass_source: np.ndarray,
    mesh: Mesh2D, p: BFParameters2D, raceway: Raceway,
    gas_exit_mask: np.ndarray | None = None,
    max_inner_iter: int = 8,
    tol_velocity: float = 1.0e-3,
    relaxation: float = 0.7,
) -> PressureSolution:
    """One outer step: iterate K(|u|) -> p -> u  until self-consistent.

    The Forchheimer correction makes K depend on |u|, so we need an inner
    loop:
      while not converged:
          1. Compute K from current |u|
          2. Solve  ∇·(ρ K ∇p) = -S_mass  for p
          3. Compute  u = -K ∇p
          4. Check ||Δu|| / ||u|| < tol

    Typical convergence: 3-6 inner iterations on a well-conditioned mesh.
    """
    u_r = u_r_old.copy()
    u_z = u_z_old.copy()
    P = P_old.copy()

    last_stats = LinearSolveStats("darcy_forchheimer", 0, 0.0, True)

    for inner in range(max_inner_iter):  # noqa: B007 -- `inner` is read
        # after the loop (line ~528, inner_iterations=inner+1); ruff's
        # unused-loop-var check only looks inside the loop body.
        # 1. Permeabilities at faces
        K_r, K_z = _compute_face_permeabilities(
            u_r, u_z, rho, mu, eps, dp_eff, mesh,
        )

        # 2. Pressure-Poisson assembly and solve
        A, b = _assemble_pressure_poisson(
            K_r, K_z, rho, mass_source, mesh, p, raceway, gas_exit_mask,
        )
        x0 = P.ravel(order="F")
        P_flat, last_stats = solve_linear_system(A, b, x0=x0)
        P_new = P_flat.reshape((mesh.nz, mesh.nr)).T
        P_new = np.clip(P_new, 2.0e4, 1.5e6)

        # 3. Velocities from pressure gradient
        u_r_new, u_z_new = _compute_velocities_from_pressure(
            P_new, K_r, K_z, mesh, raceway, gas_exit_mask, p, rho,
        )

        # 4. Check convergence in velocity (relative)
        du_r = u_r_new - u_r
        du_z = u_z_new - u_z
        norm_u = max(
            float(np.linalg.norm(u_r_new.ravel())) +
            float(np.linalg.norm(u_z_new.ravel())),
            1.0,
        )
        norm_du = (float(np.linalg.norm(du_r.ravel())) +
                   float(np.linalg.norm(du_z.ravel())))
        rel_change = norm_du / norm_u

        # Apply mild under-relaxation on velocity (Forchheimer can oscillate
        # if K changes too fast between inner iterations)
        u_r = u_r + relaxation * du_r
        u_z = u_z + relaxation * du_z
        P = P_new

        if rel_change < tol_velocity:
            break

    # Final divergence check (informational)
    div_max = 0.0
    for j in range(mesh.nz):
        for i in range(mesh.nr):
            if not mesh.active[i, j]:
                continue
            rho_w = _face_avg_r(rho, i, j, mesh.nr)
            rho_e = _face_avg_r(rho, i + 1, j, mesh.nr)
            rho_s = _face_avg_z(rho, i, j, mesh.nz)
            rho_n = _face_avg_z(rho, i, j + 1, mesh.nz)
            div = (rho_e * u_r[i + 1, j] * mesh.area_r[i + 1, j]
                   - rho_w * u_r[i, j] * mesh.area_r[i, j]
                   + rho_n * u_z[i, j + 1] * mesh.area_z[i, j + 1]
                   - rho_s * u_z[i, j] * mesh.area_z[i, j]
                   - mass_source[i, j] * mesh.vol[i, j])
            div_max = max(div_max, abs(div))

    return PressureSolution(
        u_r=u_r, u_z=u_z, P=P,
        residual_continuity=div_max,
        residual_velocity=rel_change,
        inner_iterations=inner + 1,
        linear_stats=last_stats,
    )
