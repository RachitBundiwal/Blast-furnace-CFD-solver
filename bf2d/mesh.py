"""
Axisymmetric staggered mesh for the blast-furnace solver.

Coordinate convention
---------------------
- ``r``: radial coordinate, 0 at axis, increasing outward
- ``z``: axial coordinate, 0 at tuyere-band base, increasing upward
- Staggered arrangement (Patankar style):
    * Scalars (T, P, species, rho) live at cell centres (``i, j``)
    * u_r (radial velocity) lives on east cell faces (``i+1/2, j``)
    * u_z (axial velocity) lives on north cell faces (``i, j+1/2``)

Furnace outline
---------------
Standard 4-zone profile (hearth -> bosh -> belly -> stack) constructed from
the radii and band heights supplied via ``BFParameters2D``.  Cells whose
centroid lies outside the profile are flagged ``inactive`` and are ignored
by every solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .parameters import BFParameters2D


@dataclass(frozen=True)
class Mesh2D:
    """Immutable container for mesh arrays.

    All arrays are ``numpy.ndarray`` with shape ``(nr, nz)`` for scalars,
    ``(nr+1, nz)`` for u_r and ``(nr, nz+1)`` for u_z.
    """

    # Cell centres
    r_c: np.ndarray          # (nr,)   radial centre positions
    z_c: np.ndarray          # (nz,)   axial  centre positions
    # Face coordinates
    r_f: np.ndarray          # (nr+1,) radial face positions
    z_f: np.ndarray          # (nz+1,) axial  face positions
    # Cell spacings
    dr: np.ndarray           # (nr,)
    dz: np.ndarray           # (nz,)
    # Cell volumes (annular)    V = 2 pi r dr dz
    vol: np.ndarray          # (nr, nz)
    # Face areas
    area_r: np.ndarray       # (nr+1, nz)   annular east/west faces
    area_z: np.ndarray       # (nr, nz+1)   horizontal north/south faces
    # Wall radius at each axial location
    r_wall: np.ndarray       # (nz,)
    # Active-cell mask (True inside furnace profile)
    active: np.ndarray       # (nr, nz) bool
    # Boundary-cell masks (inside active region)
    is_wall: np.ndarray      # (nr, nz)  cell touches furnace wall
    is_axis: np.ndarray      # (nr, nz)  i == 0
    is_top: np.ndarray       # (nr, nz)  j == nz-1
    is_tuyere: np.ndarray    # (nr, nz)  tuyere-injection band
    # Summary shape
    nr: int
    nz: int


# ---------------------------------------------------------------------------
def _stretched_1d(n: int, length: float, stretch: float,
                  cluster_side: str = "low") -> Tuple[np.ndarray, np.ndarray]:
    """Generate 1D faces and cell-centre positions.

    ``stretch=1`` -> uniform grid.  ``stretch>1`` clusters cells on the
    selected side using a simple geometric progression normalised to the
    requested domain length.
    """
    if stretch <= 1.0 + 1e-9:
        faces = np.linspace(0.0, length, n + 1)
    else:
        # geometric spacing
        ratios = stretch ** np.arange(n)
        widths = ratios / ratios.sum() * length
        if cluster_side == "high":
            widths = widths[::-1]
        faces = np.concatenate(([0.0], np.cumsum(widths)))
        # numerical safety: force exact bounds
        faces[0] = 0.0
        faces[-1] = length
    centres = 0.5 * (faces[:-1] + faces[1:])
    return faces, centres


# ---------------------------------------------------------------------------
def _wall_profile(z: np.ndarray, p: BFParameters2D) -> np.ndarray:
    """Piece-wise linear furnace wall radius r_wall(z).

    Zones, bottom to top:
        hearth   [0, H_h]              r = R_hearth  (straight)
        bosh     [H_h, H_h+H_b]        r linear  R_hearth -> R_bosh
        belly    [..., ...+H_bel]      r = R_belly (approx. R_bosh)
        stack    [..., top]            r linear  R_belly -> R_top
    """
    H_h = p.hearth_height
    H_b = p.bosh_height
    H_bel = p.belly_height
    H_total = p.furnace_height
    r_wall = np.empty_like(z)
    for k, zk in enumerate(z):
        if zk <= H_h:
            r_wall[k] = p.hearth_radius
        elif zk <= H_h + H_b:
            f = (zk - H_h) / max(H_b, 1e-9)
            r_wall[k] = p.hearth_radius + f * (p.bosh_radius - p.hearth_radius)
        elif zk <= H_h + H_b + H_bel:
            f = (zk - (H_h + H_b)) / max(H_bel, 1e-9)
            r_wall[k] = p.bosh_radius + f * (p.belly_radius - p.bosh_radius)
        else:
            denom = max(H_total - (H_h + H_b + H_bel), 1e-9)
            f = (zk - (H_h + H_b + H_bel)) / denom
            f = min(max(f, 0.0), 1.0)
            r_wall[k] = p.belly_radius + f * (p.stack_top_radius - p.belly_radius)
    return r_wall


# ---------------------------------------------------------------------------
def build_mesh(p: BFParameters2D) -> Mesh2D:
    """Construct an axisymmetric staggered mesh from operating parameters."""
    # Axial mesh - cluster near tuyere
    z_f, z_c = _stretched_1d(p.nz, p.furnace_height,
                             stretch=p.mesh_stretch_z, cluster_side="low")
    dz = np.diff(z_f)

    # Radial mesh - cluster near wall
    r_max = max(p.hearth_radius, p.bosh_radius,
                p.belly_radius, p.stack_top_radius)
    r_f, r_c = _stretched_1d(p.nr, r_max,
                             stretch=p.mesh_stretch_r, cluster_side="high")
    dr = np.diff(r_f)

    # Wall radius at each axial position
    r_wall = _wall_profile(z_c, p)

    # Active-cell mask
    active = r_c[:, None] <= r_wall[None, :]   # (nr, nz)

    # Annular cell volume  V = 2 pi r dr dz  (evaluated at cell centre)
    vol = 2.0 * np.pi * r_c[:, None] * dr[:, None] * dz[None, :]

    # Face areas
    # East/west: A = 2 pi r_f dz    shape (nr+1, nz)
    area_r = 2.0 * np.pi * r_f[:, None] * dz[None, :]
    # North/south: annular ring  A = pi (r_f[i+1]^2 - r_f[i]^2)    shape (nr, nz+1)
    ring = np.pi * (r_f[1:] ** 2 - r_f[:-1] ** 2)
    area_z = np.broadcast_to(ring[:, None], (p.nr, p.nz + 1)).copy()

    # Boundary flags within active domain
    is_axis = np.zeros_like(active)
    is_axis[0, :] = active[0, :]

    is_top = np.zeros_like(active)
    is_top[:, -1] = active[:, -1]

    # Wall cells: active cell whose east neighbour is inactive, OR last radial
    is_wall = np.zeros_like(active)
    for j in range(p.nz):
        for i in range(p.nr):
            if not active[i, j]:
                continue
            if i == p.nr - 1 or not active[i + 1, j]:
                is_wall[i, j] = True

    # Tuyere band - bottom ring of active cells, inner annulus on the wall side
    is_tuyere = np.zeros_like(active)
    tuyere_z_lo, tuyere_z_hi = _find_tuyere_band(z_c, p)
    r_in = (1.0 - p.tuyere_penetration_fraction) * p.hearth_radius
    for j in range(p.nz):
        if tuyere_z_lo <= z_c[j] <= tuyere_z_hi:
            for i in range(p.nr):
                if active[i, j] and r_c[i] >= r_in:
                    is_tuyere[i, j] = True

    return Mesh2D(
        r_c=r_c, z_c=z_c, r_f=r_f, z_f=z_f,
        dr=dr, dz=dz, vol=vol,
        area_r=area_r, area_z=area_z,
        r_wall=r_wall,
        active=active, is_wall=is_wall,
        is_axis=is_axis, is_top=is_top,
        is_tuyere=is_tuyere,
        nr=p.nr, nz=p.nz,
    )


def _find_tuyere_band(z_c: np.ndarray, p: BFParameters2D) -> Tuple[float, float]:
    """Approximate tuyere elevation band (+/- half a dz around the target)."""
    dz = np.diff(z_c).mean() if len(z_c) > 1 else 0.1
    z0 = max(z_c.min(), p.tuyere_elevation)
    return z0 - 0.5 * dz, z0 + 1.5 * dz
