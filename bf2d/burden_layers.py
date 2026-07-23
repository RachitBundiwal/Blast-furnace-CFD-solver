"""Layered burden structure (Fu et al. 2014 cell-type technique).

The blast furnace charge is alternating layers of ore and coke (layer
thickness 0.2-0.8 m, Fu 2014 Sect. 1).  Fu's CFD treatment assigns each
grid cell a type -- "ore cell", "coke cell", or fractional "interface
cell" -- and applies layer-specific properties and reaction source terms
per cell (their Fig. 2, Table 3/4).

This module provides:

* :func:`ore_fraction_field`  -- static periodic ore-fraction map
  (Fu assumption 3/4: layer structure and volume persist to the CZ).
* :func:`radial_ore_weight` / :func:`apply_radial_ore_bias` -- radial
  center-working charge distribution (more ore at the wall, more coke at
  the axis; Fu 2014 Sect. 2.4, Zhou 2016 Fig. 2).  This is the primary
  driver of the inverted-V cohesive zone.
* :func:`layered_bed_properties` -- porosity / particle-diameter fields
  from the ore fraction, incl. the CZ ore-layer porosity collapse
  (eps_ore -> 0.10 inside the CZ while coke slits stay at eps_coke,
  Fu Table 6) which produces the plant "coke slit" gas channelling.

Deviations from Fu (documented, revisit with Nagarnar charging data):

* Layers are horizontal.  Fu inclines them with the burden-descent
  profile; the inclination requires a burden-distribution model whose
  inputs (charging matrix, stockline profile) arrive with plant data.
* Single steady layer snapshot.  Fu averages 2-4 phase-shifted cases;
  he shows each single case already reproduces the global KPIs
  (their Sect. 3.2: "each steady state case also predicts the global
  blast furnace performance").
* One solid temperature field (Fu carries separate ore / coke energy
  equations).  Future work; noted in PHASE_HISTORY.

Secure defaults: the feature is opt-in via ``BFParameters2D.layered_burden``.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh2D
from .parameters import BFParameters2D

# Fu 2014 Table 6 layer properties
EPS_ORE: float = 0.45
EPS_ORE_CZ: float = 0.10        # fused ore inside cohesive zone
EPS_COKE: float = 0.36
DP_ORE: float = 0.012           # m
DP_COKE: float = 0.050          # m


def ore_fraction_field(
    mesh: Mesh2D,
    ore_layer_m: float,
    coke_layer_m: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """Static ore volume-fraction per cell for horizontal periodic layers.

    Layers repeat with period ``L = ore_layer_m + coke_layer_m`` measured
    downward from the stockline.  Each cell's ore fraction is the exact
    overlap of the cell's axial extent with the ore bands (so interface
    cells get the correct fractional value, matching Fu's interface-cell
    treatment).

    Parameters are validated defensively; degenerate inputs raise
    ``ValueError`` rather than producing NaN fields.
    """
    if ore_layer_m <= 0.0 or coke_layer_m <= 0.0:
        raise ValueError("layer thicknesses must be positive")
    period = ore_layer_m + coke_layer_m
    dz_med = float(np.median(mesh.dz))
    if period < 3.0 * dz_med:
        # Fewer than ~3 cells per layer pair in the bulk of the column
        # cannot resolve the structure; fail loudly so the user picks a
        # finer mesh or thicker layers.  (Median, not max: stretched
        # meshes have a few coarse cells near the top that would
        # otherwise veto realistic 0.8 m layer periods.)
        raise ValueError(
            f"layer period {period:.3f} m under-resolved by mesh "
            f"(median dz = {dz_med:.3f} m); refine nz or increase "
            f"layer thickness")

    z_top = float(mesh.z_f[mesh.nz])          # stockline
    frac = np.zeros((mesh.nr, mesh.nz))

    def ore_measure(a: float, b: float) -> float:
        """Length of [a, b] (depth-from-top coords) inside ore bands."""
        total = 0.0
        # ore band occupies [k*period, k*period + ore_layer_m)
        k0 = int(np.floor(a / period))
        k1 = int(np.floor(b / period)) + 1
        for k in range(k0, k1 + 1):
            lo = k * period + phase_offset * period
            hi = lo + ore_layer_m
            total += max(0.0, min(b, hi) - max(a, lo))
        return total

    for j in range(mesh.nz):
        d_hi = z_top - float(mesh.z_f[j])       # depth of cell bottom
        d_lo = z_top - float(mesh.z_f[j + 1])   # depth of cell top
        dz = d_hi - d_lo
        f_j = ore_measure(d_lo, d_hi) / max(dz, 1e-12)
        frac[:, j] = np.clip(f_j, 0.0, 1.0)
    return frac


def radial_ore_weight(mesh: Mesh2D, p: BFParameters2D) -> np.ndarray:
    """Radial ore-fraction weight field w(r, z), area-mean 1 per row.

    Implements the center-working charging profile (Fu 2014 Sect. 2.4
    burden distribution; Zhou 2016 Fig. 2): more ore toward the wall,
    more coke at the axis.

        w_raw(r, z) = clip(w0 + (w1 - w0) * (r / R_wall(z))^q, 0, 1.6)

    with ``w0 = 1 - bias`` (axis) and ``w1 = 1 + bias`` (wall), where
    ``bias = p.burden_radial_ore_bias`` and
    ``q = p.burden_radial_ore_exponent``.  The raw weight is renormalized
    per z-row by its area-weighted mean over ACTIVE cells so that
    multiplying any radially-uniform ore fraction by ``w`` preserves the
    row (and column) average exactly.  Inactive cells get w = 1.

    ``bias = 0`` returns an all-ones field (radially uniform charge,
    pre-existing behaviour).
    """
    bias = p.burden_radial_ore_bias
    q = p.burden_radial_ore_exponent
    w = np.ones((mesh.nr, mesh.nz))
    if bias <= 0.0:
        return w
    w0, w1 = 1.0 - bias, 1.0 + bias
    ring_area = np.pi * (mesh.r_f[1:] ** 2 - mesh.r_f[:-1] ** 2)  # (nr,)
    for j in range(mesh.nz):
        act = mesh.active[:, j]
        if not act.any():
            continue
        r_local = max(float(mesh.r_wall[j]), 1e-9)
        x = np.clip(mesh.r_c / r_local, 0.0, 1.0)
        w_raw = np.clip(w0 + (w1 - w0) * x ** q, 0.0, 1.6)
        mean = float((w_raw[act] * ring_area[act]).sum()
                     / max(ring_area[act].sum(), 1e-12))
        w[act, j] = w_raw[act] / max(mean, 1e-12)
    return w


def apply_radial_ore_bias(
    ore_frac: np.ndarray,
    weight: np.ndarray,
    mesh: Mesh2D,
) -> np.ndarray:
    """Apply the radial weight to an ore-fraction field, mass-preserving.

    The naive product ``ore_frac * weight`` can exceed 1 in pure-ore
    layers near the wall; clipping it to [0, 1] would silently REMOVE
    ore from the column and break the global mass balance.  To keep the
    column-average ore fraction exact, a scalar compensation factor c is
    found by fixed-point iteration such that the volume-weighted mean of
    ``clip(ore_frac * weight * c, 0, 1)`` over active cells equals the
    mean of the unbiased field.  The mean is monotone increasing in c,
    so the iteration converges; a defensive iteration cap raises if it
    cannot (degenerate all-saturated fields).
    """
    if ore_frac.shape != weight.shape:
        raise ValueError("ore_frac and weight shapes differ")
    act = mesh.active
    vol = mesh.vol
    v_sum = float(vol[act].sum())
    if v_sum <= 0.0:
        return np.clip(ore_frac * weight, 0.0, 1.0)
    target = float((ore_frac[act] * vol[act]).sum() / v_sum)
    if target <= 1e-12:
        return np.clip(ore_frac * weight, 0.0, 1.0)

    c = 1.0
    biased = np.clip(ore_frac * weight, 0.0, 1.0)
    for _ in range(60):
        biased = np.clip(ore_frac * weight * c, 0.0, 1.0)
        mean = float((biased[act] * vol[act]).sum() / v_sum)
        if abs(mean - target) <= 1e-4 * target:
            return biased
        if mean <= 1e-15:
            break
        c *= target / mean
    raise ValueError(
        "apply_radial_ore_bias: could not preserve column-average ore "
        f"fraction (target {target:.4f}); field too saturated for the "
        "requested bias")


def layered_bed_properties(
    ore_frac: np.ndarray,
    cz_mask: np.ndarray,
    p: BFParameters2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Porosity and Sauter diameter fields from the layer structure.

    Outside the CZ:  eps = f*EPS_ORE + (1-f)*EPS_COKE  (volume blend).
    Inside the CZ:   the ORE part fuses (EPS_ORE -> EPS_ORE_CZ) while the
    coke fraction keeps EPS_COKE -- this is the Fu "coke slit" mechanism
    that channels the ascending gas through coke layers in the CZ.
    Particle diameter blends by volume fraction.
    """
    if ore_frac.shape != cz_mask.shape:
        raise ValueError("ore_frac and cz_mask shapes differ")
    f = np.clip(ore_frac, 0.0, 1.0)
    eps_ore_local = np.where(cz_mask, EPS_ORE_CZ, EPS_ORE)
    eps = f * eps_ore_local + (1.0 - f) * EPS_COKE
    dp = f * DP_ORE + (1.0 - f) * DP_COKE
    return eps, dp
