"""
Cohesive-zone (CZ) model.

In a real blast furnace the CZ is the inverted-V region where ore softens
and melts while coke slits remain porous.  It dramatically alters gas flow
(pressure drop rises 4-6x in the softened ore layers).  This module
identifies CZ cells from the local solid temperature and returns:

* ``in_cz``     : boolean mask of cells in the cohesive zone
* ``eps_eff``   : effective porosity (lower inside CZ, preserved in coke slits)
* ``dp_eff``    : effective particle diameter (shrunk in CZ)

The reduction model uses gas temperature for indirect reduction (upper stack),
solid temperature for direct reduction (lower bosh), so CZ information feeds
through the permeability closures in ``properties.ergun_pressure_drop_coeff``.
"""
from __future__ import annotations

import numpy as np

from .parameters import BFParameters2D


def cohesive_band_mask(
    T_s: np.ndarray,
    p: BFParameters2D,
    active: np.ndarray,
) -> np.ndarray:
    """True cohesive BAND: softening < T_s < liquidus (Fu 2014 Fig. 5).

    ``detect_cohesive_zone``'s ``in_cz`` mask deliberately includes
    fully-molten cells (sigma = 1) because the porosity collapse persists
    into the dripping zone.  For SHAPE diagnostics that is wrong: the
    plant cohesive zone is the band between the ore softening isotherm
    (~1373 K) and the liquidus (~1673 K) of the solid temperature field;
    below the liquidus the material is liquid dripping through the coke
    bed (dripping zone), not cohesive burden.  This helper returns the
    band-only mask used by the cz_inversion_m KPI and diagnostics.
    """
    sigma = np.clip(
        (T_s - p.softening_temperature)
        / max(p.melting_temperature - p.softening_temperature, 1.0),
        0.0, 1.0,
    )
    return (sigma > 0.05) & (sigma < 0.95) & active


def cz_shape_metrics(
    cohesive_mask: np.ndarray,
    active: np.ndarray,
    z_c: np.ndarray,
) -> dict[str, object]:
    """Per-radius CZ band profile and inversion metric.

    For each radial index the CZ band is the lowest-to-highest active
    cell where ``cohesive_mask`` is True; its mid-height is
    ``z_mid = (z_lo + z_hi) / 2``.  The inversion metric

        inversion_m = z_mid(innermost radius) - min_i z_mid(i)

    is the apex height above the LOWEST band foot — exactly the
    distance Fu 2014 Fig. 5 shows (~3-5 m for their furnace).  It is
    POSITIVE for an inverted-V (apex high at the axis) and ~0 for a
    flat, V- or dome-shaped zone (apex not at the axis).  The foot is
    the minimum over radii rather than the outermost radius: the
    outermost radii exist only in the bosh/belly wall corner (a narrow
    z-window above the raceway), and comparing the axis against that
    geometric sliver misreads a healthy V whose feet sit at the widest
    full-height radius.

    Returns
    -------
    dict with keys ``z_lo``, ``z_hi``, ``z_mid`` (arrays (nr,), NaN
    where no band), ``inversion_m`` (float, NaN if no CZ anywhere),
    ``coverage`` (fraction of active radii that have a CZ band) and
    ``connected`` (True if the radii holding a band form ONE contiguous
    run — i.e. no doughnut holes or detached blobs.  Radii beyond the
    band's outer foot may legitimately lack a band: near-wall cells are
    stave-cooled below softening, the plant "skull").
    """
    if cohesive_mask.shape != active.shape:
        raise ValueError("cohesive_mask and active shapes differ")
    if cohesive_mask.shape[1] != z_c.shape[0]:
        raise ValueError("z_c length does not match mask axial dimension")
    nr = cohesive_mask.shape[0]
    z_lo = np.full(nr, np.nan)
    z_hi = np.full(nr, np.nan)
    for i in range(nr):
        js = np.where(cohesive_mask[i, :] & active[i, :])[0]
        if js.size:
            z_lo[i] = z_c[js.min()]
            z_hi[i] = z_c[js.max()]
    z_mid = 0.5 * (z_lo + z_hi)

    has_band = np.isfinite(z_mid)
    active_radii = active.any(axis=1)
    coverage = float(has_band[active_radii].sum()) / max(
        int(active_radii.sum()), 1)

    inversion = float("nan")
    connected = False
    if has_band.any():
        i_axis = int(np.argmax(has_band))
        i_wall = int(nr - 1 - np.argmax(has_band[::-1]))
        inversion = float(z_mid[i_axis] - np.nanmin(z_mid))
        connected = bool(has_band[i_axis:i_wall + 1].all())
    return {"z_lo": z_lo, "z_hi": z_hi, "z_mid": z_mid,
            "inversion_m": inversion, "coverage": coverage,
            "connected": connected}


def detect_cohesive_zone(
    T_s: np.ndarray,
    f_solid: dict[str, np.ndarray],
    eps_base: np.ndarray,
    dp_base: np.ndarray,
    p: BFParameters2D,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CZ mask, effective porosity and effective particle diameter.

    Arguments
    ---------
    T_s
        Solid temperature field  [K].
    f_solid
        Dictionary of solid mass fractions keyed by phase.
    eps_base, dp_base
        Reference porosity / Sauter diameter outside the CZ.
    p
        Simulation parameters (softening / melting thresholds).
    active
        Active-cell mask from the mesh.
    """
    T_soft = p.softening_temperature
    T_melt = p.melting_temperature

    # Degree of softening: 0 below softening, 1 above melting
    sigma = np.clip((T_s - T_soft) / max(T_melt - T_soft, 1.0), 0.0, 1.0)

    # Coke mass-fraction used to preserve permeability (coke slits)
    f_coke = f_solid.get("C_coke", np.zeros_like(T_s))

    # Modified porosity:
    #   above melting: epsilon collapses to CZ minimum except where coke dominates
    #   below softening: unchanged
    eps_cz = p.cz_porosity_min
    coke_window = p.cz_porosity_coke_window
    # Where coke mass fraction >= 0.5 we keep most of the base porosity
    coke_preserve = np.clip(f_coke * 2.0, 0.0, 1.0)
    eps_soft_target = (coke_preserve * coke_window +
                       (1.0 - coke_preserve) * eps_cz)
    eps_eff = (1.0 - sigma) * eps_base + sigma * eps_soft_target

    # Modified Sauter diameter (softened ore coalesces into larger lumps)
    dp_eff = dp_base * (1.0 - 0.65 * sigma)
    dp_eff = np.maximum(dp_eff, 4.0e-3)

    in_cz = (sigma > 0.05) & active

    # Zero-out inactive cells (so downstream math stays sane)
    eps_eff = np.where(active, eps_eff, eps_base)
    dp_eff = np.where(active, dp_eff, dp_base)
    return in_cz, eps_eff, dp_eff
