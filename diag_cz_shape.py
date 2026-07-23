#!/usr/bin/env python3
"""Diagnostic: cohesive-zone shape / inversion metric.

For each radial index i, finds the CZ band from ``fields.cohesive_mask``
(lowest and highest active j) and reports the band mid-height z_cz_mid(r)
plus the inversion metric

    dZ_apex = z_cz_mid(r=0) - z_cz_mid(r=wall)   [m]

Positive dZ_apex = inverted V (apex high at the axis) as in a healthy
center-working furnace (Fu 2014 Fig. 5, Zhou 2016 Fig. 8).

Usage:
    python diag_cz_shape.py [--nr 15 --nz 40 --iters 200]
                            [--ore-bias 0.0] [--ks-center 1.0]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def run_case(nr: int, nz: int, iters: int, ore_bias: float,
             ks_center: float) -> None:
    sys.path.insert(0, ".")
    from bf2d.cohesive_zone import cohesive_band_mask, cz_shape_metrics
    from bf2d.parameters import BFParameters2D
    from bf2d.simulation import BlastFurnaceSimulation

    kwargs = dict(
        nr=nr, nz=nz, max_outer_iter=iters, log_every=max(iters, 1), dt=0.5,
        n_tuyeres=32, tuyere_diameter=0.13,
        gas_exit_r_min=1.5, gas_exit_r_max=3.5,
    )
    # Only pass the new parameters when non-default so the script also runs
    # against the pre-Task-1 codebase (baseline capture).
    fields = BFParameters2D.__dataclass_fields__
    if "burden_radial_ore_bias" in fields:
        kwargs["burden_radial_ore_bias"] = ore_bias
    if "ks_center" in fields:
        kwargs["ks_center"] = ks_center

    p = BFParameters2D(**kwargs)
    sim = BlastFurnaceSimulation(p, log_level="ERROR")
    t0 = time.time()
    for it in range(iters):
        sim._one_outer_iteration(it)
        sim._adapt_relaxation()
    elapsed = time.time() - t0

    f, mesh = sim.fields, sim.mesh
    band = cohesive_band_mask(f.T_s, p, mesh.active)
    prof = cz_shape_metrics(band, mesh.active, mesh.z_c)

    print(f"\n=== CZ shape  mesh {nr}x{nz}  {iters} iters  "
          f"ore_bias={ore_bias}  ks_center={ks_center}  "
          f"({elapsed:.0f}s) ===")
    print(f"{'i':>3} {'r_c [m]':>8} {'z_lo [m]':>9} {'z_hi [m]':>9} "
          f"{'z_mid [m]':>9}")
    for i in range(mesh.nr):
        if np.isfinite(prof['z_mid'][i]):
            print(f"{i:>3} {mesh.r_c[i]:>8.3f} {prof['z_lo'][i]:>9.3f} "
                  f"{prof['z_hi'][i]:>9.3f} {prof['z_mid'][i]:>9.3f}")
        elif mesh.active[i, :].any():
            print(f"{i:>3} {mesh.r_c[i]:>8.3f} {'--- no CZ band ---':>29}")
    print(f"\ninversion dZ_apex = z_mid(axis) - z_mid(wall) "
          f"= {prof['inversion_m']:+.3f} m")
    print(f"band coverage: {prof['coverage'] * 100:.0f}% of active radii; "
          f"connected: {prof['connected']}")
    print(f"CZ band cell count: {int(band.sum())} "
          f"(cohesive_mask incl. dripping zone: "
          f"{int(f.cohesive_mask.sum())})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=15)
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ore-bias", type=float, default=0.0)
    ap.add_argument("--ks-center", type=float, default=1.0)
    args = ap.parse_args()
    run_case(args.nr, args.nz, args.iters, args.ore_bias, args.ks_center)
    return 0


if __name__ == "__main__":
    sys.exit(main())
