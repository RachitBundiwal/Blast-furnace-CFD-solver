#!/usr/bin/env python3
"""Command-line entry point for the 2D blast-furnace solver.

Example
-------
    python run_simulation.py --nr 30 --nz 80 --max-iter 200 --output output/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from bf2d.parameters import BFParameters2D, ParameterValidationError
from bf2d.simulation import BlastFurnaceSimulation
from bf2d.postprocess import plot_all, summary_metrics


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D axisymmetric blast-furnace solver (PISO, full physics)",
    )
    # Geometry
    p.add_argument("--nr", type=int, default=30)
    p.add_argument("--nz", type=int, default=80)
    p.add_argument("--height", type=float, default=28.0)
    # Blast / feed: default None = use the BFParameters2D defaults, which
    # are anchored to the Fu 2014 operating point (4993 tHM/d).  The old
    # hard-coded CLI defaults (6500 Nm3/min, 9 kg/s coke, 38 kg/s ore)
    # silently overrode those with the pre-Phase-7 inconsistent ratios.
    p.add_argument("--blast-rate", type=float, default=None,
                   help="Blast rate [Nm^3/min] (default: Fu 2014, 3640)")
    p.add_argument("--blast-temp-C", type=float, default=None,
                   help="Raceway-exit gas T [degC] (default: Fu 2014, 2000)")
    p.add_argument("--o2-enrich", type=float, default=None)
    p.add_argument("--coke-rate", type=float, default=None,
                   help="kg/s (default: Fu 2014, 23.75)")
    p.add_argument("--ore-rate", type=float, default=None,
                   help="kg/s (default: Fu 2014, 86.7)")
    # Top-gas uptake ring + radial burden distribution (the validated
    # plant configuration uses --gas-exit-rmin 1.5 --gas-exit-rmax 3.5
    # --ore-bias 0.35; see validate_plant.py)
    p.add_argument("--gas-exit-rmin", type=float, default=None,
                   help="inner radius of the uptake exit ring [m]")
    p.add_argument("--gas-exit-rmax", type=float, default=None,
                   help="outer radius of the uptake exit ring [m]")
    p.add_argument("--ore-bias", type=float, default=None,
                   help="radial burden ore bias (plant range 0.2-0.5)")
    # Numerics
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--tol", type=float, default=5.0e-5)
    # Output
    p.add_argument("--output", type=str, default="output")
    p.add_argument("--case", type=str, default="bf2d_run")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Build parameters (validation happens in __post_init__).  Optional
    # CLI overrides are only passed when given, so BFParameters2D keeps
    # its Fu-2014-anchored defaults as the single source of truth.
    overrides = {
        "blast_rate_Nm3_min": args.blast_rate,
        "blast_temperature": (args.blast_temp_C + 273.15
                              if args.blast_temp_C is not None else None),
        "oxygen_enrichment": args.o2_enrich,
        "coke_rate_kg_s": args.coke_rate,
        "ore_rate_kg_s": args.ore_rate,
        "gas_exit_r_min": args.gas_exit_rmin,
        "gas_exit_r_max": args.gas_exit_rmax,
        "burden_radial_ore_bias": args.ore_bias,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    try:
        params = BFParameters2D(
            furnace_height=args.height,
            nr=args.nr, nz=args.nz,
            max_outer_iter=args.max_iter,
            dt=args.dt,
            convergence_rms=args.tol,
            output_dir=args.output,
            case_name=args.case,
            **overrides,
        )
    except ParameterValidationError as exc:
        print(f"ERROR: invalid parameters -- {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)
    # Persist the fully-resolved parameter set for reproducibility
    with open(os.path.join(args.output, "parameters.json"), "w",
              encoding="utf-8") as fh:
        json.dump(params.to_dict(), fh, indent=2, default=str)

    sim = BlastFurnaceSimulation(params, log_level=args.log_level)
    results = sim.run()

    # Metrics
    metrics = summary_metrics(results)
    print("\n=== Summary metrics ===")
    for k, v in metrics.items():
        print(f"  {k:25s} {v:>10.3f}")

    with open(os.path.join(args.output, "metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Plots
    if not args.no_plots:
        try:
            plot_dir = plot_all(results, params,
                                output_dir=os.path.join(args.output, "plots"))
            print(f"Plots written to: {plot_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: plotting failed -- {type(exc).__name__}",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
