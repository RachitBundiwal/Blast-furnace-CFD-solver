#!/usr/bin/env python3
"""Diagnostic: per-element gas ledger (C, H, O, N) — in vs out.

H and N are inert in the gas phase (no reaction moves them between gas
and solid in this model: coke/PCI/moisture H all enter via the raceway
0D balance), so at steady state their tuyere inflow must equal the
top-ring outflow EXACTLY.  Any deficit is numerical (transport /
boundary treatment), not chemistry.  C and O additionally gain from the
in-shaft reaction sources; their closure includes the source term.

Usage:
    python diag_element_ledger.py [--nr 15 --nz 40 --iters 500]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# atoms per molecule
ATOMS = {
    "C": {"CO": 1, "CO2": 1},
    "H": {"H2": 2, "H2O": 2},
    "O": {"O2": 2, "CO": 1, "CO2": 2, "H2O": 1},
    "N": {"N2": 2},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=15)
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--ore-bias", type=float, default=0.35)
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from bf2d.constants import MW
    from bf2d.kinetics import assemble_species_sources, compute_rates
    from bf2d.parameters import BFParameters2D
    from bf2d.properties import gas_mw_mix
    from bf2d.simulation import BlastFurnaceSimulation

    p = BFParameters2D(
        nr=args.nr, nz=args.nz, max_outer_iter=args.iters,
        log_every=max(args.iters, 1), dt=0.5,
        n_tuyeres=32, tuyere_diameter=0.13,
        gas_exit_r_min=1.5, gas_exit_r_max=3.5,
        burden_radial_ore_bias=args.ore_bias,
    )
    sim = BlastFurnaceSimulation(p, log_level="ERROR")
    t0 = time.time()
    for it in range(args.iters):
        sim._one_outer_iteration(it)
        sim._adapt_relaxation()

    f, mesh, rw = sim.fields, sim.mesh, sim.raceway
    act = mesh.active
    area = np.pi * (mesh.r_f[1:] ** 2 - mesh.r_f[:-1] ** 2)
    mt = sim._gas_exit_mask[:, -1]

    # species molar flows IN (tuyere, from the 0D blast composition)
    x_b = rw.x_gas_blast
    mol_in = rw.total_blast_kg_s / sum(x_b[s] * MW[s] for s in x_b)
    n_in = {s: mol_in * x_b[s] for s in x_b}

    # species molar flows OUT (top exit ring)
    mw_top = gas_mw_mix(f.x_gas)[:, -1]
    mol_out_cells = (f.rho_g[:, -1] * f.u_z[:, mesh.nz] * area / mw_top)
    n_out = {s: float((mol_out_cells * f.x_gas[s][:, -1])[mt].sum())
             for s in x_b}

    # reaction species sources [mol/s]
    zn = np.broadcast_to((mesh.z_c / p.furnace_height)[None, :],
                         f.T_g.shape)
    r = compute_rates(f.T_g, f.T_s, f.P, f.rho_g, gas_mw_mix(f.x_gas),
                      f.x_gas, f.f_solid, f.eps, act, z_norm=zn,
                      feo_form_height_frac=p.feo_form_height_frac,
                      feo_reduce_height_frac=p.feo_reduce_height_frac,
                      dt_solid=200.0)
    src = assemble_species_sources(r)
    n_src = {s: float((src[s] * mesh.vol)[act].sum()) for s in src}

    print(f"\n=== Element gas ledger  {args.nr}x{args.nz}/{args.iters} it "
          f" ({time.time()-t0:.0f}s) ===")
    print(f"{'species':>8} {'in mol/s':>10} {'src mol/s':>10} "
          f"{'out mol/s':>10} {'out/(in+src)':>13}")
    for s in ("N2", "CO", "CO2", "H2", "H2O", "O2"):
        tot = n_in[s] + n_src.get(s, 0.0)
        ratio = n_out[s] / tot if abs(tot) > 1e-9 else float("nan")
        print(f"{s:>8} {n_in[s]:10.1f} {n_src.get(s, 0.0):10.1f} "
              f"{n_out[s]:10.1f} {ratio:13.3f}")

    print(f"\n{'element':>8} {'in':>10} {'src':>10} {'out':>10} "
          f"{'closure %':>10}")
    for el, comp in ATOMS.items():
        e_in = sum(n_in.get(s, 0.0) * a for s, a in comp.items())
        e_src = sum(n_src.get(s, 0.0) * a for s, a in comp.items())
        e_out = sum(n_out.get(s, 0.0) * a for s, a in comp.items())
        tot = e_in + e_src
        print(f"{el:>8} {e_in:10.1f} {e_src:10.1f} {e_out:10.1f} "
              f"{100.0 * e_out / max(tot, 1e-9):10.1f}")
    print("\nH and N have zero legitimate src: closure < 100% there is a")
    print("numerical loss in transport/boundaries, not chemistry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
