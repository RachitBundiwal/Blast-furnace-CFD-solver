#!/usr/bin/env python3
"""Diagnostic: shaft carbon ledger (Task 3 step 4).

Carbon IN:  charged coke fixed carbon + PCI carbon (both as kg C/s).
Carbon OUT: CO + CO2 leaving the top-gas exit ring (kg C/s), plus the
reaction-consumption view (kg C/s gasified by R1/R1b/R5/R6/R_DR).

At steady state the gas-side export must equal the reaction consumption
(the gas column holds no inventory) and both should approach the
charged carbon that is actually consumed.  Closure gaps localize where
carbon leaks from the books (e.g. the pre-Phase-9 exit-BC mass clamp).

Usage:
    python diag_carbon_ledger.py [--nr 15 --nz 80 --iters 200]
                                 [--ore-bias 0.35] [--h2o-slip 0.35]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=15)
    ap.add_argument("--nz", type=int, default=80)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ore-bias", type=float, default=0.35)
    ap.add_argument("--h2o-slip", type=float, default=None,
                    help="raceway H2O slip (default: parameter default)")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from bf2d.constants import MW
    from bf2d.kinetics import assemble_species_sources, compute_rates
    from bf2d.parameters import BFParameters2D
    from bf2d.properties import gas_mw_mix
    from bf2d.simulation import BlastFurnaceSimulation

    kwargs = dict(
        nr=args.nr, nz=args.nz, max_outer_iter=args.iters,
        log_every=max(args.iters, 1), dt=0.5,
        n_tuyeres=32, tuyere_diameter=0.13,
        gas_exit_r_min=1.5, gas_exit_r_max=3.5,
        burden_radial_ore_bias=args.ore_bias,
    )
    if args.h2o_slip is not None:
        kwargs["raceway_h2o_slip"] = args.h2o_slip
    p = BFParameters2D(**kwargs)
    sim = BlastFurnaceSimulation(p, log_level="ERROR")
    t0 = time.time()
    for it in range(args.iters):
        sim._one_outer_iteration(it)
        sim._adapt_relaxation()

    f, mesh, rw = sim.fields, sim.mesh, sim.raceway
    act = mesh.active
    area = np.pi * (mesh.r_f[1:] ** 2 - mesh.r_f[:-1] ** 2)
    mt = sim._gas_exit_mask[:, -1]

    # ---- Carbon IN -------------------------------------------------------
    c_coke_in = 0.87 * p.coke_rate_kg_s          # fixed carbon in coke
    pci_kg_s = (p.pci_rate_kg_tHM * p.productivity_tHM_day / 86400.0)
    c_pci_in = 0.85 * pci_kg_s
    c_in = c_coke_in + c_pci_in

    # ---- Carbon OUT via top gas ------------------------------------------
    mw = gas_mw_mix(f.x_gas)
    rho_top = f.rho_g[:, -1]
    u_top = f.u_z[:, mesh.nz]
    mol_out = (rho_top * u_top * area / mw[:, -1])[mt]     # mol gas/s
    x_co = f.x_gas["CO"][mt, -1]
    x_co2 = f.x_gas["CO2"][mt, -1]
    c_out_gas = float((mol_out * (x_co + x_co2)).sum()) * 0.0120

    # ---- Carbon consumed by reactions (gasification view) ----------------
    zn = np.broadcast_to((mesh.z_c / p.furnace_height)[None, :], f.T_g.shape)
    r = compute_rates(f.T_g, f.T_s, f.P, f.rho_g, mw, f.x_gas, f.f_solid,
                      f.eps, act, z_norm=zn,
                      feo_form_height_frac=p.feo_form_height_frac,
                      feo_reduce_height_frac=p.feo_reduce_height_frac,
                      dt_solid=200.0)
    def tot(rate):
        return float((rate * mesh.vol)[act].sum())
    c_rxn = (tot(r.R1) + tot(r.R1b) + tot(r.R5) + tot(r.R6)
             + tot(r.R_DR)) * 0.0120
    # raceway 0D combustion carbon (enters the gas as blast CO/CO2/H2,
    # not through the in-shaft rates above)
    x_b = rw.x_gas_blast
    mol_blast = rw.total_blast_kg_s / sum(x_b[s] * MW[s] for s in x_b)
    c_raceway = mol_blast * (x_b["CO"] + x_b["CO2"]) * 0.0120

    # net gas mass source (for reference)
    src = assemble_species_sources(r)
    m_src = sum(((src[sp] * MW[sp]) * mesh.vol)[act].sum() for sp in src)

    # per-reaction totals [mol/s] and the iron ledger
    print("\nper-reaction totals [mol/s]:")
    for name in ("R2", "R3", "R4", "R2H", "R3H", "R4H", "R5", "R6",
                 "R7", "R_DR", "R9"):
        print(f"  {name:5s}: {tot(getattr(r, name)):9.1f}")
    fe_kg_s = (tot(r.R4) + tot(r.R4H) + tot(r.R_DR)) * 0.0558
    fe_charged = 0.62 * p.ore_rate_kg_s
    thm_d = fe_kg_s / 945.0 * 86400.0
    print(f"\nFe produced: {fe_kg_s:6.2f} kg/s of {fe_charged:6.2f} charged "
          f"({100*fe_kg_s/fe_charged:5.1f}%)  -> {thm_d:7.0f} tHM/d "
          f"(plant 4993)")

    print(f"\n=== Carbon ledger  {args.nr}x{args.nz}/{args.iters} it  "
          f"bias={args.ore_bias}  slip={p.raceway_h2o_slip}  "
          f"({time.time()-t0:.0f}s) ===")
    print(f"C in, coke fixed carbon        : {c_coke_in:7.2f} kg/s")
    print(f"C in, PCI                      : {c_pci_in:7.2f} kg/s")
    print(f"C in, TOTAL charged            : {c_in:7.2f} kg/s")
    print(f"C to gas in raceway 0D         : {c_raceway:7.2f} kg/s")
    print(f"C gasified in-shaft (rates)    : {c_rxn:7.2f} kg/s")
    print(f"C out via top gas CO+CO2       : {c_out_gas:7.2f} kg/s")
    print(f"net reaction gas-mass source   : {m_src:7.2f} kg/s")
    gas_source_c = c_raceway + c_rxn
    print(f"\nclosure (top-gas C / gas-source C): "
          f"{100.0 * c_out_gas / max(gas_source_c, 1e-9):6.1f} %  "
          f"(gas-source C = raceway + in-shaft = {gas_source_c:.2f})")
    print(f"consumption vs charged           : "
          f"{100.0 * gas_source_c / max(c_in, 1e-9):6.1f} %  "
          f"(plant: ~100% of charged C leaves as top-gas CO/CO2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
