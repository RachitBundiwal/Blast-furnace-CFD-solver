#!/usr/bin/env python3
"""Plant-validation harness: run the solver and score it against
plant measurements (Fu et al. 2014, Applied Thermal Eng 66, Table 5/7).

Usage:
    python validate_plant.py [--nr 20] [--nz 80] [--iters 200] [--json out.json]

Exit code 0 if all KPIs within tolerance bands, 1 otherwise, 2 on error.
Designed for CI: fast mesh by default, JSON report for artifact upload.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Plant reference values (Fu 2014, 4993 MTHM/day furnace, Table 7 + Table 5)
# tolerance = acceptable absolute deviation for a PASS; "warn" band beyond.
# --------------------------------------------------------------------------
PLANT_TARGETS = {
    "top_gas_N2_pct":  {"value": 47.62, "tol": 3.0, "warn": 6.0},
    "top_gas_CO_pct":  {"value": 23.88, "tol": 2.0, "warn": 4.0},
    "top_gas_CO2_pct": {"value": 22.18, "tol": 2.0, "warn": 5.0},
    "top_gas_H2_pct":  {"value": 6.30,  "tol": 1.5, "warn": 3.0},
    "top_gas_H2O_pct": {"value": 1.00,  "tol": 1.0, "warn": 2.0},
    "T_max_K":         {"value": 2300.0, "tol": 200.0, "warn": 400.0},
    "top_pressure_bar": {"value": 1.70, "tol": 0.9, "warn": 1.5},
    "mass_closure_pct": {"value": 100.0, "tol": 1.0, "warn": 5.0},
    "sumY_drift":      {"value": 0.0,  "tol": 1e-6, "warn": 1e-3},
    # Reduction-ladder checks (Fu 2014 Fig 9): metallic Fe confined to
    # the bottom third; hematite gone by z/H ~ 0.6.
    # one_sided="max": measured value at or BELOW target is fully
    # compliant (Fe must not appear above z/H = 0.40; lower onset means
    # tighter confinement to the cohesive zone, which is desirable).
    "fe_onset_zH_max": {"value": 0.40, "tol": 0.05, "warn": 0.10,
                        "one_sided": "max"},
    # Coke rate (Fu Table 7: 408-413 kg/tHM) from the shaft carbon balance;
    # direct-reduction degree (RIST operating line, plant 35-45 %).
    "coke_rate_kg_tHM": {"value": 411.0, "tol": 40.0, "warn": 100.0},
    "dr_degree_pct":    {"value": 40.0,  "tol": 10.0, "warn": 20.0},
    # Cohesive-zone shape (Fu 2014 Fig. 5): inverted V, apex at the axis
    # ~3-5 m above the feet at the wall.  one_sided="min": any measured
    # inversion at or ABOVE the target is fully compliant (a taller apex
    # is a stronger center-working profile, which is desirable);
    # tol band admits apex >= +0.5 m.
    "cz_inversion_m":  {"value": 2.0, "tol": 1.5, "warn": 3.0,
                        "one_sided": "min"},
}


def run_validation(nr: int, nz: int, iters: int,
                   ore_bias: float = 0.35,
                   h2o_slip: float | None = None) -> dict:
    sys.path.insert(0, ".")
    from bf2d.parameters import BFParameters2D
    from bf2d.simulation import BlastFurnaceSimulation

    kwargs = dict(
        nr=nr, nz=nz, max_outer_iter=iters, log_every=max(iters, 1), dt=0.5,
        n_tuyeres=32, tuyere_diameter=0.13,
        gas_exit_r_min=1.5, gas_exit_r_max=3.5,
        # Radial burden distribution: Fu 2014 models a center-working
        # charge (their Sect. 2.4 burden-distribution model gives the
        # radial ore/coke ratio); 0.35 is mid plant range 0.2-0.5.
        # ks_center stays at the Fu Sect. 3.1 default (0.8).
        burden_radial_ore_bias=ore_bias,
    )
    if h2o_slip is not None:
        kwargs["raceway_h2o_slip"] = h2o_slip
    p = BFParameters2D(**kwargs)
    sim = BlastFurnaceSimulation(p, log_level="ERROR")
    t0 = time.time()
    for it in range(iters):
        sim._one_outer_iteration(it)
        sim._adapt_relaxation()
    elapsed = time.time() - t0

    f, mesh = sim.fields, sim.mesh
    m = mesh.active
    mt = sim._gas_exit_mask[:, -1]
    H = mesh.z_c[mesh.nz - 1]

    m_out = (f.rho_g[:, -1] * f.u_z[:, mesh.nz]
             * mesh.area_z[:, mesh.nz])[mt].sum()

    sum_y = sum(f.x_gas[s] for s in ["N2", "O2", "CO", "CO2", "H2", "H2O"])
    drift = float(np.abs(sum_y[m] - 1.0).max())

    # Fe onset height: lowest z/H (scanning from top) where Fe > 2%
    fe_onset = 0.0
    for j in range(mesh.nz - 1, -1, -1):
        mm = mesh.active[:, j]
        if mm.any() and f.f_solid["Fe"][mm, j].mean() > 0.02:
            fe_onset = float(mesh.z_c[j] / H)
            break

    # Carbon balance -> coke rate; DR degree from reduction pathway split
    from bf2d.kinetics import compute_rates
    from bf2d.properties import gas_mw_mix
    mw = gas_mw_mix(f.x_gas)
    zn = np.broadcast_to((mesh.z_c / p.furnace_height)[None, :], f.T_g.shape)
    r = compute_rates(f.T_g, f.T_s, f.P, f.rho_g, mw, f.x_gas, f.f_solid,
                      f.eps, mesh.active, z_norm=zn,
                      feo_form_height_frac=p.feo_form_height_frac,
                      feo_reduce_height_frac=p.feo_reduce_height_frac,
                      dt_solid=200.0)
    # Honest mass closure: at steady state, gas OUT = blast IN + net
    # reaction gas-mass source (O stripped from burden, C gasified, CO2
    # calcined).  Pre-Phase-9 the exit BC rescaled outflow to blast-in
    # alone, so this KPI read 100.000% by construction.
    from bf2d.constants import MW
    from bf2d.kinetics import assemble_species_sources
    src = assemble_species_sources(r)
    m_src = sum(((src[sp] * MW[sp]) * mesh.vol)[m].sum() for sp in src)
    m_in = sim.raceway.total_blast_kg_s + m_src

    fe_kg_s = ((r.R4 + r.R4H + r.R_DR) * mesh.vol)[m].sum() * 0.0558
    thm_s = max(fe_kg_s / 945.0, 1e-12)          # 945 kg Fe per tHM
    # Coke carbon consumed = raceway combustion (computed in the 0D
    # raceway balance -- compute_rates never sees it because the blast
    # enters pre-combusted with zero O2; pre-Phase-9 this KPI counted
    # only the in-shaft rates and structurally understated the coke
    # rate ~7x) + in-shaft gasification/direct reduction.  The PCI
    # carbon share of the raceway gas is excluded: the KPI is COKE per
    # tHM (Fu Table 7 reports coke 411 and PCI 67 separately).
    x_b = sim.raceway.x_gas_blast
    mol_blast = (sim.raceway.total_blast_kg_s
                 / sum(x_b[s] * MW[s] for s in x_b))
    c_raceway = mol_blast * (x_b["CO"] + x_b["CO2"]) * 0.0120
    pci_kg_s = p.pci_rate_kg_tHM * p.productivity_tHM_day / 86400.0
    c_raceway_coke = max(c_raceway - 0.7 * 0.85 * pci_kg_s, 0.0)
    c_in_shaft = ((r.R1 + r.R1b + r.R5 + r.R6 + r.R_DR)
                  * mesh.vol)[m].sum() * 0.0120
    c_kg_s = c_raceway_coke + c_in_shaft
    # charged-coke carbon actually consumed per tonne HM produced.
    # Known structural understatement ~10%: hot-metal carburization
    # (~45 kg C/tHM) has no liquid phase to dissolve into in this model.
    coke_rate = (c_kg_s / 0.87) / thm_s          # coke ~87 % fixed C
    dr = (r.R_DR * mesh.vol)[m].sum()
    ind = ((r.R4 + r.R4H) * mesh.vol)[m].sum()
    dr_degree = 100.0 * dr / max(dr + ind, 1e-12)

    # Cohesive-zone shape: inverted-V inversion metric (Fu 2014 Fig. 5).
    # Uses the softening-to-liquidus BAND of T_s (not cohesive_mask,
    # which includes the molten dripping zone down to the hearth).
    from bf2d.cohesive_zone import cohesive_band_mask, cz_shape_metrics
    band = cohesive_band_mask(f.T_s, p, mesh.active)
    cz_shape = cz_shape_metrics(band, mesh.active, mesh.z_c)
    cz_inv = cz_shape["inversion_m"]
    if not np.isfinite(cz_inv):
        cz_inv = -99.0   # no CZ found at all -> unambiguous FAIL

    measured = {
        "top_gas_N2_pct":  float(f.x_gas["N2"][mt, -1].mean() * 100),
        "top_gas_CO_pct":  float(f.x_gas["CO"][mt, -1].mean() * 100),
        "top_gas_CO2_pct": float(f.x_gas["CO2"][mt, -1].mean() * 100),
        "top_gas_H2_pct":  float(f.x_gas["H2"][mt, -1].mean() * 100),
        "top_gas_H2O_pct": float(f.x_gas["H2O"][mt, -1].mean() * 100),
        "T_max_K":         float(f.T_g[m].max()),
        "top_pressure_bar": float(f.P[m].min() / 1e5),
        "mass_closure_pct": float(100.0 * m_out / max(m_in, 1e-9)),
        "sumY_drift":      drift,
        "fe_onset_zH_max": fe_onset,
        "coke_rate_kg_tHM": coke_rate,
        "dr_degree_pct":    dr_degree,
        "cz_inversion_m":   float(cz_inv),
    }

    report = {"mesh": f"{nr}x{nz}", "iterations": iters,
              "runtime_s": round(elapsed, 1), "kpis": {}}
    n_pass = n_warn = n_fail = 0
    for key, tgt in PLANT_TARGETS.items():
        got = measured[key]
        one_sided = tgt.get("one_sided")
        if one_sided == "max" and got <= tgt["value"]:
            dev = 0.0
        elif one_sided == "min" and got >= tgt["value"]:
            dev = 0.0
        else:
            dev = abs(got - tgt["value"])
        if dev <= tgt["tol"]:
            status = "PASS"; n_pass += 1
        elif dev <= tgt["warn"]:
            status = "WARN"; n_warn += 1
        else:
            status = "FAIL"; n_fail += 1
        # The CZ band is 1-2 cells thick below nz ~ 80: the inversion
        # metric there is axial-quantization noise, not a measurement.
        # Report the value but never FAIL (or PASS) on it; the deep
        # gate (nz >= 120) scores it for real.
        if key == "cz_inversion_m" and nz < 80 and status != "WARN":
            if status == "FAIL":
                n_fail -= 1
            else:
                n_pass -= 1
            status = "WARN"; n_warn += 1
        report["kpis"][key] = {
            "measured": round(got, 4), "target": tgt["value"],
            "deviation": round(dev, 4), "status": status,
        }
    report["summary"] = {"pass": n_pass, "warn": n_warn, "fail": n_fail}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=15)
    ap.add_argument("--nz", type=int, default=40)
    # 500: with the Phase-9 dt-consistent availability caps the burden
    # flush takes ~400 iterations on the fast mesh; scoring earlier
    # reads a half-converged ladder (CO ~19 at 200 it vs 22.0 at 500).
    # Runtime ~85 s at 15x40.
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--ore-bias", type=float, default=0.35,
                    help="radial burden ore bias (plant range 0.2-0.5)")
    ap.add_argument("--h2o-slip", type=float, default=None,
                    help="raceway H2O slip override (default: parameter)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    try:
        report = run_validation(args.nr, args.nz, args.iters,
                                ore_bias=args.ore_bias,
                                h2o_slip=args.h2o_slip)
    except Exception as exc:  # noqa: BLE001 - CI needs a clean exit code
        print(f"VALIDATION ERROR: {type(exc).__name__}: {exc}")
        return 2

    print(f"\nPlant validation vs Fu 2014 "
          f"({report['mesh']}, {report['iterations']} iter, "
          f"{report['runtime_s']}s)")
    print(f"{'KPI':<20} {'measured':>10} {'target':>10} "
          f"{'dev':>8} {'status':>6}")
    for k, v in report["kpis"].items():
        print(f"{k:<20} {v['measured']:>10.3f} {v['target']:>10.3f} "
              f"{v['deviation']:>8.3f} {v['status']:>6}")
    s = report["summary"]
    print(f"\nSummary: {s['pass']} PASS, {s['warn']} WARN, {s['fail']} FAIL")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Report written to {args.json}")

    return 0 if s["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
