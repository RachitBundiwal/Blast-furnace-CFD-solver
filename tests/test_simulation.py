"""End-to-end integration tests and numeric-stability fuzz tests."""
from __future__ import annotations

import numpy as np
import pytest

from bf2d.parameters import BFParameters2D
from bf2d.simulation import BlastFurnaceSimulation
from bf2d.postprocess import summary_metrics


@pytest.fixture(scope="module")
def small_run():
    """Small, fast simulation used by multiple integration tests."""
    p = BFParameters2D(
        nr=10, nz=25,
        max_outer_iter=15,
        log_every=100,                  # silence in tests
        convergence_rms=1e-12,          # force full 15 iterations
    )
    sim = BlastFurnaceSimulation(p, log_level="ERROR")
    results = sim.run()
    return p, sim, results


class TestIntegrationSmallCase:
    def test_runs_without_exception(self, small_run) -> None:
        _, _, r = small_run
        assert r["T_g"].shape == (10, 25)

    def test_fields_remain_finite(self, small_run) -> None:
        _, _, r = small_run
        for key in ("T_g", "T_s", "P", "u_z", "u_r"):
            arr = r[key]
            assert np.all(np.isfinite(arr)), f"non-finite values in {key}"

    def test_temperature_within_physical_bounds(self, small_run) -> None:
        _, _, r = small_run
        active = r["active"]
        Tg = r["T_g"][active]
        Ts = r["T_s"][active]
        assert Tg.min() > 270 and Tg.max() < 2600
        assert Ts.min() > 270 and Ts.max() < 2300

    def test_pressure_within_physical_bounds(self, small_run) -> None:
        _, _, r = small_run
        active = r["active"]
        P = r["P"][active]
        assert P.min() > 4e4 and P.max() < 1e6

    def test_mole_fractions_sum_to_unity(self, small_run) -> None:
        _, _, r = small_run
        active = r["active"]
        total = sum(r["x_gas"][sp] for sp in r["x_gas"])
        assert np.allclose(total[active], 1.0, atol=1e-6)

    def test_mass_fractions_bounded(self, small_run) -> None:
        _, _, r = small_run
        for ph, field in r["f_solid"].items():
            assert field.min() >= -1e-9
            assert field.max() <= 1.0 + 1e-6

    def test_summary_metrics_populated(self, small_run) -> None:
        _, _, r = small_run
        m = summary_metrics(r)
        for key in ("T_max_g_C", "T_max_s_C", "Fe_mean_%",
                    "gas_utilization_%", "raceway_flame_T_C"):
            assert key in m
            assert np.isfinite(m[key])


# ---------------------------------------------------------------------------
# Fuzz / stability tests: perturb initial conditions; solver must not crash
# and must converge to finite fields.
# ---------------------------------------------------------------------------
class TestNumericStabilityFuzz:
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_random_pertubed_init_stable(self, seed: int) -> None:
        """Perturb initial gas composition and confirm the solver survives."""
        rng = np.random.default_rng(seed)
        p = BFParameters2D(nr=8, nz=20, max_outer_iter=6, log_every=100)
        sim = BlastFurnaceSimulation(p, log_level="ERROR")
        # Inject noise into initial composition
        for sp in sim.fields.x_gas:
            noise = rng.uniform(-0.02, 0.02, size=sim.fields.x_gas[sp].shape)
            sim.fields.x_gas[sp] = np.clip(
                sim.fields.x_gas[sp] + noise, 1e-6, 1.0)
        # Renormalise (solver will do this too, but we keep IC clean)
        total = sum(sim.fields.x_gas.values())
        for sp in sim.fields.x_gas:
            sim.fields.x_gas[sp] /= total

        results = sim.run()
        assert np.all(np.isfinite(results["T_g"]))
        assert np.all(np.isfinite(results["P"]))

    def test_extreme_o2_enrichment_survives(self) -> None:
        p = BFParameters2D(nr=8, nz=20, oxygen_enrichment=0.30,
                           max_outer_iter=5, log_every=100)
        sim = BlastFurnaceSimulation(p, log_level="ERROR")
        r = sim.run()
        assert np.all(np.isfinite(r["T_g"]))

    def test_very_cold_blast_survives(self) -> None:
        p = BFParameters2D(nr=8, nz=20, blast_temperature=500.0,
                           max_outer_iter=5, log_every=100)
        sim = BlastFurnaceSimulation(p, log_level="ERROR")
        r = sim.run()
        assert np.all(np.isfinite(r["T_g"]))
