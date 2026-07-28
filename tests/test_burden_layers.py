"""Unit + fuzz tests for the layered-burden module (Fu 2014 technique)."""
from __future__ import annotations

import numpy as np
import pytest

from bf2d.burden_layers import (
    EPS_COKE,
    EPS_ORE,
    EPS_ORE_CZ,
    layered_bed_properties,
    ore_fraction_field,
)
from bf2d.mesh import build_mesh
from bf2d.parameters import BFParameters2D


@pytest.fixture()
def fine_mesh():
    return build_mesh(BFParameters2D(nr=8, nz=120))


class TestOreFractionField:
    def test_bounded_and_finite(self, fine_mesh) -> None:
        f = ore_fraction_field(fine_mesh, 0.35, 0.45)
        assert np.isfinite(f).all()
        assert f.min() >= 0.0 and f.max() <= 1.0

    def test_volume_fraction_matches_layer_ratio(self, fine_mesh) -> None:
        # column-average ore fraction must equal L_ore / (L_ore + L_coke)
        f = ore_fraction_field(fine_mesh, 0.35, 0.45)
        dz = fine_mesh.dz
        avg = float((f[0, :] * dz).sum() / dz.sum())
        assert avg == pytest.approx(0.35 / 0.80, abs=0.02)

    def test_radially_uniform(self, fine_mesh) -> None:
        f = ore_fraction_field(fine_mesh, 0.35, 0.45)
        assert np.allclose(f, f[0:1, :])

    def test_under_resolved_mesh_rejected(self) -> None:
        coarse = build_mesh(BFParameters2D(nr=8, nz=20))
        with pytest.raises(ValueError, match="under-resolved"):
            ore_fraction_field(coarse, 0.05, 0.05)

    @pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
    def test_degenerate_thickness_rejected(self, fine_mesh, bad) -> None:
        with pytest.raises(ValueError):
            ore_fraction_field(fine_mesh, bad, 0.45)


class TestLayeredBedProperties:
    def test_pure_layers_get_pure_properties(self, fine_mesh) -> None:
        f = ore_fraction_field(fine_mesh, 0.35, 0.45)
        cz = np.zeros_like(f, dtype=bool)
        eps, dp = layered_bed_properties(f, cz, BFParameters2D())
        assert eps[f == 1.0].max() == pytest.approx(EPS_ORE)
        assert eps[f == 0.0].max() == pytest.approx(EPS_COKE)

    def test_cz_collapses_ore_only(self, fine_mesh) -> None:
        f = ore_fraction_field(fine_mesh, 0.35, 0.45)
        cz = np.ones_like(f, dtype=bool)
        eps, _ = layered_bed_properties(f, cz, BFParameters2D())
        # pure-ore cells collapse to CZ porosity; coke slits unchanged
        assert eps[f == 1.0].max() == pytest.approx(EPS_ORE_CZ)
        assert eps[f == 0.0].min() == pytest.approx(EPS_COKE)

    def test_shape_mismatch_rejected(self, fine_mesh) -> None:
        f = ore_fraction_field(fine_mesh, 0.35, 0.45)
        with pytest.raises(ValueError, match="shapes differ"):
            layered_bed_properties(f, np.zeros((2, 2), dtype=bool),
                                   BFParameters2D())


class TestFluxCalcination:
    def test_r9_produces_co2_above_onset(self) -> None:
        import bf2d.kinetics as k
        shape = (8, 20)
        hot, cold = 1200.0, 800.0
        for T, expect_active in ((hot, True), (cold, False)):
            f_solid = {ph: np.full(shape, 0.1) for ph in
                       ("Fe2O3", "Fe3O4", "FeO", "Fe", "C_coke",
                        "gangue", "flux")}
            x = {s: np.full(shape, 1 / 6) for s in
                 ("N2", "O2", "CO", "CO2", "H2", "H2O")}
            r = k.compute_rates(
                np.full(shape, T), np.full(shape, T),
                np.full(shape, 2.0e5), np.full(shape, 1.0),
                np.full(shape, 0.030), x, f_solid,
                np.full(shape, 0.42), np.ones(shape, dtype=bool))
            if expect_active:
                assert r.R9.max() > 0.0
            else:
                assert r.R9.max() == 0.0

    def test_injectant_bounds_enforced(self) -> None:
        from bf2d.parameters import ParameterValidationError
        with pytest.raises(ParameterValidationError):
            BFParameters2D(blast_h2_vol_frac=0.31)
        with pytest.raises(ParameterValidationError):
            BFParameters2D(blast_h2_vol_frac=0.2, blast_ng_vol_frac=0.2)

    def test_h2_injection_raises_bosh_h2(self) -> None:
        from bf2d.raceway import build_raceway
        p0 = BFParameters2D()
        p1 = BFParameters2D(blast_h2_vol_frac=0.10)
        g0 = build_raceway(build_mesh(p0), p0).x_gas_blast
        g1 = build_raceway(build_mesh(p1), p1).x_gas_blast
        assert g1["H2"] > g0["H2"]
        assert sum(g1.values()) == pytest.approx(1.0, abs=1e-6)
