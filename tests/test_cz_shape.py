"""Tests for the inverted-V cohesive-zone drivers (Task 1).

Covers:
* radial ore-fraction weight w(r)  (Fu 2014 Sect. 2.4 center-working charge)
* mass-preserving application of w(r) to the layered ore-fraction map
* radial descent factor Ks(r)      (Fu 2014 Sect. 3.1)
* CZ shape metrics (inversion / coverage)
* parameter validation of the new fields
* slow integration test: inverted V actually forms with the bias on
"""
from __future__ import annotations

import numpy as np
import pytest

from bf2d.burden_layers import (
    apply_radial_ore_bias,
    ore_fraction_field,
    radial_ore_weight,
)
from bf2d.cohesive_zone import cohesive_band_mask, cz_shape_metrics
from bf2d.mesh import build_mesh
from bf2d.parameters import BFParameters2D, ParameterValidationError
from bf2d.solver_species import compute_solid_descent_velocity


@pytest.fixture()
def fine_mesh():
    return build_mesh(BFParameters2D(nr=12, nz=120))


def _ring_area(mesh):
    return np.pi * (mesh.r_f[1:] ** 2 - mesh.r_f[:-1] ** 2)


# ---------------------------------------------------------------------------
class TestRadialOreWeight:
    def test_zero_bias_is_uniform(self, fine_mesh) -> None:
        p = BFParameters2D(nr=12, nz=120, burden_radial_ore_bias=0.0)
        w = radial_ore_weight(fine_mesh, p)
        assert np.all(w == 1.0)

    def test_area_weighted_row_mean_is_one(self, fine_mesh) -> None:
        p = BFParameters2D(nr=12, nz=120, burden_radial_ore_bias=0.35)
        w = radial_ore_weight(fine_mesh, p)
        area = _ring_area(fine_mesh)
        for j in range(fine_mesh.nz):
            act = fine_mesh.active[:, j]
            if not act.any():
                continue
            mean = (w[act, j] * area[act]).sum() / area[act].sum()
            assert mean == pytest.approx(1.0, abs=1e-10)

    def test_more_ore_at_wall_than_center(self, fine_mesh) -> None:
        p = BFParameters2D(nr=12, nz=120, burden_radial_ore_bias=0.35)
        w = radial_ore_weight(fine_mesh, p)
        for j in (10, 60, 110):
            act = np.where(fine_mesh.active[:, j])[0]
            assert w[act[-1], j] > w[act[0], j]

    def test_bounded_and_finite(self, fine_mesh) -> None:
        p = BFParameters2D(nr=12, nz=120, burden_radial_ore_bias=0.6)
        w = radial_ore_weight(fine_mesh, p)
        assert np.isfinite(w).all()
        assert w.min() >= 0.0


# ---------------------------------------------------------------------------
class TestApplyRadialOreBias:
    @pytest.mark.parametrize("bias", [0.2, 0.35, 0.5])
    def test_column_average_preserved(self, fine_mesh, bias) -> None:
        # Handoff acceptance: area-weighted ore fraction must equal
        # L_ore / (L_ore + L_coke) within 1% after the radial bias.
        p = BFParameters2D(nr=12, nz=120, burden_radial_ore_bias=bias)
        frac = ore_fraction_field(fine_mesh, 0.35, 0.45)
        w = radial_ore_weight(fine_mesh, p)
        biased = apply_radial_ore_bias(frac, w, fine_mesh)
        act = fine_mesh.active
        vol = fine_mesh.vol
        avg = float((biased[act] * vol[act]).sum() / vol[act].sum())
        assert avg == pytest.approx(0.35 / 0.80, rel=0.01)
        assert biased.min() >= 0.0 and biased.max() <= 1.0

    def test_wall_gets_more_ore(self, fine_mesh) -> None:
        p = BFParameters2D(nr=12, nz=120, burden_radial_ore_bias=0.35)
        frac = ore_fraction_field(fine_mesh, 0.35, 0.45)
        w = radial_ore_weight(fine_mesh, p)
        biased = apply_radial_ore_bias(frac, w, fine_mesh)
        act = fine_mesh.active
        vol = fine_mesh.vol
        # radially-resolved column averages: wall column > center column
        col = np.array([
            (biased[i, act[i]] * vol[i, act[i]]).sum()
            / max(vol[i, act[i]].sum(), 1e-12)
            for i in range(fine_mesh.nr)])
        assert col[-1] > col[0]

    def test_shape_mismatch_rejected(self, fine_mesh) -> None:
        frac = ore_fraction_field(fine_mesh, 0.35, 0.45)
        with pytest.raises(ValueError, match="shapes differ"):
            apply_radial_ore_bias(frac, np.ones((2, 2)), fine_mesh)


# ---------------------------------------------------------------------------
class TestKsDescentFactor:
    def _mass_flow_per_plane(self, mesh, u_s, rho) -> np.ndarray:
        area = _ring_area(mesh)
        flows = np.zeros(mesh.nz)
        for j in range(mesh.nz):
            act = mesh.active[:, j]
            flows[j] = (rho[act, j] * u_s[act, j] * area[act]).sum()
        return flows

    def test_mass_flow_conserved_vs_plug(self) -> None:
        # Handoff acceptance: total mass flow through any z-plane must be
        # unchanged (to 0.1%) by the Ks radial profile.
        p_plug = BFParameters2D(nr=12, nz=40, ks_center=1.0)
        p_ks = BFParameters2D(nr=12, nz=40, ks_center=0.8)
        mesh = build_mesh(p_plug)
        rho = np.full((12, 40), 1050.0)
        u_plug = compute_solid_descent_velocity(mesh, p_plug, rho)
        u_ks = compute_solid_descent_velocity(mesh, p_ks, rho)
        f_plug = self._mass_flow_per_plane(mesh, u_plug, rho)
        f_ks = self._mass_flow_per_plane(mesh, u_ks, rho)
        np.testing.assert_allclose(f_ks, f_plug, rtol=1e-3)

    def test_mass_flow_conserved_nonuniform_density(self) -> None:
        p_ks = BFParameters2D(nr=12, nz=40, ks_center=0.8)
        mesh = build_mesh(p_ks)
        rng = np.random.default_rng(42)
        rho = 900.0 + 400.0 * rng.random((12, 40))
        u_ks = compute_solid_descent_velocity(mesh, p_ks, rho)
        flows = self._mass_flow_per_plane(mesh, u_ks, rho)
        mdot = p_ks.coke_rate_kg_s + p_ks.ore_rate_kg_s + p_ks.flux_rate_kg_s
        np.testing.assert_allclose(flows, mdot, rtol=1e-3)

    def test_wall_descends_faster_than_center(self) -> None:
        p = BFParameters2D(nr=12, nz=40, ks_center=0.8)
        mesh = build_mesh(p)
        rho = np.full((12, 40), 1050.0)
        u = compute_solid_descent_velocity(mesh, p, rho)
        for j in (5, 20, 35):
            act = np.where(mesh.active[:, j])[0]
            assert u[act[-1], j] > u[act[0], j]

    def test_plug_flow_reproduced_at_ks_one(self) -> None:
        p = BFParameters2D(nr=12, nz=40, ks_center=1.0)
        mesh = build_mesh(p)
        rho = np.full((12, 40), 1050.0)
        u = compute_solid_descent_velocity(mesh, p, rho)
        for j in (5, 20, 35):
            act = mesh.active[:, j]
            assert np.ptp(u[act, j]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
class TestCzShapeMetrics:
    def test_synthetic_inverted_v(self) -> None:
        nr, nz = 5, 20
        active = np.ones((nr, nz), dtype=bool)
        z_c = np.linspace(0.5, 19.5, nz)
        mask = np.zeros((nr, nz), dtype=bool)
        # band mid falls from j=14..15 at axis to j=6..7 at wall
        for i, j0 in enumerate([14, 12, 10, 8, 6]):
            mask[i, j0:j0 + 2] = True
        out = cz_shape_metrics(mask, active, z_c)
        assert out["inversion_m"] == pytest.approx(8.0)
        assert out["coverage"] == 1.0
        assert out["connected"] is True

    def test_detached_blob_flagged_not_connected(self) -> None:
        nr, nz = 5, 20
        active = np.ones((nr, nz), dtype=bool)
        z_c = np.linspace(0.5, 19.5, nz)
        mask = np.zeros((nr, nz), dtype=bool)
        mask[0, 10:12] = True   # band at the axis ...
        mask[3, 6:8] = True     # ... detached blob at i=3, hole at i=1-2
        out = cz_shape_metrics(mask, active, z_c)
        assert out["connected"] is False

    def test_no_cz_returns_nan(self) -> None:
        active = np.ones((4, 10), dtype=bool)
        mask = np.zeros((4, 10), dtype=bool)
        out = cz_shape_metrics(mask, active, np.arange(10.0))
        assert np.isnan(out["inversion_m"])
        assert out["coverage"] == 0.0

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError):
            cz_shape_metrics(np.zeros((4, 10), dtype=bool),
                             np.ones((3, 10), dtype=bool),
                             np.arange(10.0))
        with pytest.raises(ValueError):
            cz_shape_metrics(np.zeros((4, 10), dtype=bool),
                             np.ones((4, 10), dtype=bool),
                             np.arange(9.0))

    def test_band_mask_excludes_molten_and_cold(self) -> None:
        p = BFParameters2D()
        active = np.ones((1, 4), dtype=bool)
        T_s = np.array([[1000.0,   # below softening -> out
                         1500.0,   # inside band -> in
                         1700.0,   # above liquidus (molten) -> out
                         1400.0]])  # inside band -> in
        band = cohesive_band_mask(T_s, p, active)
        assert band.tolist() == [[False, True, False, True]]


# ---------------------------------------------------------------------------
class TestParameterValidation:
    @pytest.mark.parametrize("bad", [-0.1, 0.7, float("nan")])
    def test_ore_bias_bounds(self, bad) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(burden_radial_ore_bias=bad)

    @pytest.mark.parametrize("bad", [0.3, 6.0])
    def test_ore_exponent_bounds(self, bad) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(burden_radial_ore_exponent=bad)

    @pytest.mark.parametrize("bad", [0.4, 1.1])
    def test_ks_center_bounds(self, bad) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(ks_center=bad)

    def test_defaults_valid(self) -> None:
        p = BFParameters2D()
        assert p.burden_radial_ore_bias == 0.0
        assert p.ks_center == 0.8


# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestInvertedVIntegration:
    def test_inverted_v_forms_with_radial_bias(self) -> None:
        """Handoff acceptance: inversion > 0 on 15x80 / 200 iterations
        with burden_radial_ore_bias = 0.35."""
        from bf2d.simulation import BlastFurnaceSimulation

        p = BFParameters2D(
            nr=15, nz=80, max_outer_iter=200, log_every=200, dt=0.5,
            n_tuyeres=32, tuyere_diameter=0.13,
            gas_exit_r_min=1.5, gas_exit_r_max=3.5,
            burden_radial_ore_bias=0.35,
        )
        sim = BlastFurnaceSimulation(p, log_level="ERROR")
        for it in range(200):
            sim._one_outer_iteration(it)
            sim._adapt_relaxation()
        band = cohesive_band_mask(sim.fields.T_s, p, sim.mesh.active)
        out = cz_shape_metrics(band, sim.mesh.active, sim.mesh.z_c)
        assert out["inversion_m"] > 0.0
        # Connected inverted-V band (no doughnut holes / detached blobs).
        # Coverage below 1.0 is expected: the outermost radii exist only
        # in the wall-adjacent bosh corner, which stave cooling keeps
        # below the softening temperature (the plant "skull") — so they
        # hold no CZ band.  75% still requires the band to span from the
        # axis well into the bosh.
        assert out["connected"] is True
        assert out["coverage"] >= 0.75
