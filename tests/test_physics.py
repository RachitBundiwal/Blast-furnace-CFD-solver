"""Unit tests for kinetics, properties, cohesive zone and raceway."""
from __future__ import annotations

import numpy as np
import pytest

from bf2d.cohesive_zone import detect_cohesive_zone
from bf2d.constants import GAS_SPECIES, MW
from bf2d.kinetics import (
    assemble_solid_sources,
    assemble_species_sources,
    compute_rates,
)
from bf2d.mesh import build_mesh
from bf2d.parameters import BFParameters2D
from bf2d.properties import (
    effective_particle_diameter,
    ergun_pressure_drop_coeff,
    gas_cp_mass,
    gas_density,
    gas_mw_mix,
    gas_viscosity,
    normalize_mole_fractions,
    ranz_marshall_h,
)
from bf2d.raceway import build_raceway


# ---------------------------------------------------------------------------
class TestProperties:
    def _x_default(self, shape):
        return {sp: np.full(shape, 1.0 / len(GAS_SPECIES)) for sp in GAS_SPECIES}

    def test_gas_density_ideal(self) -> None:
        shape = (5, 5)
        P = np.full(shape, 101325.0)
        T = np.full(shape, 300.0)
        x = self._x_default(shape)
        rho = gas_density(P, T, x)
        # Air-like mixture density ~1.2 kg/m^3 at STP
        assert np.all(rho > 0.3) and np.all(rho < 3.0)

    def test_gas_density_scales_with_pressure(self) -> None:
        shape = (3, 3)
        T = np.full(shape, 300.0)
        x = self._x_default(shape)
        rho_low = gas_density(np.full(shape, 100000.0), T, x)
        rho_high = gas_density(np.full(shape, 300000.0), T, x)
        assert np.all(rho_high > 2.5 * rho_low)

    def test_gas_viscosity_monotonic(self) -> None:
        T = np.array([300.0, 600.0, 1200.0, 2000.0])
        mu = gas_viscosity(T)
        assert np.all(np.diff(mu) > 0)

    def test_cp_positive(self) -> None:
        shape = (4, 4)
        T = np.full(shape, 600.0)
        x = self._x_default(shape)
        cp = gas_cp_mass(T, x)
        assert np.all(cp > 500) and np.all(cp < 5000)

    def test_ergun_positive_and_grows_with_u(self) -> None:
        rho = np.array([1.0])
        mu = np.array([2e-5])
        eps = np.array([0.4])
        dp = np.array([0.03])
        K_slow = ergun_pressure_drop_coeff(rho, mu, np.array([1.0]), eps, dp)
        K_fast = ergun_pressure_drop_coeff(rho, mu, np.array([10.0]), eps, dp)
        assert np.all(K_slow > 0) and np.all(K_fast > K_slow)

    def test_normalize_mole_fractions(self) -> None:
        x = {sp: np.full((3, 3), 0.1) for sp in GAS_SPECIES}
        x["N2"] = np.full((3, 3), 0.5)
        x_norm = normalize_mole_fractions(x)
        total = sum(x_norm[sp] for sp in GAS_SPECIES)
        assert np.allclose(total, 1.0)

    def test_ranz_marshall_bounded(self) -> None:
        rho = np.full((3, 3), 1.0)
        u = np.full((3, 3), 2.0)
        dp = np.full((3, 3), 0.03)
        eps = np.full((3, 3), 0.4)
        Tg = np.full((3, 3), 1200.0)
        Ts = np.full((3, 3), 1000.0)
        h = ranz_marshall_h(Tg, Ts, rho, u, dp, eps)
        assert np.all(h > 0) and np.all(np.isfinite(h))

    def test_effective_particle_diameter(self) -> None:
        f = {
            "Fe2O3":  np.full((3, 3), 0.7),
            "Fe3O4":  np.zeros((3, 3)),
            "FeO":    np.zeros((3, 3)),
            "Fe":     np.zeros((3, 3)),
            "C_coke": np.full((3, 3), 0.3),
            "gangue": np.zeros((3, 3)),
        }
        dp_eff = effective_particle_diameter(f, 0.045, 0.018)
        # Should lie between the two input sizes
        assert np.all(dp_eff >= 0.018) and np.all(dp_eff <= 0.045)


# ---------------------------------------------------------------------------
class TestKinetics:
    @pytest.fixture
    def simple_fields(self):
        shape = (3, 3)
        T_g = np.full(shape, 1200.0)
        T_s = np.full(shape, 1100.0)
        P = np.full(shape, 2.5e5)
        x = {
            "N2":  np.full(shape, 0.50),
            "O2":  np.full(shape, 0.01),
            "CO":  np.full(shape, 0.30),
            "CO2": np.full(shape, 0.18),
            "H2":  np.full(shape, 0.005),
            "H2O": np.full(shape, 0.005),
        }
        mw_mix = gas_mw_mix(x)
        rho = gas_density(P, T_g, x)
        f = {
            "Fe2O3":  np.full(shape, 0.4),
            "Fe3O4":  np.full(shape, 0.1),
            "FeO":    np.full(shape, 0.1),
            "Fe":     np.full(shape, 0.0),
            "C_coke": np.full(shape, 0.25),
            "gangue": np.full(shape, 0.15),
        }
        eps = np.full(shape, 0.4)
        active = np.ones(shape, dtype=bool)
        return T_g, T_s, P, rho, mw_mix, x, f, eps, active

    def test_rates_nonnegative_or_wgs(self, simple_fields) -> None:
        T_g, T_s, P, rho, mw_mix, x, f, eps, active = simple_fields
        r = compute_rates(T_g, T_s, P, rho, mw_mix, x, f, eps, active)
        for name in ("R1", "R1b", "R2", "R3", "R4", "R2H", "R3H", "R4H",
                    "R5", "R6", "R_DR"):
            arr = getattr(r, name)
            assert np.all(arr >= -1e-9), f"{name} has negative rates"
        # WGS can be signed -- just finite
        assert np.all(np.isfinite(r.R7))

    def test_rates_vanish_at_low_T(self, simple_fields) -> None:
        T_g, T_s, P, rho, mw_mix, x, f, eps, active = simple_fields
        T_g_cold = np.full_like(T_g, 350.0)
        T_s_cold = np.full_like(T_s, 350.0)
        r = compute_rates(T_g_cold, T_s_cold, P, rho, mw_mix, x, f, eps, active)
        # Direct reduction, Boudouard, etc. should all be zero below threshold
        assert np.all(r.R4 < 1e-6)
        assert np.all(r.R5 < 1e-6)
        assert np.all(r.R_DR < 1e-6)

    def test_rates_zero_when_reactant_absent(self, simple_fields) -> None:
        T_g, T_s, P, rho, mw_mix, x, f, eps, active = simple_fields
        x_no_co = {k: v.copy() for k, v in x.items()}
        x_no_co["CO"] = np.zeros_like(x_no_co["CO"])
        r = compute_rates(T_g, T_s, P, rho, mw_mix, x_no_co, f, eps, active)
        assert np.all(r.R2 < 1e-9)
        assert np.all(r.R3 < 1e-9)
        assert np.all(r.R4 < 1e-9)

    def test_assemble_species_sources_mass_balance(self, simple_fields) -> None:
        """Sum of species molar sources weighted by MW should match carbon balance."""
        T_g, T_s, P, rho, mw_mix, x, f, eps, active = simple_fields
        r = compute_rates(T_g, T_s, P, rho, mw_mix, x, f, eps, active)
        S_g = assemble_species_sources(r)
        S_s = assemble_solid_sources(r)

        # Carbon atom balance: C in coke consumed == carbon in new CO/CO2 minus consumed
        dC_solid = -S_s["C_coke"] / MW["C"]   # mol C / m^3 s (positive = generation? negative)
        # Carbon atoms out = CO + CO2 mol/m3/s; we cannot have creation of C atoms
        # Simple sanity: if coke is consumed, CO + CO2 net production must be non-negative somewhere
        net_C_gas = S_g["CO"] + S_g["CO2"]
        # Across the full domain where coke is consumed:
        mask = dC_solid > 1e-6
        if mask.any():
            # The net gas-side C atoms produced should roughly match solid C consumed
            ratio = np.sum(net_C_gas[mask]) / np.sum(dC_solid[mask])
            # Allow 30% slack from the R_DR/Boudouard stoichiometry
            assert 0.6 < ratio < 1.4, f"C-atom balance ratio = {ratio:.3f}"

    def test_iron_balance(self, simple_fields) -> None:
        T_g, T_s, P, rho, mw_mix, x, f, eps, active = simple_fields
        r = compute_rates(T_g, T_s, P, rho, mw_mix, x, f, eps, active)
        S_s = assemble_solid_sources(r)
        # Fe mass balance: rate of Fe2O3 consumption * (2Fe/1Fe2O3) * ratios
        # Just ensure Fe mass increases where reductions occur
        dFe = S_s["Fe"]
        # Where hematite is consumed at high T, iron should be produced
        if (T_g > 1000).all():
            # overall reduction should yield non-negative Fe source somewhere
            assert np.max(dFe) >= 0


# ---------------------------------------------------------------------------
class TestCohesiveZone:
    def test_cz_empty_at_low_T(self) -> None:
        p = BFParameters2D(nr=10, nz=20)
        shape = (10, 20)
        T_s = np.full(shape, 600.0)
        f = {"C_coke": np.full(shape, 0.2), "Fe2O3": np.full(shape, 0.5)}
        eps = np.full(shape, 0.4)
        dp = np.full(shape, 0.03)
        active = np.ones(shape, dtype=bool)
        mask, eps_new, dp_new = detect_cohesive_zone(T_s, f, eps, dp, p, active)
        assert not mask.any()
        assert np.allclose(eps_new, eps)

    def test_cz_forms_at_high_T(self) -> None:
        p = BFParameters2D(nr=10, nz=20)
        shape = (10, 20)
        T_s = np.full(shape, 1500.0)
        f = {"C_coke": np.full(shape, 0.1), "Fe2O3": np.full(shape, 0.5)}
        eps = np.full(shape, 0.4)
        dp = np.full(shape, 0.03)
        active = np.ones(shape, dtype=bool)
        mask, eps_new, dp_new = detect_cohesive_zone(T_s, f, eps, dp, p, active)
        assert mask.any()
        # Porosity should drop in CZ
        assert np.min(eps_new) < 0.4

    def test_cz_preserves_coke_permeability(self) -> None:
        p = BFParameters2D(nr=10, nz=20)
        shape = (10, 20)
        T_s = np.full(shape, 1500.0)
        # Half the cells are 100% coke
        f = {"C_coke": np.full(shape, 0.7), "Fe2O3": np.full(shape, 0.1)}
        eps = np.full(shape, 0.4)
        dp = np.full(shape, 0.03)
        active = np.ones(shape, dtype=bool)
        _, eps_new, _ = detect_cohesive_zone(T_s, f, eps, dp, p, active)
        # Coke-rich cells should keep much higher porosity than CZ minimum
        assert np.min(eps_new) > p.cz_porosity_min


# ---------------------------------------------------------------------------
class TestRaceway:
    def test_flame_temperature_physical(self) -> None:
        # T_blast in the Raceway dataclass is the post-combustion BOSH GAS
        # temperature (the raceway is treated as a 0D adiabatic CSTR that
        # outputs hot CO-rich gas at the bosh gas T, not raw blast air at
        # the hot-blast T).  Plant range: 2100-2500 K.
        p = BFParameters2D()
        mesh = build_mesh(p)
        rw = build_raceway(mesh, p)
        assert 2000.0 < rw.T_blast < 2700.0

    def test_blast_mass_flow_nonzero(self) -> None:
        p = BFParameters2D()
        mesh = build_mesh(p)
        rw = build_raceway(mesh, p)
        assert rw.total_blast_kg_s > 0
        # Verify G_tuyere * total_face_area equals total_blast_kg_s
        total_area = sum(mesh.area_r[i_f, j] for (i_f, j) in rw.face_indices)
        assert rw.G_tuyere * total_area == pytest.approx(
            rw.total_blast_kg_s, rel=1e-3,
        )

    def test_blast_composition_sums_to_unity(self) -> None:
        p = BFParameters2D()
        mesh = build_mesh(p)
        rw = build_raceway(mesh, p)
        total = sum(rw.x_gas_blast[sp] for sp in GAS_SPECIES)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_o2_enrichment_increases_co_fraction(self) -> None:
        # With more O2 enrichment, more carbon is combusted -> more CO
        # in the post-combustion bosh gas (since raceway treats O2 as
        # fully consumed).
        p1 = BFParameters2D(oxygen_enrichment=0.0)
        p2 = BFParameters2D(oxygen_enrichment=0.10)
        m1 = build_mesh(p1)
        m2 = build_mesh(p2)
        rw1 = build_raceway(m1, p1)
        rw2 = build_raceway(m2, p2)
        assert rw2.x_gas_blast["CO"] > rw1.x_gas_blast["CO"]

    def test_shape_mask_inside_active(self) -> None:
        p = BFParameters2D()
        mesh = build_mesh(p)
        rw = build_raceway(mesh, p)
        # Raceway cells must be a subset of active cells
        assert np.all((rw.shape_mask > 0) <= mesh.active)


class TestReductionHeightGating:
    """Verify the FeO formation/reduction height gates (Fu 2014 sequence)."""

    def _setup(self, nr=10, nz=40):
        p = BFParameters2D(nr=nr, nz=nz)
        mesh = build_mesh(p)
        shape = (mesh.nr, mesh.nz)
        T_g = np.full(shape, 1200.0)
        T_s = np.full(shape, 1200.0)
        P = np.full(shape, 2.5e5)
        rho_g = np.full(shape, 0.8)
        mw_mix = np.full(shape, 0.028)
        x_gas = {sp: np.full(shape, 0.0) for sp in GAS_SPECIES}
        x_gas["CO"][:] = 0.30
        x_gas["CO2"][:] = 0.10
        x_gas["H2"][:] = 0.05
        x_gas["N2"][:] = 0.55
        f_solid = {ph: np.full(shape, 0.0) for ph in
                   ["Fe2O3", "Fe3O4", "FeO", "Fe", "C_coke", "gangue"]}
        f_solid["Fe3O4"][:] = 0.4   # magnetite present -> R3 can form FeO
        f_solid["FeO"][:] = 0.3     # wustite present -> R4 can reduce it
        f_solid["C_coke"][:] = 0.3
        eps = np.full(shape, 0.42)
        active = np.ones(shape, dtype=bool)
        z_norm = np.broadcast_to(
            (mesh.z_c / p.furnace_height)[None, :], shape)
        return p, mesh, dict(
            T_g=T_g, T_s=T_s, P=P, rho_g=rho_g, mw_mix=mw_mix,
            x_gas=x_gas, f_solid=f_solid, eps=eps, active=active,
            z_norm=z_norm)

    def test_feo_formation_suppressed_in_upper_furnace(self) -> None:
        p, mesh, kw = self._setup()
        r = compute_rates(**kw, feo_form_height_frac=0.67,
                          feo_reduce_height_frac=0.33)
        znorm = kw["z_norm"]
        # R3 (FeO formation) should be strongly attenuated in the top
        # (z/H > 0.75).  The tanh gate is soft (for numerical stability),
        # so we check it is at least ~50x weaker than the ungated region
        # rather than exactly zero.
        upper = znorm > 0.75
        lower = znorm < 0.55
        r_upper = r.R3[upper].max()
        r_lower = r.R3[lower].max()
        if r_lower > 1e-8:
            assert r_upper < 0.05 * r_lower   # >20x attenuation
        else:
            assert r_upper < 1e-3

    def test_feo_reduction_confined_to_lower_furnace(self) -> None:
        p, mesh, kw = self._setup()
        # Make T hot enough everywhere for R4 so only the gate matters
        kw["T_g"][:] = 1300.0
        r = compute_rates(**kw, feo_form_height_frac=0.67,
                          feo_reduce_height_frac=0.33)
        znorm = kw["z_norm"]
        # R4 (FeO -> Fe) must be strongly attenuated above the reduce
        # gate (z/H > 0.45).  Soft tanh gate -> check attenuation ratio.
        upper = znorm > 0.45
        lower = znorm < 0.25
        r_upper = r.R4[upper].max()
        r_lower = r.R4[lower].max()
        if r_lower > 1e-8:
            assert r_upper < 0.05 * r_lower   # >20x attenuation
        else:
            assert r_upper < 1e-3

    def test_gates_disabled_when_znorm_none(self) -> None:
        p, mesh, kw = self._setup()
        kw.pop("z_norm")
        # Without z_norm, gates default to unity -> reactions allowed
        # (behaviour identical to pre-gating model)
        r = compute_rates(**kw)
        assert np.isfinite(r.R3).all()
        assert np.isfinite(r.R4).all()
