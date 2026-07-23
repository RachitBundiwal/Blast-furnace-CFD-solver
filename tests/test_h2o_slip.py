"""Tests for the raceway H2O-slip speciation (Task 2, Phase 9).

The 0D raceway balance previously converted ALL blast moisture to
CO + H2 (R6) inside the raceway.  `raceway_h2o_slip` routes a fraction
of that moisture to the bosh gas as unreacted H2O vapour so the shaft's
in-situ R6 generates H2 where it survives (Zhou 2016).  Element
balances must be exact for any slip value.
"""
from __future__ import annotations

import pytest

from bf2d.parameters import BFParameters2D, ParameterValidationError
from bf2d.raceway import _blast_composition as _bc


def _blast_composition(p):
    """Unpack helper: tests only need the mole-fraction dict."""
    x, _n_exit = _bc(p)
    return x


def _atom_ratios(x: dict) -> tuple[float, float]:
    """(H atoms, O atoms) per N atom of the composition dict."""
    n_atoms = 2.0 * x["N2"]
    h_atoms = 2.0 * x["H2"] + 2.0 * x["H2O"]
    o_atoms = (x["CO"] + 2.0 * x["CO2"] + x["H2O"] + 2.0 * x["O2"])
    return h_atoms / n_atoms, o_atoms / n_atoms


class TestRacewayH2OSlip:
    @pytest.mark.parametrize("slip", [0.0, 0.2, 0.35, 0.5, 1.0])
    def test_element_balance_invariant_vs_slip(self, slip) -> None:
        # Total H and O atoms relative to inert N2 must not depend on
        # the slip value (H2O carries its own 2 H + 1 O either way).
        ref = _blast_composition(BFParameters2D(raceway_h2o_slip=0.0))
        got = _blast_composition(BFParameters2D(raceway_h2o_slip=slip))
        h_ref, o_ref = _atom_ratios(ref)
        h_got, o_got = _atom_ratios(got)
        assert h_got == pytest.approx(h_ref, rel=1e-12)
        assert o_got == pytest.approx(o_ref, rel=1e-12)

    def test_composition_normalized(self) -> None:
        x = _blast_composition(BFParameters2D(raceway_h2o_slip=0.35))
        assert sum(x.values()) == pytest.approx(1.0, abs=1e-12)

    def test_slip_moves_h2_to_h2o(self) -> None:
        x0 = _blast_composition(BFParameters2D(raceway_h2o_slip=0.0))
        x5 = _blast_composition(BFParameters2D(raceway_h2o_slip=0.5))
        assert x5["H2O"] > x0["H2O"]
        assert x5["H2"] < x0["H2"]
        assert x5["CO"] < x0["CO"]        # less raceway R6 -> less CO too

    def test_zero_slip_has_dry_bosh_gas(self) -> None:
        x0 = _blast_composition(BFParameters2D(raceway_h2o_slip=0.0))
        assert x0["H2O"] == 0.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_bounds_enforced(self, bad) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(raceway_h2o_slip=bad)

    def test_coke_hydrogen_raises_bosh_h2(self) -> None:
        x0 = _blast_composition(BFParameters2D(coke_h_mass_frac=0.0))
        x5 = _blast_composition(BFParameters2D(coke_h_mass_frac=0.005))
        assert x5["H2"] > x0["H2"]
        assert sum(x5.values()) == pytest.approx(1.0, abs=1e-12)

    def test_coke_h_bounds_enforced(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(coke_h_mass_frac=0.03)
        with pytest.raises(ParameterValidationError):
            BFParameters2D(coke_h_mass_frac=-0.001)

    def test_raceway_exit_mass_basis(self) -> None:
        # The injected N2 molar flow must equal the dry blast air's N2
        # exactly (N2 is inert through the raceway).  The pre-Phase-10
        # mixed-basis mass flow (air moles x exit MW) undercounted the
        # exit stream ~25% and broke this identity.
        from bf2d.constants import MOL_VOL_NTP, MW
        from bf2d.mesh import build_mesh
        from bf2d.raceway import build_raceway
        p = BFParameters2D()
        rw = build_raceway(build_mesh(p), p)
        mw_mix = sum(rw.x_gas_blast[s] * MW[s] for s in rw.x_gas_blast)
        n_n2_injected = (rw.total_blast_kg_s / mw_mix
                         * rw.x_gas_blast["N2"])
        blast_Nm3_s = p.blast_rate_Nm3_min / 60.0
        n_n2_air = (blast_Nm3_s / MOL_VOL_NTP
                    * (1.0 - 0.21 - p.oxygen_enrichment))
        assert n_n2_injected == pytest.approx(n_n2_air, rel=1e-6)

    def test_flame_temperature_rises_with_slip(self) -> None:
        # Slipped H2O skips the endothermic raceway R6, so the raceway
        # exit temperature must not DROP with slip.
        from bf2d.mesh import build_mesh
        from bf2d.raceway import build_raceway
        p0 = BFParameters2D(raceway_h2o_slip=0.0)
        p5 = BFParameters2D(raceway_h2o_slip=0.5)
        t0 = build_raceway(build_mesh(p0), p0).T_blast
        t5 = build_raceway(build_mesh(p5), p5).T_blast
        assert t5 >= t0
