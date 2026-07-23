"""Unit tests for parameter validation."""
from __future__ import annotations

import pytest

from bf2d.parameters import BFParameters2D, ParameterValidationError


class TestParameterValidation:
    """Validation must reject physically impossible inputs."""

    def test_defaults_are_valid(self) -> None:
        p = BFParameters2D()
        assert p.nr > 0 and p.nz > 0

    def test_negative_blast_rate_rejected(self) -> None:
        with pytest.raises(ParameterValidationError, match="blast_rate"):
            BFParameters2D(blast_rate_Nm3_min=-100.0)

    def test_blast_temperature_bounds(self) -> None:
        with pytest.raises(ParameterValidationError, match="blast_temperature"):
            BFParameters2D(blast_temperature=50.0)   # far below physical
        with pytest.raises(ParameterValidationError, match="blast_temperature"):
            BFParameters2D(blast_temperature=5000.0)

    def test_void_fraction_bounds(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(void_fraction=0.0)
        with pytest.raises(ParameterValidationError):
            BFParameters2D(void_fraction=0.99)

    def test_mesh_too_coarse_rejected(self) -> None:
        with pytest.raises(ParameterValidationError, match="too coarse"):
            BFParameters2D(nr=4, nz=80)
        with pytest.raises(ParameterValidationError, match="too coarse"):
            BFParameters2D(nr=20, nz=5)

    def test_mesh_too_fine_rejected(self) -> None:
        # Guard against accidental resource exhaustion
        with pytest.raises(ParameterValidationError, match="too fine"):
            BFParameters2D(nr=1000, nz=3000)

    def test_geometry_consistency(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(bosh_radius=3.0, hearth_radius=5.0)

    def test_relaxation_bounds(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(momentum_relaxation=1.5)
        with pytest.raises(ParameterValidationError):
            BFParameters2D(energy_relaxation=0.0)

    def test_oxygen_enrichment_bounds(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(oxygen_enrichment=-0.1)
        with pytest.raises(ParameterValidationError):
            BFParameters2D(oxygen_enrichment=0.8)

    def test_feed_rates_positive(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(coke_rate_kg_s=0.0)
        with pytest.raises(ParameterValidationError):
            BFParameters2D(ore_rate_kg_s=-1.0)

    def test_top_pressure_bounds(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(top_pressure=1.0e3)       # below physical min
        with pytest.raises(ParameterValidationError):
            BFParameters2D(top_pressure=2.0e7)       # above physical max

    def test_dt_positive(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(dt=-0.1)

    def test_convergence_positive(self) -> None:
        with pytest.raises(ParameterValidationError):
            BFParameters2D(convergence_rms=0.0)

    def test_frozen_dataclass_is_immutable(self) -> None:
        p = BFParameters2D()
        with pytest.raises(Exception):
            p.nr = 100  # type: ignore[misc]

    def test_to_dict_includes_celsius(self) -> None:
        d = BFParameters2D().to_dict()
        assert "blast_temperature_C" in d
        # Default blast_temperature is 2273.15 K (2000°C), bosh gas T per Fu 2014
        assert d["blast_temperature_C"] == pytest.approx(2000.0, abs=1.0)


# ---------- Fuzz-like tests (lightweight) ------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nan_or_inf_rejected(bad: float) -> None:
    with pytest.raises((ParameterValidationError, ValueError)):
        BFParameters2D(blast_rate_Nm3_min=bad)
