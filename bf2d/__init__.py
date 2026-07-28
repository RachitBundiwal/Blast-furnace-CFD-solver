"""bf2d - 2D axisymmetric blast-furnace solver (PISO, full physics)."""
from __future__ import annotations

__version__ = "1.0.0"

from .parameters import BFParameters2D, ParameterValidationError
from .postprocess import plot_all, summary_metrics
from .simulation import BlastFurnaceSimulation

__all__ = [
    "BFParameters2D",
    "ParameterValidationError",
    "BlastFurnaceSimulation",
    "plot_all",
    "summary_metrics",
]
