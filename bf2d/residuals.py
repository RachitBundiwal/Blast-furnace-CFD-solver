"""
Per-equation L2 residual diagnostics.

Reports honest ``||A x - b||_2 / reference`` for every governing equation,
replacing the previous change-based ``\\Delta T`` convergence metric.

Reference scales are chosen so a residual of ``1.0`` corresponds to a
physically meaningful imbalance (one percent of the typical flux).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class ResidualSnapshot:
    """One outer-iteration's residuals, normalised and dimensionless."""
    iteration: int
    continuity: float = 0.0
    r_momentum: float = 0.0
    z_momentum: float = 0.0
    gas_energy: float = 0.0
    solid_energy: float = 0.0
    species_max: float = 0.0
    solid_max: float = 0.0
    # Per-species detail (optional)
    species_detail: Dict[str, float] = field(default_factory=dict)
    solid_detail: Dict[str, float] = field(default_factory=dict)

    def max(self) -> float:
        """Worst residual across all equations."""
        return max(
            self.continuity, self.r_momentum, self.z_momentum,
            self.gas_energy, self.solid_energy,
            self.species_max, self.solid_max,
        )

    def all_below(self, tol: float) -> bool:
        """True iff every equation's residual is below the tolerance."""
        return self.max() < tol

    def to_dict(self) -> dict:
        return {
            "it": self.iteration,
            "continuity": self.continuity,
            "r_momentum": self.r_momentum,
            "z_momentum": self.z_momentum,
            "gas_energy": self.gas_energy,
            "solid_energy": self.solid_energy,
            "species_max": self.species_max,
            "solid_max": self.solid_max,
            "worst": self.max(),
            "species_detail": dict(self.species_detail),
            "solid_detail": dict(self.solid_detail),
        }


# ---------------------------------------------------------------------------
def l2_normalised(residual_vec: np.ndarray, reference: float) -> float:
    """Compute ``||r||_2 / max(reference, 1e-30)`` safely."""
    if residual_vec.size == 0:
        return 0.0
    norm = float(np.linalg.norm(residual_vec))
    return norm / max(reference, 1.0e-30)


# ---------------------------------------------------------------------------
@dataclass
class ReferenceScales:
    """Physical reference quantities used to normalise residuals.

    All values computed from the parameter set at simulation start;
    stored on the simulation object and reused every outer iteration.
    """
    rho_ref: float        # kg/m^3   (tuyere gas density at blast T)
    u_ref: float          # m/s      (tuyere gas velocity)
    cp_ref: float         # J/(kg K)
    T_ref: float          # K
    P_ref: float          # Pa       (top pressure)
    L_ref: float          # m        (furnace height)
    V_total: float        # m^3      (sum of active cell volumes)
    rho_s_ref: float      # kg/m^3   (bulk solid density)

    def continuity_scale(self) -> float:
        """Expected magnitude of continuity imbalance [kg/s]."""
        return self.rho_ref * self.u_ref * self.V_total ** (2.0 / 3.0)

    def momentum_scale(self) -> float:
        """Expected magnitude of momentum RHS [kg m/s^2]."""
        return self.rho_ref * self.u_ref ** 2 * self.V_total ** (2.0 / 3.0)

    def gas_energy_scale(self) -> float:
        """Expected magnitude of gas enthalpy flux [W]."""
        return self.rho_ref * self.cp_ref * self.T_ref * \
               self.u_ref * self.V_total ** (2.0 / 3.0)

    def solid_energy_scale(self) -> float:
        """Expected magnitude of solid enthalpy flux [W]."""
        # Solid moves ~1000x slower than gas; scale accordingly
        return self.rho_s_ref * self.cp_ref * self.T_ref * \
               (self.u_ref * 1e-3) * self.V_total ** (2.0 / 3.0)

    def species_scale(self) -> float:
        """Expected magnitude of species mass flux [kg/s]."""
        return self.rho_ref * self.u_ref * self.V_total ** (2.0 / 3.0)

    def solid_species_scale(self) -> float:
        """Expected magnitude of solid phase mass flux [kg/s].

        Solid descent is ~ rho_s * u_s * A_cross where u_s ~ 2 mm/s. Build
        the reference from total burden mass-flow proxy (rho_s_ref * 2e-3
        * V^{2/3}) so a residual of 1.0 means the equation is missing
        roughly one full burden flux of imbalance -- i.e. completely broken.
        """
        return self.rho_s_ref * 2.0e-3 * self.V_total ** (2.0 / 3.0)


# ---------------------------------------------------------------------------
class ResidualHistory:
    """Ring-like container for residual trajectories + stall detection."""

    def __init__(self, stall_window: int = 5, stall_factor: float = 0.95) -> None:
        self._snaps: List[ResidualSnapshot] = []
        self._stall_window = stall_window
        self._stall_factor = stall_factor

    def add(self, snap: ResidualSnapshot) -> None:
        self._snaps.append(snap)

    def latest(self) -> Optional[ResidualSnapshot]:
        return self._snaps[-1] if self._snaps else None

    def is_stalled(self) -> bool:
        """True if worst residual hasn't dropped materially over the window."""
        if len(self._snaps) < self._stall_window + 1:
            return False
        recent = [s.max() for s in self._snaps[-self._stall_window:]]
        baseline = self._snaps[-self._stall_window - 1].max()
        if baseline < 1.0e-12:
            return False
        return min(recent) > self._stall_factor * baseline

    def is_diverging(self) -> bool:
        """True if residuals have grown for 3 consecutive iterations."""
        if len(self._snaps) < 4:
            return False
        last4 = [s.max() for s in self._snaps[-4:]]
        return (last4[1] > last4[0] and
                last4[2] > last4[1] and
                last4[3] > last4[2])

    def as_list(self) -> List[dict]:
        return [s.to_dict() for s in self._snaps]

    def __len__(self) -> int:
        return len(self._snaps)
