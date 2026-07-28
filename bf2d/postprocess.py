"""
Postprocessing utilities: 2D contour plots and summary metrics.

All plotting is done through matplotlib; no interactive GUI backend is
required.  Saves all figures to ``output_dir/plots/``.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

from .parameters import BFParameters2D


# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _masked(arr: np.ndarray, active: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~active, arr)


# ---------------------------------------------------------------------------
def plot_all(results: dict, params: BFParameters2D,
             output_dir: str | None = None) -> str:
    """Generate the full plot set and return the directory where they live."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting")

    out = output_dir or os.path.join(params.output_dir, "plots")
    _ensure_dir(out)

    r = results["r_c"]
    z = results["z_c"]
    active = results["active"]
    R, Z = np.meshgrid(r, z, indexing="ij")

    # Temperature profiles
    _plot_field(R, Z, _masked(results["T_g"] - 273.15, active),
                "Gas temperature [°C]", "gas_temperature", out, cmap="inferno")
    _plot_field(R, Z, _masked(results["T_s"] - 273.15, active),
                "Solid temperature [°C]", "solid_temperature", out, cmap="inferno")

    # Pressure
    _plot_field(R, Z, _masked(results["P"] / 1e5, active),
                "Pressure [bar]", "pressure", out, cmap="viridis")

    # Axial velocity (interpolated to centres)
    u_z_c = 0.5 * (results["u_z"][:, :-1] + results["u_z"][:, 1:])
    _plot_field(R, Z, _masked(u_z_c, active),
                "Gas axial velocity [m/s]", "u_z", out, cmap="coolwarm",
                symmetric=True)

    # Gas composition
    for sp in ("CO", "CO2", "H2", "O2"):
        _plot_field(R, Z, _masked(results["x_gas"][sp] * 100, active),
                    f"{sp} mole fraction [%]", f"xgas_{sp}", out, cmap="plasma")

    # Reduction progress
    for ph, label in [("Fe2O3", "Hematite"), ("Fe3O4", "Magnetite"),
                      ("FeO", "Wustite"), ("Fe", "Metallic iron")]:
        _plot_field(R, Z, _masked(results["f_solid"][ph] * 100, active),
                    f"{label} mass fraction [%]", f"fsol_{ph}", out, cmap="cividis")

    # Cohesive zone
    _plot_field(R, Z, _masked(results["cohesive_mask"].astype(float), active),
                "Cohesive zone (1 = softened)", "cohesive_zone", out,
                cmap="Reds", vmin=0, vmax=1)

    # Porosity
    _plot_field(R, Z, _masked(results["eps"], active),
                "Effective porosity [-]", "porosity", out, cmap="Blues_r")

    # Composite: Temperature + CZ overlay + raceway
    _plot_overlay(R, Z, results, active, params, out)

    # Convergence history
    _plot_history(results.get("history", []), out)

    return out


def _plot_field(R: np.ndarray, Z: np.ndarray, field: np.ma.MaskedArray,
                title: str, name: str, out: str, *,
                cmap: str = "viridis", symmetric: bool = False,
                vmin: float | None = None, vmax: float | None = None,
                ) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 9.5))
    if symmetric:
        vabs = np.nanmax(np.abs(field))
        vabs = max(float(vabs), 1e-3)
        vmin, vmax = -vabs, vabs
    pcm = ax.pcolormesh(R, Z, field, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax)
    # Mirror for visual axisymmetry
    ax.pcolormesh(-R, Z, field, cmap=cmap, shading="auto",
                  vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, shrink=0.85, label=title)
    ax.set_aspect("equal")
    ax.set_xlabel("Radius [m]")
    ax.set_ylabel("Height above tuyere [m]")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(R: np.ndarray, Z: np.ndarray, results: dict,
                  active: np.ndarray, params: BFParameters2D, out: str) -> None:
    """Composite plot: temperature contours + CZ hatching + raceway marker."""
    fig, ax = plt.subplots(figsize=(7.5, 10))
    Tg = _masked(results["T_g"] - 273.15, active)
    pcm = ax.pcolormesh(R, Z, Tg, cmap="inferno", shading="auto")
    ax.pcolormesh(-R, Z, Tg, cmap="inferno", shading="auto")
    fig.colorbar(pcm, ax=ax, shrink=0.8, label="Gas T [°C]")

    # Overlay cohesive zone boundary
    ax.contour(R, Z, results["cohesive_mask"].astype(float),
               levels=[0.5], colors="cyan", linewidths=1.6)
    ax.contour(-R, Z, results["cohesive_mask"].astype(float),
               levels=[0.5], colors="cyan", linewidths=1.6)

    # Velocity streamlines (axial component only, sampled)
    u_z_c = 0.5 * (results["u_z"][:, :-1] + results["u_z"][:, 1:])
    u_r_c = 0.5 * (results["u_r"][:-1, :] + results["u_r"][1:, :])
    step = max(1, R.shape[0] // 10)
    ax.quiver(R[::step, ::step], Z[::step, ::step],
              u_r_c[::step, ::step], u_z_c[::step, ::step],
              color="white", alpha=0.55, scale=80)

    ax.set_xlabel("Radius [m]")
    ax.set_ylabel("Height above tuyere [m]")
    ax.set_title("Composite: T_g contours + CZ boundary + gas flow",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "composite.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_history(history: list, out: str) -> None:
    """Plot per-equation residual trajectories.

    Phase 1 stores per-equation L2 residuals in each history entry:
    {it, worst, continuity, r_momentum, z_momentum, gas_energy,
     solid_energy, species_max, solid_max, ...}.
    Older Phase 0 history had {it, residual, T_max, T_top, Fe}; this
    function falls back gracefully when only the legacy keys are present.
    """
    if not history:
        return

    its = [h.get("it", i) for i, h in enumerate(history)]
    has_phase1 = "worst" in history[0]

    if has_phase1:
        worst = [h.get("worst", float("nan")) for h in history]
        cont  = [h.get("continuity", float("nan")) for h in history]
        rmom  = [h.get("r_momentum", float("nan")) for h in history]
        zmom  = [h.get("z_momentum", float("nan")) for h in history]
        Tg    = [h.get("gas_energy", float("nan")) for h in history]
        Ts    = [h.get("solid_energy", float("nan")) for h in history]
        spec  = [h.get("species_max", float("nan")) for h in history]
        solid = [h.get("solid_max", float("nan")) for h in history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(its, worst, "k-", lw=2, label="worst")
        ax.semilogy(its, cont,  ":",  label="continuity")
        ax.semilogy(its, rmom,  "--", label="r-momentum")
        ax.semilogy(its, zmom,  "--", label="z-momentum")
        ax.semilogy(its, Tg,    "-",  label="gas energy", alpha=0.7)
        ax.semilogy(its, Ts,    "-",  label="solid energy", alpha=0.7)
        ax.semilogy(its, spec,  "-.", label="species (max)", alpha=0.7)
        ax.semilogy(its, solid, "-.", label="solid (max)", alpha=0.7)
        ax.set_xlabel("Outer iteration")
        ax.set_ylabel("Residual (L2, normalised)")
        ax.set_title("Per-equation convergence history")
        ax.legend(loc="best", fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "history.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
        return

    # Phase 0 fallback (legacy schema)
    res = [h.get("residual", float("nan")) for h in history]
    tmax = [h.get("T_max", 0.0) - 273.15 for h in history]
    Fe = [h.get("Fe", 0.0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].semilogy(its, res, "k-")
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Convergence")
    axes[0].grid(alpha=0.3)

    axes[1].plot(its, tmax, "r-")
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel("Max T_g [°C]")
    axes[1].set_title("Peak gas temperature")
    axes[1].grid(alpha=0.3)

    axes[2].plot(its, Fe, "b-")
    axes[2].set_xlabel("Outer iteration")
    axes[2].set_ylabel("Mean Fe mass fraction [%]")
    axes[2].set_title("Iron production progress")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "history.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def summary_metrics(results: dict) -> dict[str, float]:
    """Compute integrated performance metrics."""
    active = results["active"]
    tg = results["T_g"][active]
    ts = results["T_s"][active]
    x_co = results["x_gas"]["CO"][:, -1][active[:, -1]]
    x_co2 = results["x_gas"]["CO2"][:, -1][active[:, -1]]
    f_Fe = results["f_solid"]["Fe"][active]
    gas_util = float(np.mean(x_co2) / max(np.mean(x_co) + np.mean(x_co2), 1e-9) * 100)
    return {
        "T_max_g_C": float(np.max(tg) - 273.15),
        "T_max_s_C": float(np.max(ts) - 273.15),
        "T_top_g_C_mean": float(np.mean(results["T_g"][:, -1][active[:, -1]] - 273.15))
            if active[:, -1].any() else float("nan"),
        "Fe_mean_%": float(np.mean(f_Fe) * 100),
        "Fe_bottom_%": float(np.mean(results["f_solid"]["Fe"][:, 0][active[:, 0]]) * 100)
            if active[:, 0].any() else 0.0,
        "gas_utilization_%": gas_util,
        "cohesive_zone_cells": int(results["cohesive_mask"].sum()),
        "pressure_drop_bar": float(
            (np.mean(results["P"][:, 0][active[:, 0]]) -
             np.mean(results["P"][:, -1][active[:, -1]])) / 1e5
        ) if active[:, 0].any() and active[:, -1].any() else 0.0,
        "raceway_flame_T_C": float(results["raceway_T_flame"] - 273.15),
    }
