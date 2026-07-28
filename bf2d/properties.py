"""
Thermophysical-property correlations for gas mixture and solid bed.

All vectorised over numpy arrays.  Inputs are clamped to physical ranges;
callers never need to guard against negatives.
"""
from __future__ import annotations

import numpy as np

from .constants import (
    EPS_MOLE,
    GAS_SPECIES,
    MW,
    NASA7_CP,
    R_GAS,
    SOLID_CP_COEFF,
    SOLID_PHASES,
    T_MAX_PHYSICAL,
    T_MIN_PHYSICAL,
)


# ---------------------------------------------------------------------------
# Gas mixture
# ---------------------------------------------------------------------------
def gas_mw_mix(x_gas: dict) -> np.ndarray:
    """Mean molecular weight of gas mixture [kg/mol] from mole fractions."""
    any_field = next(iter(x_gas.values()))
    mw = np.zeros_like(any_field, dtype=float)
    for sp in GAS_SPECIES:
        mw += x_gas[sp] * MW[sp]
    return np.maximum(mw, 1.0e-4)


def gas_density(P: np.ndarray, T: np.ndarray, x_gas: dict) -> np.ndarray:
    """Ideal-gas density [kg/m^3]."""
    T_safe = np.clip(T, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    return P * gas_mw_mix(x_gas) / (R_GAS * T_safe)


def gas_viscosity(T: np.ndarray) -> np.ndarray:
    """Sutherland-type viscosity [Pa s].  Good for BF gas (~90% N2+CO+CO2)."""
    T_safe = np.clip(T, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    return 1.716e-5 * (T_safe / 273.15) ** 0.67


def gas_thermal_conductivity(T: np.ndarray) -> np.ndarray:
    """Approximate thermal conductivity of BF gas [W/(m K)]."""
    T_safe = np.clip(T, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    return 0.0241 * (T_safe / 273.15) ** 0.80


def gas_cp_mass(T: np.ndarray, x_gas: dict) -> np.ndarray:
    """Mass-weighted specific heat [J/(kg K)] using NASA-7 polynomial.

    The NASA-7 coefficients in NASA7_CP are the LOW-temperature range
    (200-1000 K).  Extrapolating them to BF flame temperatures (2000-2600 K)
    can produce negative cp values, which corrupts downstream physics
    (e.g. negative effective heat-transfer coefficients in the NTU
    correction).

    To avoid this without the complexity of switching to high-T NASA
    polynomials, we clamp the per-species cp_mol to physical bounds.
    For diatomics (N2, O2, CO, H2): cp at high T → ~7/2 R = 29 J/(mol K).
    For triatomics (CO2, H2O): cp at high T → ~6 R = 50-60 J/(mol K).
    """
    T_safe = np.clip(T, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    cp_mol_mix = np.zeros_like(T_safe, dtype=float)

    # Per-species physical cp bounds [J/(mol K)]
    # Lower: 5/2 R = 20.8 (translational only, monatomic limit)
    # Upper: 8 R = 66.5 (well above any BF gas at flame T)
    CP_MIN = 20.8
    CP_MAX = 66.5

    for sp in GAS_SPECIES:
        if sp not in NASA7_CP:
            cp_mol_sp = 29.0 * np.ones_like(T_safe)  # fallback
        else:
            a1, a2, a3, a4, a5 = NASA7_CP[sp]
            cp_nd = a1 + a2 * T_safe + a3 * T_safe**2 + a4 * T_safe**3 + a5 * T_safe**4
            cp_mol_sp = cp_nd * R_GAS
            # Clamp to physical bounds (handles polynomial extrapolation
            # artifacts at flame temperatures > 2000 K)
            cp_mol_sp = np.clip(cp_mol_sp, CP_MIN, CP_MAX)
        cp_mol_mix += x_gas[sp] * cp_mol_sp
    return cp_mol_mix / gas_mw_mix(x_gas)


# ---------------------------------------------------------------------------
# Solid bed
# ---------------------------------------------------------------------------
def solid_cp_mass(T_s: np.ndarray, f_solid: dict) -> np.ndarray:
    """Mass-weighted solid specific heat [J/(kg K)]."""
    T_safe = np.clip(T_s, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    cp_mol_mix = np.zeros_like(T_safe, dtype=float)
    mw_mix = np.zeros_like(T_safe, dtype=float)
    for ph in SOLID_PHASES:
        if ph not in SOLID_CP_COEFF:
            continue
        a, b, c = SOLID_CP_COEFF[ph]
        cp_mol = a + b * T_safe + c / np.maximum(T_safe, 300.0) ** 2
        # convert mole basis to mass basis by weighting mass fraction / MW
        mw_ph = _phase_mw(ph)
        frac = f_solid.get(ph, np.zeros_like(T_safe))
        cp_mol_mix += frac / mw_ph * cp_mol
        mw_mix += frac                     # just tracks mass-fraction sum
    mw_mix = np.maximum(mw_mix, 1.0e-6)
    return cp_mol_mix / mw_mix * 1.0      # J/(kg K)


def _phase_mw(phase: str) -> float:
    """Molar mass [kg/mol] for solid phase keys used in the solver."""
    mapping = {
        "Fe2O3": MW["Fe2O3"], "Fe3O4": MW["Fe3O4"], "FeO": MW["FeO"],
        "Fe": MW["Fe"], "C_coke": MW["C"], "gangue": MW["SiO2"],
        "flux": MW["CaCO3"],
    }
    return mapping[phase]


def bed_thermal_conductivity(T_s: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Effective bed conductivity (Zehner-Schluender simplified) [W/(m K)]."""
    T_safe = np.clip(T_s, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    k_solid = 2.0 + 0.01 * (T_safe - 300.0)        # coke-ore mix
    k_gas = 0.0241 * (T_safe / 273.15) ** 0.8
    eps_safe = np.clip(eps, 0.02, 0.95)
    # Parallel model weighted by void
    return (1.0 - eps_safe) * k_solid + eps_safe * k_gas


# ---------------------------------------------------------------------------
# Packed-bed closures
# ---------------------------------------------------------------------------
def ergun_pressure_drop_coeff(rho: np.ndarray, mu: np.ndarray,
                              u_mag: np.ndarray, eps: np.ndarray,
                              dp_eff: np.ndarray) -> np.ndarray:
    """Ergun coefficient K such that dP/dL = -K * u (signed friction)."""
    eps_safe = np.clip(eps, 0.02, 0.95)
    dp_safe = np.maximum(dp_eff, 5.0e-3)
    term1 = 150.0 * mu * (1.0 - eps_safe) ** 2 / (eps_safe ** 3 * dp_safe ** 2)
    term2 = 1.75 * rho * np.abs(u_mag) * (1.0 - eps_safe) / (eps_safe ** 3 * dp_safe)
    return term1 + term2


def ranz_marshall_h(T_g: np.ndarray, T_s: np.ndarray,
                    rho: np.ndarray, u_mag: np.ndarray,
                    dp: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Volumetric gas-solid heat-transfer coefficient [W/(m^3 K)].

    Combines convective Ranz-Marshall correlation with bed radiation.

    NOTE: This returns the *full* volumetric h*a without NTU correction.
    Use ``ranz_marshall_h_ntu_corrected`` instead for cells where NTU
    can become large -- without correction, h*a values of ~10^5 W/(m^3 K)
    quench the gas to the solid temperature within a single cell, which
    breaks the thermal-reserve-zone physics of plant operation.
    """
    T_g_safe = np.clip(T_g, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    T_s_safe = np.clip(T_s, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    mu = gas_viscosity(T_g_safe)
    k_g = gas_thermal_conductivity(T_g_safe)
    dp_safe = np.maximum(dp, 5.0e-3)
    eps_safe = np.clip(eps, 0.02, 0.95)

    Re = rho * np.abs(u_mag) * dp_safe / np.maximum(mu, 1e-8)
    Re = np.clip(Re, 0.1, 1.0e5)
    Pr = 0.70
    Nu = 2.0 + 0.6 * Re ** 0.5 * Pr ** (1.0 / 3.0)
    h_conv = Nu * k_g / dp_safe

    # Radiation (Rosseland-type)
    emissivity = 0.85
    h_rad = 4.0 * 5.670374419e-8 * emissivity * \
            ((T_g_safe + T_s_safe) * 0.5) ** 3

    # Specific interfacial area a_sf = 6(1-eps)/dp
    a_sf = 6.0 * (1.0 - eps_safe) / dp_safe
    return (h_conv + h_rad) * a_sf


def ranz_marshall_h_ntu_corrected(
    T_g: np.ndarray, T_s: np.ndarray,
    rho: np.ndarray, u_mag: np.ndarray,
    dp: np.ndarray, eps: np.ndarray,
    cp_g: np.ndarray,
    dz: np.ndarray,
) -> np.ndarray:
    """NTU-corrected volumetric heat-transfer coefficient [W/(m^3 K)].

    Why this is needed
    ------------------
    The Ranz-Marshall correlation gives the *single-particle* heat-transfer
    coefficient.  Multiplied by the specific interfacial area
    a_sf = 6(1-ε)/dp, it gives the volumetric h*a in W/(m^3 K).

    For typical BF parameters (dp = 30mm, ε = 0.42, Re ~ 100), this
    full h*a is ~10^5 W/(m^3 K).  At u_z = 2 m/s and dz = 0.4 m, the
    NTU per cell is:

        NTU_cell = (h*a * V) / (m_dot * cp)
                 = (10^5 * 0.4 * A) / (1 * 2 * A * 1100)
                 = ~18

    With NTU >> 1, the gas equilibrates with the solid within a single
    cell, regardless of whether equilibration is physically reached.
    This produces:
    - T_g ≈ T_s everywhere (no thermal reserve zone)
    - Flame heat dissipates within first few cells above tuyere
    - Indirect-reduction reactions cut off above the cohesive zone
      because T_g drops below their T cutoffs (1073 K for R3)

    Plant data (Fu 2014 Fig. 9 and Zhou 2016) shows T_g remains at
    ~1100 K through most of the column ("thermal reserve zone")
    with T_g - T_s ≈ 0-50 K.  Per-cell NTU should be ~0.3-1.0.

    The NTU correction
    ------------------
    For a counter-flow heat exchanger element of length dz:

        Q = (1 - exp(-NTU)) * (m_dot * cp) * (T_in - T_wall)

    The effective h*a that achieves this same Q over the cell volume V:

        h_eff = (m_dot * cp / V) * (1 - exp(-NTU))

    Equivalently:

        h_eff = h_full * (1 - exp(-NTU)) / NTU

    As NTU → 0: h_eff → h_full (no correction, kinetics-limited)
    As NTU → ∞: h_eff → m_dot * cp / V (mass-flow-limited)

    The correction is mild for NTU < 1 and significant for NTU > 3.

    Parameters
    ----------
    cp_g : (nr, nz) gas specific heat [J/(kg K)]
    dz   : (nz,) axial cell heights [m]
    """
    T_g_safe = np.clip(T_g, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    T_s_safe = np.clip(T_s, T_MIN_PHYSICAL, T_MAX_PHYSICAL)
    mu = gas_viscosity(T_g_safe)
    k_g = gas_thermal_conductivity(T_g_safe)
    dp_safe = np.maximum(dp, 5.0e-3)
    eps_safe = np.clip(eps, 0.02, 0.95)

    # Single-particle Nusselt and h
    Re = rho * np.abs(u_mag) * dp_safe / np.maximum(mu, 1e-8)
    Re = np.clip(Re, 0.1, 1.0e5)
    Pr = 0.70
    Nu = 2.0 + 0.6 * Re ** 0.5 * Pr ** (1.0 / 3.0)
    h_conv = Nu * k_g / dp_safe

    # Radiation (Rosseland-type)
    emissivity = 0.85
    h_rad = 4.0 * 5.670374419e-8 * emissivity * \
            ((T_g_safe + T_s_safe) * 0.5) ** 3

    a_sf = 6.0 * (1.0 - eps_safe) / dp_safe
    h_full = (h_conv + h_rad) * a_sf   # [W/(m^3 K)]

    # NTU correction.  Compute per-cell NTU using the local cell height
    # and gas mass flux.  At low Re (deadman, stagnation), the correction
    # is mild; at high Re (raceway, bulk shaft), it limits h_eff to
    # m_dot * cp / V so that gas cannot equilibrate within a cell.
    nr, nz = T_g.shape
    dz_2d = np.broadcast_to(dz[None, :], (nr, nz))

    # Mass flux per unit area: G = rho * u_z (taking the magnitude as
    # representative of advective transport rate)
    G = rho * np.abs(u_mag)             # kg/(m^2 s)
    cp_safe = np.maximum(cp_g, 100.0)
    # NTU per cell: NTU = h * dz / (G * cp)
    # (h*a*V)/(m_dot*cp) where m_dot = G*A and V = A*dz simplifies to
    # h*dz / (G*cp).
    G_safe = np.maximum(G, 1.0e-3)
    NTU = h_full * dz_2d / (G_safe * cp_safe)
    NTU_safe = np.clip(NTU, 1.0e-3, 50.0)

    # Effectiveness factor: (1 - exp(-NTU)) / NTU
    # Asymptotes: factor=1 at NTU=0, factor=1/NTU at NTU>>1.
    effectiveness = (1.0 - np.exp(-NTU_safe)) / NTU_safe

    return h_full * effectiveness


def effective_particle_diameter(f_solid: dict, dp_coke: float,
                                dp_ore: float) -> np.ndarray:
    """Sauter-mean diameter of the local bed."""
    any_field = next(iter(f_solid.values()))
    f_ore = (f_solid.get("Fe2O3", np.zeros_like(any_field)) +
             f_solid.get("Fe3O4", np.zeros_like(any_field)) +
             f_solid.get("FeO",   np.zeros_like(any_field)) +
             f_solid.get("Fe",    np.zeros_like(any_field)))
    f_coke = f_solid.get("C_coke", np.zeros_like(any_field))
    denom = f_ore / max(dp_ore, 1e-4) + f_coke / max(dp_coke, 1e-4)
    denom = np.maximum(denom, 1e-4)
    dp_eff = (f_ore + f_coke) / denom
    return np.clip(dp_eff, 5.0e-3, 0.12)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalize_mole_fractions(x_gas: dict) -> dict:
    """Clip negatives to EPS_MOLE then renormalise to sum-to-unity."""
    total = None
    clipped = {}
    for sp in GAS_SPECIES:
        clipped[sp] = np.maximum(x_gas[sp], EPS_MOLE)
        total = clipped[sp] if total is None else total + clipped[sp]
    for sp in GAS_SPECIES:
        clipped[sp] = clipped[sp] / total
    return clipped
