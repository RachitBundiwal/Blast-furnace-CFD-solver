"""
Physical constants and thermochemical data for blast furnace modelling.

References
----------
- Perry's Chemical Engineers' Handbook (8th ed.)
- NIST WebBook (standard enthalpies of formation)
- Austin, P.R. et al., ISIJ Int. 37 (1997) 748-755
- Yagi, J., ISIJ Int. 33 (1993) 619-639
"""
from __future__ import annotations

from typing import Final, Mapping

# ---------------------------------------------------------------------------
# Universal constants (SI)
# ---------------------------------------------------------------------------
R_GAS: Final[float] = 8.31446261815324        # J/(mol K)
SIGMA_SB: Final[float] = 5.670374419e-8        # W/(m^2 K^4) Stefan-Boltzmann
G_ACC: Final[float] = 9.80665                  # m/s^2
P_ATM: Final[float] = 101325.0                 # Pa
T_STD: Final[float] = 273.15                   # K
MOL_VOL_NTP: Final[float] = 22.414e-3          # m^3/mol at 0 degC, 1 atm

# ---------------------------------------------------------------------------
# Molar masses (kg/mol) -- IUPAC 2021
# ---------------------------------------------------------------------------
MW: Final[Mapping[str, float]] = {
    "N2":    0.0280134,
    "O2":    0.0319988,
    "CO":    0.0280101,
    "CO2":   0.0440095,
    "H2":    0.0020159,
    "H2O":   0.0180153,
    "CH4":   0.0160425,
    "Fe":    0.0558450,
    "FeO":   0.0718444,
    "Fe3O4": 0.2315326,
    "Fe2O3": 0.1596882,
    "C":     0.0120107,
    "SiO2":  0.0600843,
    "CaO":   0.0560774,
    "MgO":   0.0403044,
    "Al2O3": 0.1019612,
    "CaCO3": 0.1000869,
}

# Gas species tracked by the solver (order matters for vectorization)
GAS_SPECIES: Final[tuple[str, ...]] = ("N2", "O2", "CO", "CO2", "H2", "H2O")

# Solid phases tracked (mass-fraction fields)
SOLID_PHASES: Final[tuple[str, ...]] = (
    "Fe2O3", "Fe3O4", "FeO", "Fe", "C_coke", "gangue", "flux",
)
# "flux" = carbonate flux (CaCO3 proxy for limestone+dolomite charge).
# Calcines in-shaft via R9: CaCO3 -> CaO + CO2 (endothermic, ~1050-1200 K).
# The CaO product is transferred to "gangue" (inert mineral pool).

# ---------------------------------------------------------------------------
# Standard enthalpies of reaction at 298 K (J/mol of limiting reactant)
# Sign convention: exothermic negative
# ---------------------------------------------------------------------------
DELTA_H_RXN: Final[Mapping[str, float]] = {
    # Fuel combustion (per mol O2)
    "R1_C_O2":     -3.935e5,   # C + O2 -> CO2            (raceway, full burn)
    "R1b_C_halfO2":-1.105e5,   # C + 0.5 O2 -> CO         (raceway)
    # Indirect reduction with CO (per mol CO consumed)
    "R2_Fe2O3_CO": -1.36e4,    # 3Fe2O3 + CO -> 2Fe3O4 + CO2   (exo)
    "R3_Fe3O4_CO": +3.79e4,    # Fe3O4 + CO -> 3FeO + CO2       (endo above ~843 K)
    "R4_FeO_CO":   -1.65e4,    # FeO + CO -> Fe + CO2           (slightly exo)
    # Indirect reduction with H2
    "R2H_Fe2O3_H2": +5.00e3,
    "R3H_Fe3O4_H2": +6.30e4,
    "R4H_FeO_H2":   +2.50e4,
    # Boudouard / solution loss
    "R5_Boudouard": +1.72e5,   # C + CO2 -> 2CO   (endo)
    # Water-gas
    "R6_WaterGas":  +1.31e5,   # C + H2O -> CO + H2
    # Flux calcination  CaCO3 -> CaO + CO2  (strongly endothermic)
    "R9_calcination": +1.78e5,
    # Water-gas shift
    "R7_WGS":       -4.10e4,   # CO + H2O -> CO2 + H2
    # Direct reduction (net, above 1000 degC)
    "R_DR":         +1.52e5,   # FeO + C -> Fe + CO
}

# ---------------------------------------------------------------------------
# NASA 7-coefficient polynomial Cp(T) = R * (a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4)
# Valid 298-3500 K (Burcat / NIST).  Kept compact for solver use.
# ---------------------------------------------------------------------------
NASA7_CP: Final[Mapping[str, tuple[float, float, float, float, float]]] = {
    # a1, a2, a3, a4, a5  (divide Cp by R_GAS to obtain dimensionless)
    "N2":  (3.298677, 1.408240e-3, -3.963222e-6,  5.641515e-9, -2.444854e-12),
    "O2":  (3.782456, -2.996734e-3, 9.847302e-6, -9.681295e-9,  3.243728e-12),
    "CO":  (3.262452, 1.511941e-3, -3.881755e-6,  5.581944e-9, -2.474951e-12),
    "CO2": (2.275725, 9.922072e-3, -1.040911e-5,  6.866687e-9, -2.117280e-12),
    "H2":  (3.298124, 8.249441e-4, -8.143015e-7, -9.475434e-11, 4.134872e-13),
    "H2O": (4.198640, -2.036402e-3, 6.520342e-6, -5.487927e-9,  1.771968e-12),
}

# ---------------------------------------------------------------------------
# Solid heat capacity coefficients  Cp_s = a + b*T + c/T^2   [J/(mol K)]
# ---------------------------------------------------------------------------
SOLID_CP_COEFF: Final[Mapping[str, tuple[float, float, float]]] = {
    "Fe2O3": (103.9, 0.00,    -1.770e6),
    "Fe3O4": (200.8, 0.00,    -4.000e6),
    "FeO":   ( 51.8, 6.780e-3,-1.590e6),
    "Fe":    ( 17.3, 0.02670,  0.000),
    "C_coke":( 17.2, 4.270e-3,-8.790e5),
    "gangue":( 80.0, 0.00,     0.000),   # lumped SiO2/CaO/Al2O3
    "flux":  (104.5, 0.02192, -2.594e6),  # CaCO3 (calcite)
}

# ---------------------------------------------------------------------------
# Default bounds for clamping / validation (physical limits)
# ---------------------------------------------------------------------------
T_MIN_PHYSICAL: Final[float] = 250.0    # K
T_MAX_PHYSICAL: Final[float] = 2600.0   # K (adiabatic flame upper bound)
P_MIN_PHYSICAL: Final[float] = 5.0e4    # Pa
P_MAX_PHYSICAL: Final[float] = 1.0e6    # Pa

# ---------------------------------------------------------------------------
# Mass-fraction floor (prevents division by zero in kinetics)
# ---------------------------------------------------------------------------
EPS_MASS: Final[float] = 1.0e-10
EPS_MOLE: Final[float] = 1.0e-12
