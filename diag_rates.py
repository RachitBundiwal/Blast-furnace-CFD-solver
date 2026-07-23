"""Trace reaction rates and species balance in the converged solver."""
import sys
sys.path.insert(0, '.')
import numpy as np
from bf2d.parameters import BFParameters2D
from bf2d.simulation import BlastFurnaceSimulation
from bf2d.kinetics import compute_rates, assemble_species_sources, assemble_solid_sources
from bf2d.properties import gas_mw_mix, gas_density
from bf2d.constants import MW

p = BFParameters2D(
    nr=10, nz=25,
    max_outer_iter=200, log_every=200,
    dt=0.5,
    convergence_rms=1e-3,
    momentum_relaxation=0.4,
    pressure_relaxation=0.3,
    energy_relaxation=0.3,
    species_relaxation=0.5,
)
sim = BlastFurnaceSimulation(p, log_level='ERROR')
for it in range(200):
    sim._one_outer_iteration(it)
    sim.residuals.add(sim.residuals.latest() if False else __import__('bf2d').residuals.ResidualSnapshot(it))  # dummy
    sim._adapt_relaxation()

f = sim.fields
mesh = sim.mesh
mw_mix = gas_mw_mix(f.x_gas)
rates = compute_rates(f.T_g, f.T_s, f.P, f.rho_g, mw_mix, f.x_gas, f.f_solid, f.eps, mesh.active)

print('Reaction rates  [mol/(m^3 s)]:')
for name in ('R1','R1b','R2','R3','R4','R5','R6','R7','R_DR'):
    arr = getattr(rates, name)
    am = arr[mesh.active]
    print(f'  {name:5s}: max={am.max():.3e}  mean={am.mean():.3e}  nz_count={int((am > 1e-12).sum())}')

print('\nTemperature info:')
print(f'  T_g[active]: min={f.T_g[mesh.active].min():.0f}  mean={f.T_g[mesh.active].mean():.0f}  max={f.T_g[mesh.active].max():.0f} K')
print(f'  T_s[active]: min={f.T_s[mesh.active].min():.0f}  mean={f.T_s[mesh.active].mean():.0f}  max={f.T_s[mesh.active].max():.0f} K')

print('\nSolid composition mean:')
for ph in ('Fe2O3','Fe3O4','FeO','Fe','C_coke','gangue'):
    print(f'  {ph:7s}: mean={f.f_solid[ph][mesh.active].mean():.4f}  max={f.f_solid[ph][mesh.active].max():.4f}')

print('\nGas composition mean:')
for sp in ('N2','O2','CO','CO2','H2','H2O'):
    print(f'  {sp:5s}: mean={f.x_gas[sp][mesh.active].mean():.4f}  max={f.x_gas[sp][mesh.active].max():.4f}')

# Where are reactions hot?  Print the cell with max R3
arg = np.unravel_index(np.argmax(rates.R3), rates.R3.shape)
i, j = arg
print(f'\nMax R3 cell: (i,j)=({i},{j})  z={mesh.z_c[j]:.2f}m  r={mesh.r_c[i]:.2f}m')
print(f'  T_g={f.T_g[i,j]:.0f}K  T_s={f.T_s[i,j]:.0f}K')
print(f'  CO={f.x_gas["CO"][i,j]:.3f}  CO2={f.x_gas["CO2"][i,j]:.3f}')
print(f'  Fe3O4={f.f_solid["Fe3O4"][i,j]:.5f}  FeO={f.f_solid["FeO"][i,j]:.5f}')
print(f'  R3={rates.R3[i,j]:.3e}  R4={rates.R4[i,j]:.3e}')
