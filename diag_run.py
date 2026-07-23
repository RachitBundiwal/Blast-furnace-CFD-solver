"""Quick convergence-trajectory diagnostic for Phase 1."""
import sys
sys.path.insert(0, '.')
import numpy as np
from bf2d.parameters import BFParameters2D
from bf2d.simulation import BlastFurnaceSimulation


def run(n_iter: int = 200,
        dt: float = 0.5,
        omega_mom: float = 0.4,
        omega_press: float = 0.3,
        omega_energy: float = 0.3,
        omega_species: float = 0.5,
        log_every: int = 10):
    p = BFParameters2D(
        nr=10, nz=25,
        max_outer_iter=n_iter,
        log_every=log_every,
        dt=dt,
        convergence_rms=1e-3,
        momentum_relaxation=omega_mom,
        pressure_relaxation=omega_press,
        energy_relaxation=omega_energy,
        species_relaxation=omega_species,
    )
    sim = BlastFurnaceSimulation(p, log_level='ERROR')
    hdr = ['it', 'worst', 'res_c', 'res_T', 'T_max', 'omega_E', 'omega_M', 'CZ']
    print('  '.join(f'{h:>9}' for h in hdr))
    for it in range(n_iter):
        snap = sim._one_outer_iteration(it)
        sim.residuals.add(snap)
        sim._adapt_relaxation()
        f = sim.fields
        if it % log_every == 0 or it == n_iter - 1:
            Tmax = f.T_g[sim.mesh.active].max()
            cz = int(f.cohesive_mask.sum())
            row = [str(it),
                   f'{snap.max():.2e}',
                   f'{snap.continuity:.2e}',
                   f'{snap.gas_energy:.2e}',
                   f'{Tmax:.0f}',
                   f'{sim._omega_energy:.3f}',
                   f'{sim._omega_mom:.3f}',
                   str(cz)]
            print('  '.join(f'{c:>9}' for c in row))
        if snap.max() > 1e6 or not np.all(np.isfinite(f.T_g)):
            print(f'DIVERGED at it {it}')
            return sim
    print()
    Tg = sim.fields.T_g[sim.mesh.active]
    P = sim.fields.P[sim.mesh.active]
    print(f'Final: T_g=[{Tg.min():.0f},{Tg.max():.0f}]K  P=[{P.min()/1e5:.2f},{P.max()/1e5:.2f}] bar')
    print(f'  u_z=[{sim.fields.u_z.min():.2f},{sim.fields.u_z.max():.2f}] m/s')
    fe3o4_max = sim.fields.f_solid['Fe3O4'][sim.mesh.active].max()
    feo_max = sim.fields.f_solid['FeO'][sim.mesh.active].max()
    fe_max = sim.fields.f_solid['Fe'][sim.mesh.active].max()
    co_top = sim.fields.x_gas['CO'][:, -1][sim.mesh.active[:, -1]].mean()
    print(f'  Fe3O4_max={fe3o4_max:.4f}  FeO_max={feo_max:.4f}  Fe_max={fe_max:.4f}')
    print(f'  CO_top_mean={co_top:.3f}')
    return sim


if __name__ == '__main__':
    run()
