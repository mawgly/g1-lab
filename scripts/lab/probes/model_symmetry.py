"""CPU only: model_symmetry.py [stage]. Is the G1 model itself left-right symmetric? Compares every left body with
its right twin (mass, COM and inertia mirrored across the x-z plane), and checks that unpaired bodies have their
COM at y = 0, in the home pose; then the whole-robot COM y. Prints only mismatches above the tolerances."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import stage_task_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402

config = stage_task_config(Path(sys.argv[1])) if len(sys.argv) > 1 else {'real_robot': True}
env = make_task('g1_velocity', 'g1_29dof', config)
raw = env.unwrapped
raw.reset(seed=0)
m, d = raw.model, raw.data
names = [m.body(i).name for i in range(m.nbody)]
pelvis = d.xpos[m.body('pelvis').id]
print('bodies', m.nbody, 'total mass', float(m.body_subtreemass[0]))
for i, name in enumerate(names):
    com = d.xipos[i] - pelvis
    if 'left' in name:
        j = names.index(name.replace('left', 'right'))
        other = d.xipos[j] - pelvis
        mass = m.body_mass[i] - m.body_mass[j]
        offset = com - other * np.array([1, -1, 1])
        inertia = m.body_inertia[i] - m.body_inertia[j]
        if abs(mass) > 1e-4 or np.abs(offset).max() > 1e-3 or np.abs(inertia).max() > 1e-6:
            print(f'{name:28s} dmass {mass:+.4f} kg  dCOM(mirrored) {offset.round(4)}  dI {inertia.round(6)}')
    elif 'right' not in name and abs(com[1]) > 1e-3:
        print(f'{name:28s} unpaired, COM y {com[1]:+.4f} m, mass {m.body_mass[i]:.3f} kg')
total = (d.xipos[1:] * m.body_mass[1:, None]).sum(0) / m.body_mass[1:].sum() - pelvis
print('whole-robot COM relative to pelvis', total.round(4))
for name in ('left_foot', 'right_foot'):
    pass
geoms = [m.geom(g).name for g in range(m.ngeom)]
left = [g for g in geoms if 'left' in g]
for g in left:
    a, b = m.geom(g).id, m.geom(g.replace('left', 'right')).id if g.replace('left', 'right') in geoms else None
    if b is None:
        print('geom without twin', g)
        continue
    if np.abs(m.geom_friction[a] - m.geom_friction[b]).max() > 1e-6 or np.abs(m.geom_size[a] - m.geom_size[b]).max() > 1e-6:
        print('geom mismatch', g, m.geom_friction[a], m.geom_friction[b], m.geom_size[a], m.geom_size[b])
for j in raw.joints:
    name = m.joint(j).name
    if 'left' in name:
        k = m.joint(name.replace('left', 'right')).id
        da = m.dof_armature[m.jnt_dofadr[j]] - m.dof_armature[m.jnt_dofadr[k]]
        dd = m.dof_damping[m.jnt_dofadr[j]] - m.dof_damping[m.jnt_dofadr[k]]
        df = m.dof_frictionloss[m.jnt_dofadr[j]] - m.dof_frictionloss[m.jnt_dofadr[k]]
        if max(abs(da), abs(dd), abs(df)) > 1e-6:
            print(f'joint {name}: d armature {da:+.5f} damping {dd:+.5f} friction {df:+.5f}')
env.close()
