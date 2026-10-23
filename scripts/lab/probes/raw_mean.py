import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\Users\Admin\Documents\Codex\Sim')
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
from humanoid_sim.core import load_stage, stage_task_config
from humanoid_sim.tasks.catalog import make_task
for st in sys.argv[1:]:
    stage = Path(st)
    model, norm = load_stage(stage)
    config = stage_task_config(stage)
    std = model.policy.log_std.exp().detach().numpy() if hasattr(model.policy, 'log_std') else None
    print('==', stage.parent.name[:12], stage.name, 'std wrist pitch L/R', None if std is None else (round(float(std[20]), 3), round(float(std[27]), 3)), 'std min/max', None if std is None else (round(float(std.min()), 3), round(float(std.max()), 3)))
    for v in (0., .5, 1.):
        env = make_task('g1_velocity', 'g1_29dof', {**config, 'heading_hold': False, 'max_episode_steps': 1000})
        raw = env.unwrapped
        obs, _ = env.reset(seed=73000, options={'command': [v, 0., 0.]})
        means = []
        for i in range(300):
            o = norm.normalize_obs(obs)
            with torch.no_grad():
                d = model.policy.get_distribution(torch.as_tensor(o[None], dtype=torch.float32))
                m = d.distribution.mean[0].numpy()
            means.append(m)
            obs, _, term, _, _ = env.step(np.clip(m, -10, 10))
            if term:
                break
        means = np.array(means[100:])
        print(f'  v{v}: raw mean L wrist pitch {means[:, 20].mean():+.2f} (min {means[:, 20].min():+.2f}, max {means[:, 20].max():+.2f}) | R {means[:, 27].mean():+.2f} | L wrist yaw {means[:, 21].mean():+.2f} R {means[:, 28].mean():+.2f} | any |mean|>10: {[(j, round(float(means[:, j].mean()), 2)) for j in range(29) if abs(means[:, j]).mean() > 9.5]}')
        env.close()
    norm.close()
