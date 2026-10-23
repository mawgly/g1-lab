"""CPU only: heel_history.py <stage> [<stage> ...]. One foot on the heel across stages? Own policy, deterministic,
zero reset noise, heading hold off. Straight 0.5 m/s from phase 0 and from half a cycle, and standing: mean ankle pitch
offset from home (L, R, rad), mean foot pitch (deg, - = toe up) over steps 100-600, yaw rate (rad/s)."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from humanoid_sim.core import load_stage, stage_task_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
import mujoco  # noqa: E402

for arg in sys.argv[1:]:
    stage = Path(arg)
    model, norm = load_stage(stage)
    config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': 700}
    parts = []
    for label, command, offset in (('0.5 L-first', [.5, 0., 0.], 0), ('0.5 R-first', [.5, 0., 0.], 15),
                                   ('stand', [0., 0., 0.], 0)):
        env = make_task('g1_velocity', 'g1_29dof', config)
        raw = env.unwrapped
        names = [raw.model.joint(j).name for j in raw.joints]
        ankle = [names.index('left_ankle_pitch_joint'), names.index('right_ankle_pitch_joint')]
        pelvis = raw.model.body('pelvis').id
        env.reset(seed=73000, options={'command': command})
        raw.data.qpos[raw.qadr] = raw.home
        raw.data.qvel[:] = 0
        mujoco.mj_forward(raw.model, raw.data)
        raw.steps = int(round(raw.gait['period'] / raw.dt / 2)) if offset else 0
        obs = raw._observation('fill')
        q, pitch, track = [], [], []
        for t in range(600):
            obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            track.append(math.atan2(raw.data.xmat[pelvis][3], raw.data.xmat[pelvis][0]))
            if t >= 100:
                q.append(raw.data.qpos[raw.qadr][ankle] - raw.home[ankle])
                pitch.append([math.degrees(math.asin(-raw.data.xmat[f][6])) for f in raw.feet])
            if term:
                break
        env.close()
        track = np.unwrap(track)
        q, p = np.mean(q, axis=0), np.mean(pitch, axis=0)
        rate = (track[-1] - track[100]) / ((len(track) - 101) * .02)
        parts.append(f'{label}: ankle L {q[0]:+.2f} R {q[1]:+.2f} foot L {p[0]:+.0f} R {p[1]:+.0f} yaw {rate:+.3f}')
    norm.close()
    print(stage.parent.name[:12], stage.name[6:], ' | '.join(parts), flush=True)
