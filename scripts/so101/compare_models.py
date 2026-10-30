"""Which copy of the real SO-101 predicts it best: on the held-out segments of real_to_data.py's data (fit.py's holdout,
the last worlds), open-loop for WINDOW control steps from the real state on the real goals, the error of our MuJoCo
model as it is, the same with sysid.py's fit, and a world model fitted on the real data (fit.py), beside standing
still; RMS degrees per joint at the first step and over the window.

    .venv\\Scripts\\python.exe scripts\\so101\\compare_models.py <data dir> --world-model <fit dir> [--sysid
        runs\\so101_real\\sysid.json] [--holdout 0.2]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab' / 'wm'))
sys.path.insert(0, str(ROOT / 'scripts' / 'so101'))

import fit  # noqa: E402
import sysid  # noqa: E402

WINDOW = sysid.WINDOW


def main():
    import torch
    from humanoid_sim import world_model as wm
    from humanoid_sim.world_model_adapters.arm import ArmReach
    parser = argparse.ArgumentParser(description='какая копия настоящей SO-101 точнее')
    parser.add_argument('data')
    parser.add_argument('--world-model', required=True)
    parser.add_argument('--sysid', default=str(ROOT / 'runs/so101_real/sysid.json'))
    parser.add_argument('--holdout', type=float, default=.2)
    args = parser.parse_args()
    meta, arrays = fit.load(args.data)
    data = fit.Data(arrays, 'cpu', args.holdout, ArmReach({}))
    total = data.total
    starts, worlds = [], []
    for w in data.held_worlds.tolist():
        for t in range(0, total - WINDOW):
            starts.append(t)
            worlds.append(w)
    starts, worlds = torch.tensor(starts), torch.tensor(worlds)
    batch = data.window(starts, worlds, WINDOW, 'cpu')
    keep = batch['alive'].all(1)
    state = batch['state'][keep].double()
    actions = batch['action'][keep].double()
    targets = batch['next'][keep].double()
    q, speed = state[:, :5].numpy(), state[:, 6:11].numpy()
    goal = (state[:, None, :5] + actions[:, :, :5]).numpy()  # the goals: angle + applied u at each step
    truth = np.degrees(targets[:, :, :5].numpy())
    rows = {}
    model, _ = wm.WorldModel.load(Path(args.world_model) / 'model.pt', 'cpu')
    model = model.double().eval()
    with torch.no_grad():
        predicted, _ = model.rollout(state, actions, torch.zeros((len(state), model.settings['context']),
                                                                 dtype=torch.float64))
    rows['модель мира (по записям руки)'] = np.degrees(predicted[:, :, :5].numpy())
    arm = sysid.Arm()

    def simulate():
        out = np.empty((len(q), WINDOW, 5))
        m, d, env = arm.model, arm.data, arm.env
        import mujoco
        for i in range(len(q)):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = env.home
            d.qpos[arm.qadr] = q[i]
            d.qvel[arm.vadr] = speed[i]
            d.ctrl[:] = env.home
            mujoco.mj_forward(m, d)
            for j in range(WINDOW):
                d.ctrl[:5] = goal[i, j]
                mujoco.mj_step(m, d, nstep=sysid.SUBSTEPS)
                out[i, j] = d.qpos[arm.qadr]
        return np.degrees(out)
    rows['MuJoCo как есть'] = simulate()
    if Path(args.sysid).is_file():
        arm.apply(json.loads(Path(args.sysid).read_text(encoding='utf-8'))['fit'])
        rows['MuJoCo после подгонки'] = simulate()
    rows['«стоять»'] = np.repeat(np.degrees(q)[:, None], WINDOW, 1)
    print(f'отложенные отрезки: {len(q)} окон по {WINDOW} шагов (40 мс), RMS, градусы — первый шаг / за окно')
    for name, values in rows.items():
        error = values - truth
        first = np.sqrt((error[:, 0] ** 2).mean(0))
        whole = np.sqrt((error ** 2).mean((0, 1)))
        print(f'  {name:30s} {np.sqrt((error[:, 0] ** 2).mean()):5.2f} / {np.sqrt((error ** 2).mean()):5.2f}   по суставам '
              + ' '.join(f'{a:.2f}/{b:.2f}' for a, b in zip(first, whole)))


if __name__ == '__main__':
    main()
