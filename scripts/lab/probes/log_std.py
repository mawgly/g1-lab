import io, json, sys, zipfile
from pathlib import Path

import torch

scale = json.load(open(r'C:\Users\Admin\Documents\Codex\Sim\assets\robots\g1\controller.json'))['action_scale']
kp = json.load(open(r'C:\Users\Admin\Documents\Codex\Sim\assets\robots\g1\controller.json'))['stiffness']
names = ['L_hip_pitch', 'L_hip_roll', 'L_hip_yaw', 'L_knee', 'L_ankle_pitch', 'L_ankle_roll',
         'R_hip_pitch', 'R_hip_roll', 'R_hip_yaw', 'R_knee', 'R_ankle_pitch', 'R_ankle_roll',
         'waist_yaw', 'waist_roll', 'waist_pitch'] + [f'arm{i}' for i in range(14)]
for stage in sys.argv[1:]:
    with zipfile.ZipFile(Path(stage) / 'policy.zip') as z:
        state = torch.load(io.BytesIO(z.read('policy.pth')), map_location='cpu', weights_only=True)
    std = state['log_std'].exp()
    print(Path(stage).name, 'geo-mean sigma', float(std.log().mean().exp()), 'legs geo-mean',
          float(std[:12].log().mean().exp()))
    for i, n in enumerate(names):
        print(f'  {n:14s} sigma {float(std[i]):.3f}  rad {float(std[i])*scale[i]:.3f}  Nm {float(std[i])*scale[i]*kp[i]:.1f}')
