"""Ladder step 0: the SO-101 arm reaching a target (robot so101, task arm_reach, arm_sim.env.ArmReachEnv), on the home
CPU in minutes: a fixed base, no floor contacts, no context. It proves the pipeline before the G1.

State: world_model.FixedLayout (6 hinges: qpos | qvel; no previous action, the arm's position actuators keep none).
Action: the applied u = ctrl − q before the step (6): clip(q + 0.08·a, ctrlrange) − q of the five axes, home − q of the
gripper, which imagination computes from (q, a). The reward is ArmReachEnv.step's, ported to torch on the gripper tip
(the gripperframe site by Kinematics.points); its hold counter is teacher-forced from the logged one before the step,
so the termination head's AUROC is shown, not judged. The gripper, held at home by the task, is left out of the report's
error (Adapter.held). Episodes end on success after ~10 steps (5–19 in the first data), so few windows live to step 8
and none to 32: the report gives each horizon's deepest step (fit.py)."""
import numpy as np

from . import Adapter, Field, Terms, Thresholds

SUCCESS_DISTANCE, SUCCESS_HOLD = .02, 5  # ArmReachEnv: within 2 cm for 5 steps
REVISION = 'models/so101/UPSTREAM_REVISION.txt'


class ArmTerms(Terms):
    """ArmReachEnv's reward and success of canonical states (FixedLayout), the target and the hold counter from the
    logged fields."""

    def __init__(self, reference, device='cpu', dtype=None):
        import torch
        from .. import world_model as wm
        model = reference.model
        dtype = torch.float32 if dtype is None else dtype
        self.layout = wm.FixedLayout(model.nq, model.nu)
        self.kinematics = wm.Kinematics(model, device, dtype)
        self.tip_body = [int(model.site_bodyid[reference.tip_id])]
        self.tip_local = torch.as_tensor(model.site_pos[reference.tip_id][None], dtype=dtype, device=device)
        # links 2-6: shoulder, upper_arm, lower_arm, wrist, gripper
        self.body_ids = torch.as_tensor([model.body(name).id for name in ('shoulder', 'upper_arm', 'lower_arm',
                                                                          'wrist', 'gripper')], device=device)

    def distance(self, state, target):
        tip = self.kinematics.points(state, self.layout, self.tip_body, self.tip_local)[:, 0]
        return (tip - target.to(state.dtype)).norm(dim=-1)

    def goal(self, batch):
        def goal(j, state):
            return self.goal_rows(state.detach(), dict(target=batch['target'][:, j],
                                                       hold_before=batch['hold_before'][:, j]))
        return goal

    def goal_rows(self, following, rows):
        """(m, 2): (d − 0.02)/0.02 and the hold counter before the step / 5."""
        import torch
        distance = self.distance(following, rows['target'])
        return torch.stack(((distance - SUCCESS_DISTANCE) / SUCCESS_DISTANCE,
                            rows['hold_before'].to(following.dtype) / SUCCESS_HOLD), -1)

    def _hold(self, following, flat):
        distance = self.distance(following, flat['target'])
        hold = (flat['hold_before'] + 1) * (distance < SUCCESS_DISTANCE)
        return distance, hold >= SUCCESS_HOLD

    def reward(self, following, state, flat):
        import torch
        distance, success = self._hold(following, flat)
        before = self.distance(state, flat['target'])
        action = flat['policy_action'].to(following.dtype)
        components = dict(progress=20 * (before - distance), distance=-10 * distance,
                          action=-.005 * (action * action).sum(-1), near=.5 * torch.exp(-50 * distance),
                          success=10 * success.to(following.dtype))
        return sum(components.values()), components

    def terminated(self, following, flat):
        return self._hold(following, flat)[1]

    def bodies(self, states):
        position, rotation, linear, angular = self.kinematics(states, self.layout)
        index = self.body_ids
        return position[:, index], rotation[:, index], linear[:, index], angular[:, index]


class ArmReach(Adapter):
    """so101 / arm_reach, CPU only (module docstring)."""

    name, robot_id, task_id = 'arm_reach', 'so101', 'arm_reach'
    # AUROC does not decide: the head reads the logged hold counter, and every termination is hold_before == 4 within
    # 2 cm, so the label is nearly an input (1.000 on the first fit whatever the dynamics).
    ladder = Thresholds(step=0, phase='all', error=.02, skill=.9, reward_r=.9, reward_set='all', auroc=.95,
                        positives=20, gating=('error', 'skill', 'reward'),
                        notes=(('auroc', 'показан, не судим: счётчик удержания берётся из записи'),))
    root, nu, joints, previous = 'fixed', 6, 6, False
    held = (5,)  # the gripper: ArmReachEnv holds it at home (ctrl[5] = home[5])
    fields = (Field('target', 'f4', (3,), 'before'), Field('hold_before', 'i4', (), 'before'),
              Field('policy_action', 'f4', (5,)))
    agreed = ('action',)
    goal = 2
    goal_keys = ('target', 'hold_before')
    defaults = {device: dict(hidden=(512, 512, 512), batch=512, minutes=10.) for device in ('cpu', 'cuda')}

    def reference(self, meta):
        from ..tasks.catalog import make_task
        return make_task('arm_reach', 'so101', meta.get('task_config') or {}).unwrapped

    def bind(self, reference, device, dtype=None):
        return ArmTerms(reference, device, dtype)

    def describe(self, reference):
        frame_skip = 20  # ArmReachEnv.step: mj_step nstep=20
        return dict(nu=int(reference.model.nu), frame_skip=frame_skip,
                    dt=float(frame_skip * reference.model.opt.timestep), home=[float(v) for v in reference.home])

    def cpu_raw(self, env):
        return np.concatenate([env.data.qpos, env.data.qvel])

    def cpu_before(self, env):
        return dict(target=env.target.copy(), hold_before=env.hold)

    def cpu_after(self, env, info, raw_before, action):
        applied = env.data.ctrl - raw_before[:env.model.nq]
        return dict(policy_action=np.clip(np.asarray(action, dtype=np.float64), -1, 1)), 0., applied


def check_revision(config):
    """ValueError (Russian) when an arm_sim run's training_config names another SO-101 model revision than the
    repository's."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    revision = (root / REVISION).read_text(encoding='utf-8').strip()
    if config.get('model_revision') != revision:
        raise ValueError('модель мира: политика руки обучена на другой ревизии модели SO-101')
