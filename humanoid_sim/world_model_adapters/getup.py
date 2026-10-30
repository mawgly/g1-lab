"""Ladder step 2: getting up by tracking one clip (g1_getup track_clip), the world model's first case (stage 1, 27.10):
world_model.TrackTerms' reward, terminations and margins, the collector's clip fields and contexts as they were, so a
getting-up run gives the numbers it gave before the adapters."""
import numpy as np

from . import Adapter, Field, Terms, Thresholds
from .contacts import CONTACT_GROUPS, GROUP_PARTS, LOCKED_WRISTS

FAMILIES = (('hands', (0, 1)), ('knees', (2, 3)), ('feet', (4, 5)))
BONES_LICENCE = ('Не публиковать: эталон слежения может быть записью BONES-SEED (лицензия запрещает '
                 'распространение).')


class GetupTerms(Terms):
    """TrackTerms through the adapter's protocol: the same calls on the same tensors."""

    def __init__(self, reference, device='cpu', dtype=None):
        import torch
        from .. import world_model as wm
        self.track = wm.TrackTerms(reference, device, torch.float32 if dtype is None else dtype)
        self.layout, self.kinematics, self.clip_last = self.track.layout, self.track.kinematics, self.track.clip_last

    def goal(self, batch):
        """Step j's termination margins of a predicted state (TrackTerms.margins at the logged clock). No gradient
        flows through them: the head learns to read them, the dynamics learn from the state error."""
        import torch
        track = self.track
        grace = track.grace_of(batch['kind'], batch['steps'])

        def goal(j, state):
            with torch.no_grad():
                return track.margins(state.detach(), batch['clock'][:, j], batch['yaw'][:, j], batch['offset'][:, j],
                                     grace[:, j])
        return goal

    def goal_rows(self, following, rows):
        grace = self.track.grace_of(rows['kind'], rows['steps'])
        return self.track.margins(following, rows['clock'], rows['yaw'], rows['offset'], grace)

    def reward(self, following, state, flat):
        return self.track.reward(following, state, flat['clock'], flat['yaw'], flat['offset'])

    def terminated(self, following, flat):
        grace = self.track.grace_of(flat['kind'], flat['steps'])
        return self.track.terminated(following, flat['clock'], flat['yaw'], flat['offset'], grace)

    def phase_masks(self, batch):
        rise = batch['clock'] <= self.clip_last
        return dict(rise=rise, hold=~rise)

    def bodies(self, states):
        return self.track.robot(states)


class GetupTrack(Adapter):
    """g1_getup track_clip (world_model.check_track_config): the state, the context world_model.CONTEXT and the clip
    fields as collect.py wrote them before the adapters."""

    name, robot_id, task_id = 'g1_getup_track', 'g1_29dof', 'g1_getup'
    ladder = Thresholds(step=2, phase='rise', error=.02, skill=None, reward_r=.9, reward_set='contact_steps',
                        auroc=.9, gating=('error', 'reward'))
    backends = frozenset({'cpu', 'cuda'})
    nu = 29
    bias = 29
    fields = (Field('clock', 'i4'), Field('yaw', 'f4'), Field('offset', 'f4', (2,)), Field('kind', 'i1'),
              Field('steps', 'i4'))
    contact_groups, group_parts, families = CONTACT_GROUPS, GROUP_PARTS, FAMILIES
    phases = (('rise', 'подъём'), ('hold', 'удержание'))
    agreed = ('action_rate',)
    excluded = ('self_contact', 'impact', 'sole_impact')
    goal_keys = ('clock', 'yaw', 'offset', 'kind', 'steps')
    licence = BONES_LICENCE

    def __init__(self, task_config=None):
        """task_config None or empty: data or a task given directly (the tests), as reference() takes them: TrackTerms
        checks the task itself."""
        from .. import world_model as wm
        if task_config:
            wm.check_track_config(task_config)
        super().__init__(task_config)
        if task_config:  # the getting-up profile always locks the wrists; None keeps the golden run's report
            # Unlike walking's they swing when the hands hit the floor (spread 0.28-1.15 rad/s): 54 % of the rise error
            # of stage 570 (0.034, without them 0.0154, 28.10). The policy cannot move them and the tracked bodies and
            # the reward barely see it (bodies 0.003, reward r 0.99), so they are judged apart (report every_coordinate).
            self.held = LOCKED_WRISTS
        self.context = tuple(wm.CONTEXT)
        self.goal = wm.TRACK_GOAL
        self.defaults = {device: dict(hidden=wm.HIDDEN, batch=wm.BATCH_WINDOWS, minutes=60.)
                         for device in ('cpu', 'cuda')}

    def reference(self, meta):
        """The CPU task of the data's task_config (the clip for TrackTerms), None without a task_config; a task_config
        TrackTerms does not know (world_model.check_track_config) is refused."""
        config = meta.get('task_config')
        if not config:
            return None
        from .. import world_model as wm
        from ..tasks.catalog import make_task
        wm.check_track_config(config)
        return make_task('g1_getup', meta.get('robot_id', 'g1_29dof'), config).unwrapped

    def bind(self, reference, device, dtype=None):
        return GetupTerms(reference, device, dtype)

    @property
    def cuda_base(self):
        from ..g1_getup_tensor_env import TensorG1GetupEnv
        return TensorG1GetupEnv

    # --- CPU: one G1GetupEnv per world ---------------------------------------------------------------------------

    def cpu_floor(self, env):
        return env.floor

    def cpu_context(self, env):
        """world_model.CONTEXT of the CPU task's episode (no COM offset on the CPU task)."""
        from ..world_model import ASSIST_SCALE
        motor = 1. if env._friction_scale is None else float(env._friction_scale[0])
        return np.array([env.friction, motor, env.delay_steps / env.frame_skip, 0., 0., 0., env.assist / ASSIST_SCALE])

    def cpu_after(self, env, info, raw_before, action):
        from ..tasks import g1_tensor_math as task
        tracking = env.clip is not None
        fields = dict(clock=env.clip_clock if tracking else 0, yaw=env.clip_yaw if tracking else 0.,
                      offset=np.array(env.clip_offset) if tracking else np.zeros(2),
                      kind=task.TRACK_START_KINDS.index(env.clip_start) if tracking and env.clip_start else 0,
                      steps=env.steps)
        components = info.get('reward_components', {})
        return fields, sum(components.get(key, 0.) for key in self.excluded), env.previous_action.copy()

    # --- CUDA: the training's TensorG1GetupEnv ---------------------------------------------------------------------

    def cuda_floor(self, env):
        return int(env.reference.floor)

    def cuda_context(self, env):
        """(n, CONTEXT) each world's episode context now."""
        import torch
        from ..world_model import ASSIST_SCALE
        n = env.n_envs
        ipos = env.wp.to_torch(env.model.body_ipos)
        if ipos.dim() == 3:  # randomize: the pelvis COM per world
            nominal = torch.as_tensor(env.reference.model.body_ipos[env.pelvis], dtype=ipos.dtype, device=env.device)
            com = ipos[:, env.pelvis] - nominal
        else:
            com = torch.zeros((n, 3), device=env.device)
        delay = env.delay_steps.float() / env.reference.frame_skip
        return torch.cat((env.friction[:, None], env.friction_scale[:, None], delay[:, None], com,
                          (env.assist / ASSIST_SCALE)[:, None]), 1)

    def cuda_after(self, env):
        """The clip's clock, yaw, offset, start kind and the episode's steps now."""
        import torch
        n = env.n_envs
        if env.clip is not None:
            clock, yaw, offset, kind = env.clip_clock.clone(), env.clip_yaw.clone(), env.clip_offset.clone(), \
                env.clip_kind.clone()
        else:
            clock = kind = torch.zeros(n, dtype=torch.long, device=env.device)
            yaw, offset = torch.zeros(n, device=env.device), torch.zeros((n, 2), device=env.device)
        return dict(clock=clock, yaw=yaw, offset=offset, kind=kind, steps=env.lengths.clone())

    def cuda_excluded(self, env):
        """(n,) the step's self_contact, impact and sole_impact terms (weighted, negative), as the step paid them."""
        import torch
        from ..tasks import g1_tensor_math as task
        weights = env.track_weights if env.clip is not None else env.weights
        part = torch.zeros(env.n_envs, device=env.device)
        if env.self_contact_force is not None and weights.get('self_contact'):
            part -= weights['self_contact'] * task.self_contact_share(env.self_contact_force)
        if env.impact_substeps is not None:
            part -= weights['impact'] * task.impact_penalty(env.impact_substeps.amax(dim=1))
        if env.sole_impact_substeps is not None:
            part -= weights['sole_impact'] * task.impact_penalty(env.sole_impact_substeps.amax(dim=1),
                                                                 task.GETUP_SOLE_IMPACT_SPEED)
        return part
