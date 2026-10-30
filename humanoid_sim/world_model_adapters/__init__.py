"""The world model's per-task adapters (Pavel and the lead, 27.10: the world model as a general tool, "copies of any
world", climbed as a ladder from the simple). humanoid_sim/world_model.py is the task-free part (state layouts,
kinematics, the network, its loss); an adapter says, for one (robot, task):
- which arrays make the state (layout: a free root or a fixed base, the actions, whether the previous action is state),
  the episode context ξ to log, the per-step exogenous inputs (a root velocity kick, a push), the encoder bias width;
- the task fields collect.py logs per step (Field: before or after the step) and how it reads them on the CPU task and,
  where the task has one, on the CUDA task;
- the reward and the termination from states by the task's own formulas (Terms, torch), which components agree by
  construction (left out of the reward comparison) and which a state cannot give (excluded: summed into the data's
  reward_contact);
- the contact groups to watch for contact-change windows, the report's phases and the ladder step's thresholds.

Steps: 0 arm_reach (SO-101, CPU only, minutes), 1 g1_walk (g1_velocity real robot, candidate 8, CUDA), 2
g1_getup_track (getting up by tracking one clip: the numbers of stage 1 as before). Data without an adapter in its meta
(collected before 27.10, the tests' data) are getting up."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Field:
    """One per-step task field collect.py logs: its name, numpy dtype, per-world shape and whether it is read before or
    after the step."""
    name: str
    dtype: str
    shape: tuple = ()
    when: str = 'after'


@dataclass(frozen=True)
class Thresholds:
    """One ladder step's gate (fit.py's report['ladder']): the 8-step «разброс» error on the phase's windows at most
    `error`, skill at least `skill` (None: shown, not judged), the reward's anomaly correlation on `reward_set`
    ('contact_steps' or 'all') at least reward_r, the termination head's AUROC at least `auroc` when the held-out
    windows hold at least `positives` terminations (fewer: n/a, not judged). gating: the criteria that decide; notes:
    (criterion, Russian reason) pairs shown beside a criterion that does not decide."""
    step: int
    phase: str
    error: float = .02
    skill: float | None = None
    reward_r: float = .9
    reward_set: str = 'contact_steps'
    auroc: float | None = None
    positives: int = 20
    gating: tuple = ('error', 'reward')
    notes: tuple = ()


class Terms:
    """The task's formulas on canonical states (torch, shape-static: no .item(), no boolean-mask indexing, so a CUDA
    graph can hold them). batch: a window batch of fit.Data.window (n, k, …); flat: the same fields flattened over the
    window's steps (m = n·k, …) with 'action', 'context' and, with inputs, 'inputs'."""

    def goal(self, batch):
        """None, or goal(j, state_j) → (n, goal) features of step j's predicted state (called inside no_grad)."""
        return None

    def goal_rows(self, following, rows):
        """(m, goal) goal features of following states (m, canonical) with their rows' fields (m, …), or None."""
        return None

    def reward(self, following, state, flat):
        raise NotImplementedError

    def terminated(self, following, flat):
        return None

    def phase_masks(self, batch):
        return {}

    def window_sets(self, batch):
        return {}

    def bodies(self, states):
        return None


class Adapter:
    """One (robot, task)'s view of its world for the world model (module docstring). Class attributes describe it;
    __init__ refuses (ValueError, Russian) a task_config whose rules it does not have."""

    name = robot_id = task_id = None
    ladder = None
    backends = frozenset({'cpu'})
    root, nu, joints, previous, kick, push = 'free', 29, None, True, False, False
    # Joints the task holds fixed (their angle and speed barely move in the data, so their «разброс» is the 1e-3·median
    # floor and noise there would decide the error): left out of the report's error coordinates, not of the training.
    held = ()
    context = ()
    inputs = ()
    bias = 0
    fields = ()
    contact_groups = ()
    group_parts = ()
    families = ()
    phases = ()
    agreed = ()
    excluded = ()
    goal = 0
    goal_keys = ()
    defaults = {}
    licence = None
    cuda_base = None

    def __init__(self, task_config=None):
        self.task_config = task_config

    def layout(self):
        from .. import world_model as wm
        return wm.layout_of(self.nu, self.root, self.joints, self.previous)

    def model_settings(self):
        """WorldModel's layout and input keyword arguments that are not their defaults ({} for getting up)."""
        values = dict(root=self.root, joints=self.joints, previous=self.previous, kick=self.kick, push=self.push)
        defaults = dict(root='free', joints=None, previous=True, kick=False, push=False)
        return {key: value for key, value in values.items() if value != defaults[key]}

    def check(self, device):
        """ValueError (Russian) when this task cannot be collected on `device`."""
        if device not in self.backends:
            raise ValueError(f'модель мира: у задачи {self.task_id} нет сборщика {device.upper()}, '
                             f'используйте --device cpu')

    def reference(self, meta):
        """The unwrapped CPU task of the data's task_config (the formulas' constants), None without one."""
        config = meta.get('task_config')
        if config is None:
            return None
        from ..tasks.catalog import make_task
        return make_task(self.task_id, meta.get('robot_id', self.robot_id), config).unwrapped

    def bind(self, reference, device, dtype=None):
        raise NotImplementedError

    def describe(self, reference):
        """meta.json's numbers of the task: actuators, frame_skip, dt, home."""
        return dict(nu=int(reference.model.nu), frame_skip=int(reference.frame_skip), dt=float(reference.dt),
                    home=[float(v) for v in reference.home])

    # --- the CPU collector -----------------------------------------------------------------------------------------

    def cpu_env(self, config):
        from ..tasks.catalog import make_task
        return make_task(self.task_id, self.robot_id, config).unwrapped

    def cpu_floor(self, env):
        from .contacts import floor_geom
        return floor_geom(env.model)

    def cpu_groups(self, env):
        """(ngeom,) each geom's contact group (−1 the rest), None without contact groups."""
        if not self.contact_groups:
            return None
        from .contacts import contact_groups
        return contact_groups(env.model, env.robot_geoms, self.contact_groups, self.group_parts)

    def cpu_contacts(self, env, groups):
        if groups is None:
            return np.zeros(0, dtype=bool), np.zeros(0)
        from .contacts import cpu_contacts
        return cpu_contacts(env, groups, self.cpu_floor(env), len(self.contact_groups))

    def cpu_raw(self, env):
        return np.concatenate([env.data.qpos, env.data.qvel, env.previous_action])

    def cpu_context(self, env):
        return np.zeros(len(self.context))

    def cpu_bias(self, env):
        return np.zeros(self.bias)

    def cpu_before(self, env):
        return {}

    def cpu_after(self, env, info, raw_before, action):
        """(the 'after' fields, the excluded components' sum, the applied action) of the step just taken."""
        raise NotImplementedError

    def cpu_inputs(self, env, raw_before):
        return np.zeros(len(self.inputs))

    # --- the CUDA collector (collect.collecting_env) ---------------------------------------------------------------

    def cuda_floor(self, env):
        from .contacts import floor_geom
        return floor_geom(env.reference.model)

    def cuda_context(self, env):
        import torch
        return torch.zeros((env.n_envs, len(self.context)), device=env.device)

    def cuda_before(self, env):
        return {}

    def cuda_after(self, env):
        return {}

    def cuda_excluded(self, env):
        import torch
        return torch.zeros(env.n_envs, device=env.device)

    def cuda_inputs(self, env, state):
        import torch
        return torch.zeros((env.n_envs, len(self.inputs)), device=env.device)


from .arm import ArmReach  # noqa: E402 (the adapters use the classes above)
from .getup import GetupTrack  # noqa: E402
from .walk import G1Walk  # noqa: E402

ADAPTERS = {('g1_29dof', 'g1_getup'): GetupTrack, ('g1_29dof', 'g1_velocity'): G1Walk,
            ('so101', 'arm_reach'): ArmReach}


def adapter_for(robot_id, task_id, task_config):
    """The adapter of a (robot, task) for a task_config; ValueError (Russian) for a pair without one or a task_config
    the adapter refuses."""
    if (robot_id, task_id) not in ADAPTERS:
        raise ValueError(f'модель мира: для робота {robot_id} и задачи {task_id} адаптера пока нет')
    return ADAPTERS[(robot_id, task_id)](task_config)


def adapter_of_meta(meta):
    """The adapter of collect.py's meta.json: meta['adapter'] by name; without it (data collected before 27.10, the
    tests' data) getting up."""
    name = meta.get('adapter')
    if name is None:
        return GetupTrack(meta.get('task_config'))
    for kind in ADAPTERS.values():
        if kind.name == name:
            return kind(meta.get('task_config'))
    raise ValueError(f'модель мира: неизвестный адаптер {name}')
