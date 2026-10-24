"""A G1 real-robot stage for Unitree's g1_ctrl (unitree_rl_mjlab deploy): policy.onnx and deploy.yaml.

    .venv\\Scripts\\python.exe -m humanoid_sim.export_g1 <stage dir | experiment dir> --out <folder> [--stand-trim]
        [--command-ranges X_MIN X_MAX Y YAW]

policy.onnx takes `obs` [1, A] (A = the actor's inputs: the deploy terms with their history, no critic extras) and
returns `actions` [1, 29], float32, static shapes, opset 17. Inside: the stage's VecNormalize ((x - mean) /
sqrt(var + epsilon), clipped to ±clip_obs), the SB3 actor (mlp_extractor.policy_net + action_net; its first layer
keeps only the columns of the actor's inputs, which is what ActorInputs' zeroing of the rest amounts to), the mean
action (no σ) and the clip to ±action_limit the environment applies before the PD target (task_config
height_command / upper_body_slot add the terms height_command and upper_body_command after last_action: A 585 at
history 5, for g1_ctrl with scripts/sim2sim/patches/g1_ctrl-height.patch); with task_config
lock_wrists, then zeros on the wrist pitch and yaw outputs (g1_tensor_math.lock_joints, as the environment), so
g1_ctrl holds those joints at default_joint_pos. g1_ctrl feeds that output back as last_action, exactly as the
environment's previous_action, and every observation term in deploy.yaml has scale 1 and clip null. task_config
arm_blind is only in the normalization's numbers (the arm columns map onto the stage's constants), so the graph's ops
and deploy.yaml stay the plain ones; the ONNX properties and a deploy.yaml comment note it.

--stand-trim (iteration 16): standing, a policy may turn on the spot from its own left-right lopsidedness (candidate
3, exp-6986f1d9544b4d00 @ 5481431040: +7…+8° in 5 s in the Unitree bridge). The graph then ends with
Where(‖command‖ < 0.1, clip(a − b, ±action_limit), a): command is the newest frame's velocity_commands (scale 1 in
deploy.yaml, so the same numbers and the same threshold as phase_stop's gait phase), b = stand_trim(stage), half the
policy's mean asymmetry π(o) − M_a·π(M_o·o) standing, within ±TRIM_LIMIT rad per joint. Walking is untouched: the
same b always on spoils the 1 m/s drift (critic: −0.003 → +0.024). b goes to stand_trim.json, deploy.yaml (comment)
and the ONNX properties; g1_ctrl sees an ordinary policy. --symmetric (experiments only) exports ½(π(o) +
M_a·π(M_o·o)) instead, which also stops the turn but breaks the foot choice against a push standing (7 of 8 falls).
ExportedPolicy runs an export folder in the CPU environment of its stage, so goal_check --export and the probes judge
exactly what goes to the robot.

deploy.yaml follows deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml: gains, default pose and action
scale from assets/robots/g1/controller.json (as the environment loads them), command ranges from the stage's
task_config (--command-ranges narrows them within it: g1_ctrl clamps the stick to these ranges, so a stage trained
up to 2 m/s can go out with the stage-1 clip −0.5…1 / ±0.5 / ±1 until floor tests; the graph is unchanged), the 7
observation terms in deploy order with the stage's history length, and the action clipped to
the environment's PD target bounds (target_low / target_high: the model's joint ranges; with task_config
knee_floor the knees' lower bound is the floor): joint_actions.h clamps scale·a + offset per joint to clip[i] =
[lo, hi], as the environment clips the PD target. That clamp acts on the target only; last_action stays the ONNX
output (action_manager.h: action() is what process_action got), as the environment's previous action keeps a knee
action below the floor. So knee_floor adds nothing to the graph.

The check runs the stage's CPU environment (stand, 0.5 and 1 m/s, turns) and compares onnxruntime with
policy.predict on the same observations, and g1_ctrl's motor target from deploy.yaml with the environment's PD target.
"""
import argparse
import copy
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution

from .checkpoint_schema import checkpoint_identity
from .core import list_stages, load_stage, read_json, stage_task_config, write_json
from .policies import ActorInputs
from .tasks import g1_tensor_math as task

OPSET = 17
# deploy_observation's terms in order (g1_tensor_math.deploy_terms gives their sizes).
TERMS = ('base_ang_vel', 'projected_gravity', 'velocity_commands', 'gait_phase', 'joint_pos_rel', 'joint_vel_rel',
         'last_action')
# Check commands as goal_check's (vx m/s, vy m/s, yaw rad/s): stand, 0.5 and 1 m/s, turns at 1 m/s.
CHECK_COMMANDS = ((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.), (1., 0., .5), (1., 0., -.5))
TOLERANCE = 1e-4
# Stand trim (critic's recipe, iteration 16): standing episodes of the stage's CPU environment (command 0, gait phase
# 0, heading hold off), the control steps averaged in each (2-6 s), and the bound per joint in rad of PD target.
TRIM_SEEDS = (73000, 73001, 73002)
TRIM_WINDOW = (100, 300)
TRIM_LIMIT = .052
TRIM_FILE = 'stand_trim.json'


class DeployedPolicy(torch.nn.Module):
    """One stage's observation normalization, actor and action clip as a single float32 graph. locked: (actions,)
    bool, the joints whose output is 0 after the clip (lock_wrists), or None (no such step in the graph).
    trim: (actions,) stand trim b in action units, subtracted (then clipped to the box again) while the norm of
    obs[command[0]:command[1]], the newest frame's command, is below task.PHASE_STOP_NORM; mirror: (obs index, obs
    sign, action index, action sign) of symmetry.Mirror over the actor inputs, for ½(π(o) + M_a·π(M_o·o)). None:
    no such step in the graph."""

    def __init__(self, mean, std, clip_obs, actor, action_limit, locked=None, trim=None, command=None, mirror=None):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
        self.clip_obs, self.action_limit = float(clip_obs), float(action_limit)
        self.actor = actor
        self.register_buffer('locked', None if locked is None else torch.as_tensor(locked, dtype=torch.bool))
        if (trim is None) != (command is None):
            raise ValueError('Подстройке стоя нужны и b, и положение команды во входе')
        self.register_buffer('trim', None if trim is None else torch.as_tensor(trim, dtype=torch.float32))
        self.command = None if command is None else (int(command[0]), int(command[1]))
        for name, dtype, value in zip(('mirror_obs', 'mirror_obs_sign', 'mirror_action', 'mirror_action_sign'),
                                      (torch.long, torch.float32, torch.long, torch.float32),
                                      (None,) * 4 if mirror is None else mirror):
            self.register_buffer(name, None if value is None else torch.as_tensor(value, dtype=dtype))

    def policy_action(self, obs):
        normalized = torch.clamp((obs - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        return torch.clamp(self.actor(normalized), -self.action_limit, self.action_limit)

    def forward(self, obs):
        action = self.policy_action(obs)
        if self.mirror_obs is not None:  # symmetric: the mean with the mirrored action for the mirrored observation
            mirrored = self.policy_action(obs[:, self.mirror_obs] * self.mirror_obs_sign)
            action = .5 * (action + mirrored[:, self.mirror_action] * self.mirror_action_sign)
        if self.locked is not None:
            action = task.lock_joints(action, self.locked)
        if self.trim is not None:  # the norm as phase_stop's (and g1_ctrl's gait_phase) reads it
            command = obs[:, self.command[0]:self.command[1]]
            standing = torch.sqrt((command * command).sum(dim=1, keepdim=True)) < task.PHASE_STOP_NORM
            action = torch.where(standing, torch.clamp(action - self.trim, -self.action_limit, self.action_limit),
                                 action)
        return action


def applied_action(action, config, raw):
    """The action the environment `raw` applies after its clip, for a (actions,) clipped policy output: with
    lock_wrists, the wrist pitch and yaw entries at 0 (g1_tensor_math.lock_joints, as the environment)."""
    if not config.get('lock_wrists'):
        return action
    return task.lock_joints(torch.from_numpy(np.asarray(action))[None], raw.wrist_mask)[0].numpy()


def command_inputs(joints, history, extra=()):
    """[start, end) of the newest frame's velocity_commands in the actor's input (history_observation layout; extra:
    the sizes of the terms after last_action)."""
    start, end = task.deploy_terms(joints, extra)[TERMS.index('velocity_commands')]
    positions = task.history_index(joints, history, extra)[history - 1][start:end]
    if positions != list(range(positions[0], positions[0] + end - start)):
        raise ValueError('Команда во входе актёра лежит не подряд')
    return positions[0], positions[0] + end - start


def mirror_maps(env):
    """(obs index, obs sign, action index, action sign) of symmetry.Mirror, numpy. Mirror resets the environment
    once (the home pose) to read the joint axes."""
    from .symmetry import Mirror
    mirror = Mirror(env)
    return tuple(t.numpy() for t in (mirror.obs_index, mirror.obs_sign, mirror.action_index, mirror.action_sign))


def reference_action(model, norm, obs, config, raw, trim=None, command=None, mirror=None):
    """What policy.onnx returns for one observation, built from the stage itself: policy.predict(normalize_obs(obs))
    clipped to ±action_limit (with mirror: averaged with the mirrored action of the mirrored observation), applied
    as the environment applies it (applied_action), then with trim the stand trim while the command is below the
    threshold."""
    limit = config['action_limit']
    act = lambda o: np.clip(model.predict(norm.normalize_obs(o), deterministic=True)[0], -limit, limit)
    action = act(obs)
    if mirror is not None:
        obs_index, obs_sign, action_index, action_sign = mirror
        action = .5 * (action + act(obs[obs_index] * obs_sign)[action_index] * action_sign)
    action = applied_action(action, config, raw)
    if trim is not None and np.linalg.norm(obs[command[0]:command[1]]) < task.PHASE_STOP_NORM:
        action = np.clip(action - trim, -limit, limit)
    return action


def stand_trim(stage_dir, seeds=TRIM_SEEDS, window=TRIM_WINDOW, limit_rad=TRIM_LIMIT):
    """The stand trim b (actions,) in action units and how it came out (dict): half the mean left-right asymmetry
    δ = π(o) − M_a·π(M_o·o) of the stage's deterministic policy (predict clipped to ±action_limit, the graph's Clip)
    over control steps window[0]..window[1] of standing episodes in its CPU environment (command 0, gait phase 0,
    heading hold off; driven by the policy's own action, applied as env.step applies it), 0 on the joints
    lock_wrists holds, and clipped per joint to ±limit_rad of PD target (±limit_rad / scale in action units)."""
    from .tasks.catalog import make_task
    stage = Path(stage_dir)
    identity = checkpoint_identity(read_json(stage / 'metadata.json'))
    config = {**stage_task_config(stage), 'heading_hold': False, 'max_episode_steps': window[1] + 1}
    limit = config['action_limit']
    model, norm = load_stage(stage)
    env = make_task(identity['task_id'], identity['robot_id'], config)
    deltas = []
    try:
        raw = env.unwrapped
        obs_index, obs_sign, action_index, action_sign = mirror_maps(env)
        act = lambda o: np.clip(model.predict(norm.normalize_obs(o), deterministic=True)[0], -limit, limit)
        for seed in seeds:
            obs, _ = env.reset(seed=seed, options={'command': [0., 0., 0.], 'phase_steps': 0})
            for step in range(window[1]):
                action = act(obs)
                if step >= window[0]:
                    deltas.append(action - act(obs[obs_index] * obs_sign)[action_index] * action_sign)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated:
                    raise ValueError(f'Этап падает стоя (seed {seed}, шаг {step + 1}): подстройку стоя не посчитать')
        names = [raw.model.joint(j).name for j in raw.joints]
        scale = np.asarray(raw.scale, dtype=float)
        locked = raw.wrist_mask.numpy() if config['lock_wrists'] else np.zeros(raw.model.nu, dtype=bool)
    finally:
        env.close()
        norm.close()
    half = .5 * np.mean(deltas, axis=0)
    half[locked] = 0.
    bound = limit_rad / scale
    trim = np.clip(half, -bound, bound)
    clipped = [names[i] for i in np.flatnonzero(np.abs(half) > bound)]
    return trim, dict(seeds=list(seeds), window_steps=list(window), limit_rad=float(limit_rad), joints=names,
                      b_rad=(trim * scale).tolist(), unclipped=half.tolist(), clipped_joints=clipped)


def deployed_policy(model, norm, actor_inputs, action_limit, locked=None, trim=None, command=None, mirror=None):
    policy = model.policy
    if not isinstance(policy.action_dist, DiagGaussianDistribution) or policy.squash_output:
        raise ValueError('Ожидалась политика PPO с гауссовым распределением без сжатия')
    extractor = policy.pi_features_extractor
    if actor_inputs < policy.observation_space.shape[0] and not (
            isinstance(extractor, ActorInputs) and extractor.actor_inputs == actor_inputs):
        raise ValueError('Актёр этапа видит не только входы робота')
    if not norm.norm_obs:
        raise ValueError('Этап без нормировки наблюдений')
    layers = [copy.deepcopy(layer) for layer in (*policy.mlp_extractor.policy_net, policy.action_net)]
    first = layers[0]
    if not isinstance(first, torch.nn.Linear):
        raise ValueError('Первый слой актёра — не линейный')
    # The actor's features are the observation with everything past actor_inputs zeroed: those columns never count.
    layers[0] = torch.nn.Linear(actor_inputs, first.out_features, bias=first.bias is not None)
    with torch.no_grad():
        layers[0].weight.copy_(first.weight[:, :actor_inputs])
        if first.bias is not None:
            layers[0].bias.copy_(first.bias)
    rms = norm.obs_rms
    std = np.sqrt(rms.var[:actor_inputs] + norm.epsilon)
    return DeployedPolicy(rms.mean[:actor_inputs], std, norm.clip_obs, torch.nn.Sequential(*layers),
                          action_limit, locked, trim, command, mirror).eval()


def write_onnx(module, path, actor_inputs, actions, properties):
    import onnx
    # The TorchScript exporter (dynamo=False): plain static ops at opset 17, which g1_ctrl's ONNX Runtime 1.22
    # loads, and no onnxscript needed. Torch marks it deprecated; that notice says nothing about this graph.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        torch.onnx.export(module, (torch.zeros(1, actor_inputs, dtype=torch.float32),), str(path),
                          input_names=['obs'], output_names=['actions'], opset_version=OPSET, dynamo=False,
                          do_constant_folding=True)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    inputs, outputs = graph.graph.input, graph.graph.output
    if ([v.name for v in inputs], shape(inputs[0])) != (['obs'], [1, actor_inputs]) or (
            outputs[0].name, shape(outputs[0])) != ('actions', [1, actions]):
        raise ValueError('Входы или выходы ONNX не совпали с ожидаемыми')
    onnx.helper.set_model_props(graph, {key: str(value) for key, value in properties.items()})
    onnx.save(graph, str(path))


def command_ranges(config, narrowed=None):
    """deploy.yaml's commands.base_velocity.ranges (lin_vel_x, lin_vel_y, ang_vel_z; g1_ctrl clamps the stick to
    them): the stage's training ranges, or narrowed = (x_min, x_max, y, yaw) for [x_min, x_max], ±y, ±yaw, each
    within the training range and with 0 inside (a centred stick must still mean stand)."""
    commands = config['commands']
    training = {'lin_vel_x': [commands['min_speed'], commands['max_speed']],
                'lin_vel_y': [-commands['lateral_speed'], commands['lateral_speed']],
                'ang_vel_z': [-commands['yaw_rate'], commands['yaw_rate']]}
    if narrowed is None:
        return training
    x_min, x_max, y, yaw = (float(v) for v in narrowed)
    wanted = {'lin_vel_x': [x_min, x_max], 'lin_vel_y': [-y, y], 'ang_vel_z': [-yaw, yaw]}
    for name, (low, high) in wanted.items():
        if not (math.isfinite(low) and math.isfinite(high) and low <= 0. <= high):
            raise ValueError(f'Диапазон команд {name} [{low:g}, {high:g}] должен включать 0')
        if low < training[name][0] or high > training[name][1]:
            raise ValueError(f'Диапазон команд {name} [{low:g}, {high:g}] шире обучения этапа '
                             f'[{training[name][0]:g}, {training[name][1]:g}]: только сужать')
    return wanted


def deploy_config(raw, config, narrowed=None):
    """deploy.yaml contents for the environment `raw` (G1VelocityEnv) under the stage's task config; narrowed:
    command_ranges' (x_min, x_max, y, yaw), or None for the training ranges."""
    nu = raw.model.nu
    floats = lambda values: [float(v) for v in values]
    # The PD target's bounds: the joint ranges, with knee_floor the knees' lower one at the floor.
    ranges = np.stack((raw.target_low, raw.target_high), axis=1)
    if not (raw.model.jnt_limited[raw.joints].all() and (ranges[:, 0] < ranges[:, 1]).all()):
        raise ValueError('У суставов модели нет пределов для clip')
    sizes = [end - start for start, end in task.deploy_terms(nu, raw.extra_sizes)]
    names = (*TERMS, *(name for name, _ in raw.extra_terms))
    params = {'velocity_commands': {'command_name': 'base_velocity'},
              'gait_phase': {'period': float(config['gait']['period'])}}
    if config.get('height_command'):  # g1_ctrl-height.patch: c from the right stick (height_params)
        params[task.HEIGHT_TERM] = height_params(config)
    return {
        'joint_ids_map': list(range(nu)),  # the model's actuators are in the SDK order, as controller.json
        'step_dt': float(raw.dt),
        'stiffness': floats(raw.kp),
        'damping': floats(raw.kd),
        'default_joint_pos': floats(raw.home),
        'commands': {'base_velocity': {'ranges': {**command_ranges(config, narrowed), 'heading': None}}},
        'actions': {'JointPositionAction': {
            'clip': [floats(pair) for pair in ranges],
            'joint_names': ['.*'],
            'scale': floats(raw.scale),
            'offset': floats(raw.home),
            'joint_ids': None}},
        'observations': {name: {'params': params.get(name, {}), 'clip': None, 'scale': [1.] * size,
                                'history_length': int(config['history'])}
                         for name, size in zip(names, sizes)},
    }


def height_params(config):
    """deploy.yaml params of the height_command term for scripts/sim2sim/patches/g1_ctrl-height.patch: c = nominal +
    (nominal − lowest)·min(0, ry) from the right stick's vertical axis (smoothed as the left stick, 0.03), at least
    walking_lowest while the command's norm is at least walking_norm (phase_stop's threshold); the term reads
    c − nominal. The stage's crouch_min: 0.78 + 0.14·min(0, ry), c ≥ 0.70 walking, with the working values."""
    lowest, walking = config['commands']['crouch_min']
    return {'nominal': float(task.BASE_HEIGHT), 'lowest': float(lowest), 'walking_lowest': float(walking),
            'walking_norm': float(task.PHASE_STOP_NORM)}


def write_yaml(value, path, header):
    """header: comment text, one '# ' line per line of it."""
    import yaml

    class Dumper(yaml.SafeDumper):
        def ignore_aliases(self, data):  # equal lists stay written out, no &anchors
            return True

    text = yaml.dump(value, Dumper=Dumper, sort_keys=False, default_flow_style=None, width=110)
    comment = ''.join(f'# {line}\n' for line in header.splitlines())
    Path(path).write_text(f'{comment}{text}', encoding='utf-8')


def trim_comment(trim, command):
    """deploy.yaml comment lines about the stand trim baked into policy.onnx (g1_ctrl ignores comments)."""
    values = [f'{v:.6f}' for v in trim]
    rows = [', '.join(values[k:k + 10]) for k in range(0, len(values), 10)]
    return '\n'.join([
        f'stand trim ({TRIM_FILE}): policy.onnx returns clip(actions - b) while the norm of obs[{command[0]}:'
        f'{command[1]}]',
        f'(the newest velocity_commands) is below {task.PHASE_STOP_NORM:g}, as gait_phase stops; nothing to set in '
        f'g1_ctrl. b, action units, joint order:',
        *(f'  {row}{"," if k < len(rows) - 1 else ""}' for k, row in enumerate(rows))])


def resolve_stage(path):
    """A stage dir as given, or an experiment dir's latest stage."""
    path = Path(path)
    if (path / 'metadata.json').is_file():
        return path
    stages = list_stages(path)
    if not stages:
        raise ValueError(f'Нет этапов: {path}')
    return stages[-1]


def export(stage_dir, out_dir, trim=None, trim_info=None, symmetric=False, commands=None):
    """Writes policy.onnx and deploy.yaml for stage_dir into out_dir; returns paths, sizes and warnings.
    trim: (actions,) stand trim b in action units (stand_trim; its info dict goes into stand_trim.json too), baked
    into the graph; symmetric: export ½(π(o) + M_a·π(M_o·o)) instead of π (experiments). Without either the graph is
    the plain one. commands: (x_min, x_max, y, yaw), deploy.yaml's command ranges narrowed within the training ones
    (command_ranges; noted in its header), or None for the training ranges; the graph is the same either way."""
    stage, out_dir = Path(stage_dir), Path(out_dir)
    metadata = read_json(stage / 'metadata.json')
    identity = checkpoint_identity(metadata)
    if (identity['robot_id'], identity['task_id']) != ('g1_29dof', 'g1_velocity'):
        raise ValueError('Выгрузка в g1_ctrl — только для G1 29 DoF (g1_velocity)')
    config = stage_task_config(stage)
    if not config['real_robot']:
        raise ValueError('Выгрузка в g1_ctrl — только для профиля real_robot (наблюдения как в deploy.yaml)')
    if trim is not None and symmetric:
        raise ValueError('Подстройка стоя и симметризация вместе не выгружаются: выберите одно')
    narrowed = None if commands is None else command_ranges(config, commands)  # refused before anything is written
    model, norm = load_stage(stage)
    try:
        env = norm.venv.envs[0]
        raw = env.unwrapped
        actor_inputs, actions = raw.actor_inputs, raw.model.nu
        if raw.frame_size * config['history'] != actor_inputs or raw.frame_size != sum(
                end - start for start, end in task.deploy_terms(raw.model.nu, raw.extra_sizes)):
            raise ValueError('Раскладка входа актёра не совпадает с термами deploy.yaml')
        if abs(raw.dt - identity['control_dt']) > 1e-12:
            raise ValueError('Шаг управления этапа не совпадает с controller.json')
        command = None
        if trim is not None:
            trim = np.asarray(trim, dtype=float)
            if trim.shape != (actions,) or not np.isfinite(trim).all():
                raise ValueError('Подстройка стоя: нужно по одному конечному числу на сустав')
            command = command_inputs(actions, config['history'], raw.extra_sizes)
        mirror = None
        if symmetric:  # the actor's inputs mirror among themselves (history frames), never into the critic extras
            obs_index, obs_sign, action_index, action_sign = mirror_maps(env)
            if obs_index[:actor_inputs].max() >= actor_inputs:
                raise ValueError('Зеркало входа актёра задевает входы критика')
            mirror = (obs_index[:actor_inputs], obs_sign[:actor_inputs], action_index, action_sign)
        module = deployed_policy(model, norm, actor_inputs, config['action_limit'],
                                 raw.wrist_mask.numpy() if config['lock_wrists'] else None, trim, command, mirror)
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path, yaml_path, trim_path = out_dir / 'policy.onnx', out_dir / 'deploy.yaml', out_dir / TRIM_FILE
        properties = {
            'stage': stage.resolve(), 'steps': metadata.get('steps'), 'robot_id': identity['robot_id'],
            'observation_schema': identity['observation_schema'], 'action_schema': identity['action_schema']}
        if trim is not None:
            properties.update(stand_trim=json.dumps([float(v) for v in trim]), stand_trim_command=json.dumps(command))
        if symmetric:
            properties['symmetric'] = 'true'
        if config.get('foot_collision'):  # training physics only: the graph and deploy.yaml are the same
            properties.update(foot_collision='true', model_revision=identity['model_revision'])
        if config.get('hand_collision'):  # likewise
            properties.update(hand_collision='true', model_revision=identity['model_revision'])
        if config.get('arm_blind'):  # in the normalization's numbers only: the graph and deploy.yaml are the same
            properties['arm_blind'] = 'true'
        for name, _ in raw.extra_terms:  # observation terms g1_ctrl knows only with g1_ctrl-height.patch
            properties[name] = 'true'
        write_onnx(module, onnx_path, actor_inputs, actions, properties)
        header = f'humanoid_sim.export_g1: {stage.resolve()} ({metadata.get("steps")} steps)'
        if trim is not None:
            header += '\n' + trim_comment(trim, command)
        if symmetric:
            header += '\nsymmetric: policy.onnx returns (pi(o) + M_a pi(M_o o)) / 2 (experiments only)'
        if config.get('foot_collision'):
            header += ('\nfoot_collision: trained with the feet colliding with each other (Unitree\'s foot capsules); '
                       'nothing to set in g1_ctrl')
        if config.get('hand_collision'):
            header += ('\nhand_collision: trained with the hands colliding with the floor and the body (Unitree\'s hand '
                       'capsules); nothing to set in g1_ctrl')
        if config.get('arm_blind'):
            header += ('\narm_blind: the arm joints\' joint_pos_rel, joint_vel_rel and last_action (every frame) are '
                       'normalized to constants inside policy.onnx\n(the policy does not see its arms); nothing to set '
                       'in g1_ctrl')
        if raw.extra_terms:
            header += ('\n' + ', '.join(name for name, _ in raw.extra_terms) + ': observation terms of '
                       'scripts/sim2sim/patches/g1_ctrl-height.patch (stock g1_ctrl refuses them); height_command '
                       '= c - 0.78 from the right stick down, upper_body_command = zeros')
        if narrowed is not None:
            training = command_ranges(config)
            header += '\ncommand ranges narrowed (--command-ranges) from the training ones:' + ''.join(
                f'\n  {name} [{training[name][0]:g}, {training[name][1]:g}] -> [{low:g}, {high:g}]'
                for name, (low, high) in narrowed.items())
        write_yaml(deploy_config(raw, config, commands), yaml_path, header)
        if trim is not None:
            write_json(trim_path, {
                'stage': str(stage.resolve()), 'steps': metadata.get('steps'),
                'rule': f'policy.onnx: clip(actions - b, +-action_limit) while the norm of obs[{command[0]}:'
                        f'{command[1]}] (the newest velocity_commands) is below {task.PHASE_STOP_NORM:g}',
                'threshold': task.PHASE_STOP_NORM, 'command_inputs': list(command), 'b': [float(v) for v in trim],
                **(trim_info or {})})
        else:  # a trim file left by an earlier export into this folder would describe a graph no longer there
            trim_path.unlink(missing_ok=True)
    finally:
        norm.close()
    notes = []
    if not config.get('phase_stop'):
        notes.append('В task_config этапа нет phase_stop: true — на роботе g1_ctrl обнуляет фазу шага при норме '
                        'команды < 0,1, а политика училась с фазой, которая тикает всегда.')
    if raw.extra_terms:
        notes.append('Во входе термы ' + ', '.join(name for name, _ in raw.extra_terms) + ': штатный g1_ctrl их не '
                     'знает, нужен патч scripts/sim2sim/patches/g1_ctrl-height.patch (высота — правый стик вниз).')
    return {'stage': stage, 'onnx': onnx_path, 'deploy': yaml_path, 'actor_inputs': actor_inputs,
            'actions': actions, 'history': config['history'], 'warnings': notes,
            'trim': None if trim is None else trim_path, 'symmetric': symmetric,
            'commands': command_ranges(config, commands)}


def onnx_properties(onnx_path):
    import onnx
    return {p.key: p.value for p in onnx.load(str(onnx_path)).metadata_props}


def deploy_target(action, deploy):
    """g1_ctrl's motor target for an action (joint_actions.h JointAction::process_actions): scale·a + offset, clamped
    per joint to clip[i] = [lo, hi], from a deploy.yaml's contents."""
    term = deploy['actions']['JointPositionAction']
    clip = np.asarray(term['clip'], dtype=float)
    target = np.asarray(action, dtype=float) * np.asarray(term['scale']) + np.asarray(term['offset'])
    return np.clip(target, clip[:, 0], clip[:, 1])


def check(stage_dir, onnx_path, samples=200, stride=5, seed_start=81000):
    """onnxruntime(policy.onnx) against policy.predict(normalize_obs(obs), deterministic) clipped to ±action_limit
    and applied as the environment applies it (applied_action: predict knows nothing of lock_wrists), with the stand
    trim or the symmetrisation the graph's properties declare (reference_action), on `samples` observations of the
    stage's CPU environment (every `stride`-th step of episodes under CHECK_COMMANDS, driven by the expected
    actions). On the same samples, g1_ctrl's motor target for the ONNX action (deploy_target, the deploy.yaml next to
    policy.onnx) against the environment's PD target for the action it applies (_target: joint ranges, knee_floor).
    With task_config height_command the second half of each command's samples comes under a crouch command (the
    stage's crouch_min, standing or walking, set mid-episode as the stick would)."""
    import onnxruntime
    import yaml
    from .tasks.catalog import make_task
    stage = Path(stage_dir)
    identity = checkpoint_identity(read_json(stage / 'metadata.json'))
    config = stage_task_config(stage)
    properties = onnx_properties(onnx_path)
    deploy_path = Path(onnx_path).with_name('deploy.yaml')
    if not deploy_path.is_file():
        raise ValueError(f'Рядом с policy.onnx нет deploy.yaml: {deploy_path}')
    deploy = yaml.safe_load(deploy_path.read_text(encoding='utf-8'))
    trim = np.array(json.loads(properties['stand_trim'])) if 'stand_trim' in properties else None
    command = json.loads(properties['stand_trim_command']) if trim is not None else None
    per_command = math.ceil(samples / len(CHECK_COMMANDS))
    session = onnxruntime.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    model, norm = load_stage(stage)
    worst, worst_target, count, resets, crouched = 0., 0., 0, 0, 0
    try:
        for number, command_now in enumerate(CHECK_COMMANDS):
            env = make_task(identity['task_id'], identity['robot_id'],
                            {**config, 'max_episode_steps': per_command * stride + 1})
            raw = env.unwrapped
            actor_inputs = raw.actor_inputs
            mirror = mirror_maps(env) if properties.get('symmetric') == 'true' else None
            moving = bool(task.command_moving(torch.tensor([command_now]))[0])
            crouch = config['commands']['crouch_min'][int(moving)] if config.get('height_command') else None
            options = {'command': list(command_now)}
            try:
                obs, _ = env.reset(seed=seed_start + number, options=options)
                for step in range(per_command * stride):
                    if crouch is not None and step == (per_command // 2) * stride:  # the stick goes down
                        options['height'] = crouch
                        raw.set_height(crouch)
                        obs = raw._observation()
                    expected = reference_action(model, norm, obs, config, raw, trim, command, mirror)
                    if step % stride == 0 and count < samples:
                        got = session.run(['actions'], {'obs': obs[None, :actor_inputs].astype(np.float32)})[0][0]
                        worst = max(worst, float(np.abs(got - expected).max()))
                        target = raw._target(raw.applied_action(expected))  # what env.step's PD follows
                        worst_target = max(worst_target, float(np.abs(deploy_target(got, deploy) - target).max()))
                        count += 1
                        crouched += raw.height_command and raw.height < task.BASE_HEIGHT
                    obs, _, terminated, truncated, _ = env.step(expected)
                    if terminated or truncated:  # a fall: go on under the same command (and height)
                        resets += 1
                        obs, _ = env.reset(seed=seed_start + 1000 * resets + number, options=options)
            finally:
                env.close()
    finally:
        norm.close()
    return {'samples': count, 'max_abs_diff': worst, 'max_target_diff': worst_target, 'tolerance': TOLERANCE,
            'resets': resets, 'crouched': int(crouched),
            'ok': count == samples and worst < TOLERANCE and worst_target < TOLERANCE}


def is_export(path):
    return (Path(path) / 'policy.onnx').is_file()


class ExportedPolicy:
    """An export folder's policy.onnx acting in the CPU environment of the stage it came from, as both the policy and
    the normalizer of load_stage: predict(obs) feeds onnxruntime the raw observation's actor inputs (normalization,
    clips, lock_wrists and the stand trim are all inside the graph; the critic extras are dropped) and returns the
    action g1_ctrl would send; normalize_obs is the identity. So code written for (model, norm) judges exactly what
    goes to the robot. stage: the stage (its environment, task config and σ), from the graph's properties."""

    def __init__(self, folder):
        import onnxruntime
        self.folder = Path(folder)
        path = self.folder / 'policy.onnx'
        if not path.is_file():
            raise ValueError(f'Нет policy.onnx: {self.folder}')
        properties = onnx_properties(path)
        if 'stage' not in properties:
            raise ValueError(f'policy.onnx без свойства stage — не выгрузка export_g1: {path}')
        self.stage = Path(properties['stage'])
        metadata = read_json(self.stage / 'metadata.json')
        if metadata is None:
            raise ValueError(f'Этап выгрузки не найден: {self.stage}')
        identity = checkpoint_identity(metadata)
        if (identity['observation_schema'], identity['action_schema']) != (
                properties.get('observation_schema'), properties.get('action_schema')):
            raise ValueError('Разъём этапа не совпадает с разъёмом выгрузки')
        self.steps = int(properties['steps'])
        self.trim = np.array(json.loads(properties['stand_trim'])) if 'stand_trim' in properties else None
        self.symmetric = properties.get('symmetric') == 'true'
        options = onnxruntime.SessionOptions()  # one observation at a time; probes run many of these in parallel
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.actor_inputs = self.session.get_inputs()[0].shape[1]

    def normalize_obs(self, obs):
        return obs

    def predict(self, obs, deterministic=True):
        """(actions, None) for one observation (obs,) or a batch (n, obs); the graph takes [1, A], so row by row.
        The graph has no noise: deterministic is always so."""
        obs = np.asarray(obs, dtype=np.float32)
        rows = obs.reshape(-1, obs.shape[-1])[:, :self.actor_inputs]
        actions = np.concatenate([self.session.run(['actions'], {'obs': np.ascontiguousarray(row[None])})[0]
                                  for row in rows])
        return actions.reshape(*obs.shape[:-1], actions.shape[-1]), None

    def close(self):
        pass


def load_policy(path):
    """(policy, normalizer, stage) for a stage dir (load_stage) or an export folder (ExportedPolicy as both)."""
    if is_export(path):
        exported = ExportedPolicy(path)
        return exported, exported, exported.stage
    model, norm = load_stage(path)
    return model, norm, Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', type=Path, help='папка этапа (или эксперимента — тогда его последний этап)')
    parser.add_argument('--out', type=Path, required=True, help='куда положить policy.onnx и deploy.yaml')
    parser.add_argument('--samples', type=int, default=200, help='наблюдений в проверке совпадения (0 — без неё)')
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--stand-trim', action='store_true',
                         help='подстройка стоя: действие − b при норме команды < 0,1 (b — половина средней '
                              'несимметрии политики стоя, не больше 0,052 рад на сустав)')
    variant.add_argument('--symmetric', action='store_true',
                         help='только для опытов: ½(π(o) + M·π(M·o)) — ломает выбор ноги при толчке стоя')
    parser.add_argument('--command-ranges', type=float, nargs=4, metavar=('X_MIN', 'X_MAX', 'Y', 'YAW'),
                        help='сузить диапазоны команд в deploy.yaml (g1_ctrl зажимает в них стик): вперёд '
                             'X_MIN…X_MAX м/с, вбок ±Y м/с, поворот ±YAW рад/с, только внутри диапазонов обучения '
                             'этапа (до проб на полу — как этап 1: -0.5 1 0.5 1)')
    args = parser.parse_args(argv)
    stage = resolve_stage(args.stage)
    trim, info = stand_trim(stage) if args.stand_trim else (None, None)
    result = export(stage, args.out, trim, info, args.symmetric, args.command_ranges)
    print(f'Этап: {stage}')
    print(f'policy.onnx: {result["onnx"]} (вход obs [1, {result["actor_inputs"]}], выход actions '
          f'[1, {result["actions"]}])')
    ranges = ', '.join(f'{name} [{low:g}, {high:g}]' for name, (low, high) in result['commands'].items())
    narrowed = ', сужены --command-ranges' if args.command_ranges else ''
    print(f'deploy.yaml: {result["deploy"]} (история {result["history"]} кадров; команды {ranges}{narrowed})')
    if trim is not None:
        worst = int(np.argmax(np.abs(info['b_rad'])))
        clipped = ', '.join(info['clipped_joints']) or 'нет'
        print(f'Подстройка стоя: {result["trim"]} — max |b| {abs(info["b_rad"][worst]):.4f} рад '
              f'({info["joints"][worst]}), обрезаны до ±{info["limit_rad"]:g} рад: {clipped}')
    if args.symmetric:
        print('Симметризованная политика ½(π(o) + M·π(M·o)) — только для опытов')
    for warning in result['warnings']:
        print(f'Внимание: {warning}')
    if args.samples > 0:
        verdict = check(stage, result['onnx'], samples=args.samples)
        same = ' (с той же подстройкой стоя)' if trim is not None else ' (симметризованным)' if args.symmetric else ''
        crouched = f' (в приседе {verdict["crouched"]})' if verdict['crouched'] else ''
        print(f'Сверка с policy.predict{same}: {verdict["samples"]} наблюдений{crouched}, max |разница| = '
              f'{verdict["max_abs_diff"]:.2e}, цели моторов по deploy.yaml {verdict["max_target_diff"]:.2e} '
              f'(порог {verdict["tolerance"]:g}) — {"совпадает" if verdict["ok"] else "НЕ СОВПАДАЕТ"}')
        if not verdict['ok']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
