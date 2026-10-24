"""Local browser dashboard. Physics stays in MuJoCo; geometry is drawn in WebGL."""
import argparse
import collections
import json
import mimetypes
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote, parse_qs

import gymnasium as gym
import mujoco
import numpy as np
import torch

from . import i18n
from .core import DEFAULT_RUN, ROOT, RunLock, list_stages, load_stage, read_json
from .lab import Laboratory
from .lab_api import LabJobs

WEB = Path(__file__).with_name('web')
SPEED_WINDOW = 2.0
# Gymnasium Humanoid-v5 reward terms reported in info (G1 labels come from its task parameters).
HUMANOID_REWARD_LABELS = dict(reward_forward='Продвижение вперёд', reward_survive='Награда за то, что стоит',
                              reward_ctrl='Штраф за усилие моторов', reward_contact='Штраф за силы контакта')


class SpeedTrack:
    """Mean horizontal speed of a body over the last SPEED_WINDOW seconds of simulation time."""

    def __init__(self):
        self.points = collections.deque()

    def clear(self):
        self.points.clear()

    def add(self, seconds, x, y):
        self.points.append((seconds, x, y))
        while seconds - self.points[0][0] > SPEED_WINDOW:
            self.points.popleft()

    def speed(self):
        if len(self.points) < 2 or self.points[-1][0] - self.points[0][0] < SPEED_WINDOW / 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = self.points[0], self.points[-1]
        return float(np.hypot(x1 - x0, y1 - y0) / (t1 - t0))


def run_task(run_dir):
    """Run configuration, checkpoint identity and TaskSpec of a run directory."""
    from .tasks.catalog import registries
    from .tasks.identity import task_identity
    config = read_json(Path(run_dir) / 'run_config.json', {})
    identity = task_identity(config.get('robot_id', 'humanoid'), config.get('task_id', 'humanoid_v5'),
                             config.get('task_config') or {})
    robots, tasks = registries()
    return config, identity, tasks.resolve(identity['task_id'], robots.get(identity['robot_id']))


def shown_geoms(model):
    """Geoms the browser draws, in model order: all but G1 foot_collision's capsules (hidden group 3, contact bit
    FOOT_COLLISION_BIT) and hand_collision's (group 3, named as HAND_CAPSULES). They sit between the other geoms of
    their links, so a race lane with them and one without would otherwise shift every later transform onto the wrong
    mesh."""
    from .tasks.g1_velocity import FOOT_COLLISION_BIT, HAND_CAPSULES
    hands = {f'{body.split("_")[0]}_hand_collision' for body in HAND_CAPSULES}
    return [i for i in range(model.ngeom) if not model.geom_group[i] == 3 or not (
        model.geom_contype[i] == FOOT_COLLISION_BIT and model.geom_conaffinity[i] == FOOT_COLLISION_BIT
        or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) in hands)]


def model_geometry(model):
    result = []
    for index, i in enumerate(shown_geoms(model)):
        geom = dict(id=index, name=model.geom(i).name, type=int(model.geom_type[i]),
                    size=model.geom_size[i].tolist(), color=model.geom_rgba[i].tolist())
        if geom['type'] == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = model.geom_dataid[i]
            va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            geom.update(vertices=model.mesh_vert[va:va+vn].reshape(-1).tolist(),
                        faces=model.mesh_face[fa:fa+fn].reshape(-1).tolist())
        result.append(geom)
    return result


def geom_transforms(model, data):
    """[x, y, z, qx, qy, qz, qw] per shown geom (shown_geoms), the layout the browser applies."""
    mujoco.mj_forward(model, data)
    transforms = []
    quat = np.zeros(4)
    for i in shown_geoms(model):
        mujoco.mju_mat2Quat(quat, data.geom_xmat[i])
        transforms.append([*data.geom_xpos[i].tolist(), float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
    return transforms


class _Looping:
    """Background tick loop that runs only while a browser stream is subscribed (client_count > 0)."""

    def _init_loop(self):
        self.speed = 1.0
        self.paused = False
        self.commands = queue.Queue()
        # A condition, so streams wake on each new frame instead of polling (Windows rounds short waits up).
        self.frame_lock = threading.Condition()
        self.stop = threading.Event()
        self.worker = None
        self.client_count = None
        self.last_access = time.monotonic()
        self.last_tick = time.monotonic()
        self.error = None
        self.seq = 0
        self._frame = {}
        self._json = {}

    def frame(self):
        with self.frame_lock:
            return self._frame

    def frame_json(self, lang='ru', localize=None):
        """(seq, JSON) of the latest frame, serialized once per language however many browsers stream it.
        localize(frame) gives the frame in `lang` (names and labels, humanoid_sim/i18n.py)."""
        with self.frame_lock:
            seq = self._frame.get('seq')
            if self._json.get(lang, (None,))[0] != seq:
                frame = self._frame if localize is None else localize(self._frame)
                self._json[lang] = (seq, json.dumps(frame, allow_nan=False))
            return self._json[lang]

    def start(self):
        def loop():
            # Ticks follow a deadline: Windows rounds short waits up to ~15 ms, so sleeping "the rest of
            # this tick" ran slower than real time. Late ticks run back to back to catch up.
            due = time.monotonic()
            try:
                while not self.stop.is_set():
                    if self.client_count == 0:
                        self.last_tick = due = time.monotonic()
                        self.stop.wait(.1)
                        continue
                    self.tick()
                    due += self.dt / self.speed
                    delay = due - time.monotonic()
                    if delay > 0:
                        self.stop.wait(delay)
                    elif delay < -.25:  # overloaded: drop the backlog instead of racing to catch up
                        due = time.monotonic()
            except Exception as exc:
                self.error = str(exc)
                self._publish()
            finally:
                self._release()
        self.worker = threading.Thread(target=loop, name=type(self).__name__.lower(), daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=10)
        else:
            self._release()


class Playback(_Looping):
    def __init__(self, run_dir):
        torch.set_num_threads(2)
        self.run_dir = Path(run_dir)
        from .tasks.catalog import make_task
        config, self.identity, self.task = run_task(self.run_dir)
        self.view = self.task.playback
        self.dt = self.identity['control_dt']
        self.env = make_task(self.identity['task_id'], self.identity['robot_id'], config.get('task_config') or {})
        self._init_loop()
        self.policy = self.norm = None
        self.stage = None
        self.follow = True
        self.episode = 0
        self.steps = 0
        self.distance = 0.0
        self.track = SpeedTrack()
        self.done_until = None
        self.tour = False
        self.stage_seconds = 10
        self.stage_elapsed = 0.0
        self.tour_mode = 'time'
        self.full_fall = False
        self.fall_seconds = 0.0
        self.landed_seconds = 0.0
        self.stages = list_stages(self.run_dir)
        self.poll_at = 0.0
        self.obs, self.initial = self.env.reset(seed=71000)
        self.info = self.initial
        if self.stages:
            self._load(self.stages[-1])
        self._publish()

    def _load(self, path):
        from .checkpoint_schema import validate_resume
        validate_resume(read_json(path / 'metadata.json'), self.identity)
        policy, norm = load_stage(path)
        from .tasks.catalog import make_task
        from .core import stage_task_config
        try:
            env = make_task(self.identity['task_id'], self.identity['robot_id'], stage_task_config(path))
        except BaseException:
            norm.close()
            raise
        self.env.close()
        self.env = env
        if self.norm is not None:
            self.norm.close()
        self.policy, self.norm, self.stage = policy, norm, path
        self.stage_elapsed = 0.0
        self._reset()

    def _reset(self):
        # Same five starting states for fair visual comparison across stages.
        self.obs, self.initial = self.env.reset(seed=71000 + self.episode % 5)
        self.info = self.initial
        self.steps = 0
        self.distance = 0.0
        self.track.clear()
        self.done_until = None
        self.fall_seconds = self.landed_seconds = 0.0
        self.time_limit = False

    def geometry(self):
        return model_geometry(self.env.unwrapped.model)

    def submit(self, body):
        if not isinstance(body, dict):
            raise ValueError('Ожидается объект команды')
        command = body.get('command')
        if command == 'stage':
            step = body.get('step')
            if type(step) is not int or step not in [int(p.name[6:]) for p in list_stages(self.run_dir)]:
                raise ValueError('Такого этапа нет')
        elif command == 'pause':
            if type(body.get('value')) is not bool:
                raise ValueError('Пауза должна быть true или false')
        elif command == 'speed':
            if type(body.get('value')) not in (int, float) or body['value'] not in (0.5, 1, 2):
                raise ValueError('Допустимая скорость: 0.5, 1, 2')
        elif command == 'tour':
            if type(body.get('seconds')) not in (int, float) or not 1 <= body['seconds'] <= 600:
                raise ValueError('Время этапа: от 1 до 600 секунд')
            if body.get('mode') not in ('time', 'attempt') or type(body.get('full_fall')) is not bool:
                raise ValueError('Некорректный режим просмотра')
            if not list_stages(self.run_dir):
                raise ValueError('Пока нет сохранённых этапов')
        elif command not in ('latest', 'reset'):
            raise ValueError('Неизвестная команда')
        self.commands.put(body.copy())

    def tick(self):
        now = time.monotonic()
        delta = min(0.25, now - self.last_tick)
        self.last_tick = now
        if now - self.poll_at > 1:
            self.stages = list_stages(self.run_dir)
            self.poll_at = now
        while not self.commands.empty():
            body = self.commands.get_nowait()
            command = body['command']
            if command == 'stage':
                self.tour = False
                self.follow = False
                self.episode = 0
                self._load(self.run_dir / f'stage_{body["step"]:012d}')
            elif command == 'latest':
                self.tour = False
                self.follow = True
                self.stages = list_stages(self.run_dir)
                if self.stages:
                    self.episode = 0
                    self._load(self.stages[-1])
            elif command == 'pause':
                self.paused = body['value']
            elif command == 'speed':
                self.speed = body['value']
            elif command == 'reset':
                self._reset()
            elif command == 'tour':
                self.tour, self.follow, self.paused = True, False, False
                self.stage_seconds, self.tour_mode, self.full_fall = body['seconds'], body['mode'], body['full_fall']
                self.episode = 0
                self._load(self.stages[0])
        if self.policy is None and self.stages:
            self._load(self.stages[-1])
        if not self.paused and self.policy is not None:
            self.stage_elapsed += delta
            if self.done_until is not None:
                # Continue physical falling with relaxed motors after Gym's early termination.
                show_fall = self.full_fall and not self.time_limit and self.view.fall_height is not None
                ready = not show_fall and self.fall_seconds >= 0.6
                if not show_fall:
                    self.fall_seconds += self.dt
                if show_fall:
                    raw = self.env.unwrapped
                    raw.data.ctrl[:] = 0
                    mujoco.mj_step(raw.model, raw.data, nstep=raw.frame_skip)
                    self.fall_seconds += raw.dt
                    if self.view.focus(raw.model, raw.data)[2] < self.view.fall_height:
                        self.landed_seconds += raw.dt
                    ready = self.landed_seconds >= 2 or self.fall_seconds >= 8
                if ready:
                    if self.tour and (self.tour_mode == 'attempt' or self.stage_elapsed >= self.stage_seconds):
                        self._next_stage()
                        self._publish()
                        return
                    self.episode += 1
                    if self.follow and self.stages and self.stage != self.stages[-1]:
                        self._load(self.stages[-1])
                    else:
                        self._reset()
            else:
                if self.tour and self.stage_elapsed >= self.stage_seconds:
                    self._next_stage()
                    self._publish()
                    return
                action = self.policy.predict(self.norm.normalize_obs(self.obs), deterministic=True)[0]
                self.obs, _, terminated, truncated, info = self.env.step(action)
                self.steps += 1
                self.info = info
                self._track()
                if terminated or truncated:
                    self.time_limit = bool(truncated and not terminated)
                    self.done_until = now + 0.6 / self.speed
        self._publish()

    def _next_stage(self):
        index = self.stages.index(self.stage)
        if index + 1 < len(self.stages):
            self.episode = 0
            self._load(self.stages[index + 1])
        else:
            self.tour = False
            self.paused = True

    def _track(self):
        if self.view.follow_body is None:
            return
        raw = self.env.unwrapped
        x, y, _ = self.view.focus(raw.model, raw.data)
        self.track.add(self.steps * self.dt, x, y)

    def _publish(self):
        model, data = self.env.unwrapped.model, self.env.unwrapped.data
        command = getattr(self.env.unwrapped, 'command', None)
        transforms = geom_transforms(model, data)
        self.seq += 1
        metrics = dict(forward_m=None, distance_m=None)
        if self.task.evaluation_metrics is not None:
            metrics.update(self.task.evaluation_metrics(self.initial, self.info, self.done_until is not None))
        focus = self.view.focus(model, data)
        frame = {'seq': self.seq, 'stage': int(self.stage.name[6:]) if self.stage else None,
                 'paused': self.paused, 'follow': self.follow, 'speed': self.speed,
                 'tour': self.tour, 'stage_remaining': max(0, self.stage_seconds - self.stage_elapsed),
                 'episode_seconds': self.steps * self.dt,
                 'robot_id': self.identity['robot_id'], 'task_id': self.identity['task_id'],
                 'forward_m': metrics['forward_m'],
                 'distance_m': metrics['distance_m'],
                 'height_m': focus[2] if self.view.follow_body else None,
                 'speed_mps': self.track.speed(),
                 'command_mps': float(np.hypot(command[0], command[1])) if command is not None else None,
                 'view': self.view.describe(),
                 'episode': self.episode + 1, 'geoms': transforms,
                 'torso': focus,
                 'ended': self.done_until is not None, 'error': self.error}
        with self.frame_lock:
            self._frame = frame
            self.frame_lock.notify_all()

    def catalog(self):
        stages = []
        for path in list_stages(self.run_dir):
            result = read_json(path / 'evaluation.json', {})
            stages.append({'step': int(path.name[6:]),
                           'reward': result.get('mean_return'), 'seconds': result.get('mean_seconds'),
                           'distance': result.get('mean_forward_m')})
        status = read_json(self.run_dir / 'status.json', {'state': 'idle', 'steps': 0})
        try:
            with RunLock(self.run_dir):
                active = False
        except RuntimeError:
            active = True
        status['active'] = active
        if not active and status.get('state') in ('training', 'saving', 'starting'):
            status['state'] = 'interrupted'
        return {'stages': stages, 'training': status}

    def _release(self):
        self.env.close()
        if self.norm is not None:
            self.norm.close()


class Cadence:
    """Live step count with the definition of motion/metrics.py ("Проверить походку"):
    a step is a new ground support (contact normal force >= 5 N) lasting at least 40 ms.
    The first landing after reset is not a step.
    """

    FORCE = 5.
    SUSTAIN_SECONDS = .04

    def __init__(self, model, foot_names, dt):
        self.feet = [model.body(name).id for name in foot_names]
        self.need = max(1, round(self.SUSTAIN_SECONDS / dt))
        self.support = [0] * len(self.feet)  # consecutive loaded control steps
        self.landed = [False] * len(self.feet)
        self.initial = [True] * len(self.feet)
        self.count = 0
        self._force = np.zeros(6)

    def update(self, model, data):
        loaded = set()
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            foot = b if a == 0 else a if b == 0 else None
            if foot not in self.feet or contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(model, data, index, self._force)
            if self._force[0] >= self.FORCE:
                loaded.add(foot)
        for k, foot in enumerate(self.feet):
            if foot in loaded:
                self.support[k] += 1
                self.landed[k] = True
                if self.support[k] == self.need and not self.initial[k]:
                    self.count += 1
            else:
                self.support[k] = 0
                if self.landed[k]:
                    self.initial[k] = False


class RaceLane:
    """One stage of the race: its own policy, normalization and physics.

    A `follow` lane switches to the newest saved stage when the next race starts.
    """

    RELAX_STEPS = 100  # after a fall, motors relax and the body keeps falling for 2 s (G1/Humanoid dt)

    def __init__(self, stage, identity, view, feet, follow=False, experiment=None):
        from .core import stage_task_config
        from .tasks.catalog import make_task
        self.stage, self.steps, self.view, self.feet = stage, int(stage.name[6:]), view, feet
        self.follow, self.experiment = follow, experiment
        self.dt = identity['control_dt']
        self.policy, self.norm = load_stage(stage)
        try:
            self.env = make_task(identity['task_id'], identity['robot_id'], stage_task_config(stage))
        except BaseException:
            self.norm.close()
            raise

    def reset(self, seed, command):
        raw = self.env.unwrapped
        self.obs, _ = self.env.reset(seed=seed, options={'command': list(command)} if command is not None else None)
        self.start_x, self.start_y, _ = self.view.focus(raw.model, raw.data)
        self.track = SpeedTrack()
        self.cadence = Cadence(raw.model, self.feet, self.dt) if self.feet else None
        self.fell_at = None
        self.finished = False
        self.relax = 0
        self.seconds = 0.
        self.reward_sums = {}
        self.reward_steps = 0

    @property
    def done(self):
        return self.fell_at is not None or self.finished

    def step(self, seconds, command):
        raw = self.env.unwrapped
        if self.fell_at is not None:
            if self.relax < self.RELAX_STEPS:
                raw.data.ctrl[:] = 0
                mujoco.mj_step(raw.model, raw.data, nstep=raw.frame_skip)
                self.relax += 1
            return
        if self.finished:
            return
        if command is not None and tuple(raw.command) != tuple(command):
            raw.set_command(command)
            self.obs = raw._observation()  # the policy must see the new command (as in evaluation.runner)
        action = self.policy.predict(self.norm.normalize_obs(self.obs), deterministic=True)[0]
        self.obs, _, terminated, truncated, info = self.env.step(action)
        self.seconds = seconds
        # G1 reports reward_components; Gymnasium Humanoid reports reward_* entries.
        components = info.get('reward_components') or {k: v for k, v in info.items() if k.startswith('reward_')}
        for key, value in components.items():
            self.reward_sums[key] = self.reward_sums.get(key, 0.) + float(value)
        self.reward_steps += 1
        x, y, _ = self.view.focus(raw.model, raw.data)
        self.track.add(seconds, x, y)
        if self.cadence is not None:
            self.cadence.update(raw.model, raw.data)
        if terminated:
            self.fell_at = seconds
        elif truncated:
            self.finished = True

    def describe(self):
        raw = self.env.unwrapped
        focus = self.view.focus(raw.model, raw.data)
        cadence = (self.cadence.count / self.seconds * 60
                   if self.cadence is not None and self.seconds >= SPEED_WINDOW else None)
        return dict(experiment=self.experiment, stage=self.steps, follow=self.follow,
                    geoms=geom_transforms(raw.model, raw.data), torso=focus,
                    distance_m=float(focus[0] - self.start_x), sideways_m=float(focus[1] - self.start_y),
                    speed_mps=None if self.done else self.track.speed(),
                    steps_per_min=cadence, fell_at=self.fell_at, finished=self.finished,
                    rewards={k: v / self.reward_steps for k, v in self.reward_sums.items()} if self.reward_steps else {})

    def close(self):
        self.env.close()
        self.norm.close()


class Race(_Looping):
    """Lockstep comparison of up to ten saved stages of one robot/task: same seeds and command,
    separate physics. Lanes may come from different branches (experiments) of the research tree.

    Lanes are offset sideways only in the browser; every robot simulates on its own floor.
    Ten G1 lanes cost about 15 ms of CPU per 20 ms control step, so 1× stays real time.
    """

    MAX_LANES = 10
    SEEDS = 5
    HOLD_SECONDS = 2.

    def __init__(self, runs, focus=None, names=None):
        """runs: {experiment id: run dir} of one robot/task; focus: branch shown by default."""
        from .evaluation.scenarios import SCENARIOS
        from .motion.trajectory import ROLES
        torch.set_num_threads(2)
        self.runs = {identity: Path(run) for identity, run in runs.items()}
        self.names = dict(names or {})
        self.stages = {identity: list_stages(run) for identity, run in self.runs.items()}
        candidates = [identity for identity in self.runs if self.stages[identity]]
        if not candidates:
            raise ValueError('Пока нет сохранённых этапов')
        focus = focus if focus in candidates else candidates[-1]
        _, self.identity, task = run_task(self.runs[focus])
        self.view = task.playback
        if self.view.follow_body is None:
            raise ValueError('Забег доступен для шагающих роботов')
        self.dt = self.identity['control_dt']
        self.feet = ROLES.get(self.identity['robot_id'], (None, ()))[1]
        self.reward_labels = {**HUMANOID_REWARD_LABELS, **{p['key']: p['label'] for p in task.parameters
                                                           if p.get('group') == 'reward_weights'}}
        self.scenarios = SCENARIOS
        self._init_loop()
        self.lanes = []
        self.round = 0
        self.scenario = None
        self.poll_at = time.monotonic()
        # Default: first and middle stages of the focus branch, plus its newest one following training.
        stages = self.stages[focus]
        fixed = sorted({stages[0], stages[len(stages) // 2]} - {stages[-1]})
        self._select([(focus, stage, False) for stage in fixed] + [(focus, stages[-1], True)])
        self.supports_command = hasattr(self.lanes[0].env.unwrapped, 'set_command')
        self.scenario = 'walk' if self.supports_command else None
        self._restart()

    def _lane(self, experiment, stage, follow):
        return RaceLane(stage, self.identity, self.view, self.feet, follow, experiment)

    def _select(self, specs):
        """specs: (experiment id, stage path, follow newest) per lane."""
        lanes = []
        try:
            for spec in specs:
                lanes.append(self._lane(*spec))
        except BaseException:
            for lane in lanes:
                lane.close()
            raise
        for lane in self.lanes:
            lane.close()
        self.lanes = lanes

    def add_runs(self, runs, names):
        """Branches created after the race opened. Applied at once, not through the command queue: the loop
        only ticks while a race page streams, and lane requests are validated against self.runs before that."""
        self.names.update(names)
        self.runs = {**self.runs, **{k: Path(v) for k, v in runs.items()}}
        self.stages = {identity: list_stages(run) for identity, run in self.runs.items()}

    def _command(self, seconds):
        return None if self.scenario is None else self.scenarios[self.scenario].command_at(seconds)

    def _restart(self):
        for index, lane in enumerate(self.lanes):
            newest = self.stages[lane.experiment][-1]
            if lane.follow and lane.stage != newest:
                replacement = self._lane(lane.experiment, newest, True)
                lane.close()
                self.lanes[index] = replacement
        self.ticks = self.hold = 0
        seed = 72000 + self.round % self.SEEDS
        for lane in self.lanes:
            lane.reset(seed, self._command(0.))
        self._publish()

    def geometry(self):
        return model_geometry(self.lanes[0].env.unwrapped.model)

    def _specs(self, lanes):
        """Validates [{experiment, stage: steps | 'latest'}, ...] from the browser."""
        if not isinstance(lanes, list) or not 1 <= len(lanes) <= self.MAX_LANES:
            raise ValueError(f'Выберите от 1 до {self.MAX_LANES} дорожек')
        specs = []
        for lane in lanes:
            if not isinstance(lane, dict) or lane.get('experiment') not in self.runs:
                raise ValueError('Ветка не относится к этому роботу')
            stages = list_stages(self.runs[lane['experiment']])
            available = {int(p.name[6:]): p for p in stages}
            stage = lane.get('stage')
            if stage == 'latest' and stages:
                specs.append((lane['experiment'], stages[-1], True))
            elif type(stage) is int and stage in available:
                specs.append((lane['experiment'], available[stage], False))
            else:
                raise ValueError('Такого сохранённого этапа нет')
        return specs

    def submit(self, body):
        if not isinstance(body, dict):
            raise ValueError('Ожидается объект команды')
        command = body.get('command')
        if command == 'pause':
            if type(body.get('value')) is not bool:
                raise ValueError('Пауза должна быть true или false')
        elif command == 'speed':
            if type(body.get('value')) not in (int, float) or body['value'] not in (0.5, 1, 2):
                raise ValueError('Допустимая скорость: 0.5, 1, 2')
        elif command == 'lanes':
            body = {**body, 'specs': self._specs(body.get('lanes'))}
        elif command == 'scenario':
            if not self.supports_command or body.get('value') not in self.scenarios:
                raise ValueError('Неизвестная команда для робота')
        elif command != 'restart':
            raise ValueError('Неизвестная команда')
        self.commands.put(body)

    def tick(self):
        if time.monotonic() - self.poll_at > 2:
            # New checkpoints of running trainers.
            self.stages = {identity: list_stages(run) for identity, run in self.runs.items()}
            self.poll_at = time.monotonic()
        while not self.commands.empty():
            body = self.commands.get_nowait()
            if body['command'] == 'pause':
                self.paused = body['value']
            elif body['command'] == 'speed':
                self.speed = body['value']
            else:
                if body['command'] == 'lanes':
                    self._select(body['specs'])
                elif body['command'] == 'scenario':
                    self.scenario = body['value']
                self.round = 0 if body['command'] != 'restart' else self.round
                self._restart()
        if not self.paused:
            if not all(lane.done for lane in self.lanes):
                command = self._command(self.ticks * self.dt)
                self.ticks += 1
                for lane in self.lanes:
                    lane.step(self.ticks * self.dt, command)
            else:
                for lane in self.lanes:
                    lane.step(self.ticks * self.dt, None)  # fallen robots finish falling
                self.hold += 1
                if self.hold * self.dt >= self.HOLD_SECONDS:
                    self.round += 1
                    self._restart()
                    return
            # 25 frames/s is enough for the browser and halves the cost of describing ten lanes.
            if (self.ticks + self.hold) % 2:
                return
        self._publish()

    def _publish(self):
        self.seq += 1
        command = self._command(self.ticks * self.dt)
        experiments = [dict(id=identity, name=self.names.get(identity, identity),
                            stages=[int(p.name[6:]) for p in stages])
                       for identity, stages in self.stages.items() if stages]
        lanes = [{**lane.describe(), 'name': self.names.get(lane.experiment, lane.experiment)} for lane in self.lanes]
        frame = {'seq': self.seq, 'paused': self.paused, 'speed': self.speed, 'seconds': self.ticks * self.dt,
                 'round': self.round + 1, 'scenario': self.scenario, 'supports_command': self.supports_command,
                 'command_mps': float(np.hypot(command[0], command[1])) if command is not None else None,
                 'robot_id': self.identity['robot_id'], 'task_id': self.identity['task_id'],
                 'experiments': experiments, 'ended': all(lane.done for lane in self.lanes),
                 'reward_labels': self.reward_labels, 'lanes': lanes, 'error': self.error}
        with self.frame_lock:
            self._frame = frame
            self.frame_lock.notify_all()

    def _release(self):
        for lane in self.lanes:
            lane.close()


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, run_dir):
        super().__init__(address, Handler)
        self.player = Playback(run_dir)
        self.player.client_count = 0
        self.lab = Laboratory(run_dir)
        self.jobs = LabJobs(self.lab)
        self.content = i18n.Content(self.lab.root)  # translated journal, branch names and notes
        self.players = {('baseline', 'main'): self.player}
        self.players_lock = threading.Lock()
        self.races = {}
        self.started = time.time()  # the page reloads itself when this changes (server restarted on new code)
        self.player.start()

    def server_close(self):
        for player in [*self.players.values(), *self.races.values()]:
            player.close()
        super().server_close()

    def robot_runs(self, robot_id, task_id):
        """Lab branches of one robot/task in creation order: ({id: run dir}, {id: name}, active id)."""
        runs, names, active = {}, {}, None
        for entry in self.lab.entries():
            config = entry.get('config', {})
            if (config.get('robot_id') or 'humanoid', config.get('task_id') or 'humanoid_v5') != (robot_id, task_id):
                continue
            runs[entry['id']] = self.lab.resolve(entry['id'])
            names[entry['id']] = entry['name']
            if entry['status'].get('state') in ('training', 'saving', 'starting'):
                active = entry['id']
        return runs, names, active

    def get_race(self, query, subscribe=False):
        """One race per robot/task; its lanes may mix stages of any branch of that robot."""
        from .tasks.catalog import registries
        robots, tasks = registries()
        robot = robots.get(query.get('robot', ['humanoid'])[0])
        compatible = [t.task_id for t in tasks.compatible(robot)]
        task_id = query.get('task', compatible[:1])[0]
        tasks.resolve(task_id, robot)
        key = robot.robot_id, task_id
        runs, names, active = self.robot_runs(*key)
        with self.players_lock:
            if key not in self.races:
                if len(self.races) >= 2:
                    unused = [(k, r) for k, r in self.races.items() if r.client_count == 0]
                    if not unused:
                        raise ValueError('Закройте один из открытых забегов')
                    old_key, old = min(unused, key=lambda pair: pair[1].last_access)
                    old.close()
                    del self.races[old_key]
                race = Race(runs, focus=query.get('focus', [active])[0], names=names)
                race.client_count = 0
                race.start()
                self.races[key] = race
            race = self.races[key]
            if set(runs) - set(race.runs):
                race.add_runs(runs, names)  # new branches
            race.last_access = time.monotonic()
            if subscribe:
                race.client_count += 1
            return race

    def get_player(self, query, subscribe=False):
        identity = query.get('experiment', ['baseline'])[0]
        channel = query.get('channel', ['main'])[0]
        if channel != 'main':
            raise ValueError('Неизвестный канал просмотра')
        run = self.lab.resolve(identity)
        key = identity, channel
        with self.players_lock:
            if key not in self.players:
                if len(self.players) >= 12:
                    unused = [(key, p) for key, p in self.players.items() if p.client_count == 0 and p is not self.player]
                    if not unused:
                        raise ValueError('Закройте один из открытых просмотров')
                    old_key, old = min(unused, key=lambda pair: pair[1].last_access)
                    old.close()
                    del self.players[old_key]
                self.players[key] = Playback(run)
                self.players[key].client_count = 0
                self.players[key].start()
            self.players[key].last_access = time.monotonic()
            if subscribe:
                self.players[key].client_count += 1
            return self.players[key]



class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def trusted(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        return host in allowed and (origin is None or origin == f'http://{host}')

    def lang(self):
        """The language the browser asked for with ?lang= (humanoid_sim/i18n.py); Russian without one."""
        return i18n.language(parse_qs(urlsplit(self.path).query))

    def localize_frame(self, frame):
        return i18n.localize_frame(frame, self.lang(), self.server.content)

    def reply(self, status, content, content_type='application/json; charset=utf-8', headers=None):
        if isinstance(content, dict) and isinstance(content.get('error'), str):
            content = {**content, 'error': i18n.translate(content['error'], self.lang())}
        data = json.dumps(content, ensure_ascii=False, allow_nan=False).encode('utf-8') if isinstance(content, (dict, list)) else content
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.trusted():
            self.reply(403, {'error': 'Только локальное подключение'})
            return
        path = unquote(urlsplit(self.path).path)
        query = parse_qs(urlsplit(self.path).query)
        lang, content = self.lang(), self.server.content
        if path.startswith('/api/lab'):
            try:
                if path == '/api/lab':
                    snapshot = i18n.localize_snapshot(self.server.jobs.snapshot(), lang, content)
                    self.reply(200, {**snapshot, 'server_started': self.server.started})
                elif path == '/api/lab/robots':
                    from .tasks.catalog import catalog
                    self.reply(200, i18n.localize_robots(catalog(), lang))
                elif path == '/api/lab/report':
                    run = self.server.lab.resolve(query.get('experiment', ['baseline'])[0])
                    self.reply(200, read_json(run / 'report.json', {'pending': True}))
                elif path == '/api/lab/journal':
                    from . import journal
                    self.reply(200, i18n.localize_journal(journal.load(self.server.lab.root), lang, content))
                elif path == '/api/lab/journal/image':
                    from . import journal
                    file = journal.image_file(self.server.lab.root, query.get('name', [''])[0])
                    if file is None:
                        self.reply(404, {'error': 'Картинка не найдена'})
                    else:
                        self.reply(200, file.read_bytes(), journal.CONTENT_TYPES[file.suffix])
                elif path == '/api/lab/journal/figure':
                    from . import journal
                    file = journal.figure_file(self.server.lab.root, query.get('name', [''])[0])
                    if file is None:
                        self.reply(404, {'error': 'Схема не найдена'})
                    else:  # a figure runs its own scripts, sandboxed: no access to the lab's origin or API
                        self.reply(200, file.read_bytes(), 'text/html; charset=utf-8',
                                   {'Content-Security-Policy': 'sandbox allow-scripts'})
                else:
                    self.reply(404, {'error': 'Не найдено'})
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})
            return
        if path == '/report.html':
            try:
                run = self.server.lab.resolve(query.get('experiment', ['baseline'])[0])
                file = run / 'report.html'
                report = read_json(run / 'report.json') if lang != 'ru' else None
                if report is not None:  # the file on disk is Russian; other languages are rendered from its data
                    from .reports import render_report
                    page = render_report({**report, 'title': content.text(report['title'], lang)}, lang)
                    self.reply(200, page.encode('utf-8'), 'text/html; charset=utf-8')
                else:
                    self.reply(200, file.read_bytes(), 'text/html; charset=utf-8') if file.exists() else self.reply(404, {'error': 'Отчёт ещё не создан'})
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})
            return
        if path.startswith('/api/race/'):
            try:
                race = self.server.get_race(query, subscribe=path == '/api/race/stream')
            except ValueError as exc:
                self.reply(400, {'error': str(exc)})
                return
            if path == '/api/race/model':
                self.reply(200, race.geometry())
            elif path == '/api/race/frame':
                self.reply(200, self.localize_frame(race.frame()))
            elif path == '/api/race/stream':
                self.stream(race)
            else:
                self.reply(404, {'error': 'Не найдено'})
            return
        try:
            player = self.server.get_player(query, subscribe=path == '/api/stream') if path in ('/api/state', '/api/model', '/api/frame', '/api/stream') else self.server.player
        except ValueError as exc:
            self.reply(400, {'error': str(exc)})
            return
        if path == '/api/health':
            self.reply(200, {'app': 'humanoid-browser', 'version': 1,
                             'run_dir': str(self.server.player.run_dir.resolve())})
        elif path == '/api/state':
            state = player.catalog()
            self.reply(200, {**state, 'training': {**state['training'],
                                                   'error': i18n.translate(state['training'].get('error'), lang)}})
        elif path == '/api/model':
            self.reply(200, player.geometry())
        elif path == '/api/frame':
            self.reply(200, self.localize_frame(player.frame()))
        elif path == '/api/stream':
            self.stream(player)
        else:
            file = (WEB / ('site.html' if path == '/' else path.lstrip('/'))).resolve()
            if WEB.resolve() not in file.parents or not file.is_file():
                self.reply(404, {'error': 'Не найдено'})
                return
            content_type = 'text/javascript; charset=utf-8' if file.suffix == '.js' else mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
            self.reply(200, file.read_bytes(), content_type)

    def stream(self, player):
        """Server-sent frames: each new frame as soon as it is published (at least once a second).

        Physics frames come at 50 Hz; polling with short waits delivered ~16 Hz on Windows (timer
        rounding plus the busy tick thread), so the stream waits on the frame condition instead. The
        browser interpolates between frames on the simulation clock."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        sent, sent_at = object(), 0.
        lang = self.lang()
        try:
            while not player.stop.is_set():
                seq, frame = player.frame_json(lang, self.localize_frame)
                if seq != sent or time.monotonic() - sent_at > 1:
                    self.wfile.write(f'data: {frame}\n\n'.encode())
                    self.wfile.flush()
                    sent, sent_at = seq, time.monotonic()
                with player.frame_lock:
                    player.frame_lock.wait_for(lambda: player._frame.get('seq') != sent or player.stop.is_set(),
                                               timeout=1.)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with self.server.players_lock:
                player.client_count -= 1
        self.close_connection = True

    def do_POST(self):
        if not self.trusted():
            self.close_connection = True
            self.reply(403, {'error': 'Недопустимый источник запроса'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Требуется небольшой JSON-запрос')
            body = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            query = parse_qs(urlsplit(self.path).query)
            if path.startswith('/api/lab/'):
                if not isinstance(body, dict):
                    raise ValueError('Ожидается объект настроек')
                lab = self.server.lab
                if path == '/api/lab/create':
                    result = lab.create(body)
                elif path == '/api/lab/series':
                    result = {'experiments': lab.create_series(body)}
                elif path == '/api/lab/enqueue':
                    result = lab.enqueue(body.get('ids'))
                elif path == '/api/lab/start':
                    if self.server.jobs.active():
                        raise ValueError('Дождитесь завершения проверки')
                    result = lab.start()
                elif path == '/api/lab/stop':
                    result = lab.stop()
                elif path == '/api/lab/job':
                    result = self.server.jobs.launch(body.get('kind'), body.get('experiment', 'baseline'))
                else:
                    raise ValueError('Неизвестное действие')
                self.reply(200, result)
            elif path == '/api/control':
                self.server.get_player(query).submit(body)
                self.reply(200, {'accepted': True})
            elif path == '/api/race/control':
                self.server.get_race(query).submit(body)
                self.reply(200, {'accepted': True})
            else:
                self.reply(404, {'error': 'Не найдено'})
        except (ValueError, TypeError) as exc:
            self.reply(400, {'error': str(exc)})
        except OSError as exc:
            self.reply(500, {'error': str(exc)})


def existing_dashboard(url, run_dir):
    try:
        with urllib.request.urlopen(url + '/api/health', timeout=1) as response:
            info = json.load(response)
    except (OSError, ValueError):
        return False
    if info.get('app') != 'humanoid-browser':
        return False
    if not info.get('run_dir') or Path(info['run_dir']).resolve() != Path(run_dir).resolve():
        raise ValueError('This port serves another experiment. Choose a different --port, e.g. --port 8767.')
    return True


def start_hidden(port, run_dir, wait=30.):
    """Starts the server as a windowless process that outlives its launcher (and the console it came from);
    its output goes to <lab>/server.log. Returns once the server answers."""
    url = f'http://127.0.0.1:{port}'
    python = Path(sys.executable)
    if python.with_name('python.exe').is_file():  # launched by pythonw: the server keeps a (hidden) console for logs
        python = python.with_name('python.exe')
    log = Laboratory(run_dir).root / 'server.log'
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    with log.open('ab') as out:
        subprocess.Popen([str(python), '-u', '-m', 'humanoid_sim.webapp', '--port', str(port), '--run-dir', str(run_dir)],
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=out, creationflags=flags)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if existing_dashboard(url, run_dir):
            return url
        time.sleep(.5)
    raise RuntimeError(f'Сервер не ответил за {wait:g} с, см. {log}')


def main():
    parser = argparse.ArgumentParser(description='Local Humanoid browser dashboard')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--background', action='store_true',
                        help='run the server as a hidden process (output in <lab>/server.log) and return')
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    try:
        if existing_dashboard(url, args.run_dir):
            print(f'Already running: {url}', flush=True)
            if args.open:
                webbrowser.open(url)
            return
        if args.background:
            print(f'Humanoid dashboard in the background: {start_hidden(args.port, args.run_dir.resolve())}', flush=True)
            if args.open:
                webbrowser.open(url)
            return
    except (ValueError, RuntimeError) as exc:
        if sys.stderr is None and sys.platform == 'win32':  # pythonw from humanoid_browser.cmd: no console to print to
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(exc), 'Robot Lab', 0x10)
        parser.exit(2, f'{exc}\n')
    server = DashboardServer(('127.0.0.1', args.port), args.run_dir.resolve())
    print(f'Humanoid dashboard: {url}', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
