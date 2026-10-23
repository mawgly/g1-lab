"""Built-in task adapters. A model and a task are independent choices.

Factories are registered locally, not imported from user-provided strings.
The public factory produces a single CPU Gym environment; batched backends
have separate adapters and must not silently fall back to CPU.
"""
from functools import lru_cache, partial
import hashlib
from pathlib import Path
import gymnasium as gym
import mujoco
from ..robots.registry import RobotRegistry, RobotSpec
from .registry import TaskRegistry, TaskSpec
from .identity import asset_identity
from ..checkpoint_schema import LEGACY
from .evaluation import locomotion_metrics, reach_metrics
from .playback import PlaybackSpec
from .config import g1_config, g1_config_identity, G1_PARAMETERS

ROOT = Path(__file__).resolve().parents[2]
CPU = frozenset({'cpu'})
LOCOMOTION_JOBS = frozenset({'report', 'diagnostics', 'motion'})


def _legacy_identity(robot):
    return dict(LEGACY)


def _humanoid(robot, config):
    return gym.make('Humanoid-v5', **config)


def _arm(robot, config):
    from arm_sim.env import ArmReachEnv
    return ArmReachEnv(**config)


# G1 variants share assets/robots/g1; the legs-only one welds waist and arms (see g1_velocity.load_model).
G1_LEGS_ONLY = {'g1_29dof': False, 'g1_12dof': True}


def _g1_model(legs_only, robot, config):
    from .g1_velocity import G1VelocityEnv
    return G1VelocityEnv(legs_only=legs_only, **config)


def _g1(robot, config):
    return _g1_model(G1_LEGS_ONLY[robot.robot_id], robot, config)


def _g1_identity(robot):
    identity = asset_identity(robot, directory=ROOT / 'assets/robots/g1', task_id='g1_velocity', control_dt=.02,
                              observation_version=2)
    if G1_LEGS_ONLY[robot.robot_id]:
        from .g1_velocity import LEGS_ONLY_REVISION
        identity['model_revision'] = hashlib.sha256(
            f"{identity['model_revision']}:{LEGS_ONLY_REVISION}".encode()).hexdigest()
    return identity


# robot_id, label, base_type, capabilities, factory whose model provides joint/actuator names
ROBOTS = (
    ('humanoid', 'Gymnasium Humanoid', 'floating', {'humanoid_v5'}, _humanoid),
    ('so101', 'SO-101', 'fixed', {'so101_reach'}, _arm),
    ('g1_29dof', 'Unitree G1 · 29 DoF', 'floating', {'g1_velocity'}, partial(_g1_model, False)),
    ('g1_12dof', 'Unitree G1 · ноги (12 DoF)', 'floating', {'g1_velocity'}, partial(_g1_model, True)),
)

TASKS = (
    TaskSpec('humanoid_v5', frozenset({'humanoid_v5'}), CPU, _humanoid, _legacy_identity,
             training_backends=frozenset({'cpu', 'gpu', 'full_gpu'}),
             evaluation_metrics=locomotion_metrics,
             playback=PlaybackSpec(follow_body='torso', target=(0., 0., .9), metric_key='forward_m', fall_height=.45),
             label='Передвижение вперёд', jobs=LOCOMOTION_JOBS),
    TaskSpec('arm_reach', frozenset({'so101_reach'}), CPU, _arm,
             partial(asset_identity, directory=ROOT / 'models/so101', task_id='arm_reach', control_dt=.04),
             episode_limit_key='max_steps', evaluation_metrics=reach_metrics,
             playback=PlaybackSpec(target=(.16, 0., .16), camera_offset=(.6, -.8, .49),
                                   metric_key='distance_m', metric_label='До цели: '),
             label='Достичь цели рукой'),
    # full_gpu: CUDA tests, lab/g1_full_gpu_benchmark.json and a save/resume run passed on 2026-10-19.
    TaskSpec('g1_velocity', frozenset({'g1_velocity'}), CPU, _g1,
             # obs-v2: linear velocity and gait clock added; v1 checkpoints are rejected.
             _g1_identity,
             training_backends=frozenset({'cpu', 'full_gpu'}),
             evaluation_metrics=locomotion_metrics,
             playback=PlaybackSpec(follow_body='pelvis', target=(0., 0., .9), camera_offset=(1.5, -2., .8),
                                   metric_key='forward_m', fall_height=.45),
             config_validator=g1_config, parameters=G1_PARAMETERS, config_identity=g1_config_identity,
             label='Следование скорости', jobs=LOCOMOTION_JOBS | {'command_suite'}),
)


def _describe(robot_id, label, base_type, env, capabilities):
    model = env.unwrapped.model
    joints = tuple(model.joint(i).name for i in range(model.njnt)
                   if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE)
    actuators = tuple(model.actuator(i).name or f'actuator_{i}' for i in range(model.nu))
    return RobotSpec(robot_id, label, base_type, joints, actuators, frozenset(capabilities))


@lru_cache(maxsize=1)
def registries():
    robots, tasks = RobotRegistry(), TaskRegistry()
    for robot_id, label, base, caps, factory in ROBOTS:
        env = factory(None, {})
        try:
            robots.register(_describe(robot_id, label, base, env, caps))
        finally:
            env.close()
    for task in TASKS:
        tasks.register(task)
    return robots, tasks


def make_task(task_id: str, robot_id: str, config: dict):
    robots, tasks = registries()
    return tasks.make(task_id, robots.get(robot_id), 'cpu', config)


def resolve_task(robot_id: str, task_id: str) -> TaskSpec:
    robots, tasks = registries()
    return tasks.resolve(task_id, robots.get(robot_id))


def catalog() -> list[dict]:
    robots, tasks = registries()
    return [dict(robot_id=r.robot_id, label=r.label, base_type=r.base_type,
                 joints=list(r.joint_names), actuators=list(r.actuator_names),
                 tasks=[dict(task_id=t.task_id, label=t.label or t.task_id, backends=sorted(t.backends),
                             race=t.playback.follow_body is not None,  # walking robots: stage race view
                             parameters=[dict(p) for p in t.parameters],
                             training_backends=sorted(t.training_backends), jobs=sorted(t.jobs))
                        for t in tasks.compatible(r)]) for r in robots.all()]
