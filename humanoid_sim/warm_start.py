"""Warm start across the G1 observation-history contract (task_config `history`).

A single-frame G1 real-robot stage becomes the first stage of a new experiment whose policy sees H frames per
term (tasks/g1_tensor_math.history_observation). The first layers of the actor and the critic keep the old
weights on the newest frame and start at zero on the older ones, and every frame takes the old normalization
statistics of its term, so the new network acts and values exactly as the old one did; training then learns to
use the older frames. Adam's moments move with the weights: the older frames start with no momentum and the
newest frame's second moment, so their first updates are of the usual size. A different observation contract,
so the result is a new root, not a branch (validate_resume rejects the old stage there).

carry_over moves a stage to a contract with the same layout; insert_terms to one with observation terms added
(task_config height_command, upper_body_slot): new columns with zero weights and fixed normalization.

    python -m humanoid_sim.warm_start <source stage dir> <new experiment dir> [--same-layout | --insert-terms]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .checkpoint_schema import validate_resume
from .core import read_json, save_stage, write_json
from .tasks import g1_tensor_math as task
from .tasks.catalog import make_task
from .tasks.config import FOOT_COLLISION_SUFFIX
from .tasks.identity import task_identity

PARAMETERS = ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'clip_range',
              'target_kl')


def source_index(raw):
    """For each entry of the history observation: the single-frame entry it copies, and whether it is the newest
    frame (or a critic extra, which has no history)."""
    frame, history = raw.frame_size, raw.history
    index, newest = [0] * raw.observation_space.shape[0], [False] * raw.observation_space.shape[0]
    for h, positions in enumerate(task.history_index(len(raw.joints), history, raw.extra_sizes)):
        for i, position in enumerate(positions):
            index[position] = i
            newest[position] = h == history - 1
    for k in range(raw.observation_space.shape[0] - frame * history):
        index[frame * history + k] = frame + k
        newest[frame * history + k] = True
    return torch.tensor(index), torch.tensor(newest)


def widen(value, index, newest, fill_older):
    """Columns of a first-layer tensor (out, single-frame inputs) laid out for the history observation."""
    wide = value[:, index].clone()
    if not fill_older:
        wide[:, ~newest] = 0
    return wide


def widen_history(source, run_dir):
    source, run_dir = Path(source), Path(run_dir)
    config = read_json(run_dir / 'run_config.json')
    robot_id, task_id, task_config = config['robot_id'], config['task_id'], config['task_config']
    saved = read_json(source / 'metadata.json')
    if (saved.get('task_config') or {}).get('history', 1) != 1 or task_config.get('history', 1) <= 1:
        raise ValueError('Пересадка идёт с одного кадра на историю кадров')
    # Everything but the history must be the same contract.
    validate_resume(saved, task_identity(robot_id, task_id, {**task_config, 'history': 1}))
    base = DummyVecEnv([lambda: make_task(task_id, robot_id, task_config)])
    try:
        raw = base.envs[0].unwrapped
        index, newest = source_index(raw)
        with open(source / 'normalize.pkl', 'rb') as file:
            norm = pickle.load(file)  # without its venv: the old observation space no longer fits
        extras = len(index) - raw.frame_size * raw.history
        if norm.obs_rms.mean.shape != (raw.frame_size + extras,):
            raise ValueError('Нормировка источника не совпадает с раскладкой одного кадра')
        norm.obs_rms.mean = norm.obs_rms.mean[index.numpy()].copy()
        norm.obs_rms.var = norm.obs_rms.var[index.numpy()].copy()
        norm.observation_space = base.observation_space
        norm.set_venv(base)
        old = PPO.load(source / 'policy.zip', device='cpu')
        model = PPO(type(old.policy), norm, device='cpu', seed=config['seed'],
                    policy_kwargs={**old.policy_kwargs, 'actor_inputs': raw.actor_inputs},
                    **{key: config[key] for key in PARAMETERS})
        names = [name for name, _ in model.policy.named_parameters()]
        if names != [name for name, _ in old.policy.named_parameters()]:
            raise ValueError('Архитектура сети источника отличается')
        target = model.policy.state_dict()
        state = {key: value if value.shape == target[key].shape else widen(value, index, newest, False)
                 for key, value in old.policy.state_dict().items()}
        model.policy.load_state_dict(state)
        optimizer = old.policy.optimizer.state_dict()
        for number, moments in optimizer['state'].items():
            if moments['exp_avg'].shape != target[names[number]].shape:
                moments['exp_avg'] = widen(moments['exp_avg'], index, newest, False)
                moments['exp_avg_sq'] = widen(moments['exp_avg_sq'], index, newest, True)
        model.policy.optimizer.load_state_dict(optimizer)
        model.num_timesteps, model._n_updates = old.num_timesteps, old._n_updates
        stage = save_stage(run_dir, model, norm)
        write_json(stage / 'warm_start.json', {'source': str(source.resolve()), 'steps': old.num_timesteps,
                                               'history': raw.history, 'method': 'newest frame = old weights, '
                                               'older frames = 0; normalization per term'})
        return stage
    finally:
        base.close()


CARRIED_SCHEMAS = ('observation_schema', 'action_schema')
# Model revision suffixes carry_over may also cross, added or removed: physics that only adds contacts between the same
# bodies (task_config foot_collision: the feet collide with each other), so the joints and the layout stay.
CARRIED_MODEL_SUFFIXES = (FOOT_COLLISION_SUFFIX,)


def carried_fields(saved, wanted):
    """The identity fields carry_over may change between `saved` and `wanted`: the observation and action schemas, and
    the model revision when the two differ by exactly one of CARRIED_MODEL_SUFFIXES at its end."""
    old, new = saved.get('model_revision'), wanted.get('model_revision')
    crossed = isinstance(old, str) and isinstance(new, str) and any(
        old + suffix == new or new + suffix == old for suffix in CARRIED_MODEL_SUFFIXES)
    return CARRIED_SCHEMAS + (('model_revision',) if crossed else ())


def check_carry_over(saved, wanted):
    """Raises unless a stage of identity `saved` may move to the contract `wanted` by carry_over: at least one of the
    observation and action schemas differs, or the model revision gains or loses a suffix of CARRIED_MODEL_SUFFIXES,
    and every other field validate_resume compares is the same."""
    fields = carried_fields(saved, wanted)
    if all(saved.get(field) == wanted[field] for field in fields):
        validate_resume(saved, wanted)  # a difference carry_over may not cross is refused by its name
        raise ValueError('Переносить нечего: контракт тот же — это обычная ветка')
    validate_resume(saved, {**wanted, **{field: saved.get(field) for field in fields}})


def carry_over(source, run_dir):
    """The source stage as the first stage of run_dir when the new contract keeps the same input and output layout
    and changes only what some inputs or outputs mean (task_config phase_stop: the gait phase reads 0 while standing;
    lock_wrists: four action entries no longer reach the motors; knee_floor: knee actions below the floor no longer
    move the knee target) or adds contacts to the physics (foot_collision: the feet collide with each other). Only the
    observation and action schemas and a model revision suffix of CARRIED_MODEL_SUFFIXES may differ; robot, the rest
    of the model revision, normalization and control step must match, the observation must keep its size and
    the action space (size and box) must be the same. Weights, normalization and optimizer state are kept as they
    are; the stage gets the new run's identity."""
    source, run_dir = Path(source), Path(run_dir)
    config = read_json(run_dir / 'run_config.json')
    robot_id, task_id, task_config = config['robot_id'], config['task_id'], config['task_config']
    saved = read_json(source / 'metadata.json')
    wanted = task_identity(robot_id, task_id, task_config)
    check_carry_over(saved, wanted)  # everything but the observation and action schemas must be the same contract
    base = DummyVecEnv([lambda: make_task(task_id, robot_id, task_config)])
    try:
        with open(source / 'normalize.pkl', 'rb') as file:
            norm = pickle.load(file)
        if norm.obs_rms.mean.shape != base.observation_space.shape:
            raise ValueError('Раскладка входа изменилась: для истории кадров — widen_history')
        if getattr(norm, 'action_space', None) != base.action_space:
            raise ValueError('Раскладка или предел действия изменились: веса так не переносятся')
        norm.observation_space = base.observation_space
        norm.set_venv(base)
        model = PPO.load(source / 'policy.zip', env=norm, device='cpu')
        stage = save_stage(run_dir, model, norm)
        write_json(stage / 'warm_start.json', {'source': str(source.resolve()), 'steps': model.num_timesteps,
                                               'method': 'same layout, new contract: weights carried over unchanged',
                                               'from_schema': saved.get('observation_schema'),
                                               'to_schema': wanted['observation_schema'],
                                               'from_action_schema': saved.get('action_schema'),
                                               'to_action_schema': wanted['action_schema'],
                                               'from_model_revision': saved.get('model_revision'),
                                               'to_model_revision': wanted['model_revision']})
        return stage
    finally:
        base.close()


def observation_keys(raw):
    """Per entry of a G1 real-robot environment's observation: (term, frame, entry), term a deploy term's number
    (0-6, deploy_observation's order) or an extra term's name (task.extra_terms), frame 0 the oldest; the critic
    extras ('critic', 0, k)."""
    nu, history, sizes = raw.model.nu, raw.history, raw.extra_sizes
    names = [*range(7), *(name for name, _ in raw.extra_terms)]
    keys = [None] * raw.observation_space.shape[0]
    for frame, positions in enumerate(task.history_index(nu, history, sizes)):
        for (start, end), name in zip(task.deploy_terms(nu, sizes), names):
            for i in range(start, end):
                keys[positions[i]] = (name, frame, i - start)
    for k in range(len(keys) - raw.frame_size * history):
        keys[raw.frame_size * history + k] = ('critic', 0, k)
    return keys


def check_insert_terms(saved, wanted):
    """Raises unless a stage of identity `saved` may move to the contract `wanted` by insert_terms: the observation
    schema differs and every other field validate_resume compares is the same."""
    if saved.get('observation_schema') == wanted['observation_schema']:
        validate_resume(saved, wanted)
        raise ValueError('Вставлять нечего: разъём наблюдения тот же')
    validate_resume(saved, {**wanted, 'observation_schema': saved.get('observation_schema')})


def term_map(old, new):
    """For each entry of `new`'s observation (G1VelocityEnv), the entry of `old`'s it copies, or −1 for an entry of a
    term `old` does not have. Every entry of `old` must be in `new` (the old terms keep their frames, entries and
    order; the critic extras stay last), and the entries new to `new` must be its fixed_normalization columns."""
    old_keys, new_keys = observation_keys(old), observation_keys(new)
    where = {key: position for position, key in enumerate(old_keys)}
    if len(where) != len(old_keys) or not set(old_keys) <= set(new_keys):
        raise ValueError('Термы источника не входят в новую раскладку наблюдения')
    if [key[0] for key in old.extra_terms] != [key[0] for key in new.extra_terms if key in old.extra_terms]:
        raise ValueError('Порядок дополнительных термов источника изменился')
    mapping = [where.get(key, -1) for key in new_keys]
    added = {position for position, source in enumerate(mapping) if source < 0}
    fixed = new.fixed_normalization
    if not added or not added <= (set() if fixed is None else set(fixed[0].tolist())):
        raise ValueError('Новые столбцы наблюдения — не столбцы новых термов с заданной нормировкой')
    return torch.tensor(mapping)


def insert_columns(value, mapping, fill):
    """Columns of a first-layer tensor (out, old inputs) laid out for the new observation: copied where mapping ≥ 0;
    the new columns `fill` (out, 1), broadcast."""
    wide = fill.expand(value.shape[0], len(mapping)).clone().to(value.dtype)
    known = mapping >= 0
    wide[:, known] = value[:, mapping[known]]
    return wide


def insert_terms(source, run_dir):
    """The source stage as the first stage of run_dir when the new observation contract adds terms (task_config
    height_command, upper_body_slot): every old entry keeps its term, frame and meaning, the critic extras move to
    the end, and the new terms' columns start at zero weights, so the new network acts and values exactly as the
    old one whenever the new terms read 0 (a height command of 0.78, an empty upper-body slot). Normalization: the
    old columns' statistics as they are (count kept), the new columns' from the environment's fixed_normalization
    (TensorNormalize keeps them frozen; the ~7e9 counted observations would never estimate them). Adam: the new
    columns start with no momentum and the mean second moment of their row's old columns, so their first updates are
    of the usual size. Steps and updates are kept. Only the observation schema may differ (check_insert_terms);
    robot, model, actions, normalization and control step must match."""
    source, run_dir = Path(source), Path(run_dir)
    config = read_json(run_dir / 'run_config.json')
    robot_id, task_id, task_config = config['robot_id'], config['task_id'], config['task_config']
    saved = read_json(source / 'metadata.json')
    wanted = task_identity(robot_id, task_id, task_config)
    check_insert_terms(saved, wanted)
    base = DummyVecEnv([lambda: make_task(task_id, robot_id, task_config)])
    old_env = make_task(task_id, robot_id, saved.get('task_config') or {})
    try:
        raw, old_raw = base.envs[0].unwrapped, old_env.unwrapped
        if old_env.action_space != base.action_space:
            raise ValueError('Раскладка или предел действия изменились: веса так не переносятся')
        mapping = term_map(old_raw, raw)
        known = mapping >= 0
        with open(source / 'normalize.pkl', 'rb') as file:
            norm = pickle.load(file)  # without its venv: the old observation space no longer fits
        if norm.obs_rms.mean.shape != old_env.observation_space.shape:
            raise ValueError('Нормировка источника не совпадает с его раскладкой наблюдения')
        positions, mean, variance = raw.fixed_normalization
        for name, values in (('mean', mean), ('var', variance)):
            wide = np.zeros(len(mapping))
            wide[known.numpy()] = getattr(norm.obs_rms, name)[mapping[known].numpy()]
            wide[positions] = values
            setattr(norm.obs_rms, name, wide)
        norm.observation_space = base.observation_space
        norm.set_venv(base)
        old = PPO.load(source / 'policy.zip', device='cpu')
        kwargs = dict(old.policy_kwargs)
        if 'actor_inputs' in kwargs:
            kwargs['actor_inputs'] = raw.actor_inputs
        model = PPO(type(old.policy), norm, device='cpu', seed=config['seed'], policy_kwargs=kwargs,
                    **{key: config[key] for key in PARAMETERS})
        names = [name for name, _ in model.policy.named_parameters()]
        if names != [name for name, _ in old.policy.named_parameters()]:
            raise ValueError('Архитектура сети источника отличается')
        target = model.policy.state_dict()
        zero = torch.zeros(1, 1)
        state = {key: value if value.shape == target[key].shape else insert_columns(value, mapping, zero)
                 for key, value in old.policy.state_dict().items()}
        model.policy.load_state_dict(state)
        optimizer = old.policy.optimizer.state_dict()
        for number, moments in optimizer['state'].items():
            if moments['exp_avg'].shape != target[names[number]].shape:
                moments['exp_avg'] = insert_columns(moments['exp_avg'], mapping, zero)
                moments['exp_avg_sq'] = insert_columns(moments['exp_avg_sq'], mapping,
                                                       moments['exp_avg_sq'].mean(dim=1, keepdim=True))
        model.policy.optimizer.load_state_dict(optimizer)
        model.num_timesteps, model._n_updates = old.num_timesteps, old._n_updates
        stage = save_stage(run_dir, model, norm)
        write_json(stage / 'warm_start.json', {
            'source': str(source.resolve()), 'steps': old.num_timesteps,
            'method': 'terms inserted: old columns kept, critic extras last, new columns zero weights; new columns '
                      'normalized as fixed_normalization; Adam m 0, v the row mean',
            'from_schema': saved.get('observation_schema'), 'to_schema': wanted['observation_schema'],
            'inserted': [name for name, _ in raw.extra_terms if name not in dict(old_raw.extra_terms)],
            'observation_size': [old_env.observation_space.shape[0], base.observation_space.shape[0]]})
        return stage
    finally:
        old_env.close()
        base.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', type=Path)
    parser.add_argument('run_dir', type=Path)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument('--same-layout', action='store_true', help='carry the weights over unchanged (carry_over)')
    layout.add_argument('--insert-terms', action='store_true',
                        help='new observation terms (height_command, upper_body_slot): insert_terms')
    args = parser.parse_args(argv)
    method = carry_over if args.same_layout else insert_terms if args.insert_terms else widen_history
    print(method(args.source, args.run_dir))


if __name__ == '__main__':
    main()
