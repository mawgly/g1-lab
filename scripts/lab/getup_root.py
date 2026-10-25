"""New g1_getup root (iteration 33): getup_root.py <name> <note> [--dry-run] [--lab <dir>] [--no-queue]

Creates an experiment of the getting-up task g1_getup on full_gpu with 4096 worlds and the PPO recipe of the latest G1
walking branches (RECIPE: those of branch 32, exp-dbbe22beb6344075, but lr 3e-4 as every G1 root trained from scratch
(exp-765d110772064ec3, exp-a8699c91ab164143; 1e-4 is the fine-tuning rate of 8 G-step walking policies): n_steps 24, batch 24576, 5 epochs,
γ 0.99, λ 0.95, clip 0.2, target_kl 0.02, entropy 0.01, bounds_coef 0.001 with range_bounds), max_std 1.0 and no
mirror loss, a stage every 10 M steps, 1 billion steps in runs of 3 h, the task's default settings (assist 200 N,
delay 20 ms, motor friction 1.25, body friction 0.3-1.0). Its first stage (0 steps) is written here: a fresh policy
with the initial σ 1.0 (log std 0; make_learner's own fresh start has σ e^-1 ≈ 0.37), asymmetric (CriticExtrasPolicy:
the actor on the 490 robot inputs, the critic on all 496), 256×256 as the walking ones, a fresh normalization; the
trainer starts from it. Queued like task_branch.py unless --no-queue (the worker then trains it after the jobs before
it). --dry-run checks everything (the lab's config schema, the environment, the first stage's policy) and prints what
it would create, writing nothing. --lab: the original run dir of another lab (a check copy in .tools/)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid_sim.core import DEFAULT_RUN, list_stages, read_json, save_stage, write_json  # noqa: E402
from humanoid_sim.lab import Laboratory, validate_config  # noqa: E402
from humanoid_sim.tasks.catalog import make_task  # noqa: E402
from humanoid_sim.tasks.identity import task_identity  # noqa: E402

RECIPE = dict(learning_rate=3e-4, n_steps=24, batch_size=24576, n_epochs=5, gamma=.99, gae_lambda=.95, clip_range=.2,
              target_kl=.02, ent_coef=.01, bounds_coef=.001, range_bounds=True)
INITIAL_STD = 1.
NET_ARCH = [256, 256]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('name')
parser.add_argument('note')
parser.add_argument('--dry-run', action='store_true', help='check and print, create nothing')
parser.add_argument('--lab', type=Path, default=DEFAULT_RUN, help='original run dir of the lab to create the root in')
parser.add_argument('--no-queue', action='store_true', help='create and annotate only')
args = parser.parse_args()

config = dict(name=args.name, backend='full_gpu', envs=4096, steps=1_000_000_000, hours=3, every=10_000_000, seed=42,
              preset='custom', robot_id='g1_29dof', task_id='g1_getup', task_config={}, max_std=1., mirror_coef=0.,
              **RECIPE)
checked = validate_config(config)


def first_stage(run_dir=None):
    """The first stage's model and normalizer (fresh, σ INITIAL_STD); saved into run_dir when given."""
    import math
    import warnings
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.policies import CriticExtrasPolicy
    base = DummyVecEnv([lambda: make_task(checked['task_id'], checked['robot_id'], checked['task_config'])])
    export = VecNormalize(base, gamma=checked['gamma'])
    try:
        raw = base.envs[0].unwrapped
        if not raw.actor_inputs < base.observation_space.shape[0]:
            raise SystemExit('у задачи нет входов только для критика')
        parameters = {key: checked[key] for key in ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                                                    'gae_lambda', 'ent_coef', 'clip_range', 'target_kl')}
        with warnings.catch_warnings():  # SB3 on the batch size against this one CPU environment: not trained here
            warnings.simplefilter('ignore', UserWarning)
            model = PPO(CriticExtrasPolicy, export, **parameters, device='cpu', seed=checked['seed'],
                        policy_kwargs={'net_arch': NET_ARCH, 'log_std_init': math.log(INITIAL_STD),
                                       'actor_inputs': raw.actor_inputs})
        info = dict(actor_inputs=raw.actor_inputs, observations=base.observation_space.shape[0],
                    std=float(model.policy.log_std.detach().exp().mean()), net_arch=NET_ARCH)
        return (save_stage(run_dir, model, export) if run_dir is not None else None), info
    finally:
        export.close()


identity = task_identity(checked['robot_id'], checked['task_id'], checked['task_config'])
ppo = {key: checked[key] for key in (*RECIPE, 'max_std', 'mirror_coef', 'envs', 'every')}
note = (f'Корень задачи вставания g1_getup (итерация 33): full_gpu, {checked["envs"]} сред, рецепт PPO ходьбы '
        f'({json.dumps(ppo, ensure_ascii=False)}), начальная σ {INITIAL_STD:g}, сеть {NET_ARCH}, настройки задачи по '
        f'умолчанию. {args.note}')
if args.dry_run:
    _, info = first_stage()
    print('dry run, nothing created:', checked['name'])
    print('observation_schema:', identity['observation_schema'], '| action_schema:', identity['action_schema'])
    print('task_config:', json.dumps(checked['task_config'], ensure_ascii=False))
    print('PPO:', json.dumps(ppo, ensure_ascii=False), '| first stage:', json.dumps(info))
    print('note:', note)
    raise SystemExit(0)
lab = Laboratory(args.lab)
entry = lab.create(checked)
experiment = entry['id']
path = lab.root / experiment
stage, info = first_stage(path)
if [s.name for s in list_stages(path)] != [stage.name]:
    raise SystemExit(f'{experiment}: ожидался один этап {stage.name}')
manifest = read_json(path / 'experiment.json')
manifest['first_stage'] = {'method': 'fresh', 'initial_std': INITIAL_STD, **info}
write_json(path / 'experiment.json', manifest)
if not args.no_queue:
    lab.enqueue([experiment])
print(experiment, stage.name, json.dumps(info))
lab.annotate(experiment, verdict='active', note=note)
