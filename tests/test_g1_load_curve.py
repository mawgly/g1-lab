"""task_config load_curve (iteration 40b): the hand load ceiling of arm_targets decides once every `block` rollout
windows on their summed flag-1 counts with its own thresholds (g1_tensor_math.LoadCurveBlock, TensorG1Env.rollout_metrics);
without it the per-window rule bit for bit; the CPU task, the contract and branching are unchanged."""
import pytest

from humanoid_sim.tasks import g1_tensor_math as task
from humanoid_sim.tasks.config import G1_PARAMETERS, g1_config, load_curve_setting
from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
from humanoid_sim.tasks.identity import task_identity
from test_g1_arm_targets import ARMS, OLD_DIGESTS, walking_digest

CURVE = {'block': 10, 'raise_falls': .04, 'lower_falls': .06}
WINDOW = 73728  # flag-1 world-steps of one rollout: 4096 worlds × 24 steps × 0.75
BLOCK = 10 * WINDOW  # 737 280, ~737 episodes of 1000 steps


def test_the_setting_needs_arm_targets_and_changes_neither_the_contract_nor_the_cpu_task():
    field = next(p for p in G1_PARAMETERS if p['key'] == 'load_curve')
    assert field['default'] is None and 'block' in field['label']
    base = {'real_robot': True, 'upper_body_slot': True, 'lock_wrists': True, 'arm_targets': True}
    assert g1_config(base)['load_curve'] is None and g1_config({**base, 'load_curve': None})['load_curve'] is None
    assert g1_config({**base, 'load_curve': {'block': 10., 'raise_falls': .04, 'lower_falls': .06}})['load_curve'] \
        == CURVE
    for bad in ({'block': 0, 'raise_falls': .04, 'lower_falls': .06}, {'block': 2.5, 'raise_falls': .04,
                'lower_falls': .06}, {'block': 10, 'raise_falls': .07, 'lower_falls': .06},
                {'block': 10, 'raise_falls': .04}, {'block': 10, 'raise_falls': .04, 'lower_falls': .06, 'x': 1},
                {'block': 10, 'raise_falls': -1., 'lower_falls': .06}, [10, .04, .06]):
        with pytest.raises(ValueError):
            load_curve_setting(bad)
    with pytest.raises(ValueError):
        g1_config({'real_robot': True, 'load_curve': CURVE})  # no arm_targets
    with pytest.raises(ValueError):
        G1VelocityEnv(real_robot=True, load_curve=CURVE)
    assert task_identity('g1_29dof', 'g1_velocity', {**ARMS, 'load_curve': CURVE}) == task_identity(
        'g1_29dof', 'g1_velocity', ARMS)
    curved = {**ARMS, 'load_curve': CURVE}
    assert walking_digest(curved, True) == OLD_DIGESTS['command']  # the CPU task as before, with the key too
    assert walking_digest(curved, False, free=True) == OLD_DIGESTS['training']


def test_a_block_decides_once_full_on_its_summed_falls():
    """Block of 10 windows, 737 280 flag-1 world-steps: 29 falls (3.93 % per episode) +0.1 kg; 30-44 hold; 45 (6.10 %)
    -0.1 kg; arm_track 0.59 raises nothing; nothing moves before the 10th window, and the counts start again."""
    def block_of(falls, track=.8, level=.5):
        block = task.LoadCurveBlock(**CURVE)
        per = [falls // 10 + (k < falls % 10) for k in range(10)]
        for k, window_falls in enumerate(per):
            full = block.add(WINDOW, window_falls, track * WINDOW)
            assert full == (k == 9)
        new, share = block.decide(level, 1000)
        assert share == pytest.approx(falls * 1000 / BLOCK)
        assert (block.windows, block.steps, block.falls, block.track) == (0, 0., 0., 0.)
        return new
    assert block_of(29) == .6
    for falls in (30, 37, 44):
        assert block_of(falls) == .5
    assert block_of(45) == .4
    assert block_of(29, track=.59) == .5 and block_of(29, track=.61) == .6
    assert block_of(0, level=2.) == 2. and block_of(80, level=0.) == 0.  # within [0, LOAD_LIMIT]
    empty = task.LoadCurveBlock(**CURVE)
    for _ in range(10):
        empty.add(0, 0, 0.)
    level, share = empty.decide(.7, 1000)
    assert level == .7 and share != share  # no flag-1 step: unchanged, NaN


def test_without_the_setting_the_per_window_rule_is_the_old_one():
    """load_curve_update with its default thresholds: up at 0 falls of a window (track ≥ 0.6), down above 3 %."""
    update = task.load_curve_update
    assert (task.LOAD_RAISE_FALLS, task.LOAD_LOWER_FALLS) == (.01, .03)
    assert update(.5, WINDOW, 0, .7, 1000) == .6 and update(.5, WINDOW, 1, .7, 1000) == .5
    assert update(.5, WINDOW, 3, .7, 1000) == .4 and update(.5, WINDOW, 2, .7, 1000) == .5
    assert update(.5, WINDOW, 0, .7, 1000, .04, .06) == .6 and update(.5, WINDOW, 4, .7, 1000, .04, .06) == .5
