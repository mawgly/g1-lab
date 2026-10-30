import argparse
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image, ImageDraw

from humanoid_sim import clip as clip_module
from humanoid_sim.clip import (CLIPS, FOCI, FPS, MARK_SECONDS, TEXTS, TILE, _keep_push, _size, _tile_font, compose,
                               configure, feet_gap, film_lane, footer, get_clip, getup_start, push_label, tag_label)
from humanoid_sim.collage import _font
from humanoid_sim.core import save_stage, write_json
from humanoid_sim.evaluation.scenarios import SCENARIOS
from humanoid_sim.tasks import g1_getup_bank as bank_module
from test_g1_getup import small_bank  # noqa: F401 (fixture)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'exp-8ce0cbd70d0a4f1d:6039109632'  # candidate 6
CYRILLIC = re.compile('[А-Яа-яЁё]')
NEW = ('back', 'side_left', 'side_right', 'turn_left', 'turn_right', 'walk_turn', 'side_stop', 'push_back',
       'push_forward', 'push_left', 'push_right', 'push_walk')
ARMS = ('arms_stand', 'arms_push', 'arms_walk', 'arms_poses')
GETUP = ('getup_back', 'getup_front', 'getup_side', 'getup_track')
HELD = ('arm_pose_stand', 'arm_pose_walk')
HEIGHT = ('crouch_stand', 'crouch_walk')
FAST = ('side_left_fast', 'side_right_fast')
# Measures of each kind as film_lane gives them: a number, arms (error, sag), getting up (head, seconds up or None),
# held poses (displacement or metres, degrees), height (pelvis, c), fast side steps (metres, smallest gap cm).
VALUES = {'arms': (.1234, -.4567), 'getup': (1.2345, 1.06), 'creep': (.84, -10.2), 'walk_heading': (2., 33.),
          'height': (.671, .64), 'left_gap': (4.2, -2.04), 'right_gap': (4.3, .31)}


def _lane(frames, fell_at=None):
    image = np.zeros((TILE[1], TILE[0], 3), dtype=np.uint8)
    return [(image, (i + 1) * .04, .04 * i, fell_at is not None and i >= fell_at) for i in range(frames)]


def test_lanes_become_tiles_four_per_row_and_a_fall_freezes_grey():
    two = compose([_lane(10), _lane(10, fell_at=4)], ['было', 'стало'])
    assert len(two) == 10 and two[0].size == (2 * TILE[0], TILE[1])
    fallen = np.asarray(two[-1])[TILE[1] // 2, TILE[0] + TILE[0] // 2]
    assert fallen.min() > 30  # greyed instead of the black test frame
    five = compose([_lane(3)] * 5, ['a', 'b', 'c', 'd', 'e'])
    assert five[0].size == (4 * TILE[0], 2 * TILE[1])


def _compose_before_languages(lanes, titles, size=TILE):
    """compose as it was before languages, scenario measures and pushes: the Russian reference."""
    font, big = _font(14), _font(16)
    columns = min(4, len(lanes))
    rows = math.ceil(len(lanes) / columns)
    result = []
    for index in range(max(len(lane) for lane in lanes)):
        sheet = Image.new('RGB', (size[0] * columns, size[1] * rows), (24, 28, 34))
        draw = ImageDraw.Draw(sheet)
        for lane_index, (lane, title) in enumerate(zip(lanes, titles)):
            image, now, forward, fell = lane[min(index, len(lane) - 1)]
            tile = Image.fromarray(image)
            if fell:
                tile = Image.blend(tile, Image.new('RGB', tile.size, (90, 90, 90)), .5)
            left, top = size[0] * (lane_index % columns), size[1] * (lane_index // columns)
            sheet.paste(tile, (left, top))
            draw.rectangle((left, top, left + size[0] - 1, top + 26), fill=(16, 18, 22))
            draw.text((left + 8, top + 4), title, font=big, fill=(235, 238, 242))
            draw.rectangle((left, top + size[1] - 24, left + size[0] - 1, top + size[1] - 1), fill=(16, 18, 22))
            status = f'упал на {now:.1f} с' if fell else f'{now:4.1f} с'
            draw.text((left + 8, top + size[1] - 21), f'{status} · вперёд {forward:+.1f} м'.replace('.', ','),
                      font=font, fill=(255, 120, 110) if fell else (235, 238, 242))
            if fell:
                draw.rectangle((left, top, left + size[0] - 1, top + size[1] - 1), outline=(220, 60, 50), width=3)
            draw.line((left, top, left, top + size[1]), fill=(24, 28, 34), width=2)
        result.append(sheet)
    return result


def test_russian_tiles_draw_exactly_as_before():
    rng = np.random.default_rng(0)
    lanes = [[(rng.integers(0, 255, (TILE[1], TILE[0], 3), dtype=np.uint8), (i + 1) * .04, 1.7 * i - 3.2,
               fell_at is not None and i >= fell_at) for i in range(12)] for fell_at in (None, 7)]
    titles = ['8ce0cbd7 · 6039 млн', 'стало']
    before = _compose_before_languages(lanes, titles)
    for frames in (compose(lanes, titles), compose([[(*f, None) for f in lane] for lane in lanes], titles)):
        assert [np.asarray(a).tobytes() for a in frames] == [np.asarray(b).tobytes() for b in before]


def test_labels_in_each_language():
    assert footer(2.5, 1.8, False) == ' 2,5 с · вперёд +1,8 м'
    assert footer(1., .4, True) == 'упал на 1,0 с · вперёд +0,4 м'
    assert footer(2.5, 1.8, False, lang='en') == ' 2.5 s · forward +1.8 m'
    assert footer(1., .4, True, lang='en') == 'fell at 1.0 s · forward +0.4 m'
    assert footer(2.5, 1.8, False, lang='zh') == ' 2.5 s · 前进 +1.8 m'
    assert footer(1., .4, True, lang='zh') == '在 1.0 s 时跌倒 · 前进 +0.4 m'
    assert footer(4., 143.4, False, 'turn_right', 'en') == ' 4.0 s · turned right +143°'
    assert footer(4., .456, False, 'push', 'ru') == ' 4,0 с · смещение 0,46 м'
    assert push_label(((-1.25, 0.), '←')) == 'толчок назад 1,25 м/с ←'
    assert push_label(((-1.25, 0.), '←'), 'en') == 'push back 1.25 m/s ←'
    assert push_label(((0., 1.), ''), 'zh') == '向左推 1.0 m/s'
    assert footer(2.5, (.15, .56), False, 'arms') == ' 2,5 с · ошибка рук 0,15 · провис плеча +0,56 рад'
    assert footer(3.1, (.3, -.02), True, 'arms', 'en') == 'fell at 3.1 s · arm error 0.30 · shoulder sag -0.02 rad'
    assert footer(.5, (.31, None), False, 'getup') == ' 0,5 с · голова 0,31 м'
    assert footer(4., (1.304, 1.06), False, 'getup') == ' 4,0 с · встал за 1,1 с · голова 1,30 м'
    assert footer(2., (.5, None), True, 'getup', 'en') == 'cut off at 2.0 s · head 0.50 m'
    assert footer(4., (1.3, 1.), False, 'getup', 'zh') == ' 4.0 s · 1.0 s 内起身 · 头部 1.30 m'
    assert footer(6., (.84, -10.2), False, 'creep') == ' 6,0 с · смещение 0,84 м · курс -10°'
    assert footer(5., (2., 33.), False, 'walk_heading', 'en') == ' 5.0 s · forward +2.0 m · heading +33°'
    assert footer(4., (.671, .64), False, 'height') == ' 4,0 с · таз 0,67 м · команда 0,64 м'
    assert footer(6., (4.2, -2.04), False, 'left_gap') == ' 6,0 с · влево +4,2 м · наименьший зазор стоп -2,0 см'
    assert footer(6., (4.3, .31), False, 'right_gap', 'en') == ' 6.0 s · right +4.3 m · smallest foot gap +0.3 cm'
    assert tag_label(('load', 1.)) == 'груз 1 кг на кисть' and tag_label(('load', .5), 'en') == 'load 0.5 kg per hand'
    assert tag_label(('load', 2.), 'zh') == '每只手负载 2 kg'
    assert tag_label(('pose', 'carry_mid')) == 'навязанная поза рук: carry_mid'
    assert tag_label(('random', 1.)) == 'навязанная поза рук ±1 рад' and tag_label(('random', 1.2), 'en') == \
        'forced arm pose ±1.2 rad'
    for lang in ('en', 'zh'):
        for measure in {clip.measure for clip in CLIPS.values()}:
            for fell in (False, True):
                text = footer(9.96, VALUES.get(measure, -12.345), fell, measure, lang)
                assert not CYRILLIC.search(text) and not re.search(r'\d,\d', text), text
        for push in (clip.push for clip in CLIPS.values() if clip.push):
            assert not CYRILLIC.search(push_label((push[1], '→'), lang))
        for tag in (('load', 1.5), ('pose', 'extended'), ('random', 1.5)):
            assert not CYRILLIC.search(tag_label(tag, lang)) and not re.search(r'\d,\d', tag_label(tag, lang))


def test_chinese_tiles_use_a_font_with_cjk_glyphs():
    def drawn(font, text):
        image = Image.new('L', (80, 30))
        ImageDraw.Draw(image).text((2, 2), text, font=font, fill=255)
        return np.asarray(image)
    font = _tile_font(16, 'zh')  # collage's Segoe UI would draw each of these as the .notdef box
    missing = drawn(font, '\U0010fffd')  # a code point no font has: the .notdef box
    assert all(not np.array_equal(drawn(font, c), missing) for c in '前进跌倒推位移左转')


def test_scenarios_and_sizes():
    assert set(SCENARIOS) | set(NEW) | set(ARMS) | set(GETUP) | set(HELD) | set(HEIGHT) | set(FAST) == set(CLIPS)
    assert all(CLIPS[name].focus == 'body' and CLIPS[name].view == 'side' and CLIPS[name].seconds == 10.
               and not CLIPS[name].episode for name in (*SCENARIOS, *NEW))
    assert all(CLIPS[name].episode for name in (*ARMS, *GETUP, *HELD, *HEIGHT))
    assert all(not CLIPS[name].episode and (CLIPS[name].focus, CLIPS[name].view) == ('feet', 'front') for name in FAST)
    assert CLIPS['side_left_fast'].scenario.command_at(0.) == (0., 1., 0.)
    assert CLIPS['side_right_fast'].scenario.command_at(5.) == (0., -1., 0.)
    assert [CLIPS['crouch_stand'].scenario.command_at(0.), CLIPS['crouch_walk'].scenario.command_at(0.)] == \
        [(0., 0., 0.), (.5, 0., 0.)]
    assert all(CLIPS[name].scenario == SCENARIOS[name] and CLIPS[name].measure == 'forward' for name in SCENARIOS)
    assert CLIPS['side_stop'].scenario.command_at(3.98) == (0., .3, 0.)
    assert CLIPS['side_stop'].scenario.command_at(4.) == (0., 0., 0.)
    assert _size('640x480') == (640, 480) and _size('1280X720') == (1280, 720)
    for bad in ('640', '640x', 'x480', '10x10', '640x48o'):
        with pytest.raises(argparse.ArgumentTypeError):
            _size(bad)
    big = compose([[(np.zeros((480, 640, 3), dtype=np.uint8), 2.04, .5, False, ((0., 1.25), ''))]], ['a'], (640, 480),
                  'zh', 'push')
    assert big[0].size == (640, 480)


def test_a_push_runs_on_through_an_unchanged_command():
    """supervise.rollout sets the command before every control step; set_command alone ends a running push."""
    from humanoid_sim.tasks.catalog import make_task
    env = make_task('g1_velocity', 'g1_29dof', {})
    try:
        raw = env.unwrapped
        env.reset(seed=1, options={'command': (0., 0., 0.)})
        raw.set_force_push((-1.25, 0.), .1)
        raw.set_command((0., 0., 0.))
        assert raw.push_left == 0
        _keep_push(raw)
        raw.set_force_push((-1.25, 0.), .1)
        left = []
        for _ in range(5):
            raw.set_command((0., 0., 0.))
            env.step(np.zeros(env.action_space.shape))
            left.append(raw.push_left)
        assert left == [40, 30, 20, 10, 0]
        raw.set_force_push((0., 1.), .1)
        raw.set_command((.5, 0., 0.))  # a new command still ends it
        assert raw.push_left == 0 and np.array_equal(raw.command, (.5, 0., 0.))
    finally:
        env.close()


@pytest.fixture(scope='module')
def candidate():
    identity, _, steps = CANDIDATE.partition(':')
    if not (ROOT / 'runs' / 'humanoid_lab' / identity / f'stage_{int(steps):012d}' / 'metadata.json').is_file():
        pytest.skip('candidate 6 is not on this machine')
    from humanoid_sim.core import DEFAULT_RUN
    from humanoid_sim.lab import Laboratory
    from humanoid_sim.supervise import find_stage
    stage = find_stage(Laboratory(DEFAULT_RUN), CANDIDATE)
    assert int(stage.name[6:]) == int(steps)
    return stage


@pytest.mark.parametrize('name', NEW)
def test_every_new_scenario_plays_on_the_candidate(candidate, name):
    clip = CLIPS[name]
    seconds = 1. if clip.push is None else clip.push[0] + .8
    frames = film_lane(candidate, clip, seconds, size=(64, 54))
    assert len(frames) == math.ceil(seconds * FPS) and frames[0][0].shape == (54, 64, 3)
    assert not any(frame[3] for frame in frames)  # no fall
    value = frames[-1][2]
    if clip.push is None:
        assert not any(frame[4] for frame in frames)
        # 1 s in the named direction (candidate 6: back 0.33 m, sideways 0.17-0.25 m, turns 21-23°, walking turn 12°)
        assert value > {'back': .1, 'left': .05, 'right': .05}.get(clip.measure, 5. if name == 'walk_turn' else 10.)
        return
    marked = [frame[1] for frame in frames if frame[4] is not None]
    start = clip.push[0]
    assert math.isclose(marked[0], start) and math.isclose(marked[-1], start + MARK_SECONDS - 1 / FPS)
    assert all(frame[4][0] == clip.push[1] for frame in frames if frame[4] is not None)
    before = next(frame[2] for frame in frames if math.isclose(frame[1], start))
    # 0.8 s after the push the pelvis is away (candidate 6: 0.53-0.85 m standing, 0.55 m to the left walking)
    assert abs(before) < .1 and value - before > .2


def test_texts_cover_every_measure():
    for lang, texts in TEXTS.items():
        assert {clip.measure for clip in CLIPS.values()} <= set(texts), lang
        assert {'getup_up', 'cutoff', 'load'} <= set(texts), lang
        assert set(texts) == set(TEXTS['ru']), lang


# Small stages (untrained 32×32 actors) of the three kinds the clips film: walking, arms on targets, getting up.
ARM_TARGETS = dict(real_robot=True, critic_extras=True, history=5, phase_stop=True, action_limit=10., heading_hold=True,
                   lock_wrists=True, height_command=True, upper_body_slot=True, arm_targets=True,
                   reward_weights={'arm_track': 2.})


def _small_stage(run, task_id, task_config, make_env, actor_inputs=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from humanoid_sim.lab import validate_config
    from humanoid_sim.policies import CriticExtrasPolicy
    config = validate_config(dict(robot_id='g1_29dof', task_id=task_id, n_steps=8, batch_size=8,
                                  task_config=task_config))
    run.mkdir(parents=True)
    write_json(run / 'run_config.json', config)
    env = VecNormalize(DummyVecEnv([lambda: make_env(config['task_config'])]))
    try:
        size = env.observation_space.shape[0]
        kwargs = dict(policy_kwargs={'net_arch': [32, 32], 'actor_inputs': size - actor_inputs}) if actor_inputs \
            else {}
        model = PPO(CriticExtrasPolicy if actor_inputs else 'MlpPolicy', env, n_steps=8, batch_size=8, n_epochs=1,
                    device='cpu', seed=1, **kwargs)
        return save_stage(run, model, env)
    finally:
        env.close()


@pytest.fixture(scope='module')
def stages(tmp_path_factory):
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    root = tmp_path_factory.mktemp('clip-stages')
    return dict(walking=_small_stage(root / 'walking', 'g1_velocity', {}, lambda config: G1VelocityEnv(**config)),
                arms=_small_stage(root / 'arms', 'g1_velocity', ARM_TARGETS, lambda config: G1VelocityEnv(**config),
                                  actor_inputs=4),
                getup=_small_stage(root / 'getup', 'g1_getup', {}, lambda config: G1GetupEnv(**config),
                                   actor_inputs=6))


class _Renderer:
    """mujoco.Renderer without OpenGL: records each frame's camera and the pelvis, renders black."""
    shots = []

    def __init__(self, model, height, width):
        self.shape, self.pelvis = (height, width, 3), model.body('pelvis').id
        self.scene = mujoco.MjvScene(model, maxgeom=64)

    def update_scene(self, data, camera, scene_option=None):
        _Renderer.shots.append(dict(distance=camera.distance, elevation=camera.elevation, azimuth=camera.azimuth,
                                    lookat=camera.lookat.copy(), pelvis=data.xpos[self.pelvis].copy(),
                                    option=scene_option, qpos=data.qpos[:7].copy(),
                                    feet=data.xpos[[data.model.body(f'{side}_ankle_roll_link').id
                                                    for side in ('left', 'right')]].copy()))

    def render(self):
        return np.zeros(self.shape, dtype=np.uint8)

    def close(self):
        pass


def test_the_default_camera_is_the_old_one_and_a_focus_moves_it(stages, monkeypatch):
    """Walking scenarios by default: the pelvis from 2.8 × 0.9 m at -8°, the side's azimuth, looking at 0.63 m above
    it on every frame, as before foci (the committed clips are made again byte for byte). The feet focus looks at the
    ankles from 1.1 m at -20°, its offset from the pelvis smoothed; the top view looks down."""
    monkeypatch.setattr(mujoco, 'Renderer', _Renderer)
    _Renderer.shots = []
    frames = film_lane(stages['walking'], CLIPS['walk'], .4)
    assert len(frames) == len(_Renderer.shots) == 10 and all(frame[5] is None for frame in frames)
    for shot in _Renderer.shots:
        assert (shot['distance'], shot['elevation'], shot['azimuth']) == (2.8 * .9, -8., 90.)
        np.testing.assert_array_equal(shot['lookat'], (shot['pelvis'][0], shot['pelvis'][1], .7 * .9))
    _Renderer.shots = []
    film_lane(stages['walking'], CLIPS['walk'], .4, view='front', focus='feet')
    first = _Renderer.shots[0]
    assert (first['distance'], first['elevation'], first['azimuth']) == (1.1, -20., 180.)
    np.testing.assert_allclose(first['lookat'], first['feet'].mean(axis=0), atol=1e-12)
    for shot in _Renderer.shots:
        assert abs(shot['lookat'][2] - shot['feet'][:, 2].mean()) < .05
        assert np.linalg.norm(shot['lookat'][:2] - shot['feet'][:, :2].mean(axis=0)) < .1
    _Renderer.shots = []
    film_lane(stages['walking'], CLIPS['stand'], .2, view='top')
    assert {(shot['elevation'], shot['azimuth']) for shot in _Renderer.shots} == {(-89., 90.)}
    with pytest.raises(ValueError, match='фокус'):
        film_lane(stages['walking'], CLIPS['walk'], .2, focus='elbows')


def test_an_arms_scenario_films_targets_loads_and_the_arms_measures(stages, monkeypatch):
    """arms_stand with 1 kg on each hand on a small arm_targets stage, filmed by OpenGL: the hand loads on every frame,
    (error, sag) finite, and per frame a target ball and a load block for each hand."""
    added = []
    add = clip_module._add_geom
    monkeypatch.setattr(clip_module, '_add_geom', lambda scene, kind, *rest: (added.append(int(kind)),
                                                                              add(scene, kind, *rest)))
    frames = film_lane(stages['arms'], configure(get_clip('arms_stand'), load=1.), .4, size=(64, 54), focus='hands')
    assert len(frames) == 10 and frames[0][0].shape == (54, 64, 3) and frames[0][0].std() > 5
    assert all(frame[5] == ('load', 1.) and len(frame[2]) == 2 and all(map(math.isfinite, frame[2])) for frame in frames)
    sphere, box = int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_BOX)
    assert added == [sphere, sphere, box, box] * len(frames)
    composed = compose([frames], ['стало'], (64, 54), 'ru', 'arms')
    assert len(composed) == 10 and composed[0].size == (64, 54)


def test_the_target_ball_is_where_the_palm_would_be_in_the_target_pose(stages):
    """Forward kinematics of the target: with the arm joints exactly at home + d the ball is on the palm; the load
    block sits on the palm turned with the hand, its half side 0.025 × kg^(1/3)."""
    from humanoid_sim.export_g1 import named_arm_poses
    from humanoid_sim.tasks import g1_tensor_math as task
    from humanoid_sim.tasks.catalog import make_task
    env = make_task('g1_velocity', 'g1_29dof', ARM_TARGETS)
    try:
        raw = env.unwrapped
        env.reset(seed=3, options={'command': (0., 0., 0.)})
        raw.set_hand_load(2.)
        raw.set_arm_targets(dict((name, q) for name, _, q in named_arm_poses(raw))['extended'], 1)
        env.step(np.zeros(env.action_space.shape))
        hands = [raw.model.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')]
        scene = mujoco.MjvScene(raw.model, maxgeom=8)
        clip_module._arm_marks(scene, raw, mujoco.MjData(raw.model), hands)
        target = raw.home + raw.previous_offset
        raw.data.qpos[raw.qadr[list(raw.track_joints)]] = target[list(raw.track_joints)]
        mujoco.mj_kinematics(raw.model, raw.data)
        assert scene.ngeom == 4 and np.allclose(clip_module.HAND_LOAD_POSITION, task.HAND_LOAD_POSITION)
        for k, body in enumerate(hands):
            np.testing.assert_allclose(scene.geoms[k].pos, clip_module._palm(raw.data, body), atol=1e-6)
            assert scene.geoms[2 + k].size[0] == pytest.approx(.025 * 2 ** (1 / 3))
        names = [raw.model.joint(joint).name for joint in raw.joints]
        shoulders = [names.index(f'{side}_shoulder_pitch_joint') for side in ('left', 'right')]
        error, sag = clip_module._arm_measure(raw, shoulders)
        assert error < 1e-9 and abs(sag) < 1e-9
        raw.data.qpos[raw.qadr[shoulders]] += .3  # both arms .3 rad lower than asked
        assert clip_module._arm_measure(raw, shoulders) == pytest.approx((.06, .3))
    finally:
        env.close()


def test_a_getting_up_scenario_starts_from_a_held_out_row_without_help(stages, small_bank, monkeypatch):  # noqa: F811
    """getup_back on a small g1_getup stage: the class's held-out row (--start picks among them), no assist, the
    lowest bound, no delay, the motor model as measured, the files' friction; (head, seconds up) per frame."""
    from humanoid_sim.tasks.g1_getup import G1GetupEnv
    monkeypatch.setattr(bank_module, 'BANK', small_bank)
    bank = bank_module.load(small_bank)
    back = np.flatnonzero(bank['holdout'] & (bank['start_class'] == bank['classes'].index('back')))
    assert getup_start(bank, 'back') == back[0]
    with pytest.raises(ValueError, match='отложенных'):
        getup_start(bank, 'back', len(back))
    seen = []
    reset, step = G1GetupEnv.reset, G1GetupEnv.step

    def spy_reset(self, **kwargs):
        seen.append(dict(kwargs['options']))
        return reset(self, **kwargs)

    def spy_step(self, action):
        seen.append(bool(np.array_equal(self.model.geom_friction[:, 0], self.nominal_friction)))
        return step(self, action)
    monkeypatch.setattr(G1GetupEnv, 'reset', spy_reset)
    monkeypatch.setattr(G1GetupEnv, 'step', spy_step)
    frames = film_lane(stages['getup'], CLIPS['getup_back'], .4, size=(64, 54))
    assert seen[0] == dict(assist_force=0., pd_bound='min', start=int(back[0]), action_delay_steps=0,
                           motor_friction_scale=1.)
    assert all(seen[1:]) and len(seen) == 1 + math.ceil(.4 / .02)
    assert len(frames) == math.ceil(.4 * FPS) and frames[0][0].std() > 5 and not any(frame[3] for frame in frames)
    assert all(len(frame[2]) == 2 and 0 <= frame[2][0] < 2 for frame in frames)


def test_scenario_changes_and_which_stages_a_scenario_films(stages):
    assert configure(get_clip('arms_stand'), pose='carry_mid', load=.5).arms == ((0., 'home'), (.5, 'carry_mid'))
    assert configure(get_clip('arms_walk')) == get_clip('arms_walk') and get_clip('arms_walk').load == 0.
    assert configure(get_clip('getup_side'), start=3).start == 3
    held = configure(get_clip('arm_pose_stand'), draw=2, amplitude=1.2)
    assert (held.held, held.draw, held.amplitude) == ((1., 'random'), 2, 1.2)
    assert configure(get_clip('arm_pose_walk'), pose='extended').held == (1., 'extended')
    for scenario, changes, message in (('walk', dict(load=1.), 'Груз'), ('arms_stand', dict(load=-1.), 'Груз'),
                                       ('arms_stand', dict(pose='waving'), 'поза'),
                                       ('arms_poses', dict(pose='extended'), 'несколько'),
                                       ('arms_stand', dict(start=1), 'старт'), ('getup_back', dict(start=-1), 'старт'),
                                       ('walk', dict(pose='extended'), 'Поза'),
                                       ('arms_stand', dict(pose='random'), 'поза'),
                                       ('arm_pose_stand', dict(pose='extended', draw=1), 'случайн'),
                                       ('arm_pose_walk', dict(amplitude=3.), 'Размах'),
                                       ('arm_pose_walk', dict(draw=-1), 'Вариант'),
                                       ('arm_pose_walk', dict(load=1.), 'Груз')):
        with pytest.raises(ValueError, match=message):
            configure(get_clip(scenario), **changes)
    for stage, scenario in (('walking', 'arms_stand'), ('walking', 'getup_back'), ('getup', 'stand')):
        with pytest.raises(ValueError, match='arm_targets|g1_getup'):
            film_lane(stages[stage], CLIPS[scenario], .1)
    assert set(FOCI) == {'body', 'torso', 'arms', 'hands', 'feet'}


def _applied_arms(monkeypatch):
    """Spies G1VelocityEnv.step: per control step the arm entries of the action it applies (applied_action)."""
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    applied, step = [], G1VelocityEnv.step

    def spy(self, action):
        applied.append(self.applied_action(action)[list(self.arm_joints)])
        return step(self, action)
    monkeypatch.setattr(G1VelocityEnv, 'step', spy)
    return applied


def test_a_held_pose_holds_the_arms_from_its_time_on_every_step(stages, monkeypatch):
    """arm_pose_*: from its time the arm joints apply the pose's action on every control step (supervise.rollout's
    set_command each step would end a segment), before it the policy's; a named pose is its angles from home over the
    scale, a random one arm_probe.py's draw; the tag names it."""
    import mujoco as mj
    from humanoid_sim.export_g1 import named_arm_poses
    from humanoid_sim.tasks.catalog import make_task
    from dataclasses import replace
    monkeypatch.setattr(mj, 'Renderer', _Renderer)
    applied = _applied_arms(monkeypatch)
    clip = replace(configure(get_clip('arm_pose_walk'), pose='carry_mid'), held=(.2, 'carry_mid'))
    frames = film_lane(stages['walking'], clip, .4)
    env = make_task('g1_velocity', 'g1_29dof', {})
    try:
        raw = env.unwrapped
        arms = list(raw.arm_joints)
        pose = dict((name, q) for name, _, q in named_arm_poses(raw))['carry_mid']
        expected = np.clip((pose - raw.home)[arms] / raw.scale[arms], -raw.action_limit, raw.action_limit)
        held = clip_module._held_action(raw, configure(get_clip('arm_pose_stand'), draw=3, amplitude=1.2))
        random = np.random.default_rng(3).uniform(-1, 1, len(arms)) * 1.2 / raw.scale[arms]
        np.testing.assert_allclose(held[arms], random)
        assert not np.delete(held, arms).any()
    finally:
        env.close()
    assert len(applied) == 20 and not any(np.allclose(a, expected) for a in applied[:10])
    for action in applied[10:]:
        np.testing.assert_allclose(action, expected)
    assert [frame[5] for frame in frames] == [None] * 5 + [('pose', 'carry_mid')] * 5
    assert all(len(frame[2]) == 2 for frame in frames)


def test_a_height_scenario_sets_the_height_command_at_its_times(stages, monkeypatch):
    """crouch_*: set_height at each of its times (here moved earlier); the measure is (pelvis height, c)."""
    import mujoco as mj
    from dataclasses import replace
    monkeypatch.setattr(mj, 'Renderer', _Renderer)
    clip = replace(get_clip('crouch_stand'), heights=((0., .78), (.2, .64), (.32, .78)))
    frames = film_lane(stages['arms'], clip, .4)
    assert [frame[2][1] for frame in frames] == [.78] * 5 + [.64] * 3 + [.78] * 2
    assert all(.4 < frame[2][0] < 1. for frame in frames)
    with pytest.raises(ValueError, match='height_command'):
        film_lane(stages['walking'], CLIPS['crouch_walk'], .1)


def test_the_feet_gap_is_the_capsules_distance():
    """feet_gap against points sampled along every capsule pair (1 mm): standing apart, and the right foot moved onto
    the left one, where the capsules overlap (negative)."""
    from humanoid_sim.tasks.catalog import make_task
    local, radius = clip_module._foot_capsules()
    env = make_task('g1_velocity', 'g1_29dof', {})
    try:
        raw = env.unwrapped
        env.reset(seed=1, options={'command': (0., 0., 0.)})

        def sampled():
            points = []
            for name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
                body = raw.model.body(name).id
                ends = raw.data.xpos[body] + local[name] @ raw.data.xmat[body].reshape(3, 3).T
                s = np.linspace(0, 1, 150)[:, None, None]
                points.append((ends[:, 0] + s * (ends[:, 1] - ends[:, 0])).reshape(-1, 3))
            return np.linalg.norm(points[0][:, None] - points[1][None], axis=-1).min() - 2 * radius
        apart = feet_gap(raw.model, raw.data, (local, radius))
        assert apart > .05 and 0 <= sampled() - apart < 2e-3
        left, right = (raw.model.body(f'{side}_ankle_roll_link').id for side in ('left', 'right'))
        raw.data.xpos[right] = raw.data.xpos[left] + (.0, .005, 0.)
        raw.data.xmat[right] = raw.data.xmat[left]
        overlap = feet_gap(raw.model, raw.data, (local, radius))
        assert overlap < 0 and 0 <= sampled() - overlap < 2e-3
    finally:
        env.close()


@pytest.mark.parametrize('name', FAST)
def test_fast_side_steps_film_the_feet_and_the_smallest_gap(candidate, name):
    clip = CLIPS[name]
    frames = film_lane(candidate, clip, 1., size=(64, 54))
    assert len(frames) == FPS and not any(frame[3] for frame in frames)
    gaps = [frame[2][1] for frame in frames]
    assert all(later <= earlier for earlier, later in zip(gaps, gaps[1:])) and all(map(math.isfinite, gaps))
    assert frames[-1][2][0] > .2  # 1 s of a 1 m/s side step in the named direction (candidate 6: 0.4-0.5 m)


def test_help_lists_the_scenarios_and_foci(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['clip', '--help'])
    with pytest.raises(SystemExit):
        clip_module.main()
    text = capsys.readouterr().out
    assert all(name in text for name in (*ARMS, *GETUP, *HELD, *HEIGHT, *FAST, 'push_walk', 'three_quarter', 'hands',
                                         'feet', 'carry_mid', 'random', '--draw', '--amplitude', '--terrain-type',
                                         '--terrain-level', '--seed', 'curbs', 'stairs_up'))


def test_a_terrain_tile_is_named_with_its_generator_number():
    """terrain_parameter from the generators' constants (generator 3 level 5: steps 2 + 18·5/9 = 12 cm, curbs 2 +
    13·5/9 = 9.2 cm; generator 2 level 9: slopes 15°); the footer in each language."""
    from humanoid_sim.clip import terrain_label, terrain_parameter
    assert terrain_parameter('stairs_up', 5, 3) == ('stairs', pytest.approx(12.))
    assert terrain_parameter('curbs', 5, 3) == ('curbs', pytest.approx(2 + 13 * 5 / 9))
    assert terrain_parameter('slope_down', 9, 2) == ('slope', pytest.approx(15.))
    assert terrain_parameter('stairs_down', 0, 2) == ('stairs', pytest.approx(2.)) and terrain_parameter('flat', 3, 3) \
        is None
    assert footer(4., ('stairs_up', 5, 3, 2.06), False, 'terrain') == \
        ' 4,0 с · лестница вверх, уровень 5 (ступень 12 см) · пройдено 2,1 м'
    assert footer(3.2, ('curbs', 5, 3, 1.5), True, 'terrain', 'en') == \
        'fell at 3.2 s · curbs, level 5 (height 9.2 cm) · walked 1.5 m'
    assert footer(4., ('flat', 0, 2, .5), False, 'terrain', 'zh') == ' 4.0 s · 平地，第 0 级 · 行进 0.5 m'
    assert terrain_label('pits', 9, 3) == 'ямы, уровень 9 (глубина 20 см)'
    from humanoid_sim.tasks.g1_terrain import GENERATORS
    kinds = {kind for spec in GENERATORS.values() for kind in spec.types}
    for lang in TEXTS:
        assert set(clip_module.TERRAIN_TEXTS[lang]['names']) == kinds
        for kind in kinds:
            for fell in (False, True):
                text = footer(9.96, (kind, 9, 3 if kind in ('pits', 'curbs') else 2, -1.234), fell, 'terrain', lang)
                if lang != 'ru':
                    assert not CYRILLIC.search(text) and not re.search(r'\d,\d', text), text


def test_terrain_options_go_with_walking_scenarios_only():
    clip = configure(get_clip('slow'), terrain_type='stairs_up', terrain_level=5)
    assert (clip.terrain, clip.measure, clip.view, clip.episode) == (('stairs_up', 5, None, None), 'terrain',
                                                                    'three_quarter', True)
    assert configure(get_clip('walk'), terrain_type='curbs', terrain_seed=1000).terrain == ('curbs', 0, 1000, None)
    assert configure(get_clip('side_left_fast'), terrain_type='flat').view == 'front'  # a scenario's own view stays
    assert configure(get_clip('walk')) == get_clip('walk')
    for scenario, changes, message in (('arms_stand', dict(terrain_type='flat'), 'ходьбы'),
                                       ('getup_back', dict(terrain_type='flat'), 'ходьбы'),
                                       ('crouch_walk', dict(terrain_type='flat'), 'ходьбы'),
                                       ('walk', dict(terrain_type='lava'), 'тип плитки'),
                                       ('walk', dict(terrain_type='flat', terrain_level=10), 'Уровень'),
                                       ('walk', dict(terrain_type='flat', terrain_level=True), 'Уровень'),
                                       ('walk', dict(terrain_level=2), 'вместе'),
                                       ('walk', dict(terrain_seed=2), 'вместе'),
                                       ('walk', dict(terrain_type='flat', terrain_seed=-1), 'Карта'),
                                       ('walk', dict(terrain_type='flat', load=1.), 'Груз')):
        with pytest.raises(ValueError, match=message):
            configure(get_clip(scenario), **changes)


@pytest.fixture(scope='module')
def terrain_stages(tmp_path_factory):
    """Small walking stages on rough ground: generator 2 on map 7, generator 3 on its default map."""
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    root = tmp_path_factory.mktemp('clip-terrain')
    return {number: _small_stage(root / f'terrain{number}', 'g1_velocity', {'terrain': terrain},
                                 lambda config: G1VelocityEnv(**config))
            for number, terrain in ((2, {'seed': 7}), (3, {'generator': 3}))}


def test_a_terrain_lane_starts_on_the_tile_facing_minus_y_with_the_camera_on_the_ground(terrain_stages, stages,
                                                                                         monkeypatch):
    """--terrain-type stairs_up --terrain-level 5 on a generator 3 stage: the reset with the command and the tile, the
    robot on the tile's centre turned to -y with heading hold there, the three-quarter camera turned with it, looking
    at the ground under the pelvis + 0.55 m from 3.2 m; the measure (type, level, generator, metres along -y); the
    box tiles drawn by their boxes, their heightfields hidden."""
    from humanoid_sim.tasks import g1_terrain
    from humanoid_sim.tasks.g1_velocity import G1VelocityEnv
    monkeypatch.setattr(mujoco, 'Renderer', _Renderer)
    seen, reset = [], G1VelocityEnv.reset

    def spy(self, **kwargs):
        seen.append(dict(kwargs['options']))
        out = reset(self, **kwargs)
        seen.append(self.data.qpos[:2].copy())
        return out
    monkeypatch.setattr(G1VelocityEnv, 'reset', spy)
    _Renderer.shots = []
    clip = configure(get_clip('slow'), terrain_type='stairs_up', terrain_level=5)
    frames = film_lane(terrain_stages[3], clip, .4)
    centre = g1_terrain.tile_centres(3)[5, g1_terrain.TERRAIN_TYPES_3.index('stairs_up')]
    assert seen[0] == {'command': (.5, 0., 0.), 'terrain_type': 'stairs_up', 'terrain_level': 5}
    np.testing.assert_array_equal(seen[1], centre)
    assert len(frames) == len(_Renderer.shots) == 10
    first = _Renderer.shots[0]
    assert (first['distance'], first['elevation'], first['azimuth']) == (3.2, -14., 45.)
    assert abs(first['lookat'][2] - .55) < .05  # the platform: the ground at the heightfield's lift
    assert np.linalg.norm(first['lookat'][:2] - first['pelvis'][:2]) < .01
    assert all(frame[2][:3] == ('stairs_up', 5, 3) and abs(frame[2][3]) < .5 for frame in frames)
    option = first['option']
    assert option.geomgroup[g1_terrain.BOX_GROUP] and not option.geomgroup[clip_module.HIDDEN_GROUP]
    yaw = [2 * math.atan2(shot['qpos'][6], shot['qpos'][3]) for shot in _Renderer.shots]
    assert all(abs(math.remainder(angle + math.pi / 2, math.tau)) < .3 for angle in yaw)
    composed = compose([frames], ['стало'], (64, 54), 'ru', clip.measure)
    assert len(composed) == 10
    with pytest.raises(ValueError, match='task_config terrain'):
        film_lane(stages['walking'], clip, .1)
    with pytest.raises(ValueError, match='генератора рельефа 2'):
        film_lane(terrain_stages[2], configure(get_clip('slow'), terrain_type='pits'), .1)


def test_the_terrain_option_draws_the_ground_the_robot_touches(terrain_stages):
    from humanoid_sim.core import stage_task_config
    from humanoid_sim.tasks import g1_terrain
    from humanoid_sim.tasks.catalog import make_task
    env = make_task('g1_velocity', 'g1_29dof', stage_task_config(terrain_stages[3]))
    try:
        model = env.unwrapped.model
        ground = set(g1_terrain.ground_geoms(model))
        boxed = [geom for geom in g1_terrain.terrain_geoms(model) if geom not in ground]
        assert boxed and g1_terrain.terrain_boxes(model)
        clip_module.terrain_option(model)
        assert {int(model.geom_group[geom]) for geom in boxed} == {clip_module.HIDDEN_GROUP}
        assert {int(model.geom_group[geom]) for geom in g1_terrain.terrain_geoms(model) if geom in ground} == {0}
    finally:
        env.close()


def test_every_lane_of_a_terrain_clip_walks_the_first_lanes_map(terrain_stages, monkeypatch, tmp_path):
    """render fills the map from the first lane's stage (or --seed) and films every lane on it."""
    filmed = []

    def film(stage, clip, *args, **kwargs):
        filmed.append(clip.terrain)
        return [(np.zeros((54, 64, 3), dtype=np.uint8), .04, (*clip.terrain[:2], clip.terrain[3], 0.), False, None,
                 None)]
    monkeypatch.setattr(clip_module, 'film_lane', film)
    items = [('a', terrain_stages[2]), ('b', terrain_stages[3])]
    clip_module.render(items, 'slow', tmp_path / 'c.webp', .04, size=(64, 54), terrain_type='stairs_down',
                       terrain_level=2)
    clip_module.render(items[::-1], 'slow', tmp_path / 'd.webp', .04, size=(64, 54), terrain_type='curbs',
                       terrain_seed=11)
    assert filmed == [('stairs_down', 2, 7, 2)] * 2 + [('curbs', 0, 11, 3)] * 2
