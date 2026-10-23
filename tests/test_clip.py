import argparse
import math
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from humanoid_sim.clip import (CLIPS, FPS, MARK_SECONDS, TEXTS, TILE, _keep_push, _size, _tile_font, compose,
                               film_lane, footer, push_label)
from humanoid_sim.collage import _font
from humanoid_sim.evaluation.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = 'exp-8ce0cbd70d0a4f1d:6039109632'  # candidate 6
CYRILLIC = re.compile('[А-Яа-яЁё]')
NEW = ('back', 'side_left', 'side_right', 'turn_left', 'turn_right', 'walk_turn', 'side_stop', 'push_back',
       'push_forward', 'push_left', 'push_right', 'push_walk')


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
    for lang in ('en', 'zh'):
        for measure in {clip.measure for clip in CLIPS.values()}:
            for fell in (False, True):
                text = footer(9.96, -12.345, fell, measure, lang)
                assert not CYRILLIC.search(text) and not re.search(r'\d,\d', text), text
        for push in (clip.push for clip in CLIPS.values() if clip.push):
            assert not CYRILLIC.search(push_label((push[1], '→'), lang))


def test_chinese_tiles_use_a_font_with_cjk_glyphs():
    def drawn(font, text):
        image = Image.new('L', (80, 30))
        ImageDraw.Draw(image).text((2, 2), text, font=font, fill=255)
        return np.asarray(image)
    font = _tile_font(16, 'zh')  # collage's Segoe UI would draw each of these as the .notdef box
    missing = drawn(font, '\U0010fffd')  # a code point no font has: the .notdef box
    assert all(not np.array_equal(drawn(font, c), missing) for c in '前进跌倒推位移左转')


def test_scenarios_and_sizes():
    assert set(SCENARIOS) | set(NEW) == set(CLIPS)
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
