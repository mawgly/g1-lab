"""scripts/robot/passport.py, the candidate passport (25.10): one page per language from the numbers a robot package
already has. A temporary package is built from package.py's own data (SIM2SIM_TABLE and the getting-up tables, as
REFERENCE.md carries them); pushes become forces F = M·Δv/T and impulses M·Δv, a field without a number stays empty,
the page comes in ru / en / zh with the catalog's labels and the numbers in each language's form."""
import importlib
import json
import math
import re
from pathlib import Path

import mujoco
import pytest

from humanoid_sim import i18n
from humanoid_sim.tasks.g1_velocity import load_model

ROOT = Path(__file__).resolve().parents[1]
BRANCH, STEPS = 'exp-28691193718e4d0e', 7221510144
EXPORT = f'{BRANCH}-{STEPS}'
GETUP = '823984128'
REAL = ROOT / 'runs/humanoid_lab/robot/g1-n8-exp-28691193718e4d0e-7221510144'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/robot'))
    return importlib.import_module('passport'), importlib.import_module('package')


def candidate(package):
    c = dict(package.N8, role='основной', slot=f'n8-{STEPS}', stage=str(STEPS),
             onnx=dict(input=['obs', [1, 490]], output=['actions', [1, 29]]), knee_clip=[0.061, 0.061],
             command_ranges=dict(lin_vel_x=[-0.5, 1.0], lin_vel_y=[-0.5, 0.5], ang_vel_z=[-1.0, 1.0]))
    return c


def make_package(folder, package, reference=True, getup=False):
    """A package folder as package.py writes it, its REFERENCE.md tables from package.py's data."""
    main = candidate(package)
    spare = dict(package.N7, role='запасной', slot='n7-6368231424', stage='6368231424')
    info = dict(kind='g1-robot-package', name=folder.name, created='2026-10-23 22:35:12',
                git=dict(commit='0' * 40, uncommitted=[]), candidates=[main, spare])
    text = '# Эталоны для первого запуска\n\n'
    if reference:
        rows = '\n'.join('| ' + ' | '.join(row) + ' |' for row in package.PACKAGES['n8']['table'])
        text += ('## Связка Unitree в симуляции (sim2sim)\n\nТекст.\n\n'
                 f'| Что | №8 (7221) | №7 (6368) |\n|---|---|---|\n{rows}\n\nЕщё текст.\n\n## Подвес в симуляции\n\n'
                 '| Физика | Опыт |\n|---|---|\n| наша | 1 |\n\n')
    if getup:
        info['getup'] = dict(export=f'exp-f19f12fb1eff4ef3-{GETUP}', stage=GETUP, slot=f'getup-{GETUP}')
        text += package.getup_reference_md(dict(info['getup'], max_joint_speed=[30.] * 29))
    text += '## Источники\n'
    (folder / 'policy' / main['slot']).mkdir(parents=True)
    (folder / 'policy' / main['slot'] / 'deploy.yaml').write_text('step_dt: 0.02\n', encoding='utf-8')
    (folder / 'package.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    (folder / 'REFERENCE.md').write_text(text, encoding='utf-8')
    (folder / 'reference').mkdir()
    bias = [dict(model='calib', draw=d, plan=plan, fell=None, yaw_rate=.02 + .001 * d, heading_deg=-5. + d,
                 bias=[.01, -.012]) for d in range(3) for plan in ('w05', 'w1', 'stand')]
    (folder / 'reference' / 'enc_bias_7221.json').write_text(json.dumps(bias), encoding='utf-8')
    return folder


def make_lab(folder, check_stage=None):
    stage = folder / BRANCH / f'stage_{STEPS:012d}'
    stage.mkdir(parents=True)
    (stage / 'metadata.json').write_text(json.dumps(dict(steps=STEPS, task_config=dict(
        force_push=1.25, arm_perturb=0.1, motor_friction=1.25, action_delay=0.0))), encoding='utf-8')
    (stage / 'evaluation.json').write_text(json.dumps(dict(episodes=[dict(fell=False, seconds=20.)] * 4 + [
        dict(fell=True, seconds=7.5)])), encoding='utf-8')
    if check_stage is not None:
        (folder / 'journal').mkdir()
        rows = [dict(label='Курс при ходьбе', value='1,0°', target='< 10°', ok=True)]
        (folder / 'journal' / 'journal.json').write_text(json.dumps(dict(check=dict(
            at=1, branch=BRANCH, stage=check_stage, title='Ветка · 7221,5 млн', items=rows)), ensure_ascii=False),
            encoding='utf-8')
    return folder


def page(passport, folder, lab, lang='ru', chatter=None):
    data = passport.collect(folder, lab, chatter)
    return data, *passport.render(data, lang)


def row(text, label):
    """The cells of the first table row whose first cell is label."""
    for line in text.splitlines():
        if line.startswith(f'| {label} |'):
            return [c.strip() for c in re.split(r'(?<!\\)\|', line)[1:-1]]
    raise AssertionError(f'no row {label!r}')


def test_pushes_are_forces_and_impulses(modules, tmp_path):
    passport, package = modules
    folder = make_package(tmp_path / 'g1-n8-test', package)
    data, text, _ = page(passport, folder, make_lab(tmp_path / 'lab'))
    mass = mujoco.mj_getTotalmass(load_model()[0])
    assert data['mass'] == pytest.approx(mass) and round(mass, 3) == 35.112
    pushes = [p for p in data['bridge']['pushes'] if not p.get('missing')]
    # SIM2SIM_TABLE: standing Δv 0.5 / 1 / 1.25 / 1.5 (8 / 8 / 20 / 8 pushes), walking 0.5 / 1 (4 / 4), 0.1 s each
    assert [(p['state'], p['dv'].value, p['falls']) for p in pushes] == [
        ('stand', .5, (0, 8)), ('stand', 1., (0, 8)), ('stand', 1.25, (0, 20)), ('stand', 1.5, (0, 8)),
        ('walk', .5, (0, 4)), ('walk', 1., (0, 4))]
    for p in pushes:
        assert p['seconds'] == passport.patch_seconds() == .1
        assert p['force'] == pytest.approx(mass * p['dv'].value / p['seconds'])
        assert p['impulse'] == pytest.approx(mass * p['dv'].value)
    # docs/sim2sim.md: «все 52 толчка №8 пришли с запрошенной Δv: 176, 351, 439 и 527 Н × 0,1 с»
    assert [round(p['force']) for p in pushes[:4]] == [176, 351, 439, 527]
    lines = [line for line in text.splitlines() if '| стоя |' in line]
    assert lines[2].endswith('| 1,25 | 0,1 | 439 | 43,9 | 0 из 20 |')
    walking = [line for line in text.splitlines() if '| на ходу 0,5 м/с |' in line]
    assert walking[1].endswith('| 1 | 0,1 | 351 | 35,1 | 0 из 4 |')


def test_numbers_from_the_tables_and_the_extrapolation(modules, tmp_path):
    passport, package = modules
    data, text, empty = page(passport, make_package(tmp_path / 'p', package), make_lab(tmp_path / 'lab'))
    assert row(text, 'Скорость при команде 0,5 м/с')[1] == '0,46–0,47 м/с'
    assert row(text, 'Увод курса без команды поворота, 0,5 м/с')[1] == '+0,032…+0,037 рад/с; курс за 10 с +17,2…+20,2°'
    # 100 m at 0.46-0.47 m/s with 0.032-0.037 rad/s: 0.032·100/0.47 to 0.037·100/0.46 rad, radius v/ω
    e = data['bridge']['walk05']['extrapolated']
    assert e['heading'] == pytest.approx((math.degrees(.032 * 100 / .47), math.degrees(.037 * 100 / .46)))
    assert e['radius'] == pytest.approx((.46 / .037, .47 / .032))
    assert row(text, 'Курс через 100 м без поправки, 0,5 м/с (экстраполяция)')[1] == '+390…+461°, дуга радиусом 12–15 м'
    assert 'экстраполяция' in row(text, 'Курс через 100 м без поправки, 1 м/с (экстраполяция)')[2]
    assert row(text, 'Торможение: путь за 3 с после стопа с 1 м/с')[1] == '0,51–0,52 м'
    assert row(text, 'Цели вне clip, все 29 суставов (сообщения lowcmd в Velocity)')[1] == '0 из 719 902'
    assert row(text, 'Прогонов в связке; падений')[1] == '31; 0'
    # the stage's own evaluation (4 of 5 stood) and the training's pushes as forces: 1.25 m/s over 0.05 s at most
    assert row(text, 'Оценка этапа со случайными командами и помехами обучения')[1] == 'падений 1 из 5, эпизоды по 20 с'
    assert f"F до {round(data['mass'] * 1.25 / .05)} Н" in row(text, 'Оценка этапа со случайными командами и помехами '
                                                                    'обучения')[2]
    assert 'Нагрев моторов при удержании груза' in empty
    # №10's column (package n10): remarks are not numbers, «в связке не мерили» is no number
    values = {what: mine for what, mine, _ in package.SIM2SIM_TABLE_N10}
    b = passport.bridge_data(values, data['mass'], .1, [])
    assert [x.value for x in b['stand30']['offset']] == [3., 3.]  # '3 см, после 2-й с 1 см'
    assert [x.value for x in b['delay'][2]] == [12., 20.]  # '12–20 мс (в стояниях 22–26)'
    assert b['turn'] is None and [p.get('missing') for p in b['pushes'] if p['state'] == 'walk'] == [True]


def test_fields_without_numbers_stay_empty(modules, tmp_path):
    """No REFERENCE.md, no lab, no chatter probe: every such field is empty, with its reason, nothing made up."""
    passport, package = modules
    folder = make_package(tmp_path / 'p', package, reference=False)
    (folder / 'reference' / 'enc_bias_7221.json').unlink()
    data, text, empty = page(passport, folder, tmp_path / 'no-lab')
    for label in ('Скорость при команде 0,5 м/с', 'Курс через 100 м без поправки, 0,5 м/с (экстраполяция)',
                  'Прогонов в связке; падений', 'Наибольший наклон корпуса при толчках стоя',
                  'Оценка этапа со случайными командами и помехами обучения', 'Смещение энкодеров',
                  'goal_check: условия цели этапа 1', 'СКЗ момента мотора выше 10 Гц, худший сустав',
                  'Среднее |Δa| за шаг управления', 'Порог и итог', 'Нагрев моторов при удержании груза'):
        assert label in empty
        assert row(text, label.replace('|', '\\|'))[1] == ''
    assert row(text, 'Скорость при команде 0,5 м/с')[2] == 'нет числа в REFERENCE.md пакета'
    assert row(text, 'Порог и итог')[2] == 'проба не приложена (--chatter)'
    assert 'измеряется на роботе' in row(text, 'Нагрев моторов при удержании груза')[2]
    assert 'Толчки, связка: стоя' in empty and '| связка unitree_mujoco + g1_ctrl | стоя |  |  |  |  |  |' in text
    assert row(text, 'Масса модели (mj_getTotalmass)')[1] == '35,112 кг'  # the model is always there
    assert 'Вставание' not in text.split('## Надёжность')[0].split('## Проверка')[1]  # no getting up in the package


def test_goal_check_only_of_this_stage(modules, tmp_path):
    passport, package = modules
    folder = make_package(tmp_path / 'p', package)
    _, text, empty = page(passport, folder, make_lab(tmp_path / 'mine', check_stage=STEPS))
    assert row(text, 'Курс при ходьбе')[1:] == ['1,0°', '< 10°', '✓']
    _, text, empty = page(passport, folder, make_lab(tmp_path / 'other', check_stage=STEPS + 1))
    assert 'goal_check: условия цели этапа 1' in empty
    assert '7221,5 млн' in row(text, 'goal_check: условия цели этапа 1')[2]


def goal_file(folder, stage, export=None):
    """goal_check's printed output (rows, the verdict, the JSON line last) for `stage`, with goal_check's own rows."""
    from humanoid_sim.goal_check import items
    scenario = lambda forward: dict(falls='0/5', forward_m=forward, sideways_m=0., heading_deg=1., torso_lean_deg=3.,
                                    clipped=0.)
    walk = dict(turn={(1., .5): .37, (1., -.5): -.33}, drift={(1., 'left'): [.032]}, pelvis=.74,
                feet=dict(lift_m=.04, stance_tilt_deg=2., by_foot={}))
    rows = items({'stand': scenario(.02), 'walk': scenario(9.4), 'stop_start': scenario(4.)}, .45, walk)
    path = folder / f'goal_{stage}.json'
    printed = [f"{'✓' if r['ok'] else '✗'} {r['label']}: {r['value']} (цель {r['target']})" for r in rows]
    line = json.dumps(dict(stage=stage, export=export, items=rows), ensure_ascii=False)
    path.write_text('\n'.join(printed + ['выполнено 10 из 10', line]) + '\n', encoding='utf-8')
    return path


def chatter_file(path, stage=f'{BRANCH} @ {STEPS}'):
    """chatter.py --json for `stage` (its label): the summary as its docstring has it, №8's numbers of 25.10 (4.57 N m
    / 0.135 rad against v0's 4.08 / 0.112), gated on torque_hf_rms_nm and action_hf_rms_rad."""
    summary = dict(action_hf_share=.4456, torque_hf_share=.42, torque_hf_rms_nm=4.568, torque_rms_peak_share=.185,
                   mean_abs_delta_action=.299, action_hf_rms_rad=.13497, torque_hf_peak_share=.046,
                   worst_joint='left_knee_joint',
                   worst_joints=dict(action_hf_share='left_shoulder_yaw_joint', torque_hf_share='left_wrist_roll_joint',
                                     torque_hf_rms_nm='left_knee_joint', torque_rms_peak_share='left_hip_pitch_joint',
                                     mean_abs_delta_action='left_ankle_pitch_joint',
                                     action_hf_rms_rad='left_ankle_pitch_joint'),
                   runs=18, falls=[], reference=dict(action_hf_share=.688, torque_hf_rms_nm=4.08,
                                                     action_hf_rms_rad=.1118), ok=True,
                   gate='худший сустав: torque_hf_rms_nm и action_hf_rms_rad <= 1.5 x Unitree v0, без падений')
    settings = dict(action_cutoff=5., torque_cutoff=10., gate=dict(keys=['torque_hf_rms_nm', 'action_hf_rms_rad'],
                                                                     factor=1.5))
    path.write_text(json.dumps(dict(stage=stage, path=str(path.parent / BRANCH / f'stage_{STEPS:012d}'),
                                    settings=settings, summary=summary)), encoding='utf-8')
    return path


def test_goal_check_file_of_this_stage_or_its_export_only(modules, tmp_path):
    """--goal-check: goal_check's printed output, its last line the JSON (stage, export, items); another stage's file
    leaves the field empty with the reason, and the journal's check is not used in its place."""
    passport, package = modules
    folder, lab = make_package(tmp_path / 'p', package), make_lab(tmp_path / 'lab', check_stage=STEPS)
    saved = lambda stage, export=None: goal_file(tmp_path, stage, export)
    data = passport.collect(folder, lab, goal_check=saved(STEPS))
    text, empty = passport.render(data, 'ru')
    assert 'goal_check: условия цели этапа 1' not in empty
    assert f'Проверка из файла `goal_{STEPS}.json`: этап {STEPS}, условий: 10' in text
    assert row(text, 'Увод без команды (0,5 и 1 м/с, старт с любой ноги)')[1:] == ['0,032 рад/с', '< 0,05', '✓']
    assert row(text, 'Ходьба 1 м/с: метров за 10 с')[1:] == ['9,4 м', '8–12 м', '✓']
    text, _ = passport.render(data, 'en')  # goal_check's labels are in the catalogs already
    assert not re.search('[А-Яа-яЁё]', row(text, i18n.translate('Ходьба 1 м/с: метров за 10 с', 'en'))[1])
    # an export of this stage counts; another stage does not, and the journal's check of this stage is not taken instead
    other = saved(STEPS + 10_000_000, str(tmp_path / EXPORT))
    text, _ = passport.render(passport.collect(folder, lab, goal_check=other), 'ru')
    assert f'выгрузка `{EXPORT}`' in text
    other = saved(STEPS + 10_000_000)
    data = passport.collect(folder, lab, goal_check=other)
    text, empty = passport.render(data, 'ru')
    assert 'goal_check: условия цели этапа 1' in empty and data['warnings']
    assert row(text, 'goal_check: условия цели этапа 1')[2] == \
        f'файл `goal_{STEPS + 10_000_000}.json` — проверка другого этапа: {STEPS + 10_000_000}'
    assert 'Курс при ходьбе' not in text
    bad = tmp_path / 'bad.json'
    bad.write_text('{"stage": 1}\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        passport.collect(folder, lab, goal_check=bad)


def test_chatter_probe_of_this_stage_or_none(modules, tmp_path):
    passport, package = modules
    folder, lab = make_package(tmp_path / 'p', package), make_lab(tmp_path / 'lab')
    probe = chatter_file(tmp_path / 'chatter.json')
    _, text, empty = page(passport, folder, lab, chatter=probe)
    assert row(text, 'СКЗ момента мотора выше 10 Гц, худший сустав — судится')[1] == \
        '4,57 Н·м (`left_knee`); Unitree v0: 4,08 Н·м; порог 1,5 × v0 = 6,12 Н·м'
    assert row(text, 'СКЗ цели ПД выше 5 Гц, худший сустав — судится')[1] == \
        '0,1350 рад (`left_ankle_pitch`); Unitree v0: 0,1118 рад; порог 1,5 × v0 = 0,1677 рад'
    assert row(text, 'Доля дисперсии цели ПД выше 5 Гц, худший сустав')[1] == \
        '0,446 (`left_shoulder_yaw`); Unitree v0: 0,688'  # not judged: no threshold
    assert row(text, 'Порог и итог')[1] == ('худший сустав: `torque_hf_rms_nm`, `action_hf_rms_rad` не больше 1,5 × '
                                           'Unitree v0, без падений; проходит')
    assert not [label for label in empty if 'Гц' in label or label == 'Порог и итог']
    _, text, _ = page(passport, folder, lab, 'en', chatter=probe)
    en = i18n.gettext('en')
    label = en('СКЗ цели ПД выше {n} Гц, худший сустав', n='5') + ' — ' + en('судится')
    assert row(text, label)[1].startswith(en('{n} рад', n='0.1350') + ' (`left_ankle_pitch`)')
    # a probe of another stage is not this candidate's
    payload = json.loads(probe.read_text(encoding='utf-8'))
    probe.write_text(json.dumps(dict(payload, stage='exp-1 @ 1', path='x')), encoding='utf-8')
    data, text, empty = page(passport, folder, lab, chatter=probe)
    assert 'Порог и итог' in empty and row(text, 'Порог и итог')[2] == 'проба другого этапа: exp-1 @ 1'
    assert data['warnings']


def test_getting_up_from_its_own_columns(modules, tmp_path):
    passport, package = modules
    folder = make_package(tmp_path / 'p', package, getup=True)
    data, text, empty = page(passport, folder, make_lab(tmp_path / 'lab'))
    physics = data['getup']['physics']
    assert [(p['id'], p['succeeded'], p['total'], p['refall']) for p in physics] == [
        ('ours', 276, 276, 0), ('delay', 276, 276, 0), ('floor', 275, 276, 0), ('noise', 274, 276, 1)]
    assert '| 275 из 276 (99,6%) | 1,06 / 2,09 / 5,92 | 0 |' in text
    assert '| связка unitree_mujoco + g1_ctrl | 18 из 18 | 1,1 / 2,3 / 2,95 |  |' in text
    assert row(text, 'Ходьба после подъёма, связка: падений')[1] == '0 из 18'
    assert row(text, 'В пакете')[1].endswith('вставание (GetUp): этап 823')
    assert 'Вставание, связка: повторных падений после подъёма' in empty


def test_three_languages(modules, tmp_path, monkeypatch):
    passport, package = modules
    folder = make_package(tmp_path / 'p', package, getup=True)
    written, _ = passport.write(folder, tmp_path / 'out', chatter=None, lab=make_lab(tmp_path / 'lab'))
    assert [path.name for path, _ in written] == ['PASSPORT.ru.md', 'PASSPORT.en.md', 'PASSPORT.zh.md']
    ru, en, zh = (path.read_text(encoding='utf-8') for path, _ in written)
    assert [len(empty) for _, empty in written] == [len(written[0][1])] * 3  # the same fields, empty in each
    assert '35,112 кг' in ru and '0,46–0,47 м/с' in ru and '+0,032…+0,037' in ru
    for text in (en, zh):
        assert '35.112' in text and '0.46–0.47' in text and '+0.032…+0.037' in text and '0,46' not in text
        assert '719,902' in text and '| 1.25 | 0.1 | 439 | 43.9 |' in text
    # labels are catalog keys: a translated one comes through, an untranslated one stays Russian
    assert f"| {i18n.gettext('en')('Значение')} |" in en and f"| {i18n.gettext('zh')('Значение')} |" in zh
    real = i18n.catalog

    def catalog(lang):
        found = real(lang)
        return i18n.Catalog({**found.exact, 'Устойчивость к толчкам': 'Push recovery'})
    monkeypatch.setattr(i18n, 'catalog', catalog)
    text, _ = passport.render(passport.collect(folder, tmp_path / 'lab'), 'en')
    assert '## Push recovery' in text and '| 0.46–0.47 m/s |' in text
    # every label of the page is a key tests/test_i18n.py checks and journal_translate export hands out
    found = i18n.server_messages()
    assert {'Паспорт кандидата: Unitree G1, {title}', 'Устойчивость к толчкам', '{k} из {n}'} <= found


def test_units_by_the_table():
    """Numbers with a unit go through i18n.unit (the table i18n.UNITS), not through catalog keys: «{d} м» would be
    covered by the catalog's «{n} м» of the same shape (i18n.missing) and yet stay Russian."""
    assert i18n.unit('0,23–0,26', 'm', 'ru') == '0,23–0,26 м' and i18n.unit('0.23–0.26', 'm', 'en') == '0.23–0.26 m'
    assert i18n.unit('+0.449 / −0.514', 'rad/s', 'zh') == '+0.449 / −0.514 rad/s'
    assert {symbol: i18n.unit('1', symbol, 'ru') for symbol in ('cm', 'mm', 's', 'ms', 'N', 'N·m', 'kg', 'm/s')} == {
        'cm': '1 см', 'mm': '1 мм', 's': '1 с', 'ms': '1 мс', 'N': '1 Н', 'N·m': '1 Н·м', 'kg': '1 кг', 'm/s': '1 м/с'}
    assert not any(re.search('[А-Яа-яЁё]', symbol) for symbol in i18n.UNITS)


def test_english_and_chinese_pages_have_no_russian_but_document_names(modules, tmp_path):
    """Everything on the en / zh page is in the catalogs or language-free; Cyrillic stays only in the name of a section
    of a Russian document (docs/sim2sim.md), marked as such. Fails while the catalogs lack a key of render()."""
    passport, package = modules
    folder, lab = make_package(tmp_path / 'p', package, getup=True), make_lab(tmp_path / 'lab')
    data = passport.collect(folder, lab, chatter_file(tmp_path / 'chatter.json'), goal_file(tmp_path, STEPS))
    name = data['candidate']['sim2sim']
    assert name == 'Этап 7221 (итерация 27)'
    for lang in ('en', 'zh'):
        text, _ = passport.render(data, lang)
        assert name in text
        assert i18n.gettext(lang)('(раздел русского документа docs/sim2sim.md)') in text
        left = sorted({line for line in text.replace(name, '').splitlines() if re.search('[А-Яа-яЁё]', line)})
        assert left == [], (lang, left)


def test_symmetric_walk_package(modules, tmp_path):
    """Package n8s (26.10): the export's name carries export_g1's suffix (…-7221510144-symwalk), its table
    (SIM2SIM_TABLE_N8S) keeps the wording of the rows the passport reads; no bridge pushes, so those stay empty."""
    passport, package = modules
    folder = make_package(tmp_path / 'g1-n8s-test', package)
    info = json.loads((folder / 'package.json').read_text(encoding='utf-8'))
    info['candidates'][0] = dict(info['candidates'][0], **{k: package.N8S[k] for k in ('label', 'title', 'export',
                                                                                         'reference', 'about')})
    (folder / 'package.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    rows = '\n'.join('| ' + ' | '.join(row) + ' |' for row in package.PACKAGES['n8s']['table'])
    (folder / 'REFERENCE.md').write_text('## Связка Unitree в симуляции (sim2sim)\n\n| Что | №8 симм. (7221sym) | №7 (6368) '
                                         f'|\n|---|---|---|\n{rows}\n\n## Источники\n', encoding='utf-8')
    data, text, empty = page(passport, folder, make_lab(tmp_path / 'lab'))
    assert (data['candidate']['branch'], data['candidate']['steps']) == (BRANCH, STEPS)
    assert row(text, 'Скорость при команде 0,5 м/с')[1] == '0,46 м/с'
    assert row(text, 'Скорость при команде 1 м/с')[1] == '1,01 м/с'
    assert row(text, 'Увод курса без команды поворота, 0,5 м/с')[1] == '0,000…+0,001 рад/с; курс за 10 с −0,8…−0,4°'
    assert row(text, 'Прогонов в связке; падений')[1] == '14; 0'
    assert row(text, 'Торможение: путь за 3 с после стопа с 0,5 м/с')[1] == '0,22–0,23 м'
    assert 'Толчки, связка: стоя' in empty and 'Скорость при команде 0,5 м/с' not in empty


@pytest.mark.skipif(not (REAL / 'package.json').is_file(), reason='the lab package №8 is not on this machine')
def test_real_package_8(modules, tmp_path):
    passport, _ = modules
    written, _ = passport.write(REAL, tmp_path, langs=('ru',))
    text = written[0][0].read_text(encoding='utf-8')
    assert written[0][0].parent == tmp_path  # --out: nothing written into the package
    assert '| стоя | 1,5 | 0,1 | 527 | 52,7 | 0 из 8 |' in text
    assert row(text, 'Скорость при команде 1 м/с')[1] == '0,99–1,00 м/с'
