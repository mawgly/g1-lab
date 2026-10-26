"""The home page's changing part as data (26.10): humanoid_sim.home (home.json, its check and its translation),
scripts/site/home_update.py (built from a robot package, skills_check and the journal, nothing measured again; a
number its source lacks stays empty with the reason), /api/lab/home and site_export's data/home.<lang>.json."""
import importlib
import json
import threading
import urllib.request

import pytest

from humanoid_sim import home, i18n, journal, journal_translate, site_export
from humanoid_sim.lab import Laboratory
from humanoid_sim.webapp import DashboardServer
from test_passport import BRANCH, GETUP, ROOT, STEPS, make_lab, make_package

WEB = i18n.WEB


def sample(**change):
    """A small home.json: one candidate with a bridge row, a skills row, an empty field and a weak spot."""
    data = dict(version=1, at=1792990000., date='2026-10-26',
                roadmap=dict(total=10, done=1, active=[dict(number=2, title='Все движения ног')]),
                stats=[dict(value='0 / 31', label='падений в связке')],
                candidates=[dict(id='walk', title='Ходьба: кандидат №8', about='Ветка exp-1, этап 7221,5 млн.',
                                 summary=[dict(label='Прогонов в связке; падений', value='31; 0', where='bridge')],
                                 skills=[dict(title='Ходьба 1 м/с', clip='walk', alt='Идёт', rows=[
                                     dict(label='Скорость при команде 1 м/с', value='0,99–1,00', unit='m/s',
                                          where='bridge'),
                                     dict(label='Курс за 10 с ходьбы 1 м/с', value='+10,2…+11,0°', where='bridge',
                                          weak=True),
                                     dict(label='Толчок стоя: падений', value='0 из 8', where='bridge'),
                                     dict(label='Назад 0,5 м/с в связке', value=None,
                                          why='нет числа в REFERENCE.md пакета', where='bridge')])])],
                missing=[], sources=[dict(file='runs/x.json', what='пакет', md5='0')])
    data.update(change)
    return data


def test_home_data_is_checked_and_saved(tmp_path):
    assert home.load(tmp_path) is None
    saved = home.save(tmp_path, sample())
    assert home.load(tmp_path) == saved and home.path(tmp_path) == tmp_path / 'journal' / 'home.json'
    row = saved['candidates'][0]['skills'][0]['rows'][3]
    assert row == dict(label='Назад 0,5 м/с в связке', value=None, why='нет числа в REFERENCE.md пакета',
                       where='bridge')
    assert home.clips(saved) == ['walk']
    bad = sample()
    bad['candidates'][0]['skills'][0]['rows'][0]['unit'] = 'furlong'
    with pytest.raises(ValueError, match='единица'):
        home.validate(bad)
    bad = sample()
    bad['candidates'][0]['skills'][0]['clip'] = '../journal'
    with pytest.raises(ValueError, match='ролик'):
        home.validate(bad)
    bad = sample()
    bad['candidates'][0]['skills'][0]['rows'][3].pop('why')
    with pytest.raises(ValueError, match='причина'):
        home.validate(bad)  # an empty field always says why


def test_home_data_in_each_language(tmp_path):
    saved = home.validate(sample())
    content = i18n.Content(tmp_path)
    russian = home.localize(saved, 'ru', content)
    rows = russian['candidates'][0]['skills'][0]['rows']
    assert rows[0]['value'] == '0,99–1,00 м/с' and 'unit' not in rows[0] and 'sources' not in russian
    folder = tmp_path / 'journal' / 'i18n'
    folder.mkdir(parents=True)
    (folder / 'en.json').write_text(json.dumps({'units': {
        i18n.unit_id('Ходьба 1 м/с'): {'text': 'Walking 1 m/s'}}}), encoding='utf-8')
    english = home.localize(saved, 'en', content)
    card = english['candidates'][0]['skills'][0]
    assert card['title'] == 'Walking 1 m/s'  # lab text: journal_translate's cache
    assert card['rows'][0]['label'] == 'Speed at command 1 m/s'  # the passport's label: the catalog's
    assert card['rows'][0]['value'] == '0.99–1.00 m/s' and card['rows'][1]['value'] == '+10.2…+11.0°'
    assert card['rows'][2]['value'] == '0 of 8' and card['rows'][1]['weak'] is True
    assert card['rows'][3]['why'] == i18n.translate('нет числа в REFERENCE.md пакета', 'en')
    assert english['candidates'][0]['about'] == 'Ветка exp-1, этап 7221,5 млн.'  # untranslated yet: Russian


def test_home_text_goes_to_the_translator(tmp_path):
    lab = Laboratory(tmp_path / 'original')
    home.save(lab.root, sample())
    units = journal_translate.units(lab.root)
    assert i18n.unit_id('Ходьба: кандидат №8') in units and i18n.unit_id('Курс за 10 с ходьбы 1 м/с') in units
    # catalog strings (the passport's labels) and values need no paragraph translation
    assert i18n.unit_id('Скорость при команде 1 м/с') not in units and i18n.unit_id('0 из 8') not in units


def test_api_and_static_site_serve_the_home_data(tmp_path):
    server = DashboardServer(('127.0.0.1', 0), tmp_path / 'original')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'

    def get(path):
        with urllib.request.urlopen(url + path) as response:
            return json.load(response)
    try:
        assert get('/api/lab/home') == {'empty': True}
        home.save(server.lab.root, sample())
        reply = get('/api/lab/home?lang=zh')
        assert reply['candidates'][0]['skills'][0]['rows'][0]['value'] == '0.99–1.00 m/s'
        assert 'sources' not in reply and get('/api/lab/home')['stats'][0]['value'] == '0 / 31'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    lab = server.lab
    for page in site_export.site_pages():
        for clip in site_export.CLIP.findall((WEB / page).read_text(encoding='utf-8')) + ['walk']:
            for lang in i18n.LANGUAGES:
                (journal.folder(lab.root) / f'site-{lang}-{clip}.webp').write_bytes(b'clip')
                (journal.folder(lab.root) / f'site-{lang}-{clip}-poster.jpg').write_bytes(b'poster')
    out = tmp_path / 'site'
    summary = site_export.export(out, lab.original)
    for lang in i18n.LANGUAGES:
        data = json.loads((out / 'data' / f'home.{lang}.json').read_text(encoding='utf-8'))
        assert data['date'] == '2026-10-26' and 'sources' not in data
    assert [name for name in summary['missing'] if name.startswith('media/')] == []


def test_home_page_reads_the_data():
    page = (WEB / 'site.html').read_text(encoding='utf-8')
    script = (WEB / 'home.js').read_text(encoding='utf-8')
    for marker in ('id="home-updated"', 'id="home-stats"', 'id="home-candidates"'):
        assert marker in page
    assert 'Обновлено 23 октября' not in page and 'Что умеет кандидат 8' not in page  # no hand-written numbers left
    assert "data('home')" in script and 'wire(candidates.querySelectorAll' in script
    assert 'row.value !== null' in script and '<details' in script  # empty fields hidden, the rest folded away
    for key in ('физика Unitree', 'физика Unitree, со столкновением стоп', 'наша симуляция', 'пакет для робота'):
        assert f"t('{key}')" in script  # home.WHERE's tags
    assert set(home.WHERE) == {'bridge', 'bridge_copy', 'skills', 'goal', 'package'}


# --- scripts/site/home_update.py -----------------------------------------------------------------------------------


@pytest.fixture
def update(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'robot'))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'site'))
    module = importlib.import_module('home_update')
    checked = []
    monkeypatch.setattr(module, 'licence', lambda package, slot: checked.append(slot))
    module.checked = checked
    return module, importlib.import_module('package')


def skills_json(path):
    rows = [dict(key='speed:1', skill='Скорость', label='Вперёд 1 м/с: скорость к команде, худшее зерно',
                 target='≥ 0,9', ok=True, note='порог и эталон', value='1,01', reference='1,01', v0='0,94',
                 v0_vs='better'),
            dict(key='speed:2', skill='Скорость', label='Вперёд 2 м/с: скорость к команде, худшее зерно',
                 target='≥ 0,9', ok=True, note='порог и эталон', value='0,98', reference='0,98', v0='0,89',
                 v0_vs='better'),
            dict(key='brake:0.5', skill='Торможение', label='После 0,5 м/с: путь за 3 с, м', target='≤ 0,25',
                 ok=True, note='порог и эталон', value='0,17', reference='0,17', v0='0,13', v0_vs='same'),
            dict(key='armp:turn', skill='Руки', label='Стоя: худший поворот, °', target='≤ 10', ok=True,
                 note='порог не взят, как и у эталона', value='55,6', reference='55,6', v0='н/п (у v0 нет)'),
            dict(key='chatter:action_hf_rms_rad', skill='Дрожание', label='Цель выше 5 Гц, худший сустав, рад',
                 target='≤ 1,5 × v0 (0,112)', ok=True, note='порог и эталон', value='0,135', reference='0,135',
                 v0='0,112', v0_vs='worse'),
            dict(key='height:0.70', skill='Присед', label='Стоя на 0,70: таз, м', target='≤ 0,72', ok=None,
                 note='н/п: нет команды высоты', value='—', reference='н/п')]
    path.write_text(json.dumps(dict(stage=dict(label=f'{BRANCH} @ {STEPS}'), reference=dict(label='тот же'),
                                    rows=rows), ensure_ascii=False), encoding='utf-8')
    return path


def getup_card(lab):
    items = [dict(skill='Вставание: готовность к роботу', label='Старт на спине, все физики', target='не судится',
                  value='72 из 72', reference='72 из 72', note='не хуже эталона', ok=True),
             dict(skill='Вставание: стойка после подъёма', label='Тяга крена бёдер внутрь, Н·м', target='≤ 6',
                  value='9,9', reference='16,2', note='порог не взят, как и у эталона', ok=True),
             dict(skill='Вставание: готовность к роботу', label='Подъём, медиана, наша физика, с',
                  target='не судится', value='1,66', reference='1,11', note='хуже эталона', ok=False)]
    saved = journal.load(lab)
    saved['skills'] = dict(at=1, branch='exp-f19f12fb1eff4ef3', stage=int(GETUP), title='вставание',
                           against=dict(branch='exp-0', stage=753795072, title='753'), items=items)
    saved['roadmap'] = [dict(title='Ходьба', status='done', detail='—', story=None),
                        dict(title='Все движения ног', status='active', detail='—', story=None),
                        dict(title='Руки по целям', status='active', detail='—', story=None),
                        dict(title='Кисти', status='next', detail='—', story=None)]
    (journal.folder(lab) / 'journal.json').write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')


def rows_of(data, card_title):
    card = next(c for candidate in data['candidates'] for c in candidate['skills'] if c['title'] == card_title)
    return {r['label']: r for r in card['rows']}


def test_home_update_builds_from_the_package(update, tmp_path):
    module, package = update
    folder = make_package(tmp_path / 'g1-n8-test', package, getup=True)
    lab = make_lab(tmp_path / 'lab', check_stage=STEPS)
    getup_card(lab)
    data = module.build(lab, folder, walk_skills=skills_json(tmp_path / 'skills.json'), render=False)
    clean = home.validate(data)  # what home.save writes
    assert module.checked == [f'n8-{STEPS}', f'getup-{GETUP}']  # both policies pass package.py's LAFAN1 check
    assert clean['roadmap'] == dict(total=4, done=1, active=[dict(number=2, title='Все движения ног'),
                                                             dict(number=3, title='Руки по целям')])
    stats = [s['value'] for s in clean['stats']]
    assert stats[:2] == ['0 / 31', '52 / 52'] and stats[-1] == '0,98' and clean['stats'][-1]['accent']
    assert 'стоя до Δv 1,5 м/с (527 Н, 0,1 с)' in clean['stats'][1]['label']
    walk = rows_of(clean, 'Ходьба 1 м/с')
    assert walk['Скорость при команде 1 м/с'] == dict(label='Скорость при команде 1 м/с', value='0,99–1,00',
                                                       unit='m/s', where='bridge')
    # the journal's goal_check gives the heading limit (< 10°): +10,2…+11,0° over 10 s in the bridge misses it
    assert walk['Курс за 10 с ходьбы 1 м/с']['weak'] is True
    assert walk['Вперёд 1 м/с: скорость к команде, худший из 2 прогонов']['v0'] == '0,94'
    # skills_check's brake limit (≤ 0,25) against the bridge's 0,23–0,26 m
    assert rows_of(clean, 'Стоять, идти, остановиться')['Торможение: путь за 3 с после стопа с 0,5 м/с']['weak']
    back = rows_of(clean, 'Назад 0,5 м/с')['Назад 0,5 м/с в физике Unitree']
    assert back['value'] is None and back['why'] == 'нет числа в REFERENCE.md пакета'
    sprint = rows_of(clean, 'Бег, команда 2 м/с')['Бег в физике Unitree']
    assert sprint['value'] is None and 'deploy.yaml вперёд до 1 м/с' in sprint['why']
    assert rows_of(clean, 'Руки в случайной позе')['Стоя: худший поворот, °']['weak'] is True  # порог не взят
    chatter = rows_of(clean, 'Моторы: дрожание моментов')
    assert chatter['Цель выше 5 Гц, худший сустав, рад']['weak'] is True  # Unitree v0 does better
    labels = {r['label'] for c in clean['candidates'] for card in c['skills'] for r in card['rows']}
    assert 'Стоя на 0,70: таз, м' not in labels  # н/п rows stay out
    # the lab's own words stay off the public page (PUBLIC)
    shown = ' '.join([c['title'] for c in clean['candidates']] + [s['label'] for s in clean['stats']]
                     + [card['title'] for c in clean['candidates'] for card in c['skills']] + list(labels))
    for word in ('связк', 'зерно', 'наша физика', 'Наша физика', 'копия', 'goal_check', 'getup_probe'):
        assert word not in shown, word
    getup = clean['candidates'][1]
    assert getup['title'] == 'Вставание после падения: этап 823'
    assert rows_of(clean, 'Встаёт со спины')['Старт на спине, все варианты физики']['value'] == '72 из 72'
    assert rows_of(clean, 'Стойка после подъёма')['Тяга крена бёдер внутрь, Н·м']['weak'] is True
    # slower than the previous stage is no weak spot: the home page judges by thresholds
    assert 'weak' not in rows_of(clean, 'Готовность к роботу')['Подъём, медиана, наша симуляция, с']
    assert rows_of(clean, 'В физике Unitree (unitree_mujoco + g1_ctrl)')['Встал в физике Unitree']['value'] == '18 из 18'
    assert getup['summary'][0]['value'] == '2 из 3'
    # clips: none rendered here and none on disk, so the cards go without and say so
    assert all('clip' not in card for c in clean['candidates'] for card in c['skills'])
    assert any(m['what'] == 'ролик walk' for m in clean['missing'])


def test_home_update_without_skills_check(update, tmp_path):
    module, package = update
    folder = make_package(tmp_path / 'g1-n8-test', package)
    lab = make_lab(tmp_path / 'lab')
    data = home.validate(module.build(lab, folder, render=False))
    summary = {r['label']: r for r in data['candidates'][0]['summary']}
    assert summary['Проверка навыков: выполнено']['value'] is None
    assert 'skills_check' in summary['Проверка навыков: выполнено']['why']
    assert list(rows_of(data, 'Бег, команда 2 м/с')) == ['Бег в физике Unitree']  # skills_check's rows are not invented
    assert [s['value'] for s in data['stats']] == ['0 / 31', '52 / 52']  # no made-up speed card
    assert len(data['candidates']) == 1 and any(m['what'] == 'Вставание' for m in data['missing'])
    with pytest.raises(SystemExit, match='не 7221510144'):
        other = skills_json(tmp_path / 'other.json')
        other.write_text(other.read_text(encoding='utf-8').replace(str(STEPS), '1'), encoding='utf-8')
        module.build(lab, folder, walk_skills=other, render=False)


def test_thresholds_and_weak_spots(update):
    module, _ = update
    assert module.limit('≤ 0,25') == ('≤', .25) and module.limit('< 10°') == ('<', 10.)
    assert module.limit('не меньше ±0,25') is None and module.limit('не судится') is None
    assert module.misses(.26, ('≤', .25)) and not module.misses(.25, ('≤', .25))
    assert module.misses(10., ('<', 10.)) and module.misses(.89, ('≥', .9)) and not module.misses(1, None)
    assert module.skill_weak(dict(note='порог не взят', v0_vs='same'))
    assert module.skill_weak(dict(note='порог и эталон', v0_vs='worse'))
    assert not module.skill_weak(dict(note='хуже эталона', v0_vs=None))


def test_a_lafan_stage_never_reaches_the_candidates(monkeypatch, tmp_path):
    """The candidates' cards come from robot packages alone, through package.py's licence check: a package whose
    getting-up policy was trained with a LAFAN1 reward stops the update instead of reaching the page."""
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'robot'))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts' / 'site'))
    module, package = importlib.import_module('home_update'), importlib.import_module('package')
    folder = make_package(tmp_path / 'g1-n8-test', package, getup=True)
    lab = make_lab(tmp_path / 'lab', check_stage=STEPS)
    walk = lab / BRANCH / f'stage_{STEPS:012d}'
    (folder / 'policy' / f'n8-{STEPS}' / 'deploy.yaml').write_text(
        f'# humanoid_sim.export_g1: {walk} ({STEPS} steps)\nstep_dt: 0.02\n', encoding='utf-8')
    lafan = lab / 'exp-f19f12fb1eff4ef3' / f'stage_{int(GETUP):012d}'
    lafan.mkdir(parents=True)
    (lafan / 'metadata.json').write_text(json.dumps(dict(task_config=dict(reward_weights=dict(style=1.)))),
                                         encoding='utf-8')
    (folder / 'policy' / f'getup-{GETUP}').mkdir(parents=True)
    (folder / 'policy' / f'getup-{GETUP}' / 'deploy.yaml').write_text(
        f'# humanoid_sim.export_g1: {lafan} ({GETUP} steps)\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='LAFAN1'):
        module.build(lab, folder, render=False)


def test_research_is_a_block_of_its_own(update, tmp_path):
    """RESEARCH (the LAFAN1 «getting up like a person» clip) goes into home.json «research» only, with its licence
    line, and only while its journal section still has its numbers and its clip is there."""
    module, package = update
    folder = make_package(tmp_path / 'g1-n8-test', package)
    lab = make_lab(tmp_path / 'lab')
    block = module.RESEARCH[0]
    assert data_without_research(module, lab, folder)['research'] == []  # no journal section: left out
    section = dict(title=block['journal'] + ': вставание по записи человека',
                   text='Встаёт за 2,8 с; 95–100% во всех четырёх физиках; стопы 0,57 м (у 1164 было 1,2 м).')
    saved = journal.load(lab)
    saved['roadmap'] = [dict(title='Все движения ног', status='active', detail='—',
                             story=dict(sections=[section], path=[], races=[]))]
    journal.folder(lab).mkdir(parents=True, exist_ok=True)
    (journal.folder(lab) / 'journal.json').write_text(json.dumps(saved, ensure_ascii=False), encoding='utf-8')
    assert data_without_research(module, lab, folder)['research'] == []  # the clip is not there yet
    for lang in i18n.LANGUAGES:
        (journal.folder(lab) / f'site-{lang}-{block["clip"]}.webp').write_bytes(b'clip')
    data = home.validate(module.build(lab, folder, render=False))
    [shown] = data['research']
    assert shown['licence'].startswith('Исследовательский эксперимент. Движение по записи LAFAN1 © Ubisoft La Forge')
    assert shown['stage'] == 'exp-4131bbebbba849a4:150405120' and block['clip'] in home.clips(data)
    cards = [card.get('clip') for c in data['candidates'] for card in c['skills']]
    assert block['clip'] not in cards and 'exp-4131bbebbba849a4' not in json.dumps(data['candidates'])
    assert block['title'] in home.texts(data)  # translated like the rest


def data_without_research(module, lab, folder):
    return home.validate(module.build(lab, folder, render=False))
