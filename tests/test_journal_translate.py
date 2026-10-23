import json
import shutil

from humanoid_sim import i18n, journal
from humanoid_sim import journal_translate as jt

RUSSIAN = 'Ветка exp-facb20de4bfe472c (feet_flat 0,5) за 4975,4 млн шагов шла 1,0 м/с; увод 0,054 рад/с.'
ENGLISH = 'Branch exp-facb20de4bfe472c (feet_flat 0.5) walked 1.0 m/s over 4975.4 M steps; drift 0.054 rad/s.'


def test_a_faithful_translation_passes():
    assert jt.problems(RUSSIAN, ENGLISH, 'en') == []
    chinese = '分支 exp-facb20de4bfe472c（feet_flat 0.5）在 4975.4 M 步时以 1.0 m/s 行走；漂移 0.054 rad/s。'
    assert jt.problems(RUSSIAN, chinese, 'zh') == []
    assert jt.problems('Пауза 2 с, затем 10 000 шагов.', 'A 2 秒 pause, then 10,000 steps.', 'zh') == []
    # Cases the journal raised: «с» as "with", numbers inside words, units with a slash, Cyrillic with "_",
    # plural loanwords, a date whose month becomes a number in Chinese.
    for russian, translation, lang in (
            ('Корень 4 с исправленным обучением и ветка от 5451,4 с lock_wrists.',
             'Root 4 with fixed training and a branch from 5451.4 with lock_wrists.', 'en'),
            ('Через WSL2 с Ubuntu; версия G1 с 23 моторами.', 'Through WSL2 with Ubuntu; the G1 with 23 motors.', 'en'),
            ('Шаг: 34 мм/2,1°, арм_пресс 0,1, разброс по seed 0,01.',
             'Step: 34 mm/2.1°, arm_press 0.1, spread across seeds 0.01.', 'en'),
            ('20 октября, 03:25: за 10 с без падений.', '10月20日，03:25：10 s 内没有跌倒。', 'zh')):
        assert jt.problems(russian, translation, lang) == [], russian


def test_changed_numbers_identifiers_units_or_lines_are_rejected():
    def why(text, lang='en'):
        return ' / '.join(jt.problems(RUSSIAN, text, lang))
    assert 'numbers differ' in why(ENGLISH.replace('0.054', '0.05'))
    assert 'Latin words' in why(ENGLISH.replace('feet_flat', 'feet flat'))
    assert 'Latin words' in why(ENGLISH.replace('exp-facb20de4bfe472c', 'the branch'))
    assert 'units' in why(ENGLISH.replace('4975.4 M steps', '4975.4 million steps'))
    assert 'decimal comma' in why(ENGLISH.replace('1.0 m/s', '1,0 m/s'))
    assert 'Cyrillic' in why(ENGLISH + ' Ещё.')
    assert 'line breaks' in jt.problems('Раз\nдва', 'One two', 'en')[0]
    # A decimal followed by another number is two numbers, not one with thousands groups.
    assert 'numbers differ' in jt.problems('стоит на 0,093 200 млн шагов', 'stands at 0.093200 M steps', 'en')[0]
    assert jt.problems('стоит на 0,093 200 млн шагов', 'stands at 0.093, 200 M steps', 'en') == []
    assert 'placeholders' in jt.problems('Выберите от 1 до {n} дорожек', 'Choose 1 to n lanes', 'en')[0]
    assert jt.problems(RUSSIAN, '', 'en') == ['empty']


def answer(path, rows):
    path.write_text(''.join(f'@@ {key} {lang}\n{text}\n\n' for (key, lang), text in rows.items()), encoding='utf-8')
    return path


def test_export_submit_caches_good_sections_and_flags_repeated_failures(tmp_path):
    root, work = tmp_path / 'lab', tmp_path / 'work'
    journal.add_entry(root, 'one', 'Пошёл', 'Робот пошёл на 1,0 м/с.\n\nВторой абзац.', at=1)
    (root / 'exp-0000000000000001').mkdir()
    (root / 'exp-0000000000000001' / 'experiment.json').write_text(json.dumps({'name': 'Осторожный PPO'}))
    paths = jt.export(root, work)
    assert [p.name for p in paths] == ['batch-001.md']
    text = paths[0].read_text(encoding='utf-8')
    assert 'увод = drift = 漂移' in text  # the glossary rides along
    batch = {body: (key, langs) for key, langs, body in jt.sections(text)}
    # The interface catalog already translates the default name; each paragraph is asked in both languages.
    assert sorted(batch) == ['Второй абзац.', 'Пошёл', 'Робот пошёл на 1,0 м/с.']
    assert all(langs == ['en', 'zh'] for _, langs in batch.values())
    walks, title, second = (batch[t][0] for t in ('Робот пошёл на 1,0 м/с.', 'Пошёл', 'Второй абзац.'))
    rejected, stored, _ = jt.submit(root, [answer(work / 'batch-001.answer.md', {
        (title, 'en'): 'Walking', (walks, 'en'): 'The robot walks at 1.5 m/s.', (second, 'en'): 'Second paragraph.',
        (title, 'zh'): '走了', (walks, 'zh'): '机器人以 1.0 m/s 行走。', (second, 'zh'): '第二段。'})])
    assert stored == 5 and list(rejected) == [(walks, 'en')] and 'numbers differ' in rejected[(walks, 'en')][0]
    content = i18n.Content(root)
    entry = i18n.localize_journal(journal.load(root), 'en', content)['entries'][0]
    assert entry['title'] == 'Walking' and entry['text'] == 'Робот пошёл на 1,0 м/с.\n\nSecond paragraph.'
    assert jt.status(root)['en']['missing'] == 1 and jt.status(root)['zh']['missing'] == 0
    # The next batch asks only the rejected section, with the reason; submitting the same answer again is no new
    # attempt; after MAX_ATTEMPTS different failures the paragraph is flagged and left out.
    retry = jt.export(root, work)[0].read_text(encoding='utf-8')
    assert f'@@ {walks} en\n' in retry and 'previous en answer rejected: numbers differ' in retry
    assert 'Второй абзац.' not in retry
    jt.import_answers(root, work)
    assert jt.load_flags(root)['en'][walks]['attempts'] == 1
    for attempt in range(2, jt.MAX_ATTEMPTS + 1):
        answer(work / 'batch-001.answer.md', {(walks, 'en'): f'The robot walks at {attempt}.5 m/s.'})
        jt.import_answers(root, work)
    assert jt.status(root)['en']['flagged'] == [walks] and jt.export(root, work) == []
    (work / 'batch-001.answer.md').unlink()
    rejected, stored, _ = jt.submit(root, [answer(work / 'en-001.fix.md', {(walks, 'en'): 'The robot walks at 1.0 m/s.'})])
    assert (rejected, stored) == ({}, 1)
    entry = i18n.localize_journal(journal.load(root), 'en', content)['entries'][0]
    assert entry['text'] == 'The robot walks at 1.0 m/s.\n\nSecond paragraph.'
    assert jt.status(root)['en'] == dict(units=3, translated=3, missing=0, missing_chars=0, flagged=[], reviewed=1,
                                         interface=0)


def test_review_corrects_stored_translations_and_redo_translates_the_rest_again(tmp_path):
    root, work = tmp_path / 'lab', tmp_path / 'work'
    journal.add_entry(root, 'one', 'Пошёл', 'Жду увод меньше 0,05 рад/с.', at=1)
    batch = {body: key for key, _, body in jt.sections(jt.export(root, work, langs=('zh',))[0].read_text('utf-8'))}
    title, wait = batch['Пошёл'], batch['Жду увод меньше 0,05 рад/с.']
    jt.submit(root, [answer(work / 'batch-001.answer.md', {(title, 'zh'): '走了', (wait, 'zh'): '等待偏航小于 0.05 rad/s。'})])
    review = jt.review(root, work, langs=('zh',), sample=1, seed=1)
    assert [p.name for p in review] == ['zh-review-001.md'] and len(jt.sections(review[0].read_text('utf-8'))) == 1
    text = jt.review(root, work, langs=('zh',))[0].read_text(encoding='utf-8')
    assert '[ru] Жду увод меньше 0,05 рад/с.\n[zh] 等待偏航小于 0.05 rad/s。' in text and '预期' in text  # the terms
    rejected, stored, _ = jt.submit(root, [answer(work / 'zh-review-001.fix.md', {(wait, 'zh'): '预期漂移小于 0.05 rad/s。'})],
                                 reviewer='claude-opus-5-5')
    assert (rejected, stored) == ({}, 1)
    units = jt.load_cache(root, 'zh')['units']
    assert units[wait]['text'] == '预期漂移小于 0.05 rad/s。' and units[wait]['by'] == 'claude-opus-5-5'
    assert all(unit['reviewed'] for unit in units.values()) and jt.status(root)['zh']['reviewed'] == 2
    assert jt.review(root, work, langs=('zh',)) == []  # nothing left to review
    # --redo asks again for stored translations nobody reviewed; reviewed ones keep their text.
    journal.add_entry(root, 'two', 'Ещё', 'Новый абзац.', at=2)
    jt.submit(root, [answer(work / 'batch-001.answer.md', {(i18n.unit_id('Ещё'), 'zh'): '又一个',
                                                            (i18n.unit_id('Новый абзац.'), 'zh'): '旧译文。'})])
    redo = jt.sections(jt.export(root, work, langs=('zh',), redo=('zh',))[0].read_text(encoding='utf-8'))
    assert sorted(body for _, _, body in redo) == ['Ещё', 'Новый абзац.']
    matched = jt.export(root, work, langs=('zh',), redo=('zh',), match='Ещ')[0].read_text(encoding='utf-8')
    assert [body for _, _, body in jt.sections(matched)] == ['Ещё']  # one term the glossary changed
    jt.submit(root, [answer(work / 'batch-001.answer.md', {(i18n.unit_id('Новый абзац.'), 'zh'): '新段落。',
                                                            (wait, 'zh'): '等待。'})])
    units = jt.load_cache(root, 'zh')['units']
    assert units[i18n.unit_id('Новый абзац.')]['text'] == '新段落。' and units[wait]['text'] == '预期漂移小于 0.05 rad/s。'


def test_new_interface_strings_go_through_the_same_batches_into_the_catalogs(tmp_path, monkeypatch):
    catalogs = tmp_path / 'i18n'
    shutil.copytree(i18n.CATALOGS, catalogs)
    monkeypatch.setattr(i18n, 'CATALOGS', catalogs)
    i18n.catalog.cache_clear()
    new = 'Выберите от 1 до {n} новых дорожек'
    real = i18n.missing
    monkeypatch.setattr(i18n, 'missing', lambda lang: ([], [new]) if new not in i18n.catalog(lang).exact else real(lang))
    try:
        root, work = tmp_path / 'lab', tmp_path / 'work'
        text = jt.export(root, work)[0].read_text(encoding='utf-8')
        [(key, langs, body)] = jt.sections(text)
        assert (body, langs) == (new, ['en', 'zh']) and 'interface string: keep each {placeholder}' in text
        rejected, stored, _ = jt.submit(root, [answer(work / 'batch-001.answer.md', {
            (key, 'en'): 'Choose 1 to n new lanes', (key, 'zh'): '请选择 1 到 {n} 条新跑道'})])
        assert stored == 1 and 'placeholders' in rejected[(key, 'en')][0]
        assert i18n.translate('Выберите от 1 до 3 новых дорожек', 'zh') == '请选择 1 到 3 条新跑道'
        jt.submit(root, [answer(work / 'batch-001.answer.md', {(key, 'en'): 'Choose 1 to {n} new lanes'})])
        assert i18n.translate('Выберите от 1 до 3 новых дорожек', 'en') == 'Choose 1 to 3 new lanes'
        assert jt.export(root, work) == [] and json.loads((catalogs / 'en.json').read_text('utf-8'))[new]
        assert b'\r' not in (catalogs / 'en.json').read_bytes()  # LF as committed, also on Windows
    finally:
        i18n.catalog.cache_clear()
