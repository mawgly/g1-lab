"""scripts/robot/package.py, package n8s (26.10): candidate 8's symmetric-walk export as the main policy. Its standing
branch is the plain export's graph (symmetric_contract), REFERENCE.md carries the reviewer's skills_check with its counts,
and the packages n8 / n9 / n10 keep their candidates and tables."""
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / 'runs/humanoid_lab/export'
SYM = EXPORTS / 'exp-28691193718e4d0e-7221510144-symwalk'
PLAIN = EXPORTS / 'exp-28691193718e4d0e-7221510144'


@pytest.fixture
def package(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/robot'))
    return importlib.import_module('package')


def test_n8s_candidates_and_the_old_packages(package):
    n8s = package.PACKAGES['n8s']
    assert [(c['label'], c['role']) for c in n8s['candidates']] == [('n8s', 'основной'), ('n7', 'запасной')]
    assert n8s['candidates'][0]['symmetric_walk'] == PLAIN.name
    assert [c['label'] for c in package.PACKAGES['n8']['candidates']] == ['n8', 'n7']
    for key in ('n8', 'n9', 'n10'):
        assert not package.PACKAGES[key].get('skills') and not package.PACKAGES[key].get('caveats')
        assert not any(c.get('symmetric_walk') for c in package.PACKAGES[key]['candidates'])
    # the rows passport.py reads keep SIM2SIM_TABLE's wording
    plain = {what for what, *_ in package.SIM2SIM_TABLE}
    mine = {what for what, *_ in n8s['table']}
    assert {'Прогонов; падений', 'Прямо 0,5 м/с: скорость; увод после 2 с; курс за 10 с',
            'Прямо 1 м/с: скорость; увод; курс за 10 с', 'Стояние 30 с: поворот; смещение'} <= plain & mine


def test_skills_table_counts(package):
    path = package.HERE / 'reference' / package.SKILLS_N8S['file']
    text = package.skills_md(path, '№8 симм.', '№8')
    assert '№8 симм.: 36 из 37 строк; против v0 — лучше 9,\nтак же 21, хуже 1.' in text
    assert '| Торможение: После 0,5 м/с: путь за 3 с, м | ≤ 0,25 | 0,19 | 0,17 | 0,13 | да: порог и эталон | хуже |' in text


@pytest.mark.skipif(not (SYM / 'policy.onnx').is_file() or not (PLAIN / 'policy.onnx').is_file(),
                    reason='the lab exports of candidate 8 are not on this machine')
def test_symmetric_export_stands_as_the_plain_one(package):
    s = package.symmetric_contract(SYM / 'policy.onnx', PLAIN / 'policy.onnx', 490, n=16)
    assert s['symmetric_walk'] and s['command'] == [42, 45] and s['threshold'] == 0.1
    assert s['standing'] <= package.SYMMETRIC_TOL < s['walking']
    # the plain export against itself: no symmetry property, so no columns to check
    assert package.symmetric_contract(PLAIN / 'policy.onnx', PLAIN / 'policy.onnx', 490)['standing'] is None
