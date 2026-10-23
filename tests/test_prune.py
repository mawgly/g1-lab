import pytest

from humanoid_sim import journal, prune
from humanoid_sim.core import write_json
from humanoid_sim.lab import Laboratory


def _stage(run, steps):
    stage = run / f'stage_{steps:012d}'
    stage.mkdir(parents=True)
    for name in ('policy.zip', 'normalize.pkl', 'metadata.json'):
        (stage / name).write_text('x')


def test_spread_keeps_first_last_and_even_progress():
    stages = list(range(0, 100, 2))
    assert prune.spread(stages, 5) == {0, 24, 48, 74, 98}
    assert prune.spread([4, 6], 5) == {4, 6}


def test_plan_keeps_every_referenced_stage_and_never_touches_baseline(tmp_path):
    lab = Laboratory(tmp_path / 'original')
    branch = lab.create({'name': 'Ветка'})['id']
    run = lab.resolve(branch)
    for steps in range(2_000_000, 202_000_000, 2_000_000):  # 100 stages, one every 2 M
        _stage(run, steps)
    write_json(run / 'report.json', {'summary': {'last_steps': 200_000_000, 'best': {'steps': 150_000_000}}})
    child = 'exp-' + '0' * 16  # a manifest is enough: create() would load the (fake) source stage
    write_json(lab.root / child / 'experiment.json', {'id': child, 'name': 'Дочь', 'created_at': 1,
                                                      'config': {'source_id': branch, 'source_stage': 120_000_000}})
    journal.add_entry(lab.root, 'entry', 'Заголовок', 'Текст', race={
        'robot': 'humanoid', 'task': 'humanoid_v5', 'lanes': [{'experiment': branch, 'stage': 40_000_000}]})
    lab.annotate(branch, note='На 88,1 млн пошёл; 999 млн — не этап этой ветки.')
    lane = lambda steps: {'robot': 'humanoid', 'task': 'humanoid_v5', 'lanes': [{'experiment': branch, 'stage': steps}]}
    journal.set_roadmap(lab.root, [{'title': 'Этап', 'status': 'active', 'story': {  # «было → стало» in the story
        'path': [{'title': 'Шаг', 'change': 'Что', 'result': 'Вышло', 'race': lane(60_000_000)}],
        'races': [{'title': 'Сравнение', **lane(70_000_000)}]}}])
    kept, removed = prune.plan(lab, branch, keep=5)
    assert {2_000_000, 200_000_000, 150_000_000, 120_000_000, 40_000_000, 88_000_000, 60_000_000, 70_000_000} <= kept
    assert len(kept) == 10 and len(removed) == 90  # spread 2, 52, 102, 150, 200 M (best, last among them) + 5 more
    assert all(int(p.name[6:]) not in kept for p in removed)
    with pytest.raises(ValueError):
        prune.plan(lab, 'baseline')
