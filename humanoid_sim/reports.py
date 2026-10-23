"""Evidence-backed reports. Original session and stage artifacts are read-only."""

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import pickle
from pathlib import Path
import statistics
import zipfile

from .core import read_json, write_json
from .i18n import gettext

REPORT_CSS = '''
body { margin: 0; background: #101821; color: #e5edf5; font: 16px/1.6 system-ui, sans-serif }
main { max-width: 1120px; margin: auto; padding: 48px 26px }
h1 { font-size: 40px; line-height: 1.15; max-width: 900px }
h2 { margin-top: 44px; font-size: 25px }
p { max-width: 920px; color: #c6d1df }
.eyebrow { color: #79dbc1; letter-spacing: .1em }
.lead { font-size: 21px }
.chart { background: #182432; border-radius: 14px; padding: 20px; margin: 20px 0 }
svg { width: 100%; height: auto }
svg text { fill: #aebdce; font-size: 12px }
table { border-collapse: collapse; width: 100%; font-size: 14px }
th, td { text-align: left; padding: 12px; border-bottom: 1px solid #334254; vertical-align: top }
th { color: #79dbc1 }
.scroll { overflow: auto }
pre { white-space: pre-wrap; word-break: break-word; font-size: 12px }
summary { cursor: pointer; color: #79dbc1 }
.note { border-left: 3px solid #e9bd74; padding: 12px 20px; background: #211f1c }
a { color: #79dbc1 }
'''


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _summary(values):
    values = [v for v in values if v is not None]
    return {
        'count': len(values),
        'min': min(values) if values else None,
        'max': max(values) if values else None,
        'median': statistics.median(values) if values else None,
        'last': values[-1] if values else None,
    }


def build_report(run_dir):
    run = Path(run_dir)
    experiment = read_json(run / 'experiment.json', {})
    sources, stages, sessions, logs = [], [], [], []

    def source(path):
        sources.append(
            {
                'path': str(path.relative_to(run)),
                'bytes': path.stat().st_size,
                'modified_at': path.stat().st_mtime,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    if (run / 'experiment.json').exists():
        source(run / 'experiment.json')
    for folder in sorted(run.glob('stage_*')):
        metadata = read_json(folder / 'metadata.json')
        if metadata is None:
            continue
        source(folder / 'metadata.json')
        evaluation = read_json(folder / 'evaluation.json')
        if evaluation is not None:
            source(folder / 'evaluation.json')
        stages.append(
            {
                'name': folder.name,
                'steps': metadata.get('steps'),
                'created_at': metadata.get('created_at'),
                'task_config': metadata.get('task_config', {}),
                'mean_return': evaluation.get('mean_return') if evaluation else None,
                'mean_seconds': evaluation.get('mean_seconds') if evaluation else None,
                'mean_forward_m': evaluation.get('mean_forward_m') if evaluation else None,
                'episodes': evaluation.get('episodes', []) if evaluation else [],
            }
        )
    stages.sort(key=lambda row: row['steps'] if row['steps'] is not None else -1)
    for folder in sorted(run.glob('session_*')):
        config = read_json(folder / 'config.json', {})
        if (folder / 'config.json').exists():
            source(folder / 'config.json')
        rows = []
        if (folder / 'progress.csv').exists():
            source(folder / 'progress.csv')
            with (folder / 'progress.csv').open(newline='', encoding='utf-8') as stream:
                rows = [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            logs.extend(dict(row, session=folder.name) for row in rows)
        sessions.append(
            {
                'name': folder.name,
                'config': config,
                'logged_rows': len(rows),
                'first_logged_steps': rows[0].get('time/total_timesteps') if rows else None,
                'last_logged_steps': rows[-1].get('time/total_timesteps') if rows else None,
            }
        )
    evaluated = [s for s in stages if s['mean_return'] is not None]
    best = max(evaluated, key=lambda s: s['mean_return']) if evaluated else None
    audit = {
        'historical_mujoco_version': None,
        'historical_mujoco_version_note': 'Not recorded in checkpoint system_info.txt',
        'ppo': None,
        'checkpoint_system_info': None,
        'code_is_current_not_historical': True,
    }
    if stages and (run / stages[-1]['name'] / 'policy.zip').exists():
        path = run / stages[-1]['name'] / 'policy.zip'
        source(path)
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read('data'))
            audit['ppo'] = {
                k: data.get(k)
                for k in (
                    'learning_rate',
                    'n_envs',
                    'n_steps',
                    'batch_size',
                    'n_epochs',
                    'gamma',
                    'gae_lambda',
                    'ent_coef',
                    'vf_coef',
                    'max_grad_norm',
                    'clip_range_vf',
                    'normalize_advantage',
                    'target_kl',
                    'use_sde',
                    'sde_sample_freq',
                    'seed',
                    '_n_updates',
                    'num_timesteps',
                    '_stats_window_size',
                    '_current_progress_remaining',
                    '_total_timesteps',
                    '_num_timesteps_at_start',
                )
            }
            audit['policy_kwargs'] = {
                k: v for k, v in data.get('policy_kwargs', {}).items() if not k.startswith(':')
            }
            audit['checkpoint_system_info'] = (
                archive.read('system_info.txt').decode() if 'system_info.txt' in archive.namelist() else None
            )
            audit['clip_range_logged'] = _summary([r.get('train/clip_range') for r in logs])
        norm_path = run / stages[-1]['name'] / 'normalize.pkl'
        if norm_path.exists():
            source(norm_path)
            with norm_path.open('rb') as stream:
                normalizer = pickle.load(stream)
            audit['training_normalization_saved'] = {
                k: getattr(normalizer, k, None)
                for k in (
                    'norm_obs',
                    'norm_reward',
                    'training',
                    'clip_obs',
                    'clip_reward',
                    'gamma',
                    'epsilon',
                )
            }
        audit['actual_rollout_size'] = data.get('n_envs', 1) * data.get('n_steps', 1)
        audit['semantics'] = {
            'steps': 'environment transitions across all parallel environments',
            'checkpoint_interval': 'minimum transitions; actual save occurs at rollout boundary',
            'seed_on_resume': 'new environment resets use session seed; policy RNG and optimizer restored by PPO.load',
            'reward_training': 'normalized and clipped using saved discounted-return statistics',
            'reward_evaluation': 'raw environment reward; observations use frozen saved normalization',
            'source_revision': None,
        }

    diagnostics = read_json(run / 'diagnostics.json')
    motion = read_json(run / 'motion.json')
    command_suite = read_json(run / 'command_suite.json')
    if command_suite:
        source(run / 'command_suite.json')
    if motion:
        source(run / 'motion.json')
    if diagnostics:
        source(run / 'diagnostics.json')
    error = run / 'training-error.log'
    audit['winerror5'] = None
    if error.exists() and 'WinError 5' in error.read_text(encoding='utf-8', errors='replace'):
        source(error)
        observed = read_json(run / 'interval-change-state.json', {})
        resume_steps = [
            int(Path(s['config']['resume'].replace('\\', '/')).name.split('_')[-1])
            for s in sessions
            if s['config'].get('resume')
        ]
        restart = resume_steps[-1] if resume_steps else None
        audit['winerror5'] = {
            'source': error.name,
            'observed_steps': observed.get('steps'),
            'resumed_steps': restart,
            'observed_unsaved_steps': observed['steps'] - restart
            if observed.get('steps') and restart is not None
            else None,
            'note': 'Observed gap is a lower bound; pending status can be later than interval-change snapshot.',
        }
        if observed:
            source(run / 'interval-change-state.json')
    stats = {
        key: _summary([r.get(key) for r in logs])
        for key in (
            'train/approx_kl',
            'train/clip_fraction',
            'train/std',
            'train/entropy_loss',
            'rollout/ep_rew_mean',
            'rollout/ep_len_mean',
        )
    }
    cutoff = max(
        [s['created_at'] for s in stages if s.get('created_at')] + [diagnostics['created_at']]
        if diagnostics
        else [s['created_at'] for s in stages if s.get('created_at')],
        default=None,
    )
    return {
        'schema_version': 1,
        'title': experiment.get('name')
        or (
            'Эксперимент 1 · Аудит обучения Humanoid'
            if run.name == 'humanoid'
            else f'Эксперимент · {run.name}'
        ),
        'experiment': experiment,
        'asOf': datetime.fromtimestamp(cutoff, timezone.utc).isoformat() if cutoff else None,
        'run_dir': str(run.resolve()),
        'summary': {
            'stage_count': len(stages),
            'last_steps': stages[-1]['steps'] if stages else None,
            'best': best,
            'baseline': stages[0] if stages else None,
            'last': stages[-1] if stages else None,
        },
        'stages': stages,
        'sessions': sessions,
        'training_logs': logs,
        'training_statistics': stats,
        'audit': audit,
        'diagnostics': diagnostics,
        'motion': motion,
        'command_suite': command_suite,
        'provenance': sources,
        'metric_definitions': {
            'mean_return': 'Sum of raw environment rewards, averaged across episodes; larger is better.',
            'mean_seconds': 'Time until termination or time limit, not a posture classifier.',
            'original_evaluation': 'Deterministic actions; same seeds 60000–60004 repeatedly used for checkpoint selection.',
            'fresh_evaluation': 'Deterministic actions; held-out seeds 80000 onward, shared across baseline/best/last.',
            'fall_rate': 'Share terminated by Humanoid health condition; healthy z interval, not visual assessment.',
            'training_metrics': 'Logged rollout means from stochastic training policy; not equivalent to deterministic evaluation.',
        },
    }


def _chart(rows, key, label, color='#79dbc1', _=gettext('ru')):
    points = [(r['steps'], r.get(key)) for r in rows if r.get('steps') is not None and r.get(key) is not None]
    if not points:
        return '<p>' + _('Нет данных') + '</p>'
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymax = max(p[1] for p in points) * 1.1 or 1
    coords = ' '.join(
        f'{50 + (x - xmin) / max(xmax - xmin, 1) * 880:.2f},{230 - y / ymax * 200:.2f}' for x, y in points
    )
    first, last = _('{n} млн', n=f'{xmin / 1e6:.1f}'), _('{n} млн шагов', n=f'{xmax / 1e6:.1f}')
    return f'<svg viewBox="0 0 960 270" role="img" aria-label="{label}"><path d="M50 30V230H930" fill="none" stroke="#596678"/><polyline points="{coords}" stroke="{color}" fill="none" stroke-width="2"/><text x="8" y="35">{ymax:.1f}</text><text x="25" y="230">0</text><text x="50" y="255">{first}</text><text x="850" y="255">{last}</text></svg>'


def render_report(report, lang='ru'):
    """The report page in Russian (the file written next to report.json) or, for the browser, in English or
    Chinese (humanoid_sim/i18n.py: the Russian text is the key of its translation)."""
    esc = html.escape
    _ = gettext(lang)

    def fmt(v, digits=2):
        return _('неизвестно') if v is None else f'{v:,.{digits}f}'.replace(',', ' ')

    def table(headers, rows):
        return (
            '<div class="scroll"><table><thead><tr>'
            + ''.join(f'<th>{esc(str(x))}</th>' for x in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{esc(str(x))}</td>' for x in row) + '</tr>' for row in rows)
            + '</tbody></table></div>'
        )

    s, audit = report['summary'], report['audit']
    best, last = s['best'], s['last']
    intro = (
        _('Нет сохранённых оценок.')
        if not best or not last
        else _(
            'Лучший результат на исходных пяти seed достигнут на {n1} шагах: {n2} награды. Последний этап: {n3}. Разница последнего и лучшего этапа: {n4}.',
            n1=fmt(best['steps'], 0),
            n2=fmt(best['mean_return']),
            n3=fmt(last['mean_return']),
            n4=fmt(last['mean_return'] - best['mean_return']),
        )
    )
    diag = report.get('diagnostics')
    fresh = '<p>' + _('Дополнительная проверка ещё не выполнена.') + '</p>'
    if diag:
        fresh = table(
            [
                _('Этап'),
                _('Эпизоды'),
                _('Награда ± SD'),
                _('Время, с'),
                _('Вперёд, м'),
                _('Падения'),
                _('Точный повтор'),
            ],
            [
                [
                    role,
                    d['episode_count'],
                    f'{fmt(d["mean_return"])} ± {fmt(d["std_return"])}',
                    fmt(d['mean_seconds']),
                    fmt(d['mean_forward_m']),
                    (_('не применимо') if d['fall_rate'] is None else f'{100 * d["fall_rate"]:.0f}%'),
                    str(d['deterministic_replay_equal']),
                ]
                for role, d in diag['checkpoints'].items()
            ],
        )
        last_seed = 80000 + next(iter(diag['checkpoints'].values()))['episode_count'] - 1
        fresh += (
            '<p>'
            + _(
                'Проверка использует новые общие seed 80000–{n}. «Best» выбран по исходным пяти seed. Разница средних на новой выборке сама по себе не доказывает превосходство; все эпизоды завершились нарушением условия здоровья, если доля падений равна 100%.',
                n=last_seed,
            )
            + '</p>'
        )
        fresh += (
            '<p>'
            + _(
                'Награды разных задач несопоставимы. Для руки падения и продвижение не применимы; оценивайте достижение цели и расстояние до неё.'
            )
            + '</p>'
        )
        fresh += table(
            [_('Этап'), _('Компонент'), _('Средняя сумма за эпизод')],
            [
                [role, k, fmt(v)]
                for role, d in diag['checkpoints'].items()
                for k, v in d['mean_components'].items()
            ],
        )
    stats = report['training_statistics']
    motion_html = '<h2>' + _('Контакты и движение') + '</h2>'
    suite = report.get('command_suite')
    if suite:
        names = {
            'stand': _('Стояние'),
            'slow': _('Ходьба 0,5 м/с'),
            'walk': _('Ходьба 1 м/с'),
            'brisk': _('Ходьба 1,4 м/с'),
            'stop_start': _('Старт, остановка, повторный старт'),
        }
        rows = []
        for name, result in suite['scenarios'].items():
            means = result['means']
            interval = result.get('statistics', {}).get('fell', {}).get('wilson95')
            rows.append(
                [
                    names.get(name, name),
                    len(result['episodes']),
                    fmt(means.get('duration_s')),
                    fmt(100 * means['fell']) + '%' if means.get('fell') is not None else '—',
                    '–'.join(fmt(100 * v) for v in interval) + '%' if interval else '—',
                    fmt(means.get('velocity_error_rmse_m_s')),
                ]
            )
        motion_html += (
            '<h2>'
            + _('Проверка команд')
            + '</h2><p>'
            + esc(suite['protocol'])
            + ' · '
            + esc(suite['checkpoint'])
            + '</p>'
        )
        motion_html += table(
            [
                _('Сценарий'),
                _('Попыток'),
                _('Время, с'),
                _('Падения'),
                _('Интервал 95%'),
                _('Ошибка скорости, м/с'),
            ],
            rows,
        )
        motion_html += (
            '<p>'
            + _(
                'Предварительная проверка последнего сохранения. Интервал показывает неопределённость доли падений; отсутствие падений в пяти попытках не доказывает надёжность. Финальные seed здесь не используются.'
            )
            + '</p>'
        )
    motion = report.get('motion')
    if motion:
        motion_html += (
            '<p>'
            + _(
                'Отдельные измерения, не общий балл естественности. Низкие затраты или плавность не компенсируют падение.'
            )
            + '</p>'
        )
        labels = {
            'duration_s': _('Длительность, с'),
            'fell': _('Доля падений'),
            'contact_slip_speed_m_s': _('Скорость контактных точек, м/с'),
            'tilt_rms_rad': _('Наклон RMS, рад'),
            'step_count': _('Контактов-шагов'),
            'cadence_steps_per_min': _('Контактов-шагов в минуту'),
            'mean_step_length_m': _('Расстояние между контактами разных стоп, м'),
            'alternation_fraction': _('Доля чередующихся контактов'),
            'action_rate_rms_s_inv': _('Скорость изменения действий RMS, 1/с'),
            'absolute_mechanical_work_j': _('Абсолютная механическая работа, Дж'),
            'velocity_error_rmse_m_s': _('Ошибка команды скорости RMS, м/с'),
        }
        for role, result in motion['checkpoints'].items():
            motion_html += '<h3>' + esc(role) + '</h3><p>' + esc(result['checkpoint']) + '</p>'
            motion_html += (
                '<p>'
                + _(
                    'Seed: {n1}; попыток: {n2}; лимит: {n3} с.',
                    n1=result['seed_start'],
                    n2=len(result['episodes']),
                    n3=result['seconds'],
                )
                + '</p>'
            )
            motion_html += table(
                [_('Показатель'), _('Среднее')],
                [[label, fmt(result['means'].get(key))] for key, label in labels.items()],
            )
            motion_html += '<p>' + esc('; '.join(result['limits'])) + '</p>'
    else:
        motion_html += '<p>' + _('Сравнение контактов между сохранениями ещё не выполнено.') + '</p>'
    warning = _(
        'В журнале median approx_kl = {n1}, median clip_fraction = {n2}. Это измеренный признак больших изменений политики; target_kl = {value}. Нормализация и clipping награды настроены, но частота clipping не записана: связать их с высоким KL по сохранённым данным нельзя. Причина не установлена: нужен отдельный контролируемый запуск с меньшими обновлениями. Высокая доля clipping сама по себе не доказывает программную ошибку.',
        n1=fmt(stats['train/approx_kl']['median']),
        n2=fmt(stats['train/clip_fraction']['median']),
        value=audit.get('ppo', {}).get('target_kl') if audit.get('ppo') else _('неизвестно'),
    )
    if stats['train/approx_kl']['median'] is None or stats['train/clip_fraction']['median'] is None:
        warning = _(
            'Недостаточно журналов PPO для оценки величины обновлений. Вывод об их стабильности не сделан.'
        )
    unknown = _('неизвестно')
    sessions = table(
        [_('Сессия'), 'Seed', _('Среды'), _('Интервал'), _('Возобновление'), _('Строк CSV')],
        [
            [
                r['name'],
                r['config'].get('seed', unknown),
                r['config'].get('envs', unknown),
                r['config'].get('checkpoint_interval', unknown),
                r['config'].get('resume') or _('с нуля'),
                r['logged_rows'],
            ]
            for r in report['sessions']
        ],
    )
    allstages = table(
        [_('Шаги'), _('Награда'), _('Время, с'), _('Вперёд, м')],
        [
            [fmt(r['steps'], 0), fmt(r['mean_return']), fmt(r['mean_seconds']), fmt(r['mean_forward_m'])]
            for r in report['stages']
        ],
    )
    payload = esc(
        json.dumps(
            {
                'audit': audit,
                'experiment': report.get('experiment'),
                'session_configs': report['sessions'],
                'diagnostics_settings': {k: v for k, v in (diag or {}).items() if k != 'checkpoints'},
                'actual_models': {
                    k: {
                        'model': v.get('actual_model'),
                        'settings_consistency': v.get('settings_consistency'),
                        'reward_components_max_abs_error': v.get('reward_components_max_abs_error'),
                    }
                    for k, v in (diag or {}).get('checkpoints', {}).items()
                },
                'evaluation_environment': next(iter(diag['checkpoints'].values()))['environment']
                if diag
                else None,
                'metric_definitions': report['metric_definitions'],
                'sources': report['provenance'],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    tag = {'ru': 'ru', 'en': 'en', 'zh': 'zh-Hans'}.get(lang, 'ru')
    reward_chart = _chart(report['stages'], 'mean_return', _('Награда по шагам'), _=_)
    time_chart = _chart(report['stages'], 'mean_seconds', _('Время эпизода по шагам'), '#e9bd74', _=_)
    stages_line = _(
        '{n1} этапов · {n2} шагов среды. PPO обучает управление выбранным роботом. Состав награды зависит от задачи. Время эпизода само по себе не доказывает качество навыка.',
        n1=s['stage_count'],
        n2=fmt(s['last_steps'], 0),
    )
    method = _(
        'Оценка: deterministic=True, замороженная нормализация наблюдений, сырая награда. Обучение: нормализация наблюдений и наград. PPO и версии ниже извлечены из последнего checkpoint; текущий исходный код не является доказательством всех исторических настроек. Историческая версия MuJoCo не записана в checkpoint и остаётся неизвестной. Дата отсечения доказательств: {date}.',
        date=esc(str(report['asOf'])),
    )
    return f'''<!doctype html><html lang="{tag}">
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(report['title'])}</title>
<style>{REPORT_CSS}</style>
<main>
<div class="eyebrow">{_('ROBOT LAB · ПРОВЕРКА ЭКСПЕРИМЕНТА')}</div>
<h1>{esc(report['title'])}</h1>
<p class="lead">{intro}</p>
<p>{stages_line}</p>
<h2>{_('Результаты по ходу обучения')}</h2>
<div class="chart"><b>{_('Награда на исходных пяти seed')}</b>{reward_chart}</div>
<div class="chart"><b>{_('Время до конца эпизода, секунды')}</b>{time_chart}</div>
<p>{_('Эти seed многократно использовались для выбора лучшего этапа; такая оценка оптимистична относительно новых начальных состояний. Обучающая награда в CSV измерена при случайных действиях политики и не сопоставляется напрямую с этим графиком.')}</p>
<h2>{_('Новая проверка: baseline, best и last')}</h2>
{fresh}{motion_html}
<h2>{_('Диагностика обновлений PPO')}</h2>
<p class="note">{warning}</p>
<h2>{_('Сессии и перезапуски')}</h2>
{sessions}
<p>{_('Интервал задаёт минимальный порог: сохранение на границе rollout даёт фактические промежутки больше порога. Перезапуск продолжает параметры и статистики нормализации сохранённого этапа, но не восстанавливает состояния физических сред и незавершённый rollout.')}</p>
<p>{_('Сбой записи WinError 5: {value}', value=esc(json.dumps(audit['winerror5'], ensure_ascii=False)))}</p>
<h2>{_('Метод и воспроизводимость')}</h2>
<p>{method}</p>
<details><summary>{_('Все этапы и результаты')}</summary>{allstages}</details>
<details><summary>{_('Параметры, версии, определения и проверяемые источники')}</summary><pre>{payload}</pre></details>
</main>
</html>'''


def write_report(run_dir):
    run = Path(run_dir)
    report = build_report(run)
    write_json(run / 'report.json', report)
    (run / 'report.html').write_text(render_report(report), encoding='utf-8')
    return report
