# Humanoid Lab Implementation Plan

**Goal:** локальная браузерная лаборатория с отчётом первого эксперимента и проверенной GPUфизикой.
**Architecture:** существующий HTTPсервер+независимые workers, JSONmanifest/queue, сохранения SB3; браузерHTML/JS/Three.
**Spec:** humanoid-lab-spec.md

## Tasks
- [x] Reports/diagnostics: reports.py build_report/write_report; diagnostics.py run_diagnostics; реальный baseline/best/last на30seeds. Тесты агрегатов и сохранности исходников.
- [x] GPU: gpu*.py create_gpu_env(n_envs,seed) SB3VecEnv, isolated .venv-gpu, benchmark; smoke физики и сопоставимости.
- [x] Experiments: lab.py IDs/config validation/immutable manifests, lab_worker.py очередь и job lifecycle. Tests invalidvalues/pathtraversal/duplicate start/stop.
- [x] Training: --config JSON settings, CPU/GPU selection, stop timer, callback status and500k snapshots. Реальный короткий запуск и suite.
- [x] Browser implementation: /lab.html default, reports/diagnostic/queue APIs, selected experiment view and independent best/latest channels; staticJS andDOM-IDvalidation.
- [x] Integration: reviewer, pytest, real GPU benchmark, lab launcher, documentation.
- [ ] Visual browser QA/screenshot blocked by browser tool URL policy, no workaround attempted.

## Review focus
Старые модели не меняются. Отсутствующие данные не равны нулю. Запуск сервера не возобновляет очередь. Остановка во время запуска не теряется. Смена просмотра не перенаправляет остановку чужогоjob. GPU недоступность объясняется, без тихой подмены CPU. Нагрузка отображается фактическая.

## Execution record
Параллельные GPU и report задачи назначены в непересекающихся файлах; интеграция/веб выполняется основным агентом. Git в текущем проекте не настроен; файлы изменяются на месте в общей рабочей папке.

## Results
Baseline frozen:232stages,103578120steps. New diagnostics90episodes all fell, deterministic replay and normalization checks passed. High approximateKL is observed, causality remains hypothesis.
GPU physics benchmark16677/30802/52788/81796transitions/s at128/256/512/1024worlds; excludesPPO. NativeWindows, Newton variant, CPU policy. Actual65536step training completed~3.5ksteps/s, saved and reportgenerated.
Nine GPU experiments launched sequentially:3presets×3seeds,2msteps each,256worlds×32rolloutsteps; every500k; perrun1hour safety timer with sleepinhibition. User can stop from browser.
Review fixed legacybaselinewrite endpoint, orphantrainer detection+stop, unusedplaybackLRU+idlepause, gamma normalization consistency, queueeditcontention and durablejobownership locks. Static reviewfollowup finalorphanstopbutton fixed. Browservisuallimit must be disclosed.
