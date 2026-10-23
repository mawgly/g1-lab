"""Start the user-approved three continuations once; retain a manifest."""
from humanoid_sim.core import ROOT, read_json, write_json, list_stages
from humanoid_sim.lab import Laboratory
from humanoid_sim.lab_api import LabJobs


def main():
    lab = Laboratory(ROOT / 'runs/humanoid')
    manifest = ROOT / 'lab/long_training_series.json'
    if manifest.exists():
        raise RuntimeError('Series already registered; inspect long_training_series.json before starting another.')
    if lab.worker_active() or lab.active_trainers() or LabJobs(lab).active():
        raise RuntimeError('Another compute task is active')
    sources = [
        ('exp-e4ae6917dd2c41a4', 'Длинное обучение · осторожное · +10 млн'),
        ('exp-76dc32173b7c4391', 'Длинное обучение · больше поиска · +10 млн'),
        ('exp-0183f7e4077145e6', 'Длинное обучение · полный GPU · +10 млн'),
    ]
    series = {'additional_steps_per_run': 10_000_000, 'checkpoint_interval': 500_000,
              'reward_changed': False, 'experiments': []}
    for source_id, name in sources:
        source = lab.resolve(source_id)
        stage = list_stages(source)[-1]
        config = read_json(source / 'run_config.json')
        config.update(name=name, steps=10_000_000, every=500_000, hours=24,
                      source_id=source_id, source_stage=int(stage.name[6:]))
        entry = lab.create(config)
        series['experiments'].append(entry)
        write_json(manifest, series)
        print(entry['id'], flush=True)
    lab.enqueue([entry['id'] for entry in series['experiments']])
    print(lab.start(), flush=True)


if __name__ == '__main__':
    main()
