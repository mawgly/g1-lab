import argparse
from pathlib import Path
from .core import DEFAULT_RUN, list_stages, read_json


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description='Humanoid-v5: train and watch saved learning stages')
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--steps', type=positive, default=2000000)
    train_parser.add_argument('--envs', type=positive, default=8)
    train_parser.add_argument('--every', type=positive, default=1000000)
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--duration-seconds', type=positive)
    train_parser.add_argument('--config', type=Path)
    train_parser.add_argument('--parent-stop', type=Path)
    watch_parser = sub.add_parser('watch')
    watch_parser.add_argument('--stage', type=int)
    watch_parser.add_argument('--seconds', type=positive)
    watch_parser.add_argument('--screenshot', type=Path)
    for p in (train_parser, watch_parser, sub.add_parser('list'), sub.add_parser('stop')):
        p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    try:
        if args.command == 'train':
            from .train import train
            if args.config is None:
                # A manual run competes with the lab queue for the machine: hold the lab compute slot.
                from .core import RunLock
                from .lab import Laboratory
                try:
                    slot = RunLock(Laboratory(DEFAULT_RUN).root / 'compute').__enter__()
                except RuntimeError:
                    raise RuntimeError('Лаборатория сейчас занята очередью или проверкой. '
                                       'Остановите её или дождитесь окончания.') from None
                try:
                    train(args)
                finally:
                    slot.__exit__(None, None, None)
            else:
                train(args)
        elif args.command == 'watch':
            from .viewer import watch
            watch(args)
        elif args.command == 'stop':
            args.run_dir.mkdir(parents=True, exist_ok=True)
            (args.run_dir / 'stop.request').touch()
            print('Stop requested. Trainer will save the current stage and exit.')
        else:
            for stage in list_stages(args.run_dir):
                metrics = read_json(stage / 'evaluation.json', {})
                print(stage.name, metrics.get('mean_return', 'not evaluated'), metrics.get('mean_seconds', ''))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
