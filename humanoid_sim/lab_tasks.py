import argparse
from pathlib import Path
from .core import RunLock, read_json, write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['report', 'diagnostics', 'motion', 'command_suite', 'benchmark', 'full_gpu_benchmark'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lab-root', type=Path, required=True)
    args = parser.parse_args()
    code = 1
    try:
        with RunLock(args.lab_root / 'job'), RunLock(args.lab_root / 'compute'):
            if args.kind == 'benchmark':
                from .gpu_benchmark import benchmark
                benchmark([128, 256, 512, 1024], 100, 500, str(args.lab_root / 'gpu_benchmark.json'))
            elif args.kind == 'full_gpu_benchmark':
                from .full_gpu_benchmark import benchmark
                benchmark(output=args.lab_root / 'full_gpu_benchmark.json')
            else:
                if args.kind == 'command_suite':
                    from .evaluation.runner import run_suite_diagnostics
                    run_suite_diagnostics(args.run_dir)
                if args.kind == 'motion':
                    from .evaluation.runner import run_motion_diagnostics
                    run_motion_diagnostics(args.run_dir)
                if args.kind == 'diagnostics':
                    from .diagnostics import run_diagnostics
                    run_diagnostics(args.run_dir)
                from .reports import write_report
                write_report(args.run_dir)
        code = 0
    finally:
        state = read_json(args.lab_root / 'job.json', {})
        write_json(args.lab_root / 'job.json', {**state, 'exit_code': code})
