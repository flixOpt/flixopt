#!/usr/bin/env python
"""Sweep benchmarks across multiple git commits.

Runs benchmarks on each commit by checking out, installing, and invoking
`python -m benchmarks.run` as a subprocess. Each commit uses its own
benchmark code, so API changes can be handled per-branch.

Usage:
    python -m benchmarks.sweep 97e35f6,HEAD --model simple --phase build --sizes 168 --iterations 3
    python -m benchmarks.sweep HEAD~10..HEAD --model simple district --phase build memory
    python -m benchmarks.sweep main,feature/perf --all --quick
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.config import CACHE_DIR, DEFAULT_SWEEP_ITERATIONS

RESULTS_DIR = CACHE_DIR


def _git(*cmd: str, check: bool = True) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(['git', *cmd], capture_output=True, text=True, check=check)
    return result.stdout.strip()


def resolve_revisions(rev_range: str) -> list[str]:
    """Resolve a git revision range or comma-separated refs to a list of commit SHAs."""
    if '..' in rev_range:
        output = _git('rev-list', '--reverse', rev_range)
    else:
        output = '\n'.join(_git('rev-parse', r.strip()) for r in rev_range.split(','))
    return [line.strip() for line in output.splitlines() if line.strip()]


def check_working_directory_clean() -> bool:
    """Check if working directory has uncommitted changes. Returns True if clean."""
    status = _git('status', '--porcelain')
    return not status


def print_commit_info(shas: list[str]) -> None:
    """Print information about commits that would be benchmarked."""
    print(f'Commits to benchmark ({len(shas)}):')
    for i, sha in enumerate(shas):
        short = sha[:8]
        subject = _git('log', '--format=%s', '-1', sha)
        print(f'  [{i + 1}] {short} {subject[:60]}')


def run_benchmark_subprocess(
    args_list: list[str],
    *,
    name: str,
    env_override: dict | None = None,
) -> bool:
    """Run benchmarks.run as a subprocess. Returns True on success."""
    cmd = [sys.executable, '-m', 'benchmarks.run', '--name', name, *args_list]
    print(f'  Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, env=env_override)
    return result.returncode == 0


def sweep(
    rev_range: str,
    benchmark_args: list[str],
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Benchmark across multiple git commits.

    For each commit:
    1. git checkout <sha>
    2. uv pip install -e .
    3. python -m benchmarks.run --name <short_sha> <benchmark_args>
    4. Collect JSON results

    Returns paths to all generated JSON files.
    """
    shas = resolve_revisions(rev_range)
    if not shas:
        print(f'No commits found for range: {rev_range}')
        sys.exit(1)

    # Dry run: just show what would be benchmarked
    if dry_run:
        print_commit_info(shas)
        print(f'\nBenchmark args: {" ".join(benchmark_args)}')
        print(f'Results would be saved to: {RESULTS_DIR}')
        return []

    # Check for uncommitted changes before starting
    if not check_working_directory_clean():
        print('ERROR: Working directory has uncommitted changes.')
        print('Commit or stash them before running a sweep.')
        print('(Sweeping checks out different commits, which would lose your changes.)')
        sys.exit(1)

    original_ref = _git('rev-parse', '--abbrev-ref', 'HEAD')
    if original_ref == 'HEAD':
        original_ref = _git('rev-parse', 'HEAD')

    print(f'Sweeping {len(shas)} commits: {shas[0][:8]}..{shas[-1][:8]}')
    print(f'Results directory: {RESULTS_DIR}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_result_files: list[Path] = []

    try:
        for i, sha in enumerate(shas):
            short = sha[:8]
            subject = _git('log', '--format=%s', '-1', sha)
            print(f'\n{"=" * 70}')
            print(f'[{i + 1}/{len(shas)}] {short} {subject[:60]}')
            print('=' * 70)

            # Checkout
            _git('checkout', sha)

            # Check if benchmarks/ exists on this commit
            if not Path('benchmarks/run.py').exists():
                print(f'  SKIP: benchmarks/ not found on {short}')
                continue

            # Install
            print('  Installing...')
            subprocess.run(['uv', 'pip', 'install', '-e', '.', '--quiet'], check=True)

            # Run benchmarks
            success = run_benchmark_subprocess(benchmark_args, name=short)
            if not success:
                print(f'  WARN: Benchmark failed on {short}')
                continue

            # Collect results and add commit metadata
            for json_file in Path('benchmarks/results').glob(f'{short}_*.json'):
                with open(json_file) as f:
                    data = json.load(f)
                data['metadata']['commit'] = sha
                data['metadata']['commit_short'] = short
                data['metadata']['commit_subject'] = subject

                out_path = RESULTS_DIR / json_file.name
                with open(out_path, 'w') as f:
                    json.dump(data, f, indent=2)
                all_result_files.append(out_path)
                print(f'  → {out_path}')

    finally:
        print(f'\nRestoring {original_ref}...')
        _git('checkout', original_ref)
        subprocess.run(['uv', 'pip', 'install', '-e', '.', '--quiet'], check=True)

    print(f'\nSweep complete. {len(all_result_files)} result files in {RESULTS_DIR}/')

    # Generate table and plot
    if all_result_files:
        from benchmarks.compare import load_results, sweep_plot, sweep_table

        grouped = load_results([str(p) for p in all_result_files])
        sweep_table(grouped)
        sweep_plot(grouped)

    return all_result_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sweep benchmarks across git commits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmarks.sweep HEAD~5..HEAD --model simple --phase build --sizes 168
  python -m benchmarks.sweep main,feature/perf --all --quick --iterations 3
  python -m benchmarks.sweep v1.0,v2.0,HEAD --model district --phase build memory
        """,
    )
    parser.add_argument('revisions', help='Git revision range (HEAD~5..HEAD) or comma-separated refs')
    parser.add_argument('--model', nargs='+', help='Model(s) to run')
    parser.add_argument('--phase', nargs='+', help='Phase(s) to run')
    parser.add_argument('--sizes', type=int, nargs='+', help='Override timestep sizes')
    parser.add_argument(
        '--iterations',
        type=int,
        default=DEFAULT_SWEEP_ITERATIONS,
        help=f'Number of iterations (default: {DEFAULT_SWEEP_ITERATIONS})',
    )
    parser.add_argument('--all', action='store_true', help='Run all models × phases')
    parser.add_argument('--quick', action='store_true', help='Use QUICK_SIZES')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be benchmarked without running anything',
    )

    args = parser.parse_args()

    # Build the args to pass to benchmarks.run
    benchmark_args = []
    if args.all:
        benchmark_args.append('--all')
    if args.model:
        benchmark_args.extend(['--model', *args.model])
    if args.phase:
        benchmark_args.extend(['--phase', *args.phase])
    if args.sizes:
        benchmark_args.extend(['--sizes', *[str(s) for s in args.sizes]])
    if args.quick:
        benchmark_args.append('--quick')
    benchmark_args.extend(['--iterations', str(args.iterations)])

    if not args.all and not args.model and not args.phase:
        parser.error('Specify --all, --model, or --phase')

    sweep(args.revisions, benchmark_args, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
