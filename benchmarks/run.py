"""Benchmark orchestrator: iterate models × phases × sizes.

Usage:
    python -m benchmarks.run --name main --all
    python -m benchmarks.run --name main --model simple district --phase build
    python -m benchmarks.run --name main --all --sizes 24 168
    python -m benchmarks.run --name main --all --quick
    python -m benchmarks.run --list
    python -m benchmarks.run --sweep HEAD~5..HEAD --model simple --phase build --sizes 168 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.models import discover_models
from benchmarks.runners import discover_runners

RESULTS_DIR = Path(__file__).parent / 'results'


def run_single(
    name: str,
    model_name: str,
    phase: str,
    iterations: int,
    *,
    sizes: list[int] | None = None,
    quick: bool = False,
) -> dict:
    """Run a single model × phase benchmark and save JSON."""
    models = discover_models()
    runners = discover_runners()

    if model_name not in models:
        raise ValueError(f'Unknown model {model_name!r}. Available: {list(models)}')
    if phase not in runners:
        raise ValueError(f'Unknown phase {phase!r}. Available: {list(runners)}')

    model_mod = models[model_name]
    runner_mod = runners[phase]

    if sizes is not None:
        run_sizes = sizes
    elif quick:
        run_sizes = model_mod.QUICK_SIZES
    else:
        run_sizes = model_mod.SIZES

    print(f'[{model_name}/{phase}] sizes={run_sizes}, iterations={iterations}')

    results_by_size = {}
    for size in run_sizes:
        print(f'  size={size} ...', end=' ', flush=True)
        result = runner_mod.run(model_mod, size, iterations)
        results_by_size[size] = result
        unit = result.get('unit')
        if unit:
            print(f'{result["median"]:.1f} {unit}')
        else:
            print(f'{result["median"] * 1000:.1f}ms')

    output = {
        'name': name,
        'model': model_name,
        'model_label': getattr(model_mod, 'LABEL', model_name),
        'phase': phase,
        'iterations': iterations,
        'quick': quick,
        'sizes': run_sizes,
        'results': {str(s): r for s, r in results_by_size.items()},
        'metadata': {
            'timestamp': datetime.now(UTC).isoformat(),
            'python': sys.version,
            'platform': platform.platform(),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f'{name}_{model_name}_{phase}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  → {out_path}')
    return output


def run_matrix(
    name: str,
    iterations: int,
    *,
    model_names: list[str] | None = None,
    phase_names: list[str] | None = None,
    sizes: list[int] | None = None,
    quick: bool = False,
) -> list[dict]:
    """Run selected models × phases (defaults to all)."""
    models = discover_models()
    runners = discover_runners()

    selected_models = sorted(model_names) if model_names else sorted(models)
    selected_phases = sorted(phase_names) if phase_names else sorted(runners)

    results = []
    for model_name in selected_models:
        for phase in selected_phases:
            result = run_single(name, model_name, phase, iterations, sizes=sizes, quick=quick)
            results.append(result)
    return results


def _git(*cmd: str) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(['git', *cmd], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def resolve_revisions(rev_range: str) -> list[str]:
    """Resolve a git revision range (e.g. HEAD~5..HEAD) to a list of commit SHAs, oldest first."""
    if '..' in rev_range:
        output = _git('rev-list', '--reverse', rev_range)
    else:
        # Treat as a single ref or comma-separated list
        output = '\n'.join(_git('rev-parse', r.strip()) for r in rev_range.split(','))
    return [line.strip() for line in output.splitlines() if line.strip()]


def sweep(
    rev_range: str,
    iterations: int,
    *,
    model_names: list[str] | None = None,
    phase_names: list[str] | None = None,
    sizes: list[int] | None = None,
    quick: bool = False,
) -> list[Path]:
    """Benchmark across multiple git commits.

    Checks out each commit, installs, runs benchmarks, then restores the original branch.
    Returns paths to all generated JSON files.
    """
    original_ref = _git('rev-parse', '--abbrev-ref', 'HEAD')
    if original_ref == 'HEAD':
        # Detached HEAD — save the exact SHA
        original_ref = _git('rev-parse', 'HEAD')

    shas = resolve_revisions(rev_range)
    if not shas:
        print(f'No commits found for range: {rev_range}')
        sys.exit(1)

    print(f'Sweeping {len(shas)} commits: {shas[0][:8]}..{shas[-1][:8]}')

    all_result_files: list[Path] = []

    try:
        for i, sha in enumerate(shas):
            short = sha[:8]
            # Get commit subject for display
            subject = _git('log', '--format=%s', '-1', sha)
            print(f'\n{"=" * 60}')
            print(f'[{i + 1}/{len(shas)}] {short} {subject}')
            print('=' * 60)

            _git('checkout', sha)
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'],
                check=True,
            )

            results = run_matrix(
                short,
                iterations,
                model_names=model_names,
                phase_names=phase_names,
                sizes=sizes,
                quick=quick,
            )

            # Tag each result with commit info for sweep plots
            for r in results:
                r['metadata']['commit'] = sha
                r['metadata']['commit_short'] = short
                r['metadata']['commit_subject'] = subject
                # Re-save with commit metadata
                out_path = RESULTS_DIR / f'{short}_{r["model"]}_{r["phase"]}.json'
                with open(out_path, 'w') as f:
                    json.dump(r, f, indent=2)
                all_result_files.append(out_path)

    finally:
        print(f'\nRestoring {original_ref}...')
        _git('checkout', original_ref)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', '.', '--quiet'],
            check=True,
        )

    print(f'\nSweep complete. {len(all_result_files)} result files in {RESULTS_DIR}/')
    return all_result_files


def list_available() -> None:
    """Print available models and phases."""
    models = discover_models()
    runners = discover_runners()

    print('Models:')
    for name, mod in sorted(models.items()):
        label = getattr(mod, 'LABEL', name)
        print(f'  {name:20s} {label} (sizes: {mod.SIZES})')

    print('\nPhases (runners):')
    for name, mod in sorted(runners.items()):
        doc = (mod.__doc__ or '').strip().split('\n')[0]
        print(f'  {name:20s} {doc}')


def main() -> None:
    parser = argparse.ArgumentParser(description='flixopt benchmark runner')
    parser.add_argument('--name', default='current', help='Label for this benchmark run')
    parser.add_argument('--model', nargs='+', help='Model(s) to run (default: all)')
    parser.add_argument('--phase', nargs='+', help='Phase(s)/runner(s) to run (default: all)')
    parser.add_argument('--sizes', type=int, nargs='+', help='Override timestep sizes')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--all', action='store_true', help='Run all models × phases')
    parser.add_argument('--quick', action='store_true', help='Use QUICK_SIZES instead of SIZES')
    parser.add_argument('--list', action='store_true', help='List available models and phases')
    parser.add_argument('--sweep', metavar='REV_RANGE', help='Sweep across git commits (e.g. HEAD~5..HEAD)')

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if args.sweep:
        result_files = sweep(
            args.sweep,
            args.iterations,
            model_names=args.model,
            phase_names=args.phase,
            sizes=args.sizes,
            quick=args.quick,
        )
        # Auto-generate sweep plot
        from benchmarks.compare import load_results, sweep_plot

        grouped = load_results([str(p) for p in result_files])
        sweep_plot(grouped)
        return

    if args.all or args.model or args.phase:
        run_matrix(
            args.name,
            args.iterations,
            model_names=args.model,
            phase_names=args.phase,
            sizes=args.sizes,
            quick=args.quick,
        )
    else:
        parser.error('Specify --all, --model, --phase, --sweep, or --list')


if __name__ == '__main__':
    main()
