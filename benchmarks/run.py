"""Benchmark orchestrator: iterate models × phases × sizes.

Usage:
    python -m benchmarks.run --name main --all
    python -m benchmarks.run --name main --model simple --phase build
    python -m benchmarks.run --name main --all --quick
    python -m benchmarks.run --list
"""

from __future__ import annotations

import argparse
import json
import platform
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
    sizes = model_mod.QUICK_SIZES if quick else model_mod.SIZES

    print(f'[{model_name}/{phase}] sizes={sizes}, iterations={iterations}')

    results_by_size = {}
    for size in sizes:
        print(f'  size={size} ...', end=' ', flush=True)
        result = runner_mod.run(model_mod, size, iterations)
        results_by_size[size] = result
        print(f'{result["median"] * 1000:.1f}ms')

    output = {
        'name': name,
        'model': model_name,
        'model_label': getattr(model_mod, 'LABEL', model_name),
        'phase': phase,
        'iterations': iterations,
        'quick': quick,
        'sizes': sizes,
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


def run_all(name: str, iterations: int, *, quick: bool = False) -> list[dict]:
    """Run all models × all phases."""
    models = discover_models()
    runners = discover_runners()
    results = []
    for model_name in sorted(models):
        for phase in sorted(runners):
            result = run_single(name, model_name, phase, iterations, quick=quick)
            results.append(result)
    return results


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
    parser.add_argument('--model', help='Specific model to run')
    parser.add_argument('--phase', help='Specific phase/runner to run')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--all', action='store_true', help='Run all models × phases')
    parser.add_argument('--quick', action='store_true', help='Use QUICK_SIZES instead of SIZES')
    parser.add_argument('--list', action='store_true', help='List available models and phases')

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if args.all:
        run_all(args.name, args.iterations, quick=args.quick)
    elif args.model and args.phase:
        run_single(args.name, args.model, args.phase, args.iterations, quick=args.quick)
    else:
        parser.error('Specify --all, --list, or both --model and --phase')


if __name__ == '__main__':
    main()
