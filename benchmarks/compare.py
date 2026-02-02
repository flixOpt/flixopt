"""Load 2+ benchmark JSON results and produce comparison plots.

Usage:
    python -m benchmarks.compare results/main_*.json results/ref_*.json
    python -m benchmarks.compare --sweep results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / 'results'


def load_results(paths: list[str]) -> dict[str, list[dict]]:
    """Load JSON results grouped by run name."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        by_name[data['name']].append(data)
    return dict(by_name)


def _extract_series(results: list[dict]) -> dict[str, dict[int, float]]:
    """Extract {model_phase: {size: median}} from a list of result dicts."""
    series = {}
    for r in results:
        key = f'{r["model"]}/{r["phase"]}'
        medians = {}
        for size_str, stats in r['results'].items():
            medians[int(size_str)] = stats['median']
        series[key] = medians
    return series


def compare_plot(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Create a 4-panel comparison plot for 2+ runs."""
    names = list(grouped.keys())
    if len(names) < 2:
        print('Need at least 2 run names to compare. Got:', names)
        sys.exit(1)

    all_series = {name: _extract_series(results) for name, results in grouped.items()}

    # Collect all benchmark keys
    all_keys = set()
    for s in all_series.values():
        all_keys.update(s.keys())
    all_keys = sorted(all_keys)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Benchmark Comparison: {" vs ".join(names)}', fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # Panel 1: Log-log overview (all benchmarks)
    ax = axes[0, 0]
    ax.set_title('Overview (log-log)')
    for idx, name in enumerate(names):
        series = all_series[name]
        for key in all_keys:
            if key in series:
                sizes = sorted(series[key].keys())
                medians = [series[key][s] for s in sizes]
                ax.plot(sizes, medians, 'o-', color=colors[idx], label=f'{name}: {key}', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Time (s)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Speedup ratio (name[0] / name[1])
    ax = axes[0, 1]
    base_name, comp_name = names[0], names[1]
    ax.set_title(f'Speedup: {base_name} / {comp_name}')
    base_series = all_series[base_name]
    comp_series = all_series[comp_name]
    for key in all_keys:
        if key in base_series and key in comp_series:
            common_sizes = sorted(set(base_series[key]) & set(comp_series[key]))
            if common_sizes:
                ratios = [base_series[key][s] / comp_series[key][s] for s in common_sizes]
                ax.plot(common_sizes, ratios, 'o-', label=key, alpha=0.7)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel(f'Ratio ({base_name} / {comp_name})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Small models (linear scale, sizes <= 720)
    ax = axes[1, 0]
    ax.set_title('Small models (linear)')
    for idx, name in enumerate(names):
        series = all_series[name]
        for key in all_keys:
            if key in series:
                sizes = sorted(s for s in series[key] if s <= 720)
                if sizes:
                    medians = [series[key][s] for s in sizes]
                    ax.plot(sizes, medians, 'o-', color=colors[idx], label=f'{name}: {key}', alpha=0.7)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Time (s)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 4: Large models (log scale, sizes >= 720)
    ax = axes[1, 1]
    ax.set_title('Large models (log)')
    for idx, name in enumerate(names):
        series = all_series[name]
        for key in all_keys:
            if key in series:
                sizes = sorted(s for s in series[key] if s >= 720)
                if sizes:
                    medians = [series[key][s] for s in sizes]
                    ax.plot(sizes, medians, 'o-', color=colors[idx], label=f'{name}: {key}', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Time (s)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output is None:
        output = RESULTS_DIR / f'compare_{"_vs_".join(names)}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f'Saved comparison plot: {output}')
    plt.close(fig)


def sweep_plot(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Create a timeline plot showing performance across commits.

    Each run name is treated as a commit (ordered by timestamp).
    One subplot per unique model/phase+size combination.
    """
    # Build ordered commit list from timestamps
    commit_info: dict[str, dict] = {}  # name -> {timestamp, short, subject}
    for name, results in grouped.items():
        meta = results[0].get('metadata', {})
        commit_info[name] = {
            'timestamp': meta.get('timestamp', ''),
            'short': meta.get('commit_short', name),
            'subject': meta.get('commit_subject', ''),
        }

    # Sort names by timestamp (oldest first)
    ordered_names = sorted(commit_info, key=lambda n: commit_info[n]['timestamp'])

    # Collect all (model/phase, size) combos
    combos: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))  # key -> {size -> [values per commit]}
    for name in ordered_names:
        series = _extract_series(grouped[name])
        for key, size_map in series.items():
            for size, median in size_map.items():
                combos[key][size].append(median)

    # Determine which keys have units (memory)
    units: dict[str, str] = {}
    for _name, results in grouped.items():
        for r in results:
            key = f'{r["model"]}/{r["phase"]}'
            for _size_str, stats in r['results'].items():
                if 'unit' in stats:
                    units[key] = stats['unit']

    # Group by phase for subplot layout
    all_keys = sorted(combos.keys())
    n_plots = sum(len(sizes) for sizes in combos.values())

    if n_plots == 0:
        print('No data to plot.')
        return

    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    x_labels = [commit_info[n]['short'] for n in ordered_names]
    x = np.arange(len(ordered_names))

    plot_idx = 0
    for key in all_keys:
        for size in sorted(combos[key]):
            values = combos[key][size]
            if len(values) != len(ordered_names):
                continue  # Skip incomplete data

            row, col = divmod(plot_idx, n_cols)
            ax = axes[row][col]

            unit = units.get(key)
            if unit:
                ax.plot(x, values, 'o-', linewidth=2, markersize=5)
                ax.set_ylabel(unit)
            else:
                ax.plot(x, [v * 1000 for v in values], 'o-', linewidth=2, markersize=5)
                ax.set_ylabel('ms')

            ax.set_title(f'{key} (n={size})', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Highlight min value
            arr = np.array(values)
            best = arr.argmin()
            ax.annotate(
                x_labels[best],
                xy=(best, arr[best] * (1 if unit else 1000)),
                fontsize=7,
                color='green',
                ha='center',
                va='top',
            )

            plot_idx += 1

    # Hide unused axes
    for idx in range(plot_idx, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f'Performance sweep: {x_labels[0]} → {x_labels[-1]} ({len(ordered_names)} commits)', fontsize=13)
    plt.tight_layout()

    if output is None:
        output = RESULTS_DIR / f'sweep_{x_labels[0]}_{x_labels[-1]}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f'Saved sweep plot: {output}')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('files', nargs='+', help='JSON result files to compare')
    parser.add_argument('--output', type=Path, help='Output PNG path')
    parser.add_argument('--sweep', action='store_true', help='Generate sweep timeline plot instead of comparison')
    args = parser.parse_args()

    grouped = load_results(args.files)

    if args.sweep:
        sweep_plot(grouped, args.output)
    else:
        compare_plot(grouped, args.output)


if __name__ == '__main__':
    main()
