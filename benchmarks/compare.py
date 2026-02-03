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

from benchmarks.config import RESULTS_DIR


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
    """Create an interactive timeline plot showing performance across commits using plotly."""
    import pandas as pd
    import plotly.express as px

    # Build ordered commit list from timestamps
    commit_info: dict[str, dict] = {}
    for name, results in grouped.items():
        meta = results[0].get('metadata', {})
        commit_info[name] = {
            'timestamp': meta.get('timestamp', ''),
            'short': meta.get('commit_short', name),
            'subject': meta.get('commit_subject', ''),
        }

    ordered_names = sorted(commit_info, key=lambda n: commit_info[n]['timestamp'])

    # Determine which keys have units (memory)
    units: dict[str, str] = {}
    for _name, results in grouped.items():
        for r in results:
            key = f'{r["model"]}/{r["phase"]}'
            for _size_str, stats in r['results'].items():
                if 'unit' in stats:
                    units[key] = stats['unit']

    # Build long-form DataFrame for plotly
    rows = []
    for name in ordered_names:
        info = commit_info[name]
        series = _extract_series(grouped[name])
        for key, size_map in series.items():
            for size, median in size_map.items():
                unit = units.get(key)
                rows.append(
                    {
                        'commit': info['short'],
                        'subject': info['subject'][:60],
                        'benchmark': f'{key} (n={size})',
                        'value': median if unit else median * 1000,
                        'unit': unit or 'ms',
                    }
                )

    if not rows:
        print('No data to plot.')
        return

    df = pd.DataFrame(rows)

    # Preserve commit order
    df['commit'] = pd.Categorical(
        df['commit'],
        categories=[commit_info[n]['short'] for n in ordered_names],
        ordered=True,
    )

    fig = px.line(
        df,
        x='commit',
        y='value',
        color='benchmark',
        facet_row='unit',
        markers=True,
        hover_data=['subject'],
        title=f'Performance sweep ({len(ordered_names)} commits)',
        labels={'value': '', 'commit': 'Commit'},
    )
    fig.update_yaxes(matches=None)  # Independent y-axes per facet
    fig.for_each_yaxis(lambda y: y.update(title=''))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

    if output is None:
        first = commit_info[ordered_names[0]]['short']
        last = commit_info[ordered_names[-1]]['short']
        output = RESULTS_DIR / f'sweep_{first}_{last}.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f'Saved sweep plot: {output}')


def sweep_table(grouped: dict[str, list[dict]], output: Path | None = None) -> str:
    """Generate a markdown table showing performance across commits with speedup vs first commit.

    Returns the markdown string and optionally writes it to a file.
    """
    import pandas as pd

    # Build ordered commit list from timestamps
    commit_info: dict[str, dict] = {}
    for name, results in grouped.items():
        meta = results[0].get('metadata', {})
        commit_info[name] = {
            'timestamp': meta.get('timestamp', ''),
            'short': meta.get('commit_short', name),
            'subject': meta.get('commit_subject', ''),
        }

    ordered_names = sorted(commit_info, key=lambda n: commit_info[n]['timestamp'])

    # Determine which keys have units (memory)
    units: dict[str, str] = {}
    for _name, results in grouped.items():
        for r in results:
            key = f'{r["model"]}/{r["phase"]}'
            for _size_str, stats in r['results'].items():
                if 'unit' in stats:
                    units[key] = stats['unit']

    # Build rows: one per commit
    rows = []
    for name in ordered_names:
        info = commit_info[name]
        subject = info['subject']
        if len(subject) > 60:
            subject = subject[:57] + '...'
        row = {'Commit': info['short'], 'Description': subject}

        series = _extract_series(grouped[name])
        for key, size_map in series.items():
            for size, median in size_map.items():
                unit = units.get(key)
                if unit:
                    row[f'{key} n={size} ({unit})'] = median
                else:
                    row[f'{key} n={size} (ms)'] = round(median * 1000, 1)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add speedup columns (base = first row)
    metric_cols = [c for c in df.columns if c not in ('Commit', 'Description')]
    for col in metric_cols:
        base_val = df[col].iloc[0]
        df[f'{col} speedup'] = (base_val / df[col]).map(lambda x: f'{x:.2f}x')

    # Interleave: metric, speedup, metric, speedup, ...
    ordered_cols = ['Commit', 'Description']
    for col in metric_cols:
        ordered_cols.extend([col, f'{col} speedup'])
    df = df[ordered_cols]

    # Bold the first row commit
    df.loc[0, 'Commit'] = df.loc[0, 'Commit'] + ' (base)'

    md = df.to_markdown(index=False)

    if output is None:
        first = commit_info[ordered_names[0]]['short']
        last = commit_info[ordered_names[-1]]['short']
        output = RESULTS_DIR / f'sweep_{first}_{last}.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(md + '\n')
    print(f'Saved sweep table: {output}')
    print()
    print(md)

    return md


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('files', nargs='+', help='JSON result files to compare')
    parser.add_argument('--output', type=Path, help='Output PNG path')
    parser.add_argument('--sweep', action='store_true', help='Generate sweep timeline plot instead of comparison')
    parser.add_argument('--table', action='store_true', help='Generate sweep markdown table')
    args = parser.parse_args()

    grouped = load_results(args.files)

    if args.table:
        sweep_table(grouped, args.output)
    elif args.sweep:
        sweep_plot(grouped, args.output)
    else:
        compare_plot(grouped, args.output)


if __name__ == '__main__':
    main()
