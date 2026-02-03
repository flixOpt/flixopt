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


def plot(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Create a unified benchmark plot.

    Layout:
        - Rows: Phases (build, io, lp_write, memory)
        - Columns: Models (Simple, District, Synthetic XL)
        - X-axis: Timesteps (log scale)
        - Y-axis: Value (ms or MB depending on phase)
        - Color/lines: Runs/commits (single run = 1 line, sweep = multiple lines)
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Build ordered run list from timestamps
    run_info: dict[str, dict] = {}
    for name, results in grouped.items():
        meta = results[0].get('metadata', {})
        run_info[name] = {
            'timestamp': meta.get('timestamp', ''),
            'short': meta.get('commit_short', name),
            'subject': meta.get('commit_subject', ''),
        }

    ordered_runs = sorted(run_info, key=lambda n: run_info[n]['timestamp'])

    # Build long-form DataFrame
    rows = []
    for name in ordered_runs:
        info = run_info[name]
        for r in grouped[name]:
            model = r.get('model_label', r['model'])
            phase = r['phase']
            for size_str, stats in r['results'].items():
                size = int(size_str)
                unit = stats.get('unit', 's')
                value = stats['median']
                if unit == 's':
                    value = value * 1000
                    unit = 'ms'
                rows.append(
                    {
                        'run': name,
                        'run_label': info['short'],
                        'subject': info['subject'][:40] if info['subject'] else name,
                        'model': model,
                        'phase': phase,
                        'size': size,
                        'value': value,
                        'unit': unit,
                    }
                )

    if not rows:
        print('No data to plot.')
        return

    df = pd.DataFrame(rows)

    # Get unique values (preserve order)
    models = df['model'].unique().tolist()
    phases = df['phase'].unique().tolist()
    runs = [run_info[r]['short'] for r in ordered_runs]

    # Create subplots: rows = phases, cols = models
    fig = make_subplots(
        rows=len(phases),
        cols=len(models),
        row_titles=phases,
        column_titles=models,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    # Color map for runs: sequential colorscale (old = light, new = dark)
    import plotly.express as px

    if len(runs) == 1:
        run_colors = {runs[0]: px.colors.qualitative.Plotly[0]}
    else:
        # Use a sequential colorscale: light (old) -> dark (new)
        colorscale = px.colors.sequential.Blues
        # Sample colors from the scale (skip very light colors)
        n = len(runs)
        indices = [int(2 + i * (len(colorscale) - 3) / max(n - 1, 1)) for i in range(n)]
        run_colors = {run: colorscale[idx] for run, idx in zip(runs, indices, strict=False)}

    # Add traces
    for row_idx, phase in enumerate(phases, 1):
        df_phase = df[df['phase'] == phase]
        unit = df_phase['unit'].iloc[0] if len(df_phase) > 0 else 'ms'

        for col_idx, model in enumerate(models, 1):
            df_cell = df_phase[df_phase['model'] == model]

            for run_label in runs:
                df_run = df_cell[df_cell['run_label'] == run_label].sort_values('size')
                if len(df_run) == 0:
                    continue

                # Get hover text
                subject = df_run['subject'].iloc[0]
                hover = f'{run_label}: {subject}' if subject != run_label else run_label

                fig.add_trace(
                    go.Scatter(
                        x=df_run['size'].tolist(),
                        y=df_run['value'].tolist(),
                        mode='lines+markers',
                        name=run_label,
                        legendgroup=run_label,
                        showlegend=(row_idx == 1 and col_idx == 1),
                        line=dict(color=run_colors[run_label]),
                        marker=dict(size=6),
                        hovertemplate=f'{hover}<br>n=%{{x}}<br>%{{y:.1f}} {unit}<extra></extra>',
                    ),
                    row=row_idx,
                    col=col_idx,
                )

        # Set y-axis label for this row
        fig.update_yaxes(title_text=unit, row=row_idx, col=1)

    # Update x-axes to log scale
    fig.update_xaxes(type='log', title_text='Timesteps')

    # Layout
    n_runs = len(runs)
    title = f'Benchmark Results ({n_runs} run{"s" if n_runs > 1 else ""})'
    if n_runs > 1 and run_info[ordered_runs[0]]['subject']:
        title = f'Benchmark Sweep ({n_runs} commits)'

    fig.update_layout(
        title=title,
        height=220 * len(phases) + 80,
        width=320 * len(models) + 100,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
    )

    if output is None:
        run_names = '_'.join(runs[:3])
        if len(runs) > 3:
            run_names += f'_+{len(runs) - 3}'
        output = RESULTS_DIR / f'plot_{run_names}.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f'Saved plot: {output}')


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


def results_plot(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Create an interactive faceted plot showing results per model.

    Layout: rows = models, cols = unit (time vs memory), color = phase, x = timesteps
    """
    import pandas as pd
    import plotly.express as px

    # Build long-form DataFrame
    rows = []
    for name, results in grouped.items():
        for r in results:
            model = r.get('model_label', r['model'])
            phase = r['phase']
            for size_str, stats in r['results'].items():
                size = int(size_str)
                unit = stats.get('unit', 's')
                value = stats['median']
                # Convert seconds to ms for better readability
                if unit == 's':
                    value = value * 1000
                    unit = 'ms'
                rows.append(
                    {
                        'run': name,
                        'model': model,
                        'phase': phase,
                        'timesteps': size,
                        'value': value,
                        'unit': unit,
                        'min': stats.get('min', value) * (1000 if unit == 'ms' else 1),
                        'max': stats.get('max', value) * (1000 if unit == 'ms' else 1),
                    }
                )

    if not rows:
        print('No data to plot.')
        return

    df = pd.DataFrame(rows)

    # Separate time and memory for different y-axes
    df_time = df[df['unit'] == 'ms'].copy()
    df_memory = df[df['unit'] == 'MB'].copy()

    # Determine layout
    has_time = len(df_time) > 0
    has_memory = len(df_memory) > 0
    models = df['model'].unique().tolist()

    if has_time and has_memory:
        # Two columns: time and memory
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=len(models),
            cols=2,
            subplot_titles=[f'{m} - Time' for m in models] + [f'{m} - Memory' for m in models],
            row_titles=models,
            column_titles=['Time (ms)', 'Memory (MB)'],
            shared_xaxes=True,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
        )

        colors = px.colors.qualitative.Plotly
        phase_colors = {phase: colors[i % len(colors)] for i, phase in enumerate(df['phase'].unique())}

        for row_idx, model in enumerate(models, 1):
            # Time column
            model_time = df_time[df_time['model'] == model]
            for phase in model_time['phase'].unique():
                phase_data = model_time[model_time['phase'] == phase].sort_values('timesteps')
                fig.add_scatter(
                    x=phase_data['timesteps'],
                    y=phase_data['value'],
                    mode='lines+markers',
                    name=phase,
                    legendgroup=phase,
                    showlegend=(row_idx == 1),
                    line=dict(color=phase_colors[phase]),
                    marker=dict(size=8),
                    row=row_idx,
                    col=1,
                )

            # Memory column
            model_mem = df_memory[df_memory['model'] == model]
            for phase in model_mem['phase'].unique():
                phase_data = model_mem[model_mem['phase'] == phase].sort_values('timesteps')
                fig.add_scatter(
                    x=phase_data['timesteps'],
                    y=phase_data['value'],
                    mode='lines+markers',
                    name=phase,
                    legendgroup=phase,
                    showlegend=False,
                    line=dict(color=phase_colors[phase]),
                    marker=dict(size=8),
                    row=row_idx,
                    col=2,
                )

        fig.update_xaxes(title_text='Timesteps', type='log')
        fig.update_yaxes(title_text='ms', col=1)
        fig.update_yaxes(title_text='MB', col=2)

    else:
        # Single metric type - use plotly express facet
        df_plot = df_time if has_time else df_memory
        unit_label = 'ms' if has_time else 'MB'

        fig = px.line(
            df_plot,
            x='timesteps',
            y='value',
            color='phase',
            facet_row='model',
            markers=True,
            log_x=True,
            labels={'value': unit_label, 'timesteps': 'Timesteps'},
        )
        fig.update_yaxes(matches=None)  # Independent y-axes per facet
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))

    run_names = list(grouped.keys())
    title = f'Benchmark Results: {", ".join(run_names)}'
    fig.update_layout(
        title=title,
        height=300 * len(models),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    if output is None:
        output = RESULTS_DIR / f'results_{"_".join(run_names)}.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f'Saved results plot: {output}')


def sweep_results_plot(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Create per-model faceted sweep plot with dropdown for timestep size.

    Layout: rows = models, cols = unit (time vs memory), x = commits, color = phase
    Dropdown lets user select which timestep size to view.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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
    ordered_commits = [commit_info[n]['short'] for n in ordered_names]

    # Build long-form DataFrame
    rows = []
    for name in ordered_names:
        info = commit_info[name]
        for r in grouped[name]:
            model = r.get('model_label', r['model'])
            phase = r['phase']
            for size_str, stats in r['results'].items():
                size = int(size_str)
                unit = stats.get('unit', 's')
                value = stats['median']
                if unit == 's':
                    value = value * 1000
                    unit = 'ms'
                rows.append(
                    {
                        'commit': info['short'],
                        'subject': info['subject'][:40],
                        'model': model,
                        'phase': phase,
                        'size': size,
                        'value': value,
                        'unit': unit,
                    }
                )

    if not rows:
        print('No data to plot.')
        return

    df = pd.DataFrame(rows)

    # Get unique values
    models = df['model'].unique().tolist()
    sizes = sorted(df['size'].unique().tolist())
    phases = df['phase'].unique().tolist()

    # Check what units we have
    has_time = (df['unit'] == 'ms').any()
    has_memory = (df['unit'] == 'MB').any()
    n_cols = (1 if has_time else 0) + (1 if has_memory else 0)

    # Create subplots: rows = models, cols = time/memory
    col_titles = []
    if has_time:
        col_titles.append('Time (ms)')
    if has_memory:
        col_titles.append('Memory (MB)')

    fig = make_subplots(
        rows=len(models),
        cols=n_cols,
        row_titles=models,
        column_titles=col_titles,
        shared_xaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    # Color map for phases
    import plotly.express as px

    colors = px.colors.qualitative.Plotly
    phase_colors = {phase: colors[i % len(colors)] for i, phase in enumerate(phases)}

    # Add traces for each size (will toggle visibility with dropdown)
    trace_info = []  # Track which traces belong to which size

    for size in sizes:
        df_size = df[df['size'] == size]

        for row_idx, model in enumerate(models, 1):
            df_model = df_size[df_size['model'] == model]

            # Time column
            if has_time:
                col_idx = 1
                df_time = df_model[df_model['unit'] == 'ms']
                for phase in phases:
                    df_phase = df_time[df_time['phase'] == phase]
                    if len(df_phase) > 0:
                        # Sort by commit order
                        df_phase = df_phase.set_index('commit').reindex(ordered_commits).dropna()
                        fig.add_trace(
                            go.Scatter(
                                x=df_phase.index.tolist(),
                                y=df_phase['value'].tolist(),
                                mode='lines+markers',
                                name=phase,
                                legendgroup=phase,
                                showlegend=(row_idx == 1 and col_idx == 1 and size == sizes[0]),
                                line=dict(color=phase_colors[phase]),
                                marker=dict(size=8),
                                hovertemplate=f'{phase}<br>%{{x}}<br>%{{y:.1f}} ms<extra></extra>',
                                visible=(size == sizes[0]),  # Only first size visible initially
                            ),
                            row=row_idx,
                            col=col_idx,
                        )
                        trace_info.append(size)

            # Memory column
            if has_memory:
                col_idx = 2 if has_time else 1
                df_mem = df_model[df_model['unit'] == 'MB']
                for phase in phases:
                    df_phase = df_mem[df_mem['phase'] == phase]
                    if len(df_phase) > 0:
                        df_phase = df_phase.set_index('commit').reindex(ordered_commits).dropna()
                        fig.add_trace(
                            go.Scatter(
                                x=df_phase.index.tolist(),
                                y=df_phase['value'].tolist(),
                                mode='lines+markers',
                                name=phase,
                                legendgroup=phase,
                                showlegend=False,
                                line=dict(color=phase_colors[phase]),
                                marker=dict(size=8),
                                hovertemplate=f'{phase}<br>%{{x}}<br>%{{y:.1f}} MB<extra></extra>',
                                visible=(size == sizes[0]),
                            ),
                            row=row_idx,
                            col=col_idx,
                        )
                        trace_info.append(size)

    # Create dropdown buttons for size selection
    buttons = []
    for size in sizes:
        visibility = [s == size for s in trace_info]
        buttons.append(
            dict(
                label=f'n={size}',
                method='update',
                args=[{'visible': visibility}],
            )
        )

    fig.update_layout(
        title=f'Sweep Results ({len(ordered_names)} commits)',
        height=280 * len(models) + 100,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction='down',
                showactive=True,
                x=0.0,
                xanchor='left',
                y=1.15,
                yanchor='top',
            )
        ],
        annotations=[
            dict(
                text='Timesteps:',
                x=0.0,
                xref='paper',
                xanchor='right',
                xshift=-10,
                y=1.15,
                yref='paper',
                yanchor='top',
                showarrow=False,
            )
        ],
    )

    fig.update_xaxes(tickangle=45)

    if output is None:
        first = ordered_commits[0]
        last = ordered_commits[-1]
        output = RESULTS_DIR / f'sweep_results_{first}_{last}.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f'Saved sweep results plot: {output}')


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
    parser.add_argument('--output', type=Path, help='Output path')
    parser.add_argument('--table', action='store_true', help='Generate markdown table')
    parser.add_argument(
        '--legacy-compare',
        action='store_true',
        help='Legacy 4-panel matplotlib comparison plot',
    )
    args = parser.parse_args()

    grouped = load_results(args.files)

    if args.table:
        sweep_table(grouped, args.output)
    elif args.legacy_compare:
        compare_plot(grouped, args.output)
    else:
        # Default: unified plot (works for single run or sweep)
        plot(grouped, args.output)


if __name__ == '__main__':
    main()
