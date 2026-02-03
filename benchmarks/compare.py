"""Load benchmark JSON results and produce comparison tables.

Usage:
    python -m benchmarks.compare results/*.json
    python -m benchmarks.compare results/*.json --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(paths: list[str]) -> dict[str, list[dict]]:
    """Load JSON results grouped by run name."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        by_name[data['name']].append(data)
    return dict(by_name)


def build_dataframe(grouped: dict[str, list[dict]]) -> tuple[list[dict], list[str]]:
    """Build rows for table output.

    Returns (rows, column_order) where each row is a dict.
    """
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

    # Collect all (model, phase, size) combinations and their units
    all_keys: set[tuple[str, str, int]] = set()
    units: dict[tuple[str, str], str] = {}

    for name in ordered_runs:
        for r in grouped[name]:
            model = r['model']
            phase = r['phase']
            for size_str, stats in r['results'].items():
                size = int(size_str)
                all_keys.add((model, phase, size))
                if 'unit' in stats:
                    units[(model, phase)] = stats['unit']

    # Sort keys: by model, then phase, then size
    sorted_keys = sorted(all_keys, key=lambda k: (k[0], k[1], k[2]))

    # Build rows
    rows = []
    for name in ordered_runs:
        info = run_info[name]
        subject = info['subject'][:40] if info['subject'] else ''

        row = {
            'run': info['short'],
            'description': subject,
        }

        # Index results by (model, phase, size)
        result_map: dict[tuple[str, str, int], float] = {}
        for r in grouped[name]:
            model = r['model']
            phase = r['phase']
            for size_str, stats in r['results'].items():
                size = int(size_str)
                value = stats['median']
                unit = units.get((model, phase), 's')
                # Convert to ms for time values
                if unit == 's':
                    value = value * 1000
                result_map[(model, phase, size)] = value

        for key in sorted_keys:
            model, phase, size = key
            unit = units.get((model, phase), 's')
            unit_label = 'MB' if unit == 'MB' else 'ms'
            col_name = f'{model}/{phase}/n={size} ({unit_label})'
            row[col_name] = result_map.get(key)

        rows.append(row)

    # Column order
    columns = ['run', 'description']
    for key in sorted_keys:
        model, phase, size = key
        unit = units.get((model, phase), 's')
        unit_label = 'MB' if unit == 'MB' else 'ms'
        columns.append(f'{model}/{phase}/n={size} ({unit_label})')

    return rows, columns


def print_table(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Print markdown table to stdout and optionally save to file."""
    rows, columns = build_dataframe(grouped)

    if not rows:
        print('No data.')
        return

    # Format values
    def fmt(v):
        if v is None:
            return '-'
        if isinstance(v, float):
            return f'{v:.1f}'
        return str(v)

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(fmt(row.get(col, ''))))

    # Print header
    header = ' | '.join(col.ljust(widths[col]) for col in columns)
    separator = ' | '.join('-' * widths[col] for col in columns)
    print(header)
    print(separator)

    # Print rows
    for row in rows:
        line = ' | '.join(fmt(row.get(col, '')).ljust(widths[col]) for col in columns)
        print(line)

    # Save to file if specified
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            f.write(header + '\n')
            f.write(separator + '\n')
            for row in rows:
                line = ' | '.join(fmt(row.get(col, '')).ljust(widths[col]) for col in columns)
                f.write(line + '\n')
        print(f'\nSaved to {output}')


def print_csv(grouped: dict[str, list[dict]], output: Path | None = None) -> None:
    """Print CSV to stdout and optionally save to file."""
    rows, columns = build_dataframe(grouped)

    if not rows:
        print('No data.')
        return

    # Write to stdout
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)

    # Save to file if specified
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved to {output}', file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('files', nargs='+', help='JSON result files')
    parser.add_argument('--output', '-o', type=Path, help='Output file path')
    parser.add_argument('--csv', action='store_true', help='Output as CSV instead of markdown')
    args = parser.parse_args()

    grouped = load_results(args.files)

    if args.csv:
        print_csv(grouped, args.output)
    else:
        print_table(grouped, args.output)


if __name__ == '__main__':
    main()
