"""Build the pre-built example FlowSystems hosted for the advanced notebooks (08-09).

This regenerates the realistic example systems (which need ``demandlib``/``pvlib`` and
the raw input CSVs under ``docs/notebooks/data``), serialises each one with
``FlowSystem.to_netcdf`` and writes a ``registry.txt`` with sha256 hashes.

The resulting ``*.nc`` files **and** ``registry.txt`` are uploaded as assets to the
GitHub release tagged ``flixopt.tutorials._examples.DATA_RELEASE``; at runtime
``flixopt.tutorials.load_example`` downloads them from there. Run this whenever the
example systems change, then re-upload the assets (the CI workflow does this on demand).

Usage:
    python scripts/build_tutorial_datasets.py [--out-dir dist/tutorial_datasets]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'docs' / 'notebooks' / 'data'


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=REPO_ROOT / 'dist' / 'tutorial_datasets',
        help='Directory to write the *.nc artefacts and registry.txt into.',
    )
    args = parser.parse_args()

    sys.path.insert(0, str(DATA_DIR))
    import generate_example_systems as ges  # noqa: E402

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from flixopt.tutorials import list_examples  # noqa: E402

    registry_lines = []
    for name in list_examples():
        func_name = f'create_{name}_system'
        print(f'Building {name} via {func_name}() ...', flush=True)
        fs = getattr(ges, func_name)()
        path = out_dir / f'{name}.nc'
        fs.to_netcdf(path)
        digest = _sha256(path)
        registry_lines.append(f'{name}.nc sha256:{digest}')
        print(f'  -> {path.name} ({path.stat().st_size:,} bytes) sha256:{digest}')

    registry_path = out_dir / 'registry.txt'
    registry_path.write_text('\n'.join(registry_lines) + '\n')
    print(f'\nWrote {registry_path} with {len(registry_lines)} entries.')
    print('Upload every *.nc and registry.txt as assets to the GitHub release.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
