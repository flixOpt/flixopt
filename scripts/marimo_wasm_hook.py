"""mkdocs hook: export marimo notebooks as WASM apps after site build."""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger('mkdocs.hooks.marimo_wasm')

NOTEBOOKS = {
    'docs/wasm/quickstart.py': 'playground/quickstart',
}


def on_post_build(config, **kwargs):
    try:
        subprocess.run(['marimo', '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.warning('marimo not installed — skipping WASM playground build')
        return

    site_dir = config['site_dir']

    for notebook, output_path in NOTEBOOKS.items():
        output_dir = os.path.join(site_dir, output_path)

        log.info(f'Exporting WASM notebook: {notebook} -> {output_dir}')
        result = subprocess.run(
            ['marimo', 'export', 'html-wasm', notebook, '-o', output_dir, '--mode', 'edit'],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warning(f'Failed to export {notebook}: {result.stderr}')
        else:
            log.info(f'WASM notebook exported to {output_dir}')
