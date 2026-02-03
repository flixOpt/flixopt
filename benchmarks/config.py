"""Shared configuration for the benchmark suite."""

from __future__ import annotations

from pathlib import Path

# Results directory within the repo
RESULTS_DIR = Path(__file__).parent / 'results'

# Cache directory outside the repo (survives git operations during sweeps)
CACHE_DIR = Path.home() / '.cache' / 'flixopt-benchmarks'

# Default iteration counts
DEFAULT_ITERATIONS = 10
DEFAULT_SWEEP_ITERATIONS = 5
