"""Runner: measure peak memory during build_model()."""

from __future__ import annotations

import gc
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

DEFAULT_ITERATIONS = 3


def run(model_module: ModuleType, size: int, iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Measure peak memory of build_model() for *model_module* at *size* timesteps."""
    # Warmup
    fs = model_module.build(size)
    fs.build_model()
    del fs
    gc.collect()

    peaks = []
    for _ in range(iterations):
        fs = model_module.build(size)
        gc.collect()

        tracemalloc.start()
        fs.build_model()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peaks.append(peak)
        del fs
        gc.collect()

    arr = np.array(peaks) / 1e6  # Convert to MB
    return {
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'unit': 'MB',
        'times': arr.tolist(),
    }
