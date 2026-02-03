"""Runner: time fs.build_model()."""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

DEFAULT_ITERATIONS = 10


def run(model_module: ModuleType, size: int, iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Benchmark build_model() for *model_module* at *size* timesteps."""
    # Warmup
    fs = model_module.build(size)
    fs.build_model()
    del fs
    gc.collect()

    times = []
    for _ in range(iterations):
        fs = model_module.build(size)
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        fs.build_model()
        elapsed = time.perf_counter() - t0
        gc.enable()
        times.append(elapsed)
        del fs

    arr = np.array(times)
    return {
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'times': times,
    }
