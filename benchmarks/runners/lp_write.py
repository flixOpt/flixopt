"""Runner: time model LP file writing."""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

DEFAULT_ITERATIONS = 5


def run(model_module: ModuleType, size: int, iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Build once, then benchmark repeated to_lp() writes."""
    fs = model_module.build(size)
    fs.build_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        lp_path = Path(tmpdir) / 'model.lp'

        # Warmup
        fs.model.to_file(lp_path, progress=False)

        times = []
        for _ in range(iterations):
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            fs.model.to_file(lp_path, progress=False)
            elapsed = time.perf_counter() - t0
            gc.enable()
            times.append(elapsed)

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
