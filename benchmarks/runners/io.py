"""Runner: time to_dataset() + from_dataset() round-trip."""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

import flixopt as fx

DEFAULT_ITERATIONS = 5


def run(model_module: ModuleType, size: int, iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Benchmark to_dataset() and from_dataset() for *model_module* at *size* timesteps."""
    fs = model_module.build(size)

    # Warmup
    ds = fs.to_dataset()
    fx.FlowSystem.from_dataset(ds)

    times_to = []
    times_from = []
    for _ in range(iterations):
        gc.collect()
        gc.disable()

        t0 = time.perf_counter()
        ds = fs.to_dataset()
        times_to.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fx.FlowSystem.from_dataset(ds)
        times_from.append(time.perf_counter() - t0)

        gc.enable()

    to_arr = np.array(times_to)
    from_arr = np.array(times_from)
    total_arr = to_arr + from_arr

    return {
        'median': float(np.median(total_arr)),
        'q25': float(np.percentile(total_arr, 25)),
        'q75': float(np.percentile(total_arr, 75)),
        'times': total_arr.tolist(),
        'detail': {
            'to_dataset': {
                'median': float(np.median(to_arr)),
                'q25': float(np.percentile(to_arr, 25)),
                'q75': float(np.percentile(to_arr, 75)),
            },
            'from_dataset': {
                'median': float(np.median(from_arr)),
                'q25': float(np.percentile(from_arr, 25)),
                'q75': float(np.percentile(from_arr, 75)),
            },
        },
    }
