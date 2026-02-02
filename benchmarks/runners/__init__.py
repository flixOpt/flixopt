"""Auto-discovery registry for benchmark runners (phases).

Each runner module must export:
    - run(model_module, size: int, iterations: int) -> dict
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def discover_runners() -> dict[str, ModuleType]:
    """Find all runner modules that have a run() function."""
    runners = {}
    package = importlib.import_module('benchmarks.runners')
    for _importer, name, _ispkg in pkgutil.iter_modules(package.__path__):
        if name.startswith('_'):
            continue
        mod = importlib.import_module(f'benchmarks.runners.{name}')
        if hasattr(mod, 'run'):
            runners[name] = mod
    return runners
