"""Auto-discovery registry for benchmark models.

Each model module must export:
    - build(size: int) -> fx.FlowSystem
    - SIZES: list[int]
    - QUICK_SIZES: list[int]
    - LABEL: str
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def discover_models() -> dict[str, ModuleType]:
    """Find all model modules that have build() and SIZES."""
    models = {}
    package = importlib.import_module('benchmarks.models')
    for _importer, name, _ispkg in pkgutil.iter_modules(package.__path__):
        if name.startswith('_'):
            continue
        mod = importlib.import_module(f'benchmarks.models.{name}')
        if hasattr(mod, 'build') and hasattr(mod, 'SIZES'):
            models[name] = mod
    return models
