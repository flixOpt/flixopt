"""Accessor pattern components for flixopt statistics plotting.

This package provides the infrastructure for PyPSA-style accessor patterns,
enabling clean, chainable API for statistics and visualization.

Architecture:
- Base plotter with common functionality (DRY principle)
- Specialized plotters for domain-specific visualizations
- Automatic plotter selection based on statistic method
"""

from .plotly_charts import (
    InteractivePlotter,
    StorageStatePlotter,
    get_plotter_class,
)
from .plotter import StatisticPlotter
from .wrapper import MethodHandlerWrapper

__all__ = [
    'StatisticPlotter',
    'MethodHandlerWrapper',
    'InteractivePlotter',
    'StorageStatePlotter',
    'get_plotter_class',
]
