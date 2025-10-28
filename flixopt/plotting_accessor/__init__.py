"""Accessor pattern components for flixopt statistics and results plotting.

This package provides the infrastructure for PyPSA-style accessor patterns,
enabling clean, chainable API for statistics and visualization.

Architecture:
- Base plotters with common functionality (DRY principle)
- Specialized plotters for domain-specific visualizations
- Automatic plotter selection based on statistic method
- Results accessors for ComponentResults, BusResults, and CalculationResults
- Data transformation utilities for xarray → DataFrame conversions
"""

from .data_transformer import DataTransformer
from .plotly_charts import (
    InteractivePlotter,
    StorageStatePlotter,
    get_plotter_class,
)
from .plotter import StatisticPlotter
from .results_accessor import (
    BusPlotAccessor,
    CalculationResultsPlotAccessor,
    ComponentPlotAccessor,
    SegmentedCalculationResultsPlotAccessor,
)
from .results_plotters import (
    ChargeStatePlotter,
    HeatmapPlotter,
    NodeBalancePlotter,
    PieChartPlotter,
    ResultsPlotterBase,
)
from .wrapper import MethodHandlerWrapper

__all__ = [
    'StatisticPlotter',
    'MethodHandlerWrapper',
    'DataTransformer',
    'InteractivePlotter',
    'StorageStatePlotter',
    'get_plotter_class',
    # Results plotters
    'ResultsPlotterBase',
    'NodeBalancePlotter',
    'PieChartPlotter',
    'ChargeStatePlotter',
    'HeatmapPlotter',
    # Results accessors
    'ComponentPlotAccessor',
    'BusPlotAccessor',
    'CalculationResultsPlotAccessor',
    'SegmentedCalculationResultsPlotAccessor',
]
