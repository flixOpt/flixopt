"""
This module bundles all common functionality of flixopt and sets up the logging
"""

from importlib.metadata import version

__version__ = version('flixopt')

from .commons import (
    CONFIG,
    AggregatedCalculation,
    AggregationParameters,
    Bus,
    Effect,
    Flow,
    FlowSystem,
    FullCalculation,
    InvestParameters,
    LinearConverter,
    OnOffParameters,
    Piece,
    Piecewise,
    PiecewiseConversion,
    PiecewiseEffects,
    SegmentedCalculation,
    Sink,
    Source,
    SourceAndSink,
    Storage,
    TimeSeriesData,
    Transmission,
    linear_converters,
    plotting,
    results,
    solvers,
)
