"""
This module bundles all common functionality of flixopt and sets up the logging
"""

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
    change_logging_level,
    linear_converters,
    plotting,
    results,
    solvers,
)

CONFIG.load_config()
