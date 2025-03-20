"""
This module bundles all common functionality of flixOpt and sets up the logging
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
    Medium,
    MediumCategories,
    OnOffParameters,
    SegmentedCalculation,
    Sink,
    Source,
    SourceAndSink,
    Storage,
    TimeSeriesData,
    Transmission,
    change_logging_level,
    create_datetime_array,
    linear_converters,
    math_modeling,
    plotting,
    results,
    solvers,
)

CONFIG.load_config()
