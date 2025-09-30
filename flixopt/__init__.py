"""
This module bundles all common functionality of flixopt and sets up the logging
"""

import warnings
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
    change_logging_level,
    linear_converters,
    plotting,
    results,
    solvers,
)

CONFIG.load_config()


# Suppress noisy third-party warnings that users can't fix
warnings.filterwarnings(
    'ignore',
    message='A value is trying to be set on a copy of a DataFrame or Series through chained assignment',
    module='tsam',
)
warnings.filterwarnings('ignore', message='Specify future_stack=True to adopt the new implementation', module='tsam')
warnings.filterwarnings('ignore', message='Coordinates across variables not equal', module='linopy')
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)
warnings.filterwarnings(
    'ignore', message="default value for join will change from join='outer' to join='exact'.", module='linopy'
)
