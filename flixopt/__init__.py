"""
This module bundles all common functionality of flixopt and sets up the logging
"""

import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('flixopt')
except (PackageNotFoundError, TypeError):
    # Package is not installed (development mode without editable install)
    __version__ = '0.0.0.dev0'

# Import commonly used classes and functions
from . import linear_converters, plotting, results, solvers
from .aggregation import AggregationParameters
from .calculation import AggregatedCalculation, FullCalculation, SegmentedCalculation
from .components import (
    LinearConverter,
    Sink,
    Source,
    SourceAndSink,
    Storage,
    Transmission,
)
from .config import CONFIG, change_logging_level, get_logger
from .core import TimeSeriesData
from .effects import Effect
from .elements import Bus, Flow
from .flow_system import FlowSystem
from .interface import InvestParameters, OnOffParameters, Piece, Piecewise, PiecewiseConversion, PiecewiseEffects

__all__ = [
    'TimeSeriesData',
    'CONFIG',
    'change_logging_level',
    'get_logger',
    'Flow',
    'Bus',
    'Effect',
    'Source',
    'Sink',
    'SourceAndSink',
    'Storage',
    'LinearConverter',
    'Transmission',
    'FlowSystem',
    'FullCalculation',
    'SegmentedCalculation',
    'AggregatedCalculation',
    'InvestParameters',
    'OnOffParameters',
    'Piece',
    'Piecewise',
    'PiecewiseConversion',
    'PiecewiseEffects',
    'AggregationParameters',
    'plotting',
    'results',
    'linear_converters',
    'solvers',
]

# === Runtime warning suppression for third-party libraries ===
# These warnings are from dependencies and cannot be fixed by end users.
# They are suppressed at runtime to provide a cleaner user experience.
# These filters match the test configuration in pyproject.toml for consistency.

# tsam: Time series aggregation library
# - UserWarning: Informational message about minimal value constraints during clustering.
warnings.filterwarnings('ignore', category=UserWarning, message='.*minimal value.*exceeds.*', module='tsam')
# TODO: Might be able to fix it in flixopt?

# linopy: Linear optimization library
# - UserWarning: Coordinate mismatch warnings that don't affect functionality and are expected.
warnings.filterwarnings(
    'ignore', category=UserWarning, message='Coordinates across variables not equal', module='linopy'
)
# - FutureWarning: join parameter default will change in future versions
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message="In a future version of xarray the default value for join will change from join='outer' to join='exact'",
    module='linopy',
)

# numpy: Core numerical library
# - RuntimeWarning: Binary incompatibility warnings from compiled extensions (safe to ignore). numpy 1->2
warnings.filterwarnings('ignore', category=RuntimeWarning, message='numpy\\.ndarray size changed')
