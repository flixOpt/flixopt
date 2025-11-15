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

# Type system for dimension-aware type hints
from .types import Numeric_TPS, Numeric_PS, Numeric_S, Bool_TPS, Bool_PS, Bool_S, Effect_TPS, Effect_PS, Effect_S, Scalar

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
