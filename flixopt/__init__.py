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

# === Runtime warning suppression for third-party libraries ===
# These warnings are from dependencies and cannot be fixed by end users.
# They are suppressed at runtime to provide a cleaner user experience.
# These filters match the test configuration in pyproject.toml for consistency.

# tsam: Time series aggregation library
# - FutureWarning: Upcoming API changes in tsam (will be fixed in future tsam releases)
warnings.filterwarnings('ignore', category=FutureWarning, module='tsam')
# - UserWarning: Informational message about minimal value constraints
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
    message="default value for join will change from join='outer' to join='exact'",
    module='linopy',
)

# numpy: Core numerical library
# - RuntimeWarning: Binary incompatibility warnings from compiled extensions (safe to ignore). numpy 1->2
warnings.filterwarnings('ignore', category=RuntimeWarning, message='numpy\\.ndarray size changed')
