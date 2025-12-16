"""
This module bundles all common functionality of flixopt and sets up the logging
"""

import logging
import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('flixopt')
except (PackageNotFoundError, TypeError):
    # Package is not installed (development mode without editable install)
    __version__ = '0.0.0.dev0'

# Import commonly used classes and functions
from . import linear_converters, plotting, results, solvers
from .carrier import Carrier, CarrierContainer
from .clustering import ClusteringIndices, ClusteringParameters
from .components import (
    LinearConverter,
    Sink,
    Source,
    SourceAndSink,
    Storage,
    Transmission,
)
from .config import CONFIG
from .core import TimeSeriesData
from .effects import PENALTY_EFFECT_LABEL, Effect
from .elements import Bus, Flow
from .flow_system import FlowSystem
from .interface import InvestParameters, Piece, Piecewise, PiecewiseConversion, PiecewiseEffects, StatusParameters
from .optimization import ClusteredOptimization, Optimization, SegmentedOptimization
from .plot_result import PlotResult

__all__ = [
    'TimeSeriesData',
    'CONFIG',
    'Carrier',
    'CarrierContainer',
    'Flow',
    'Bus',
    'Effect',
    'PENALTY_EFFECT_LABEL',
    'Source',
    'Sink',
    'SourceAndSink',
    'Storage',
    'LinearConverter',
    'Transmission',
    'FlowSystem',
    'Optimization',
    'ClusteredOptimization',
    'SegmentedOptimization',
    'InvestParameters',
    'StatusParameters',
    'Piece',
    'Piecewise',
    'PiecewiseConversion',
    'PiecewiseEffects',
    'ClusteringIndices',
    'ClusteringParameters',
    'PlotResult',
    'plotting',
    'results',
    'linear_converters',
    'solvers',
]

# Initialize logger with default configuration (silent: WARNING level, NullHandler)
logger = logging.getLogger('flixopt')
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())

# === Runtime warning suppression for third-party libraries ===
# These warnings are from dependencies and cannot be fixed by end users.
# They are suppressed at runtime to provide a cleaner user experience.
# These filters match the test configuration in pyproject.toml for consistency.

# tsam: Time series aggregation library
# - UserWarning: Informational message about minimal value constraints during clustering.
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='.*minimal value.*exceeds.*',
    module='tsam.timeseriesaggregation',  # More specific if possible
)
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
