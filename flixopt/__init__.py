"""
This module bundles all common functionality of flixopt and sets up the logging
"""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('flixopt')
except (PackageNotFoundError, TypeError):
    # Package is not installed (development mode without editable install)
    __version__ = '0.0.0.dev0'

# Import commonly used classes and functions
from . import clustering, linear_converters, plotting, results, solvers

# Register xr.Dataset.fxplot accessor (import triggers registration via decorator)
from . import dataset_plot_accessor as _  # noqa: F401
from .carrier import Carrier, CarrierContainer
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
from .optimization import Optimization, SegmentedOptimization
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
    'SegmentedOptimization',
    'InvestParameters',
    'StatusParameters',
    'Piece',
    'Piecewise',
    'PiecewiseConversion',
    'PiecewiseEffects',
    'PlotResult',
    'clustering',
    'plotting',
    'results',
    'linear_converters',
    'solvers',
]

# Initialize logger with default configuration (silent: WARNING level, NullHandler)
logger = logging.getLogger('flixopt')
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())
