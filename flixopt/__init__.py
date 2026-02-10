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

# Monkey-patch linopy with disjunctive PWL support (no-op when linopy ships it natively)
from ._linopy_compat import patch_linopy_model as _patch_linopy

_patch_linopy()
del _patch_linopy

# Import commonly used classes and functions
# Register xarray accessors:
# - xr.Dataset.plotly / xr.DataArray.plotly (from xarray_plotly package)
# - xr.Dataset.fxstats (from stats_accessor)
import xarray_plotly as _xpx  # noqa: F401, E402

from . import clustering, linear_converters, plotting, results, solvers  # noqa: E402
from . import stats_accessor as _fxstats  # noqa: F401, E402
from .carrier import Carrier, CarrierContainer  # noqa: E402
from .comparison import Comparison  # noqa: E402
from .components import (  # noqa: E402
    LinearConverter,
    Sink,
    Source,
    SourceAndSink,
    Storage,
    Transmission,
)
from .config import CONFIG  # noqa: E402
from .core import TimeSeriesData  # noqa: E402
from .effects import PENALTY_EFFECT_LABEL, Effect  # noqa: E402
from .elements import Bus, Flow  # noqa: E402
from .flow_system import FlowSystem  # noqa: E402
from .flow_system_status import FlowSystemStatus  # noqa: E402
from .interface import (  # noqa: E402
    InvestParameters,
    Piece,
    Piecewise,
    PiecewiseConversion,
    PiecewiseEffects,
    StatusParameters,
)
from .optimization import Optimization, SegmentedOptimization  # noqa: E402
from .plot_result import PlotResult  # noqa: E402

__all__ = [
    'TimeSeriesData',
    'CONFIG',
    'Carrier',
    'CarrierContainer',
    'Comparison',
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
    'FlowSystemStatus',
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

# Initialize logger with default configuration (silent: WARNING level, NullHandler).
logger = logging.getLogger('flixopt')
logger.setLevel(logging.WARNING)
logger.addHandler(logging.NullHandler())
