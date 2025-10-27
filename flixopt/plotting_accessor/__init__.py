"""Accessor pattern components for flixopt statistics plotting.

This package provides the infrastructure for PyPSA-style accessor patterns,
enabling clean, chainable API for statistics and visualization.
"""

from .plotter import StatisticPlotter
from .wrapper import MethodHandlerWrapper

__all__ = ['StatisticPlotter', 'MethodHandlerWrapper']
