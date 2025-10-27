"""Statistics accessor module for flixopt CalculationResults and node results.

This package provides PyPSA-style accessor patterns for calculating and
visualizing statistics from optimization results at both system-level
(CalculationResults) and node-level (ComponentResults/BusResults).
"""

from .accessor import StatisticsAccessor
from .node_statistics import NodeStatisticsAccessor

__all__ = ['StatisticsAccessor', 'NodeStatisticsAccessor']
