"""
Optimization accessor for FlowSystem.

This module provides the OptimizeAccessor class that enables the
`flow_system.optimize(...)` pattern with extensible optimization methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .flow_system import FlowSystem
    from .solvers import _Solver


class OptimizeAccessor:
    """
    Accessor for optimization methods on FlowSystem.

    This class provides the optimization API for FlowSystem, accessible via
    `flow_system.optimize`. It supports both direct calling (standard optimization)
    and method access for specialized optimization modes.

    Examples:
        Standard optimization (via __call__):

        >>> flow_system.optimize(solver)
        >>> print(flow_system.solution)

        Future specialized modes:

        >>> flow_system.optimize.clustered(solver, aggregation=params)
        >>> flow_system.optimize.mga(solver, alternatives=5)
    """

    def __init__(self, flow_system: FlowSystem) -> None:
        """
        Initialize the accessor with a reference to the FlowSystem.

        Args:
            flow_system: The FlowSystem to optimize.
        """
        self._fs = flow_system

    def __call__(self, solver: _Solver, normalize_weights: bool = True) -> FlowSystem:
        """
        Build and solve the optimization model in one step.

        This is a convenience method that combines `build_model()` and `solve()`.
        Use this for simple optimization workflows. For more control (e.g., inspecting
        the model before solving, or adding custom constraints), use `build_model()`
        and `solve()` separately.

        Args:
            solver: The solver to use (e.g., HighsSolver, GurobiSolver).
            normalize_weights: Whether to normalize scenario/period weights to sum to 1.

        Returns:
            The FlowSystem, for method chaining.

        Examples:
            Simple optimization:

            >>> flow_system.optimize(HighsSolver())
            >>> print(flow_system.solution['Boiler(Q_th)|flow_rate'])

            Access element solutions directly:

            >>> flow_system.optimize(solver)
            >>> boiler = flow_system.components['Boiler']
            >>> print(boiler.solution)

            Method chaining:

            >>> solution = flow_system.optimize(solver).solution
        """
        self._fs.build_model(normalize_weights)
        self._fs.solve(solver)
        return self._fs

    # Future methods can be added here:
    #
    # def clustered(self, solver: _Solver, aggregation: AggregationParameters,
    #               normalize_weights: bool = True) -> FlowSystem:
    #     """Clustered optimization with time aggregation."""
    #     ...
    #
    # def mga(self, solver: _Solver, alternatives: int = 5) -> FlowSystem:
    #     """Modeling to Generate Alternatives."""
    #     ...
