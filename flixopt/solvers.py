"""
This module contains the solvers of the flixopt framework, making them available to the end user in a compact way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from flixopt.config import CONFIG

logger = logging.getLogger('flixopt')


@dataclass
class _Solver:
    """
    Abstract base class for solvers.

    Args:
        mip_gap: Acceptable relative optimality gap in [0.0, 1.0]. Defaults to CONFIG.Solving.mip_gap.
        time_limit_seconds: Time limit in seconds. Defaults to CONFIG.Solving.time_limit_seconds.
        log_to_console: If False, no output to console. Defaults to CONFIG.Solving.log_to_console.
        extra_options: Additional solver options merged into `options`.
    """

    name: ClassVar[str]
    mip_gap: float = field(default_factory=lambda: CONFIG.Solving.mip_gap)
    time_limit_seconds: int = field(default_factory=lambda: CONFIG.Solving.time_limit_seconds)
    log_to_console: bool = field(default_factory=lambda: CONFIG.Solving.log_to_console)
    extra_options: dict[str, Any] = field(default_factory=dict)

    @property
    def options(self) -> dict[str, Any]:
        """Return a dictionary of solver options."""
        return {key: value for key, value in {**self._options, **self.extra_options}.items() if value is not None}

    @property
    def _options(self) -> dict[str, Any]:
        """Return a dictionary of solver options, translated to the solver's API."""
        raise NotImplementedError


class GurobiSolver(_Solver):
    """
    Gurobi solver configuration.

    Args:
        mip_gap: Acceptable relative optimality gap in [0.0, 1.0]; mapped to Gurobi `MIPGap`.
        time_limit_seconds: Time limit in seconds; mapped to Gurobi `TimeLimit`.
        log_to_console: If False, no output to console.
        extra_options: Additional solver options merged into `options`.
    """

    name: ClassVar[str] = 'gurobi'

    @property
    def _options(self) -> dict[str, Any]:
        return {
            'MIPGap': self.mip_gap,
            'TimeLimit': self.time_limit_seconds,
            'LogToConsole': 1 if self.log_to_console else 0,
        }


class HighsSolver(_Solver):
    """
    HiGHS solver configuration.

    Attributes:
        mip_gap: Acceptable relative optimality gap in [0.0, 1.0]; mapped to HiGHS `mip_rel_gap`.
        time_limit_seconds: Time limit in seconds; mapped to HiGHS `time_limit`.
        log_to_console: If False, no output to console.
        extra_options: Additional solver options merged into `options`.
        threads (int | None): Number of threads to use. If None, HiGHS chooses.
    """

    threads: int | None = None
    name: ClassVar[str] = 'highs'

    @property
    def _options(self) -> dict[str, Any]:
        return {
            'mip_rel_gap': self.mip_gap,
            'time_limit': self.time_limit_seconds,
            'threads': self.threads,
            'log_to_console': self.log_to_console,
        }
