"""
This module contains the solvers of the flixopt framework, making them available to the end user in a compact way.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional

logger = logging.getLogger('flixopt')


@dataclass
class _Solver:
    """
    Abstract base class for solvers.

    Args:
        mip_gap: Solver's mip gap setting. The MIP gap describes the accepted (MILP) objective,
            and the lower bound, which is the theoretically optimal solution (LP)
        time_limit_seconds: Solver's time limit in seconds.
        extra_options: Additional solver options.
    """

    name: ClassVar[str]
    mip_gap: float
    time_limit_seconds: int
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @property
    def options(self) -> Dict[str, Any]:
        """Return a dictionary of solver options."""
        return {key: value for key, value in {**self._options, **self.extra_options}.items() if value is not None}

    @property
    def _options(self) -> Dict[str, Any]:
        """Return a dictionary of solver options, translated to the solver's API."""
        raise NotImplementedError


class GurobiSolver(_Solver):
    """
    Args:
        mip_gap: Solver's mip gap setting. The MIP gap describes the accepted (MILP) objective,
            and the lower bound, which is the theoretically optimal solution (LP)
        time_limit_seconds: Solver's time limit in seconds.
        extra_options: Additional solver options.
    """

    name: ClassVar[str] = 'gurobi'

    @property
    def _options(self) -> Dict[str, Any]:
        return {
            'MIPGap': self.mip_gap,
            'TimeLimit': self.time_limit_seconds,
        }


class HighsSolver(_Solver):
    """
    HiGHS solver configuration.

    Attributes:
        mip_gap: Solver's mip gap setting. The MIP gap describes the accepted (MILP) objective,
            and the lower bound, which is the theoretically optimal solution (LP)
        time_limit_seconds: Solver's time limit in seconds.
        extra_options: Additional solver options.
        threads (Optional[int]): Number of threads to use. Defaults to None.
    """

    threads: Optional[int] = None
    name: ClassVar[str] = 'highs'

    @property
    def _options(self) -> Dict[str, Any]:
        return {
            'mip_rel_gap': self.mip_gap,
            'time_limit': self.time_limit_seconds,
            'threads': self.threads,
        }
