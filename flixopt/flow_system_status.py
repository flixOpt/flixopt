"""FlowSystem lifecycle status tracking and invalidation.

This module provides explicit status tracking for FlowSystem instances,
replacing implicit flag checks with a clear state machine.

The lifecycle progresses through these statuses:

    INITIALIZED → CONNECTED → MODEL_CREATED → MODEL_BUILT → SOLVED

Each status has specific preconditions and postconditions. Certain operations
(like adding elements) invalidate the status back to an earlier point,
clearing appropriate caches.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flixopt.flow_system import FlowSystem

__all__ = ['FlowSystemStatus', 'get_status', 'invalidate_to_status']


class FlowSystemStatus(IntEnum):
    """Lifecycle status of a FlowSystem.

    Statuses are ordered by progression, allowing comparisons like
    ``status >= FlowSystemStatus.CONNECTED``.

    Attributes:
        INITIALIZED: FlowSystem created, elements can be added.
            No data transformation has occurred yet.
        CONNECTED: Network topology connected, element data transformed
            to xarray DataArrays aligned with model coordinates.
        MODEL_CREATED: linopy Model instantiated (empty shell).
        MODEL_BUILT: Variables and constraints populated in the model.
        SOLVED: Optimization complete, solution exists.
    """

    INITIALIZED = auto()
    CONNECTED = auto()
    MODEL_CREATED = auto()
    MODEL_BUILT = auto()
    SOLVED = auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'FlowSystemStatus.{self.name}'


def get_status(fs: FlowSystem) -> FlowSystemStatus:
    """Derive current status from FlowSystem flags.

    This computes the status from existing internal flags, providing
    backwards compatibility during the transition to explicit status tracking.

    Args:
        fs: The FlowSystem to check.

    Returns:
        The current lifecycle status.
    """
    if fs._solution is not None:
        return FlowSystemStatus.SOLVED
    if fs.model is not None and getattr(fs.model, '_is_built', False):
        return FlowSystemStatus.MODEL_BUILT
    if fs.model is not None:
        return FlowSystemStatus.MODEL_CREATED
    if fs._connected_and_transformed:
        return FlowSystemStatus.CONNECTED
    return FlowSystemStatus.INITIALIZED


def invalidate_to_status(fs: FlowSystem, target: FlowSystemStatus) -> None:
    """Invalidate FlowSystem down to target status, clearing appropriate caches.

    This clears all data/caches associated with statuses above the target.
    If the FlowSystem is already at or below the target status, this is a no-op.

    Args:
        fs: The FlowSystem to invalidate.
        target: The target status to invalidate down to.
    """
    current = get_status(fs)
    if target >= current:
        return  # Already at or below target, nothing to do

    # Clear in reverse order (highest status first)
    if current >= FlowSystemStatus.SOLVED and target < FlowSystemStatus.SOLVED:
        _clear_solved(fs)

    if current >= FlowSystemStatus.MODEL_BUILT and target < FlowSystemStatus.MODEL_BUILT:
        _clear_model_built(fs)

    if current >= FlowSystemStatus.MODEL_CREATED and target < FlowSystemStatus.MODEL_CREATED:
        _clear_model_created(fs)

    if current >= FlowSystemStatus.CONNECTED and target < FlowSystemStatus.CONNECTED:
        _clear_connected(fs)


def _clear_solved(fs: FlowSystem) -> None:
    """Clear artifacts from SOLVED status."""
    fs._solution = None
    fs._statistics = None


def _clear_model_built(fs: FlowSystem) -> None:
    """Clear artifacts from MODEL_BUILT status."""
    # Clear element variable/constraint name registries
    fs._element_variable_names.clear()
    fs._element_constraint_names.clear()
    # Reset the model-built flag so status downgrades to MODEL_CREATED
    if fs.model is not None:
        fs.model._is_built = False


def _clear_model_created(fs: FlowSystem) -> None:
    """Clear artifacts from MODEL_CREATED status."""
    fs.model = None


def _clear_connected(fs: FlowSystem) -> None:
    """Clear artifacts from CONNECTED status."""
    fs._connected_and_transformed = False
    fs._topology = None
    fs._flow_carriers = None
    fs._batched = None
