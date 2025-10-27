"""Method handler wrapper for accessor pattern.

This module provides the core decorator that enables the accessor pattern
by intercepting method returns and wrapping them in handler objects.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class MethodHandlerWrapper:
    """Decorator that wraps a method's return value with a handler class.

    This decorator is the key component of the accessor pattern. It intercepts
    the return value of statistics methods and wraps them in a handler object
    (typically StatisticPlotter) that provides additional functionality like
    plotting capabilities.

    The wrapper creates a bound method that captures the original method call
    and its arguments, but delays execution until the data is actually needed.
    This enables lazy evaluation and clean API chaining.

    Args:
        handler_class: Class to wrap the method with. Must accept (bound_method, parent_object, method_name)
            as constructor arguments.

    Examples:
        >>> @MethodHandlerWrapper(handler_class=StatisticPlotter)
        ... def energy_balance(self):
        ...     return calculate_energy_balance()
        >>> # Returns StatisticPlotter, not raw data
        >>> result = model.statistics.energy_balance()
        >>> # Access data by calling
        >>> data = result()
        >>> # Or plot directly
        >>> fig = result.plot.bar()
    """

    def __init__(self, handler_class: type):
        """Initialize the wrapper with a handler class.

        Args:
            handler_class: The class that will wrap method returns. Must support the interface
                expected by the accessor pattern (accepting bound_method, parent_object,
                and method_name in its constructor).
        """
        self.handler_class = handler_class

    def __call__(self, method: Callable) -> Callable:
        """Wrap the method to return handler instance instead of raw result.

        Args:
            method: The method to wrap. This is typically a statistics calculation method
                that returns an xarray.Dataset.

        Returns:
            Wrapped method that returns handler_class instance instead of raw result.
        """

        @wraps(method)
        def wrapper(accessor_self, *args, **kwargs):
            # Create a bound method that captures the call but delays execution
            # This enables lazy evaluation - the statistics are only calculated
            # when actually needed (e.g., when plotting or explicitly called)
            def bound_method():
                return method(accessor_self, *args, **kwargs)

            # Return handler instance with the bound method and parent reference
            # The handler (e.g., StatisticPlotter) will provide the .plot interface
            return self.handler_class(
                bound_method=bound_method, parent_object=accessor_self._parent, method_name=method.__name__
            )

        return wrapper
