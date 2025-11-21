from __future__ import annotations

import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import MappingProxyType
from typing import Literal

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

__all__ = ['CONFIG', 'get_logger', 'change_logging_level']

# Add custom SUCCESS level (between INFO and WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')


def _success(self, message, *args, **kwargs):
    """Log a message with severity 'SUCCESS'."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Add success() method to Logger class
logging.Logger.success = _success


class MultilineFormatter(logging.Formatter):
    """Custom formatter that handles multi-line messages with box-style borders."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default format with time
        if not self._fmt:
            self._fmt = '%(asctime)s %(levelname)-8s │ %(message)s'
            self._style = logging.PercentStyle(self._fmt)

    def format(self, record):
        """Format multi-line messages with box-style borders for better readability."""
        # Split into lines
        lines = record.getMessage().split('\n')

        # Format time
        time_str = self.formatTime(record, '%H:%M:%S')

        # Single line - return standard format
        if len(lines) == 1:
            level_str = f'{record.levelname: <8}'
            return f'{time_str} {level_str} │ {lines[0]}'

        # Multi-line - use box format
        level_str = f'{record.levelname: <8}'
        result = f'{time_str} {level_str} │ ┌─ {lines[0]}'
        indent = ' ' * 8  # 8 spaces for time
        for line in lines[1:-1]:
            result += f'\n{indent} {" " * 8} │ │  {line}'
        result += f'\n{indent} {" " * 8} │ └─ {lines[-1]}'

        return result


if COLORLOG_AVAILABLE:

    class ColoredMultilineFormatter(colorlog.ColoredFormatter):
        """Colored formatter with multi-line message support."""

        def format(self, record):
            """Format multi-line messages with colors and box-style borders."""
            # Split into lines
            lines = record.getMessage().split('\n')

            # Format time (dimmed)
            from colorlog.escape_codes import escape_codes
            dim = escape_codes.get('thin', '')
            reset = escape_codes['reset']
            time_str = self.formatTime(record, '%H:%M:%S')
            time_formatted = f'{dim}{time_str}{reset}'

            # Get the color for this level
            log_colors = self.log_colors
            level_name = record.levelname
            color_name = log_colors.get(level_name, '')
            color = escape_codes.get(color_name, '')

            level_str = f'{level_name: <8}'

            # Single line - return standard colored format
            if len(lines) == 1:
                return f'{time_formatted} {color}{level_str}{reset} │ {lines[0]}'

            # Multi-line - use box format with colors
            result = f'{time_formatted} {color}{level_str}{reset} │ {color}┌─ {lines[0]}{reset}'
            indent = ' ' * 8  # 8 spaces for time
            for line in lines[1:-1]:
                result += f'\n{dim}{indent}{reset} {" " * 8} │ {color}│  {line}{reset}'
            result += f'\n{dim}{indent}{reset} {" " * 8} │ {color}└─ {lines[-1]}{reset}'

            return result


def get_logger(name: str = 'flixopt') -> logging.Logger:
    """Get flixopt logger.

    Args:
        name: Logger name (default: 'flixopt')

    Returns:
        Logger instance that can be configured via standard logging module.

    Examples:
        ```python
        # Get the logger
        logger = get_logger()
        logger.info('Starting optimization')

        # Configure manually with standard logging
        import logging

        logging.basicConfig(level=logging.DEBUG)
        ```
    """
    return logging.getLogger(name)


# SINGLE SOURCE OF TRUTH - immutable to prevent accidental modification
_DEFAULTS = MappingProxyType(
    {
        'config_name': 'flixopt',
        'modeling': MappingProxyType(
            {
                'big': 10_000_000,
                'epsilon': 1e-5,
                'big_binary_bound': 100_000,
            }
        ),
        'plotting': MappingProxyType(
            {
                'default_show': True,
                'default_engine': 'plotly',
                'default_dpi': 300,
                'default_facet_cols': 3,
                'default_sequential_colorscale': 'turbo',
                'default_qualitative_colorscale': 'plotly',
            }
        ),
        'solving': MappingProxyType(
            {
                'mip_gap': 0.01,
                'time_limit_seconds': 300,
                'log_to_console': True,
                'log_main_results': True,
            }
        ),
    }
)


class CONFIG:
    """Configuration for flixopt library.

    Note:
        flixopt uses standard Python logging. Configure it via:
        - Quick helpers: ``CONFIG.Logging.enable_console('INFO')``
        - Standard logging: ``import logging; logging.basicConfig(level=logging.DEBUG)``

    Attributes:
        Logging: Logging configuration helpers.
        Modeling: Optimization modeling parameters.
        Solving: Solver configuration and default parameters.
        Plotting: Plotting configuration.
        config_name: Configuration name.

    Examples:
        Quick colored console output:

        ```python
        CONFIG.Logging.enable_console('DEBUG')
        ```

        File logging with rotation:

        ```python
        CONFIG.Logging.enable_file('INFO', 'app.log')
        ```

        Both console and file:

        ```python
        CONFIG.Logging.enable_console('INFO')
        CONFIG.Logging.enable_file('DEBUG', 'debug.log')
        ```

        Full control (advanced):

        ```python
        import logging

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('app.log'), logging.StreamHandler()],
        )
        ```
    """

    class Logging:
        """Minimal logging helpers.

        For advanced configuration, use Python's logging module directly.
        flixopt is silent by default (WARNING level, no handlers).

        Examples:
            Quick colored console output:

                CONFIG.Logging.enable_console('INFO')

            File logging with rotation:

                CONFIG.Logging.enable_file('DEBUG', 'app.log',
                                          max_bytes=10*1024*1024, backup_count=3)

            Both console and file:

                CONFIG.Logging.enable_console('INFO')
                CONFIG.Logging.enable_file('DEBUG', 'debug.log')

            Disable colors:

                CONFIG.Logging.enable_console('INFO', colored=False)

            Full control (advanced):

                import logging
                logging.getLogger('flixopt').addHandler(your_custom_handler)
        """

        @classmethod
        def enable_console(cls, level: str | int = 'INFO', colored: bool = True, stream=None) -> None:
            """Enable colored console logging.

            Args:
                level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or logging constant)
                colored: Use colored output if colorlog is available (default: True)
                stream: Output stream (default: sys.stdout). Can be sys.stdout or sys.stderr.

            Note:
                For full control over formatting, use logging.basicConfig() instead.

            Examples:
                ```python
                # Colored output to stdout (default)
                CONFIG.Logging.enable_console('INFO')

                # Plain text output
                CONFIG.Logging.enable_console('INFO', colored=False)

                # Log to stderr instead
                import sys
                CONFIG.Logging.enable_console('INFO', stream=sys.stderr)

                # Using logging constants
                import logging

                CONFIG.Logging.enable_console(logging.DEBUG)
                ```
            """
            import sys

            logger = logging.getLogger('flixopt')

            # Convert string level to logging constant
            if isinstance(level, str):
                level = getattr(logging, level.upper())

            logger.setLevel(level)

            # Default to stdout
            if stream is None:
                stream = sys.stdout

            # Remove existing console handlers to avoid duplicates
            logger.handlers = [
                h
                for h in logger.handlers
                if not isinstance(h, logging.StreamHandler) or isinstance(h, RotatingFileHandler)
            ]

            if colored and COLORLOG_AVAILABLE:
                handler = colorlog.StreamHandler(stream)
                handler.setFormatter(
                    ColoredMultilineFormatter(
                        '%(log_color)s%(levelname)-8s%(reset)s %(message)s',
                        log_colors={
                            'DEBUG': 'cyan',
                            'INFO': 'white',
                            'SUCCESS': 'green',
                            'WARNING': 'yellow',
                            'ERROR': 'red',
                            'CRITICAL': 'bold_red',
                        },
                    )
                )
            else:
                handler = logging.StreamHandler(stream)
                handler.setFormatter(MultilineFormatter('%(levelname)-8s %(message)s'))

            logger.addHandler(handler)
            logger.propagate = False  # Don't propagate to root

        @classmethod
        def enable_file(
            cls,
            level: str | int = 'INFO',
            path: str | Path = 'flixopt.log',
            max_bytes: int = 10 * 1024 * 1024,
            backup_count: int = 5,
        ) -> None:
            """Enable file logging with rotation.

            Args:
                level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or logging constant)
                path: Path to log file (default: 'flixopt.log')
                max_bytes: Maximum file size before rotation in bytes (default: 10MB)
                backup_count: Number of backup files to keep (default: 5)

            Note:
                For full control over formatting and handlers, use logging module directly.

            Examples:
                ```python
                # Basic file logging
                CONFIG.Logging.enable_file('INFO', 'app.log')

                # With custom rotation
                CONFIG.Logging.enable_file('DEBUG', 'debug.log', max_bytes=50 * 1024 * 1024, backup_count=10)
                ```
            """
            logger = logging.getLogger('flixopt')

            # Convert string level to logging constant
            if isinstance(level, str):
                level = getattr(logging, level.upper())

            logger.setLevel(level)

            # Remove existing file handlers to avoid duplicates
            logger.handlers = [
                h
                for h in logger.handlers
                if not isinstance(h, (logging.FileHandler, RotatingFileHandler))
                or isinstance(h, logging.StreamHandler)
                and not isinstance(h, RotatingFileHandler)
            ]

            # Create log directory if needed
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

            logger.addHandler(handler)
            logger.propagate = False  # Don't propagate to root

        @classmethod
        def disable(cls) -> None:
            """Disable all flixopt logging.

            Examples:
                ```python
                CONFIG.Logging.disable()
                ```
            """
            logger = logging.getLogger('flixopt')
            logger.handlers.clear()
            logger.setLevel(logging.CRITICAL)

    class Modeling:
        """Optimization modeling parameters.

        Attributes:
            big: Large number for big-M constraints.
            epsilon: Tolerance for numerical comparisons.
            big_binary_bound: Upper bound for binary constraints.
        """

        big: int = _DEFAULTS['modeling']['big']
        epsilon: float = _DEFAULTS['modeling']['epsilon']
        big_binary_bound: int = _DEFAULTS['modeling']['big_binary_bound']

    class Solving:
        """Solver configuration and default parameters.

        Attributes:
            mip_gap: Default MIP gap tolerance for solver convergence.
            time_limit_seconds: Default time limit in seconds for solver runs.
            log_to_console: Whether solver should output to console.
            log_main_results: Whether to log main results after solving.

        Examples:
            ```python
            # Set tighter convergence and longer timeout
            CONFIG.Solving.mip_gap = 0.001
            CONFIG.Solving.time_limit_seconds = 600
            CONFIG.Solving.log_to_console = False
            ```
        """

        mip_gap: float = _DEFAULTS['solving']['mip_gap']
        time_limit_seconds: int = _DEFAULTS['solving']['time_limit_seconds']
        log_to_console: bool = _DEFAULTS['solving']['log_to_console']
        log_main_results: bool = _DEFAULTS['solving']['log_main_results']

    class Plotting:
        """Plotting configuration.

        Configure backends via environment variables:
        - Matplotlib: Set `MPLBACKEND` environment variable (e.g., 'Agg', 'TkAgg')
        - Plotly: Set `PLOTLY_RENDERER` or use `plotly.io.renderers.default`

        Attributes:
            default_show: Default value for the `show` parameter in plot methods.
            default_engine: Default plotting engine.
            default_dpi: Default DPI for saved plots.
            default_facet_cols: Default number of columns for faceted plots.
            default_sequential_colorscale: Default colorscale for heatmaps and continuous data.
            default_qualitative_colorscale: Default colormap for categorical plots (bar/line/area charts).

        Examples:
            ```python
            # Configure default export and color settings
            CONFIG.Plotting.default_dpi = 600
            CONFIG.Plotting.default_sequential_colorscale = 'plasma'
            CONFIG.Plotting.default_qualitative_colorscale = 'Dark24'
            ```
        """

        default_show: bool = _DEFAULTS['plotting']['default_show']
        default_engine: Literal['plotly', 'matplotlib'] = _DEFAULTS['plotting']['default_engine']
        default_dpi: int = _DEFAULTS['plotting']['default_dpi']
        default_facet_cols: int = _DEFAULTS['plotting']['default_facet_cols']
        default_sequential_colorscale: str = _DEFAULTS['plotting']['default_sequential_colorscale']
        default_qualitative_colorscale: str = _DEFAULTS['plotting']['default_qualitative_colorscale']

    config_name: str = _DEFAULTS['config_name']

    @classmethod
    def reset(cls) -> None:
        """Reset all configuration values to defaults."""
        for key, value in _DEFAULTS['modeling'].items():
            setattr(cls.Modeling, key, value)

        for key, value in _DEFAULTS['solving'].items():
            setattr(cls.Solving, key, value)

        for key, value in _DEFAULTS['plotting'].items():
            setattr(cls.Plotting, key, value)

        cls.config_name = _DEFAULTS['config_name']

    @classmethod
    def to_dict(cls) -> dict:
        """Convert the configuration class into a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the current configuration.
        """
        return {
            'config_name': cls.config_name,
            'modeling': {
                'big': cls.Modeling.big,
                'epsilon': cls.Modeling.epsilon,
                'big_binary_bound': cls.Modeling.big_binary_bound,
            },
            'solving': {
                'mip_gap': cls.Solving.mip_gap,
                'time_limit_seconds': cls.Solving.time_limit_seconds,
                'log_to_console': cls.Solving.log_to_console,
                'log_main_results': cls.Solving.log_main_results,
            },
            'plotting': {
                'default_show': cls.Plotting.default_show,
                'default_engine': cls.Plotting.default_engine,
                'default_dpi': cls.Plotting.default_dpi,
                'default_facet_cols': cls.Plotting.default_facet_cols,
                'default_sequential_colorscale': cls.Plotting.default_sequential_colorscale,
                'default_qualitative_colorscale': cls.Plotting.default_qualitative_colorscale,
            },
        }

    @classmethod
    def silent(cls) -> type[CONFIG]:
        """Configure for silent operation.

        Disables all logging, solver output, and result logging
        for clean production runs. Does not show plots.
        """
        cls.Logging.disable()
        cls.Plotting.default_show = False
        cls.Solving.log_to_console = False
        cls.Solving.log_main_results = False
        return cls

    @classmethod
    def debug(cls) -> type[CONFIG]:
        """Configure for debug mode with verbose output.

        Enables console logging at DEBUG level and all solver output for troubleshooting.
        """
        cls.Logging.enable_console('DEBUG')
        cls.Solving.log_to_console = True
        cls.Solving.log_main_results = True
        return cls

    @classmethod
    def exploring(cls) -> type[CONFIG]:
        """Configure for exploring flixopt.

        Enables console logging at INFO level and all solver output.
        Also enables browser plotting for plotly with showing plots per default.
        """
        cls.Logging.enable_console('INFO')
        cls.Solving.log_to_console = True
        cls.Solving.log_main_results = True
        cls.browser_plotting()
        return cls

    @classmethod
    def browser_plotting(cls) -> type[CONFIG]:
        """Configure for interactive usage with plotly to open plots in browser.

        Sets plotly.io.renderers.default = 'browser'. Useful for running examples
        and viewing interactive plots. Does NOT modify CONFIG.Plotting settings.

        Respects FLIXOPT_CI environment variable if set.
        """
        cls.Plotting.default_show = True

        # Only set to True if environment variable hasn't overridden it
        if 'FLIXOPT_CI' not in os.environ:
            import plotly.io as pio

            pio.renderers.default = 'browser'

        return cls


def change_logging_level(level_name: str | int) -> None:
    """Change the logging level for the flixopt logger.

    .. deprecated:: 5.0.0
        Use ``CONFIG.Logging.enable_console(level)`` instead.
        This function will be removed in version 6.0.0.

    Args:
        level_name: The logging level to set (DEBUG, INFO, WARNING, ERROR, CRITICAL or logging constant).

    Examples:
        >>> change_logging_level('DEBUG')  # deprecated
        >>> # Use this instead:
        >>> CONFIG.Logging.enable_console('DEBUG')
    """
    warnings.warn(
        'change_logging_level is deprecated and will be removed in version 6.0.0. '
        'Use CONFIG.Logging.enable_console(level) instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    CONFIG.Logging.enable_console(level_name)


# Initialize logger with default configuration (silent: WARNING level, no handlers)
_flixopt_logger = logging.getLogger('flixopt')
_flixopt_logger.setLevel(logging.WARNING)
