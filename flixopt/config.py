from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from loguru import logger

__all__ = ['CONFIG', 'change_logging_level']


# SINGLE SOURCE OF TRUTH - immutable to prevent accidental modification
_DEFAULTS = MappingProxyType(
    {
        'config_name': 'flixopt',
        'logging': MappingProxyType(
            {
                'level': 'INFO',
                'file': None,
                'console': False,
                'max_file_size': 10_485_760,  # 10MB
                'backup_count': 5,
                'verbose_tracebacks': False,
            }
        ),
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

    Always call ``CONFIG.apply()`` after changes.

    Note:
        flixopt uses `loguru <https://loguru.readthedocs.io/>`_ for logging.

    Attributes:
        Logging: Logging configuration.
        Modeling: Optimization modeling parameters.
        Solving: Solver configuration and default parameters.
        Plotting: Plotting configuration.
        config_name: Configuration name.

    Examples:
        ```python
        CONFIG.Logging.console = True
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.apply()
        ```

        Load from YAML file:

        ```yaml
        logging:
          level: DEBUG
          console: true
          file: app.log
        solving:
          mip_gap: 0.001
          time_limit_seconds: 600
        ```
    """

    class Logging:
        """Logging configuration.

        Silent by default. Enable via ``console=True`` or ``file='path'``.

        Attributes:
            level: Logging level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL).
            file: Log file path for file logging (None to disable).
            console: Enable console output (True/'stdout' or 'stderr').
            max_file_size: Max file size in bytes before rotation.
            backup_count: Number of backup files to keep.
            verbose_tracebacks: Show detailed tracebacks with variable values.

        Examples:
            ```python
            # Enable console logging
            CONFIG.Logging.console = True
            CONFIG.Logging.level = 'DEBUG'
            CONFIG.apply()

            # File logging with rotation
            CONFIG.Logging.file = 'app.log'
            CONFIG.Logging.max_file_size = 5_242_880  # 5MB
            CONFIG.apply()

            # Console to stderr
            CONFIG.Logging.console = 'stderr'
            CONFIG.apply()
            ```

        Note:
            For advanced formatting or custom loguru configuration,
            use loguru's API directly after calling CONFIG.apply():

            ```python
            from loguru import logger

            CONFIG.apply()  # Basic setup
            logger.add('custom.log', format='{time} {message}')
            ```
        """

        level: Literal['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = _DEFAULTS['logging']['level']
        file: str | None = _DEFAULTS['logging']['file']
        console: bool | Literal['stdout', 'stderr'] = _DEFAULTS['logging']['console']
        max_file_size: int = _DEFAULTS['logging']['max_file_size']
        backup_count: int = _DEFAULTS['logging']['backup_count']
        verbose_tracebacks: bool = _DEFAULTS['logging']['verbose_tracebacks']

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
            CONFIG.apply()
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
            # Set consistent theming
            CONFIG.Plotting.plotly_template = 'plotly_dark'
            CONFIG.apply()

            # Configure default export and color settings
            CONFIG.Plotting.default_dpi = 600
            CONFIG.Plotting.default_sequential_colorscale = 'plasma'
            CONFIG.Plotting.default_qualitative_colorscale = 'Dark24'
            CONFIG.apply()
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
    def reset(cls):
        """Reset all configuration values to defaults."""
        for key, value in _DEFAULTS['logging'].items():
            setattr(cls.Logging, key, value)

        for key, value in _DEFAULTS['modeling'].items():
            setattr(cls.Modeling, key, value)

        for key, value in _DEFAULTS['solving'].items():
            setattr(cls.Solving, key, value)

        for key, value in _DEFAULTS['plotting'].items():
            setattr(cls.Plotting, key, value)

        cls.config_name = _DEFAULTS['config_name']
        cls.apply()

    @classmethod
    def apply(cls):
        """Apply current configuration to logging system."""
        valid_levels = ['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.Logging.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level '{cls.Logging.level}'. Must be one of: {', '.join(valid_levels)}")

        if cls.Logging.max_file_size <= 0:
            raise ValueError('max_file_size must be positive')

        if cls.Logging.backup_count < 0:
            raise ValueError('backup_count must be non-negative')

        if cls.Logging.console not in (False, True, 'stdout', 'stderr'):
            raise ValueError(f"console must be False, True, 'stdout', or 'stderr', got {cls.Logging.console}")

        _setup_logging(
            default_level=cls.Logging.level,
            log_file=cls.Logging.file,
            console=cls.Logging.console,
            max_file_size=cls.Logging.max_file_size,
            backup_count=cls.Logging.backup_count,
            verbose_tracebacks=cls.Logging.verbose_tracebacks,
        )

    @classmethod
    def load_from_file(cls, config_file: str | Path):
        """Load configuration from YAML file and apply it.

        Args:
            config_file: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        # Import here to avoid circular import
        from . import io as fx_io

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_file}')

        config_dict = fx_io.load_yaml(config_path)
        cls._apply_config_dict(config_dict)

        cls.apply()

    @classmethod
    def _apply_config_dict(cls, config_dict: dict):
        """Apply configuration dictionary to class attributes.

        Args:
            config_dict: Dictionary containing configuration values.
        """
        for key, value in config_dict.items():
            if key == 'logging' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if hasattr(cls.Logging, nested_key):
                        setattr(cls.Logging, nested_key, nested_value)
            elif key == 'modeling' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    setattr(cls.Modeling, nested_key, nested_value)
            elif key == 'solving' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    setattr(cls.Solving, nested_key, nested_value)
            elif key == 'plotting' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    setattr(cls.Plotting, nested_key, nested_value)
            elif hasattr(cls, key):
                setattr(cls, key, value)

    @classmethod
    def to_dict(cls) -> dict:
        """Convert the configuration class into a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the current configuration.
        """
        return {
            'config_name': cls.config_name,
            'logging': {
                'level': cls.Logging.level,
                'file': cls.Logging.file,
                'console': cls.Logging.console,
                'max_file_size': cls.Logging.max_file_size,
                'backup_count': cls.Logging.backup_count,
                'verbose_tracebacks': cls.Logging.verbose_tracebacks,
            },
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

        Disables console logging, solver output, and result logging
        for clean production runs. Does not show plots. Automatically calls apply().
        """
        cls.Logging.console = False
        cls.Plotting.default_show = False
        cls.Logging.file = None
        cls.Solving.log_to_console = False
        cls.Solving.log_main_results = False
        cls.apply()
        return cls

    @classmethod
    def debug(cls) -> type[CONFIG]:
        """Configure for debug mode with verbose output.

        Enables console logging at DEBUG level, verbose tracebacks,
        and all solver output for troubleshooting. Automatically calls apply().
        """
        cls.Logging.console = True
        cls.Logging.level = 'DEBUG'
        cls.Logging.verbose_tracebacks = True
        cls.Solving.log_to_console = True
        cls.Solving.log_main_results = True
        cls.apply()
        return cls

    @classmethod
    def exploring(cls) -> type[CONFIG]:
        """Configure for exploring flixopt

        Enables console logging at INFO level and all solver output.
        Also enables browser plotting for plotly with showing plots per default
        """
        cls.Logging.console = True
        cls.Logging.level = 'INFO'
        cls.Solving.log_to_console = True
        cls.Solving.log_main_results = True
        cls.browser_plotting()
        cls.apply()
        return cls

    @classmethod
    def browser_plotting(cls) -> type[CONFIG]:
        """Configure for interactive usage with plotly to open plots in browser.

        Sets plotly.io.renderers.default = 'browser'. Useful for running examples
        and viewing interactive plots. Does NOT modify CONFIG.Plotting settings.

        Respects FLIXOPT_CI environment variable if set.
        """
        cls.Plotting.default_show = True
        cls.apply()

        # Only set to True if environment variable hasn't overridden it
        if 'FLIXOPT_CI' not in os.environ:
            import plotly.io as pio

            pio.renderers.default = 'browser'

        return cls


def _format_multiline(record):
    """Format multi-line messages with box-style borders for better readability.

    Single-line messages use standard format.
    Multi-line messages use boxed format with ┌─, │, └─ characters.

    Note: Escapes curly braces in messages to prevent format string errors.
    """
    # Escape curly braces in message to prevent format string errors
    message = record['message'].replace('{', '{{').replace('}', '}}')
    lines = message.split('\n')

    # Format timestamp and level
    time_str = record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # milliseconds
    level_str = f'{record["level"].name: <8}'

    # Single line messages - standard format
    if len(lines) == 1:
        result = f'<dim>{time_str}</dim> | <level>{level_str}</level> | <level>{message}</level>\n'
        if record['exception']:
            result += '{exception}'
        return result

    # Multi-line messages - boxed format
    indent = ' ' * len(time_str)  # Match timestamp length

    # Build the boxed output
    result = f'<dim>{time_str}</dim> | <level>{level_str}</level> | <level>┌─ {lines[0]}</level>\n'
    for line in lines[1:-1]:
        result += f'<dim>{indent}</dim> | <level>{" " * 8}</level> | <level>│  {line}</level>\n'
    result += f'<dim>{indent}</dim> | <level>{" " * 8}</level> | <level>└─ {lines[-1]}</level>\n'

    # Add exception info if present
    if record['exception']:
        result += '\n{exception}'

    return result


def _setup_logging(
    default_level: Literal['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    log_file: str | None = None,
    console: bool | Literal['stdout', 'stderr'] = False,
    max_file_size: int = 10_485_760,
    backup_count: int = 5,
    verbose_tracebacks: bool = False,
) -> None:
    """Internal function to setup logging - use CONFIG.apply() instead.

    Configures loguru logger with console and/or file handlers.
    Multi-line messages are automatically formatted with box-style borders.

    Args:
        default_level: Logging level for the logger.
        log_file: Path to log file (None to disable file logging).
        console: Enable console logging (True/'stdout' or 'stderr').
        max_file_size: Maximum log file size in bytes before rotation.
        backup_count: Number of backup log files to keep.
        verbose_tracebacks: If True, show detailed tracebacks with variable values.
    """
    # Remove all existing handlers
    logger.remove()

    # Console handler with multi-line formatting
    if console:
        stream = sys.stdout if console is True or console == 'stdout' else sys.stderr
        logger.add(
            stream,
            format=_format_multiline,
            level=default_level.upper(),
            colorize=True,
            backtrace=verbose_tracebacks,
            diagnose=verbose_tracebacks,
            enqueue=False,
        )

    # File handler with rotation (plain format for files)
    if log_file:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Cannot create log directory '{log_path.parent}': Permission denied") from e

        logger.add(
            log_file,
            format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}',
            level=default_level.upper(),
            colorize=False,
            rotation=max_file_size,
            retention=backup_count,
            encoding='utf-8',
            backtrace=verbose_tracebacks,
            diagnose=verbose_tracebacks,
            enqueue=False,
        )


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']):
    """Change the logging level for the flixopt logger.

    .. deprecated:: 2.1.11
        Use ``CONFIG.Logging.level = level_name`` and ``CONFIG.apply()`` instead.
        This function will be removed in version 3.0.0.

    Args:
        level_name: The logging level to set.

    Examples:
        >>> change_logging_level('DEBUG')  # deprecated
        >>> # Use this instead:
        >>> CONFIG.Logging.level = 'DEBUG'
        >>> CONFIG.apply()
    """
    warnings.warn(
        'change_logging_level is deprecated and will be removed in version 3.0.0. '
        'Use CONFIG.Logging.level = level_name and CONFIG.apply() instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    CONFIG.Logging.level = level_name.upper()
    CONFIG.apply()


# Initialize default config
CONFIG.apply()
