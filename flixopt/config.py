from __future__ import annotations

import logging
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style
from rich.theme import Theme

__all__ = ['CONFIG', 'change_logging_level']

logger = logging.getLogger('flixopt')


# SINGLE SOURCE OF TRUTH - immutable to prevent accidental modification
_DEFAULTS = MappingProxyType(
    {
        'config_name': 'flixopt',
        'logging': MappingProxyType(
            {
                'level': 'INFO',
                'file': None,
                'rich': False,
                'console': False,
                'max_file_size': 10_485_760,  # 10MB
                'backup_count': 5,
                'date_format': '%Y-%m-%d %H:%M:%S',
                'format': '%(message)s',
                'console_width': 120,
                'show_path': False,
                'show_logger_name': False,
                'colors': MappingProxyType(
                    {
                        'DEBUG': '\033[90m',  # Bright Black/Gray
                        'INFO': '\033[0m',  # Default/White
                        'WARNING': '\033[33m',  # Yellow
                        'ERROR': '\033[31m',  # Red
                        'CRITICAL': '\033[1m\033[31m',  # Bold Red
                    }
                ),
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
    }
)


class CONFIG:
    """Configuration for flixopt library.

    Always call ``CONFIG.apply()`` after changes.

    Attributes:
        Logging: Logging configuration.
        Modeling: Optimization modeling parameters.
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
        ```
    """

    class Logging:
        """Logging configuration.

        Silent by default. Enable via ``console=True`` or ``file='path'``.

        Attributes:
            level: Logging level.
            file: Log file path for file logging.
            console: Enable console output.
            rich: Use Rich library for enhanced output.
            max_file_size: Max file size before rotation.
            backup_count: Number of backup files to keep.
            date_format: Date/time format string.
            format: Log message format string.
            console_width: Console width for Rich handler.
            show_path: Show file paths in messages.
            show_logger_name: Show logger name in messages.
            Colors: ANSI color codes for log levels.

        Examples:
            ```python
            # File logging with rotation
            CONFIG.Logging.file = 'app.log'
            CONFIG.Logging.max_file_size = 5_242_880  # 5MB
            CONFIG.apply()

            # Rich handler with stdout
            CONFIG.Logging.console = True  # or 'stdout'
            CONFIG.Logging.rich = True
            CONFIG.apply()

            # Console output to stderr
            CONFIG.Logging.console = 'stderr'
            CONFIG.apply()
            ```
        """

        level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = _DEFAULTS['logging']['level']
        file: str | None = _DEFAULTS['logging']['file']
        rich: bool = _DEFAULTS['logging']['rich']
        console: bool | Literal['stdout', 'stderr'] = _DEFAULTS['logging']['console']
        max_file_size: int = _DEFAULTS['logging']['max_file_size']
        backup_count: int = _DEFAULTS['logging']['backup_count']
        date_format: str = _DEFAULTS['logging']['date_format']
        format: str = _DEFAULTS['logging']['format']
        console_width: int = _DEFAULTS['logging']['console_width']
        show_path: bool = _DEFAULTS['logging']['show_path']
        show_logger_name: bool = _DEFAULTS['logging']['show_logger_name']

        class Colors:
            """ANSI color codes for log levels.

            Attributes:
                DEBUG: ANSI color for DEBUG level.
                INFO: ANSI color for INFO level.
                WARNING: ANSI color for WARNING level.
                ERROR: ANSI color for ERROR level.
                CRITICAL: ANSI color for CRITICAL level.

            Examples:
                ```python
                CONFIG.Logging.Colors.INFO = '\\033[32m'  # Green
                CONFIG.Logging.Colors.ERROR = '\\033[1m\\033[31m'  # Bold red
                CONFIG.apply()
                ```

            Common ANSI codes:
                - '\\033[30m' - Black
                - '\\033[31m' - Red
                - '\\033[32m' - Green
                - '\\033[33m' - Yellow
                - '\\033[34m' - Blue
                - '\\033[35m' - Magenta
                - '\\033[36m' - Cyan
                - '\\033[37m' - White
                - '\\033[90m' - Bright Black/Gray
                - '\\033[0m' - Reset to default
                - '\\033[1m\\033[3Xm' - Bold (replace X with color code 0-7)
                - '\\033[2m\\033[3Xm' - Dim (replace X with color code 0-7)
            """

            DEBUG: str = _DEFAULTS['logging']['colors']['DEBUG']
            INFO: str = _DEFAULTS['logging']['colors']['INFO']
            WARNING: str = _DEFAULTS['logging']['colors']['WARNING']
            ERROR: str = _DEFAULTS['logging']['colors']['ERROR']
            CRITICAL: str = _DEFAULTS['logging']['colors']['CRITICAL']

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
            if key == 'colors':
                # Reset nested Colors class
                for color_key, color_value in value.items():
                    setattr(cls.Logging.Colors, color_key, color_value)
            else:
                setattr(cls.Logging, key, value)

        for key, value in _DEFAULTS['modeling'].items():
            setattr(cls.Modeling, key, value)

        cls.config_name = _DEFAULTS['config_name']
        cls.apply()

    @classmethod
    def apply(cls):
        """Apply current configuration to logging system."""
        # Convert Colors class attributes to dict
        colors_dict = {
            'DEBUG': cls.Logging.Colors.DEBUG,
            'INFO': cls.Logging.Colors.INFO,
            'WARNING': cls.Logging.Colors.WARNING,
            'ERROR': cls.Logging.Colors.ERROR,
            'CRITICAL': cls.Logging.Colors.CRITICAL,
        }
        valid_levels = list(colors_dict)
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
            use_rich_handler=cls.Logging.rich,
            console=cls.Logging.console,
            max_file_size=cls.Logging.max_file_size,
            backup_count=cls.Logging.backup_count,
            date_format=cls.Logging.date_format,
            format=cls.Logging.format,
            console_width=cls.Logging.console_width,
            show_path=cls.Logging.show_path,
            show_logger_name=cls.Logging.show_logger_name,
            colors=colors_dict,
        )

    @classmethod
    def load_from_file(cls, config_file: str | Path):
        """Load configuration from YAML file and apply it.

        Args:
            config_file: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_file}')

        with config_path.open() as file:
            config_dict = yaml.safe_load(file) or {}
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
                    if nested_key == 'colors' and isinstance(nested_value, dict):
                        # Handle nested colors under logging
                        for color_key, color_value in nested_value.items():
                            setattr(cls.Logging.Colors, color_key, color_value)
                    else:
                        setattr(cls.Logging, nested_key, nested_value)
            elif key == 'modeling' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    setattr(cls.Modeling, nested_key, nested_value)
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
                'rich': cls.Logging.rich,
                'console': cls.Logging.console,
                'max_file_size': cls.Logging.max_file_size,
                'backup_count': cls.Logging.backup_count,
                'date_format': cls.Logging.date_format,
                'format': cls.Logging.format,
                'console_width': cls.Logging.console_width,
                'show_path': cls.Logging.show_path,
                'show_logger_name': cls.Logging.show_logger_name,
                'colors': {
                    'DEBUG': cls.Logging.Colors.DEBUG,
                    'INFO': cls.Logging.Colors.INFO,
                    'WARNING': cls.Logging.Colors.WARNING,
                    'ERROR': cls.Logging.Colors.ERROR,
                    'CRITICAL': cls.Logging.Colors.CRITICAL,
                },
            },
            'modeling': {
                'big': cls.Modeling.big,
                'epsilon': cls.Modeling.epsilon,
                'big_binary_bound': cls.Modeling.big_binary_bound,
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


class MultilineFormatter(logging.Formatter):
    """Formatter that handles multi-line messages with consistent prefixes.

    Args:
        fmt: Log message format string.
        datefmt: Date/time format string.
        show_logger_name: Show logger name in log messages.
    """

    def __init__(self, fmt: str = '%(message)s', datefmt: str | None = None, show_logger_name: bool = False):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.show_logger_name = show_logger_name

    def format(self, record) -> str:
        record.message = record.getMessage()
        message_lines = self._style.format(record).split('\n')
        timestamp = self.formatTime(record, self.datefmt)
        log_level = record.levelname.ljust(8)

        if self.show_logger_name:
            # Truncate long logger names for readability
            logger_name = record.name if len(record.name) <= 20 else f'...{record.name[-17:]}'
            log_prefix = f'{timestamp} | {log_level} | {logger_name.ljust(20)} |'
        else:
            log_prefix = f'{timestamp} | {log_level} |'

        indent = ' ' * (len(log_prefix) + 1)  # +1 for the space after prefix

        lines = [f'{log_prefix} {message_lines[0]}']
        if len(message_lines) > 1:
            lines.extend([f'{indent}{line}' for line in message_lines[1:]])

        return '\n'.join(lines)


class ColoredMultilineFormatter(MultilineFormatter):
    """Formatter that adds ANSI colors to multi-line log messages.

    Args:
        fmt: Log message format string.
        datefmt: Date/time format string.
        colors: Dictionary of ANSI color codes for each log level.
        show_logger_name: Show logger name in log messages.
    """

    RESET = '\033[0m'

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        colors: dict[str, str] | None = None,
        show_logger_name: bool = False,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, show_logger_name=show_logger_name)
        self.COLORS = (
            colors
            if colors is not None
            else {
                'DEBUG': '\033[90m',
                'INFO': '\033[0m',
                'WARNING': '\033[33m',
                'ERROR': '\033[31m',
                'CRITICAL': '\033[1m\033[31m',
            }
        )

    def format(self, record):
        lines = super().format(record).splitlines()
        log_color = self.COLORS.get(record.levelname, self.RESET)
        formatted_lines = [f'{log_color}{line}{self.RESET}' for line in lines]
        return '\n'.join(formatted_lines)


def _create_console_handler(
    use_rich: bool = False,
    stream: Literal['stdout', 'stderr'] = 'stdout',
    console_width: int = 120,
    show_path: bool = False,
    show_logger_name: bool = False,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
    colors: dict[str, str] | None = None,
) -> logging.Handler:
    """Create a console logging handler.

    Args:
        use_rich: If True, use RichHandler with color support.
        stream: Output stream
        console_width: Width of the console for Rich handler.
        show_path: Show file paths in log messages (Rich only).
        show_logger_name: Show logger name in log messages.
        date_format: Date/time format string.
        format: Log message format string.
        colors: Dictionary of ANSI color codes for each log level.

    Returns:
        Configured logging handler (RichHandler or StreamHandler).
    """
    # Determine the stream object
    stream_obj = sys.stdout if stream == 'stdout' else sys.stderr

    if use_rich:
        # Convert ANSI codes to Rich theme
        if colors:
            theme_dict = {}
            for level, ansi_code in colors.items():
                # Rich can parse ANSI codes directly!
                try:
                    style = Style.from_ansi(ansi_code)
                    theme_dict[f'logging.level.{level.lower()}'] = style
                except Exception:
                    # Fallback to default if parsing fails
                    pass

            theme = Theme(theme_dict) if theme_dict else None
        else:
            theme = None

        console = Console(width=console_width, theme=theme, file=stream_obj)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            show_path=show_path,
            log_time_format=date_format,
        )
        handler.setFormatter(logging.Formatter(format))
    else:
        handler = logging.StreamHandler(stream=stream_obj)
        handler.setFormatter(
            ColoredMultilineFormatter(
                fmt=format,
                datefmt=date_format,
                colors=colors,
                show_logger_name=show_logger_name,
            )
        )

    return handler


def _create_file_handler(
    log_file: str,
    max_file_size: int = 10_485_760,
    backup_count: int = 5,
    show_logger_name: bool = False,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
) -> RotatingFileHandler:
    """Create a rotating file handler to prevent huge log files.

    Args:
        log_file: Path to the log file.
        max_file_size: Maximum size in bytes before rotation.
        backup_count: Number of backup files to keep.
        show_logger_name: Show logger name in log messages.
        date_format: Date/time format string.
        format: Log message format string.

    Returns:
        Configured RotatingFileHandler (without colors).
    """

    # Ensure parent directory exists
    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot create log directory '{log_path.parent}': Permission denied") from e

    try:
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8',
        )
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to log file '{log_file}': Permission denied. "
            f'Choose a different location or check file permissions.'
        ) from e

    handler.setFormatter(
        MultilineFormatter(
            fmt=format,
            datefmt=date_format,
            show_logger_name=show_logger_name,
        )
    )
    return handler


def _setup_logging(
    default_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    log_file: str | None = None,
    use_rich_handler: bool = False,
    console: bool | Literal['stdout', 'stderr'] = False,
    max_file_size: int = 10_485_760,
    backup_count: int = 5,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
    console_width: int = 120,
    show_path: bool = False,
    show_logger_name: bool = False,
    colors: dict[str, str] | None = None,
) -> None:
    """Internal function to setup logging - use CONFIG.apply() instead.

    Configures the flixopt logger with console and/or file handlers.
    If no handlers are configured, adds NullHandler (library best practice).

    Args:
        default_level: Logging level for the logger.
        log_file: Path to log file (None to disable file logging).
        use_rich_handler: Use Rich for enhanced console output.
        console: Enable console logging.
        max_file_size: Maximum log file size before rotation.
        backup_count: Number of backup log files to keep.
        date_format: Date/time format for log messages.
        format: Log message format string.
        console_width: Console width for Rich handler.
        show_path: Show file paths in log messages (Rich only).
        show_logger_name: Show logger name in log messages.
        colors: ANSI color codes for each log level.
    """
    logger = logging.getLogger('flixopt')
    logger.setLevel(getattr(logging, default_level.upper()))
    logger.propagate = False  # Prevent duplicate logs
    logger.handlers.clear()

    # Handle console parameter: False = disabled, True = stdout, 'stdout' = stdout, 'stderr' = stderr
    if console:
        # Convert True to 'stdout', keep 'stdout'/'stderr' as-is
        stream = 'stdout' if console is True else console
        logger.addHandler(
            _create_console_handler(
                use_rich=use_rich_handler,
                stream=stream,
                console_width=console_width,
                show_path=show_path,
                show_logger_name=show_logger_name,
                date_format=date_format,
                format=format,
                colors=colors,
            )
        )

    if log_file:
        logger.addHandler(
            _create_file_handler(
                log_file=log_file,
                max_file_size=max_file_size,
                backup_count=backup_count,
                show_logger_name=show_logger_name,
                date_format=date_format,
                format=format,
            )
        )

    # Library best practice: NullHandler if no handlers configured
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    """Change the logging level for the flixopt logger and all its handlers.

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
    logger = logging.getLogger('flixopt')
    logging_level = getattr(logging, level_name.upper())
    logger.setLevel(logging_level)
    for handler in logger.handlers:
        handler.setLevel(logging_level)


# Initialize default config
CONFIG.apply()
