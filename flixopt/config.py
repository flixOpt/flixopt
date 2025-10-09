from __future__ import annotations

import logging
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
                'file': 'flixopt.log',
                'rich': False,
                'console': True,
                'max_file_size': 10_485_760,  # 10MB
                'backup_count': 5,
                'date_format': '%Y-%m-%d %H:%M:%S',
                'format': '%(message)s',
                'console_width': 120,
                'show_path': False,
                'console_level': 'INFO',
                'file_level': 'INFO',
                'colors': MappingProxyType(
                    {
                        'DEBUG': '\033[32m',  # Green
                        'INFO': '\033[34m',  # Blue
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
    }
)


class CONFIG:
    """Configuration for flixopt library.

    The CONFIG class provides centralized configuration for logging and modeling parameters.
    All changes require calling ``CONFIG.apply()`` to take effect.

    By default, logging outputs to console at INFO level and to file at INFO level.

    Attributes:
        Logging: Nested class containing all logging configuration options.
            Colors: Nested subclass for ANSI color codes
        Modeling: Nested class containing optimization modeling parameters.
        config_name (str): Name of the configuration (default: 'flixopt').

    Logging Attributes:
        file (str | None): Log file path. Default: 'flixopt.log'.
            Set to None to disable file logging.
        console (bool): Enable console (stdout) logging. Default: False
        console_level (str): Logging level for console handler.
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. Default: 'INFO'
        file_level (str): Logging level for file handler.
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. Default: 'INFO'
        rich (bool): Use Rich library for enhanced console output. Default: False
        max_file_size (int): Maximum log file size in bytes before rotation.
            Default: 10485760 (10MB)
        backup_count (int): Number of backup log files to keep. Default: 5
        date_format (str): Date/time format for log messages.
            Default: '%Y-%m-%d %H:%M:%S'
        format (str): Log message format string. Default: '%(message)s'
        console_width (int): Console width for Rich handler. Default: 120
        show_path (bool): Show file paths in log messages. Default: False

    Colors Attributes:
        DEBUG (str): ANSI color code for DEBUG level. Default: '\\033[32m' (green)
        INFO (str): ANSI color code for INFO level. Default: '\\033[34m' (blue)
        WARNING (str): ANSI color code for WARNING level. Default: '\\033[33m' (yellow)
        ERROR (str): ANSI color code for ERROR level. Default: '\\033[31m' (red)
        CRITICAL (str): ANSI color code for CRITICAL level. Default: '\\033[1m\\033[31m' (bold red)

        Works with both Rich and standard console handlers.
        Rich automatically converts ANSI codes using Style.from_ansi().

        Common ANSI codes:

        - '\\033[30m' - Black
        - '\\033[31m' - Red
        - '\\033[32m' - Green
        - '\\033[33m' - Yellow
        - '\\033[34m' - Blue
        - '\\033[35m' - Magenta
        - '\\033[36m' - Cyan
        - '\\033[37m' - White
        - '\\033[1m\\033[3Xm' - Bold color (replace X with color code 0-7)
        - '\\033[2m\\033[3Xm' - Dim color (replace X with color code 0-7)

        Examples:

        - Magenta: '\\033[35m'
        - Bold cyan: '\\033[1m\\033[36m'
        - Dim green: '\\033[2m\\033[32m'

        Note: Use regular strings, not raw strings (r'...'). ANSI Codes start with a SINGLE BACKSLASH \
        Correct: '\\033[35m'
        Incorrect: r'\\033[35m'

    Modeling Attributes:
        big (int): Large number for optimization constraints. Default: 10000000
        epsilon (float): Small tolerance value. Default: 1e-5
        big_binary_bound (int): Upper bound for binary variable constraints.
            Default: 100000

    Examples:
        Basic configuration::

            from flixopt import CONFIG

            CONFIG.Logging.console = True
            CONFIG.Logging.console_level = 'DEBUG'
            CONFIG.apply()

        Use different log levels for console and file::

            CONFIG.Logging.console_level = 'DEBUG'  # Console shows DEBUG+
            CONFIG.Logging.file_level = 'WARNING'   # File only logs WARNING+
            CONFIG.apply()

        Customize log colors::

            CONFIG.Logging.Colors.INFO = '\\033[35m'  # Magenta
            CONFIG.Logging.Colors.DEBUG = '\\033[36m'  # Cyan
            CONFIG.Logging.Colors.ERROR = '\\033[1m\\033[31m'  # Bold red
            CONFIG.apply()

        Use Rich handler with custom colors::

            CONFIG.Logging.console = True
            CONFIG.Logging.rich = True
            CONFIG.Logging.console_width = 100
            CONFIG.Logging.show_path = True
            CONFIG.Logging.Colors.INFO = '\\033[36m'  # Cyan
            CONFIG.apply()

        Load from YAML file::

            CONFIG.load_from_file('config.yaml')

        Example YAML config file:

        .. code-block:: yaml

            logging:
              console: true
              file: app.log
              rich: true
              max_file_size: 5242880  # 5MB
              backup_count: 3
              date_format: '%H:%M:%S'
              console_width: 100
              show_path: true
              handlers:
                console_level: DEBUG
                file_level: WARNING
              colors:
                DEBUG: "\\033[36m"              # Cyan
                INFO: "\\033[32m"               # Green
                WARNING: "\\033[33m"            # Yellow
                ERROR: "\\033[31m"              # Red
                CRITICAL: "\\033[1m\\033[31m"  # Bold red

            modeling:
              big: 20000000
              epsilon: 1e-6
              big_binary_bound: 200000

        Reset to defaults::

            CONFIG.reset()

        Export current configuration::

            config_dict = CONFIG.to_dict()
            import yaml

            with open('my_config.yaml', 'w') as f:
                yaml.dump(config_dict, f)
    """

    class Logging:
        file: str | None = _DEFAULTS['logging']['file']
        rich: bool = _DEFAULTS['logging']['rich']
        console: bool = _DEFAULTS['logging']['console']
        max_file_size: int = _DEFAULTS['logging']['max_file_size']
        backup_count: int = _DEFAULTS['logging']['backup_count']
        date_format: str = _DEFAULTS['logging']['date_format']
        format: str = _DEFAULTS['logging']['format']
        console_width: int = _DEFAULTS['logging']['console_width']
        show_path: bool = _DEFAULTS['logging']['show_path']
        console_level: str = _DEFAULTS['logging']['console_level']
        file_level: str = _DEFAULTS['logging']['file_level']

        class Colors:
            """ANSI color codes for each log level.

            These colors work with both Rich and standard console handlers.
            """

            DEBUG: str = _DEFAULTS['logging']['colors']['DEBUG']
            INFO: str = _DEFAULTS['logging']['colors']['INFO']
            WARNING: str = _DEFAULTS['logging']['colors']['WARNING']
            ERROR: str = _DEFAULTS['logging']['colors']['ERROR']
            CRITICAL: str = _DEFAULTS['logging']['colors']['CRITICAL']

    class Modeling:
        big: int = _DEFAULTS['modeling']['big']
        epsilon: float = _DEFAULTS['modeling']['epsilon']
        big_binary_bound: int = _DEFAULTS['modeling']['big_binary_bound']

    config_name: str = _DEFAULTS['config_name']

    @classmethod
    def reset(cls):
        """Reset all configuration values to defaults."""
        for key, value in _DEFAULTS['logging'].items():
            if key == 'colors':
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
        _setup_logging(
            console_level=cls.Logging.console_level,
            file_level=cls.Logging.file_level,
            log_file=cls.Logging.file,
            use_rich_handler=cls.Logging.rich,
            console=cls.Logging.console,
            max_file_size=cls.Logging.max_file_size,
            backup_count=cls.Logging.backup_count,
            date_format=cls.Logging.date_format,
            format=cls.Logging.format,
            console_width=cls.Logging.console_width,
            show_path=cls.Logging.show_path,
            colors={
                'DEBUG': cls.Logging.Colors.DEBUG,
                'INFO': cls.Logging.Colors.INFO,
                'WARNING': cls.Logging.Colors.WARNING,
                'ERROR': cls.Logging.Colors.ERROR,
                'CRITICAL': cls.Logging.Colors.CRITICAL,
            },
        )

    @classmethod
    def load_from_file(cls, config_file: str | Path):
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_file}')
        with config_path.open() as file:
            config_dict = yaml.safe_load(file)
            cls._apply_config_dict(config_dict)
        cls.apply()

    @classmethod
    def _apply_config_dict(cls, config_dict: dict):
        for key, value in config_dict.items():
            if key == 'logging' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if nested_key == 'colors' and isinstance(nested_value, dict):
                        for color_key, color_value in nested_value.items():
                            setattr(cls.Logging.Colors, color_key, color_value)
                    elif nested_key == 'level':
                        # DEPRECATED: Handle old 'level' attribute for backward compatibility
                        warnings.warn(
                            "The 'level' attribute in config files is deprecated and will be removed in version 3.0.0. "
                            "Use 'logging.console_level' and 'logging.file_level' instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        # Apply to both handlers for backward compatibility
                        cls.Logging.console_level = nested_value
                        cls.Logging.file_level = nested_value
                    else:
                        setattr(cls.Logging, nested_key, nested_value)
            elif key == 'modeling' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    setattr(cls.Modeling, nested_key, nested_value)
            elif hasattr(cls, key):
                setattr(cls, key, value)

    @classmethod
    def to_dict(cls):
        return {
            'config_name': cls.config_name,
            'logging': {
                'file': cls.Logging.file,
                'rich': cls.Logging.rich,
                'console': cls.Logging.console,
                'max_file_size': cls.Logging.max_file_size,
                'backup_count': cls.Logging.backup_count,
                'date_format': cls.Logging.date_format,
                'format': cls.Logging.format,
                'console_width': cls.Logging.console_width,
                'show_path': cls.Logging.show_path,
                'console_level': cls.Logging.console_level,
                'file_level': cls.Logging.file_level,
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
        }


class MultilineFormatter(logging.Formatter):
    def format(self, record):
        message_lines = record.getMessage().split('\n')
        timestamp = self.formatTime(record, self.datefmt)
        log_level = record.levelname.ljust(8)
        prefix = f'{timestamp} | {log_level} |'
        lines = [f'{prefix} {line}' for line in message_lines]
        return '\n'.join(lines)


class ColoredMultilineFormatter(MultilineFormatter):
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, colors=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.COLORS = colors or {
            'DEBUG': '\033[32m',
            'INFO': '\033[34m',
            'WARNING': '\033[33m',
            'ERROR': '\033[31m',
            'CRITICAL': '\033[1m\033[31m',
        }

    def format(self, record):
        lines = super().format(record).splitlines()
        color = self.COLORS.get(record.levelname, self.RESET)
        return '\n'.join(f'{color}{line}{self.RESET}' for line in lines)


def _create_console_handler(
    use_rich: bool = False,
    console_width: int = 120,
    show_path: bool = False,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
    colors: dict[str, str] | None = None,
) -> logging.Handler:
    """Create a console (stdout) logging handler.

    Args:
        use_rich: If True, use RichHandler with color support.
        console_width: Width of the console for Rich handler.
        show_path: Show file paths in log messages (Rich only).
        date_format: Date/time format string.
        format: Log message format string.
        colors: Dictionary of ANSI color codes for each log level.

    Returns:
        Configured logging handler (RichHandler or StreamHandler).
    """
    if use_rich:
        theme = None
        if colors:
            theme_dict = {}
            for level, ansi_code in colors.items():
                try:
                    theme_dict[f'logging.level.{level.lower()}'] = Style.from_ansi(ansi_code)
                except Exception:
                    pass
            theme = Theme(theme_dict) if theme_dict else None

        console = Console(width=console_width, theme=theme)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            show_path=show_path,
            log_time_format=date_format,
        )
        handler.setFormatter(logging.Formatter(format))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredMultilineFormatter(fmt=format, datefmt=date_format, colors=colors))
    return handler


def _create_file_handler(
    log_file: str,
    max_file_size: int = 10_485_760,
    backup_count: int = 5,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
) -> RotatingFileHandler:
    """Create a rotating file handler to prevent huge log files.

    Args:
        log_file: Path to the log file.
        max_file_size: Maximum size in bytes before rotation.
        backup_count: Number of backup files to keep.
        date_format: Date/time format string.
        format: Log message format string.

    Returns:
        Configured RotatingFileHandler (without colors).
    """
    handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count, encoding='utf-8')
    handler.setFormatter(MultilineFormatter(fmt=format, datefmt=date_format))
    return handler


def _setup_logging(
    console_level: str = 'INFO',
    file_level: str = 'INFO',
    log_file: str | None = None,
    use_rich_handler: bool = False,
    console: bool = False,
    max_file_size: int = 10_485_760,
    backup_count: int = 5,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
    console_width: int = 120,
    show_path: bool = False,
    colors: dict[str, str] | None = None,
):
    """Internal function to setup logging - use CONFIG.apply() instead.

    Configures the flixopt logger with console and/or file handlers.
    The logger level is automatically set to the minimum level needed by any handler.
    If no handlers are configured, adds NullHandler (library best practice).

    Args:
        console_level: Logging level for console handler.
        file_level: Logging level for file handler.
        log_file: Path to log file (None to disable file logging).
        use_rich_handler: Use Rich for enhanced console output.
        console: Enable console logging.
        max_file_size: Maximum log file size before rotation.
        backup_count: Number of backup log files to keep.
        date_format: Date/time format for log messages.
        format: Log message format string.
        console_width: Console width for Rich handler.
        show_path: Show file paths in log messages (Rich only).
        colors: ANSI color codes for each log level.
    """
    logger = logging.getLogger('flixopt')
    levels = []
    if console:
        levels.append(getattr(logging, console_level.upper()))
    if log_file:
        levels.append(getattr(logging, file_level.upper()))
    logger.setLevel(min(levels) if levels else logging.CRITICAL)
    logger.propagate = False
    logger.handlers.clear()

    if console:
        handler = _create_console_handler(
            use_rich=use_rich_handler,
            console_width=console_width,
            show_path=show_path,
            date_format=date_format,
            format=format,
            colors=colors,
        )
        handler.setLevel(getattr(logging, console_level.upper()))
        logger.addHandler(handler)

    if log_file:
        handler = _create_file_handler(
            log_file=log_file,
            max_file_size=max_file_size,
            backup_count=backup_count,
            date_format=date_format,
            format=format,
        )
        handler.setLevel(getattr(logging, file_level.upper()))
        logger.addHandler(handler)

    # Library best practice: NullHandler if no handlers configured
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    """
    Change the logging level for the flixopt logger and all its handlers.

    .. deprecated:: 2.1.11
        Use ``CONFIG.Logging.console_level`` and ``CONFIG.Logging.file_level`` instead.
        This function will be removed in version 3.0.0.

    Parameters
    ----------
    level_name : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        The logging level to set.

    Examples
    --------
    >>> change_logging_level('DEBUG')  # deprecated
    >>> # Use this instead:
    >>> CONFIG.Logging.console_level = 'DEBUG'
    >>> CONFIG.Logging.file_level = 'DEBUG'
    >>> CONFIG.apply()
    """
    warnings.warn(
        'change_logging_level is deprecated and will be removed in version 3.0.0. '
        'Use CONFIG.Logging.console_level and CONFIG.Logging.file_level instead.',
        DeprecationWarning,
        stacklevel=2,
    )

    # Apply to both for backward compatibility
    CONFIG.Logging.console_level = level_name
    CONFIG.Logging.file_level = level_name
    CONFIG.apply()


# Initialize default config
CONFIG.apply()
