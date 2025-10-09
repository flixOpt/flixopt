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
from rich.theme import Theme

__all__ = ['CONFIG', 'change_logging_level']

logger = logging.getLogger('flixopt')

# Default Rich theme for log levels
DEFAULT_THEME = Theme(
    {
        'logging.level.debug': 'green',
        'logging.level.info': 'blue',
        'logging.level.warning': 'yellow',
        'logging.level.error': 'red',
        'logging.level.critical': 'bold red',
    }
)


# SINGLE SOURCE OF TRUTH - immutable to prevent accidental modification
_DEFAULTS = MappingProxyType(
    {
        'config_name': 'flixopt',
        'logging': MappingProxyType(
            {
                'level': 'INFO',
                'file': 'flixopt.log',
                'rich': False,
                'console': False,
                'max_file_size': 10_485_760,  # 10MB
                'backup_count': 5,
                'date_format': '%Y-%m-%d %H:%M:%S',
                'format': '%(message)s',
                'console_width': 120,
                'show_path': False,
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

    Library is SILENT by default (best practice for libraries).

    The CONFIG class provides centralized configuration for logging and modeling parameters.
    All changes require calling ``CONFIG.apply()`` to take effect.

    Attributes:
        Logging: Nested class containing all logging configuration options.
        Modeling: Nested class containing optimization modeling parameters.
        config_name (str): Name of the configuration (default: 'flixopt').

    Logging Attributes:
        level (str): Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            Default: 'INFO'
        file (str | None): Log file path. Default: 'flixopt.log'.
            Set to None to disable file logging.
        console (bool): Enable console (stdout) logging. Default: False
        rich (bool): Use Rich library for enhanced console output. Default: False
        max_file_size (int): Maximum log file size in bytes before rotation.
            Default: 10485760 (10MB)
        backup_count (int): Number of backup log files to keep. Default: 5
        date_format (str): Date/time format for log messages.
            Default: '%Y-%m-%d %H:%M:%S'
        format (str): Log message format string. Default: '%(message)s'
        console_width (int): Console width for Rich handler. Default: 120
        show_path (bool): Show file paths in log messages. Default: False
        colors (dict[str, str]): ANSI color codes for each log level.
            Keys: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            Default colors:

            - DEBUG: green ('\\033[32m')
            - INFO: blue ('\\033[34m')
            - WARNING: yellow ('\\033[33m')
            - ERROR: red ('\\033[31m')
            - CRITICAL: bold red ('\\033[1m\\033[31m')

    Modeling Attributes:
        big (int): Large number for optimization constraints. Default: 10000000
        epsilon (float): Small tolerance value. Default: 1e-5
        big_binary_bound (int): Upper bound for binary variable constraints.
            Default: 100000

    Examples:
        Basic configuration::

            from flixopt import CONFIG

            CONFIG.Logging.console = True
            CONFIG.Logging.level = 'DEBUG'
            CONFIG.apply()

        Configure log file rotation::

            CONFIG.Logging.file = 'myapp.log'
            CONFIG.Logging.max_file_size = 5_242_880  # 5 MB
            CONFIG.Logging.backup_count = 3
            CONFIG.apply()

        Customize log colors::

            CONFIG.Logging.colors['INFO'] = '\\033[35m'  # Magenta
            CONFIG.Logging.colors['DEBUG'] = '\\033[36m'  # Cyan
            CONFIG.apply()

        Use Rich handler with custom width::

            CONFIG.Logging.console = True
            CONFIG.Logging.rich = True
            CONFIG.Logging.console_width = 100
            CONFIG.Logging.show_path = True
            CONFIG.apply()

        Load from YAML file::

            CONFIG.load_from_file('config.yaml')

        Example YAML config file:

        .. code-block:: yaml

            logging:
              level: DEBUG
              console: true
              file: app.log
              rich: false
              max_file_size: 5242880  # 5MB
              backup_count: 3
              date_format: '%H:%M:%S'
              console_width: 100
              show_path: true
              colors:
                DEBUG: "\\033[36m"    # Cyan
                INFO: "\\033[32m"     # Green
                WARNING: "\\033[33m"
                ERROR: "\\033[31m"
                CRITICAL: "\\033[1m\\033[31m"

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
        level: str = _DEFAULTS['logging']['level']
        file: str | None = _DEFAULTS['logging']['file']
        rich: bool = _DEFAULTS['logging']['rich']
        console: bool = _DEFAULTS['logging']['console']
        max_file_size: int = _DEFAULTS['logging']['max_file_size']
        backup_count: int = _DEFAULTS['logging']['backup_count']
        date_format: str = _DEFAULTS['logging']['date_format']
        format: str = _DEFAULTS['logging']['format']
        console_width: int = _DEFAULTS['logging']['console_width']
        show_path: bool = _DEFAULTS['logging']['show_path']
        colors: dict[str, str] = dict(_DEFAULTS['logging']['colors'])

    class Modeling:
        big: int = _DEFAULTS['modeling']['big']
        epsilon: float = _DEFAULTS['modeling']['epsilon']
        big_binary_bound: int = _DEFAULTS['modeling']['big_binary_bound']

    config_name: str = _DEFAULTS['config_name']

    @classmethod
    def reset(cls):
        """Reset all configuration values to defaults."""
        # Dynamically reset from _DEFAULTS - no repetition!
        for key, value in _DEFAULTS['logging'].items():
            setattr(cls.Logging, key, value)

        for key, value in _DEFAULTS['modeling'].items():
            setattr(cls.Modeling, key, value)

        cls.config_name = _DEFAULTS['config_name']

        # Apply the reset configuration
        cls.apply()

    @classmethod
    def apply(cls):
        """Apply current configuration to logging system."""
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
            colors=cls.Logging.colors,
        )

    @classmethod
    def load_from_file(cls, config_file: str | Path):
        """Load configuration from YAML file and apply it."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_file}')

        with config_path.open() as file:
            config_dict = yaml.safe_load(file)
            cls._apply_config_dict(config_dict)

        cls.apply()

    @classmethod
    def _apply_config_dict(cls, config_dict: dict):
        """Apply configuration dictionary to class attributes."""
        for key, value in config_dict.items():
            if key == 'logging' and isinstance(value, dict):
                # Apply logging config
                for nested_key, nested_value in value.items():
                    setattr(cls.Logging, nested_key, nested_value)
            elif key == 'modeling' and isinstance(value, dict):
                # Apply modeling config
                for nested_key, nested_value in value.items():
                    setattr(cls.Modeling, nested_key, nested_value)
            elif hasattr(cls, key):
                # Simple attribute
                setattr(cls, key, value)

    @classmethod
    def to_dict(cls):
        """Convert the configuration class into a dictionary for JSON serialization."""
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
                'colors': dict(cls.Logging.colors),  # Ensure it's a regular dict
            },
            'modeling': {
                'big': cls.Modeling.big,
                'epsilon': cls.Modeling.epsilon,
                'big_binary_bound': cls.Modeling.big_binary_bound,
            },
        }


class MultilineFormater(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        message_lines = record.getMessage().split('\n')

        # Prepare the log prefix (timestamp + log level)
        timestamp = self.formatTime(record, self.datefmt)
        log_level = record.levelname.ljust(8)  # Align log levels for consistency
        log_prefix = f'{timestamp} | {log_level} |'

        # Format all lines
        first_line = [f'{log_prefix} {message_lines[0]}']
        if len(message_lines) > 1:
            lines = first_line + [f'{log_prefix} {line}' for line in message_lines[1:]]
        else:
            lines = first_line

        return '\n'.join(lines)


class ColoredMultilineFormater(MultilineFormater):
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, colors=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        # Use provided colors or fall back to defaults
        self.COLORS = (
            colors
            if colors is not None
            else {
                'DEBUG': '\033[32m',  # Green
                'INFO': '\033[34m',  # Blue
                'WARNING': '\033[33m',  # Yellow
                'ERROR': '\033[31m',  # Red
                'CRITICAL': '\033[1m\033[31m',  # Bold Red
            }
        )

    def format(self, record):
        lines = super().format(record).splitlines()
        log_color = self.COLORS.get(record.levelname, self.RESET)

        # Create a formatted message for each line separately
        formatted_lines = []
        for line in lines:
            formatted_lines.append(f'{log_color}{line}{self.RESET}')

        return '\n'.join(formatted_lines)


def _ansi_to_rich_style(ansi_code: str) -> str:
    """Convert ANSI color codes to Rich style strings.

    Args:
        ansi_code: ANSI escape sequence (e.g., '\\033[32m')

    Returns:
        Rich style string (e.g., 'green')
    """
    ansi_to_rich_map = {
        '\033[32m': 'green',
        '\033[34m': 'blue',
        '\033[33m': 'yellow',
        '\033[31m': 'red',
        '\033[35m': 'magenta',
        '\033[36m': 'cyan',
        '\033[1m\033[31m': 'bold red',
        '\033[1m\033[32m': 'bold green',
        '\033[1m\033[33m': 'bold yellow',
        '\033[1m\033[34m': 'bold blue',
        '\033[1m\033[35m': 'bold magenta',
        '\033[1m\033[36m': 'bold cyan',
    }
    return ansi_to_rich_map.get(ansi_code, 'default')


def _create_console_handler(
    use_rich: bool = False,
    console_width: int = 120,
    show_path: bool = False,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
    colors: dict[str, str] | None = None,
) -> logging.Handler:
    """Create a console (stdout) logging handler."""
    if use_rich:
        # Create custom theme from ANSI colors if provided
        if colors:
            theme_dict = {}
            for level, ansi_code in colors.items():
                rich_style = _ansi_to_rich_style(ansi_code)
                theme_dict[f'logging.level.{level.lower()}'] = rich_style
            theme = Theme(theme_dict)
        else:
            theme = DEFAULT_THEME

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
        handler.setFormatter(ColoredMultilineFormater(fmt=format, datefmt=date_format, colors=colors))
    return handler


def _create_file_handler(
    log_file: str,
    max_file_size: int = 10_485_760,
    backup_count: int = 5,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    format: str = '%(message)s',
) -> RotatingFileHandler:
    """Create a rotating file handler to prevent huge log files."""
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8',
    )
    handler.setFormatter(MultilineFormater(fmt=format, datefmt=date_format))
    return handler


def _setup_logging(
    default_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
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
    """Internal function to setup logging - use CONFIG.apply() instead."""
    logger = logging.getLogger('flixopt')
    logger.setLevel(getattr(logging, default_level.upper()))
    logger.propagate = False  # Prevent duplicate logs
    logger.handlers.clear()

    # Only log to console if explicitly requested
    if console:
        logger.addHandler(
            _create_console_handler(
                use_rich=use_rich_handler,
                console_width=console_width,
                show_path=show_path,
                date_format=date_format,
                format=format,
                colors=colors,
            )
        )

    # Add file handler if specified
    if log_file:
        logger.addHandler(
            _create_file_handler(
                log_file=log_file,
                max_file_size=max_file_size,
                backup_count=backup_count,
                date_format=date_format,
                format=format,
            )
        )

    # IMPORTANT: If no handlers, use NullHandler (library best practice)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    """
    Change the logging level for the flixopt logger and all its handlers.

    .. deprecated:: 2.1.11
        Use ``CONFIG.Logging.level = level_name`` and ``CONFIG.apply()`` instead.
        This function will be removed in version 3.0.0.

    Parameters
    ----------
    level_name : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        The logging level to set.

    Examples
    --------
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
