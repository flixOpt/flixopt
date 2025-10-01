from __future__ import annotations

import logging
import types
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import yaml
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger('flixopt')


def merge_configs(defaults: dict, overrides: dict) -> dict:
    """
    Merge the default configuration with user-provided overrides.
    Args:
        defaults: Default configuration dictionary.
        overrides: User configuration dictionary.
    Returns:
        Merged configuration dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in defaults and isinstance(defaults[key], dict):
            # Recursively merge nested dictionaries
            defaults[key] = merge_configs(defaults[key], value)
        else:
            # Override the default value
            defaults[key] = value
    return defaults


class CONFIG:
    """Configuration using simple nested classes."""

    class Logging:
        level: str = 'INFO'
        file: str | None = None
        rich: bool = False
        console: bool = False  # Libraries should be silent by default

    class Modeling:
        big: int = 10_000_000
        epsilon: float = 1e-5
        big_binary_bound: int = 100_000

    config_name: str = 'flixopt'

    @classmethod
    def load_from_file(cls, config_file: str | Path):
        """Load and apply configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_file}')

        with config_path.open() as file:
            config_dict = yaml.safe_load(file)
            cls._apply_config_dict(config_dict)

        # Re-setup logging with new config
        cls._setup_logging()

    @classmethod
    def _setup_logging(cls):
        """Setup logging based on current configuration."""
        setup_logging(
            default_level=cls.Logging.level,
            log_file=cls.Logging.file,
            use_rich_handler=cls.Logging.rich,
            console=cls.Logging.console,
        )

    @classmethod
    def _apply_config_dict(cls, config_dict: dict):
        """Apply configuration dictionary to class attributes."""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                target = getattr(cls, key)
                if hasattr(target, '__dict__') and isinstance(value, dict):
                    # It's a nested class, apply recursively
                    for nested_key, nested_value in value.items():
                        setattr(target, nested_key, nested_value)
                else:
                    # Simple attribute
                    setattr(cls, key, value)

    @classmethod
    def to_dict(cls):
        """
        Convert the configuration class into a dictionary for JSON serialization.
        """
        config_dict = {}
        for attribute, value in cls.__dict__.items():
            # Only consider attributes (not methods, etc.)
            if (
                not attribute.startswith('_')
                and not isinstance(value, (types.FunctionType, types.MethodType))
                and not isinstance(value, classmethod)
            ):
                if hasattr(value, '__dict__') and not isinstance(value, type):
                    # It's a nested class instance
                    config_dict[attribute] = {
                        k: v for k, v in value.__dict__.items() if not k.startswith('_') and not callable(v)
                    }
                elif isinstance(value, type):
                    # It's a nested class definition
                    config_dict[attribute] = {
                        k: v
                        for k, v in value.__dict__.items()
                        if not k.startswith('_') and not callable(v) and not isinstance(v, classmethod)
                    }
                else:
                    # Simple attribute
                    config_dict[attribute] = value

        return config_dict


class MultilineFormater(logging.Formatter):
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
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[32m',  # Green
        'INFO': '\033[34m',  # Blue
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[1m\033[31m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        lines = super().format(record).splitlines()
        log_color = self.COLORS.get(record.levelname, self.RESET)

        # Create a formatted message for each line separately
        formatted_lines = []
        for line in lines:
            formatted_lines.append(f'{log_color}{line}{self.RESET}')

        return '\n'.join(formatted_lines)


def _create_console_handler(use_rich: bool = False) -> logging.Handler:
    """Create a console (stdout) logging handler."""
    if use_rich:
        console = Console(width=120)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            show_path=False,
            log_time_format='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredMultilineFormater(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    return handler


def _create_file_handler(log_file: str) -> RotatingFileHandler:
    """Create a rotating file handler to prevent huge log files."""
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10_485_760,  # 10MB max file size
        backupCount=5,  # Keep 5 backup files
        encoding='utf-8',
    )
    handler.setFormatter(MultilineFormater(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    return handler


def setup_logging(
    default_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    log_file: str | None = None,
    use_rich_handler: bool = False,
    console: bool = False,
):
    """Setup logging - silent by default for library use."""
    logger = logging.getLogger('flixopt')
    logger.setLevel(getattr(logging, default_level.upper()))
    logger.propagate = False  # Prevent duplicate logs
    logger.handlers.clear()

    # Only log to console if explicitly requested
    if console:
        logger.addHandler(_create_console_handler(use_rich=use_rich_handler))

    # Add file handler if specified
    if log_file:
        logger.addHandler(_create_file_handler(log_file))

    # IMPORTANT: If no handlers, use NullHandler (library best practice)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def change_logging_level(level_name: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']):
    """Change the logging level for the flixopt logger and all its handlers."""
    logger = logging.getLogger('flixopt')
    log_level = getattr(logging, level_name.upper())
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
