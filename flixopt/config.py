from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import yaml
from rich.console import Console
from rich.logging import RichHandler

__all__ = ['CONFIG']

logger = logging.getLogger('flixopt')


class CONFIG:
    """
    Configuration for flixopt library.

    Library is SILENT by default (best practice for libraries).

    Usage:
        # Change config and apply
        CONFIG.Logging.console = True
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.apply()

        # Load from YAML file (auto-applies)
        CONFIG.load_from_file('config.yaml')

        # Reset to defaults
        CONFIG.reset()
    """

    class Logging:
        level: str = 'INFO'
        file: str | None = None
        rich: bool = False
        console: bool = False

    class Modeling:
        big: int = 10_000_000
        epsilon: float = 1e-5
        big_binary_bound: int = 100_000

    config_name: str = 'flixopt'

    @classmethod
    def reset(cls):
        """Reset all configuration values to defaults."""
        # Reset Logging config
        cls.Logging.level = 'INFO'
        cls.Logging.file = None
        cls.Logging.rich = False
        cls.Logging.console = False

        # Reset Modeling config
        cls.Modeling.big = 10_000_000
        cls.Modeling.epsilon = 1e-5
        cls.Modeling.big_binary_bound = 100_000

        # Reset config name
        cls.config_name = 'flixopt'

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
            },
            'modeling': {
                'big': cls.Modeling.big,
                'epsilon': cls.Modeling.epsilon,
                'big_binary_bound': cls.Modeling.big_binary_bound,
            },
        }


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


def _setup_logging(
    default_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO',
    log_file: str | None = None,
    use_rich_handler: bool = False,
    console: bool = False,
):
    """Internal function to setup logging - use CONFIG.apply() instead."""
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


# Initialize default config
CONFIG.apply()
