from __future__ import annotations

import logging
import os
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import TextIO

try:
    import colorlog
    from colorlog.escape_codes import escape_codes

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False
    escape_codes = None

__all__ = ['CONFIG', 'MultilineFormatter', 'SUCCESS_LEVEL', 'DEPRECATION_REMOVAL_VERSION']

if COLORLOG_AVAILABLE:
    __all__.append('ColoredMultilineFormatter')

# Add custom SUCCESS level (between INFO and WARNING)
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')

# Deprecation removal version - update this when planning the next major version
DEPRECATION_REMOVAL_VERSION = '7.0.0'


class MultilineFormatter(logging.Formatter):
    """Custom formatter that handles multi-line messages with box-style borders.

    Uses Unicode box-drawing characters for prettier output, with a fallback
    to simple formatting if any encoding issues occur.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default format with time
        if not self._fmt:
            self._fmt = '%(asctime)s %(levelname)-8s │ %(message)s'
            self._style = logging.PercentStyle(self._fmt)

    def format(self, record):
        """Format multi-line messages with box-style borders for better readability."""
        try:
            # Split into lines
            lines = record.getMessage().split('\n')

            # Add exception info if present (critical for logger.exception())
            if record.exc_info:
                lines.extend(self.formatException(record.exc_info).split('\n'))
            if record.stack_info:
                lines.extend(record.stack_info.rstrip().split('\n'))

            # Format time with date and milliseconds (YYYY-MM-DD HH:MM:SS.mmm)
            # formatTime doesn't support %f, so use datetime directly
            import datetime

            dt = datetime.datetime.fromtimestamp(record.created)
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # Single line - return standard format
            if len(lines) == 1:
                level_str = f'{record.levelname: <8}'
                return f'{time_str} {level_str} │ {lines[0]}'

            # Multi-line - use box format
            level_str = f'{record.levelname: <8}'
            result = f'{time_str} {level_str} │ ┌─ {lines[0]}'
            indent = ' ' * 23  # 23 spaces for time with date (YYYY-MM-DD HH:MM:SS.mmm)
            for line in lines[1:-1]:
                result += f'\n{indent} {" " * 8} │ │  {line}'
            result += f'\n{indent} {" " * 8} │ └─ {lines[-1]}'

            return result

        except Exception as e:
            # Fallback to simple formatting if anything goes wrong (e.g., encoding issues)
            return f'{record.created} {record.levelname} - {record.getMessage()} [Formatting Error: {e}]'


if COLORLOG_AVAILABLE:

    class ColoredMultilineFormatter(colorlog.ColoredFormatter):
        """Colored formatter with multi-line message support.

        Uses Unicode box-drawing characters for prettier output, with a fallback
        to simple formatting if any encoding issues occur.
        """

        def format(self, record):
            """Format multi-line messages with colors and box-style borders."""
            try:
                # Split into lines
                lines = record.getMessage().split('\n')

                # Add exception info if present (critical for logger.exception())
                if record.exc_info:
                    lines.extend(self.formatException(record.exc_info).split('\n'))
                if record.stack_info:
                    lines.extend(record.stack_info.rstrip().split('\n'))

                # Format time with date and milliseconds (YYYY-MM-DD HH:MM:SS.mmm)
                import datetime

                # Use thin attribute for timestamp
                dim = escape_codes['thin']
                reset = escape_codes['reset']
                # formatTime doesn't support %f, so use datetime directly
                dt = datetime.datetime.fromtimestamp(record.created)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
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
                result = f'{time_formatted} {color}{level_str}{reset} │ {color}┌─{reset} {lines[0]}'
                indent = ' ' * 23  # 23 spaces for time with date (YYYY-MM-DD HH:MM:SS.mmm)
                for line in lines[1:-1]:
                    result += f'\n{dim}{indent}{reset} {" " * 8} │ {color}│{reset}  {line}'
                result += f'\n{dim}{indent}{reset} {" " * 8} │ {color}└─{reset} {lines[-1]}'

                return result

            except Exception as e:
                # Fallback to simple formatting if anything goes wrong (e.g., encoding issues)
                return f'{record.created} {record.levelname} - {record.getMessage()} [Formatting Error: {e}]'


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
                'default_line_shape': 'hv',
            }
        ),
        'solving': MappingProxyType(
            {
                'mip_gap': 0.01,
                'time_limit_seconds': 300,
                'log_to_console': True,
                'log_main_results': True,
                'compute_infeasibilities': True,
            }
        ),
    }
)


class CONFIG:
    """Configuration for flixopt library.

    Attributes:
        Logging: Logging configuration (see CONFIG.Logging for details).
        Modeling: Optimization modeling parameters.
        Solving: Solver configuration and default parameters.
        Plotting: Plotting configuration.
        config_name: Configuration name.

    Examples:
        ```python
        # Quick logging setup
        CONFIG.Logging.enable_console('INFO')

        # Or use presets (affects logging, plotting, solver output)
        CONFIG.exploring()  # Interactive exploration
        CONFIG.debug()  # Troubleshooting
        CONFIG.production()  # Production deployment
        CONFIG.silent()  # No output

        # Adjust other settings
        CONFIG.Solving.mip_gap = 0.001
        CONFIG.Plotting.default_dpi = 600
        ```
    """

    class Logging:
        """Logging configuration helpers.

        flixopt is silent by default (WARNING level, no handlers).

        Quick Start - Use Presets:
            These presets configure logging along with plotting and solver output:

            | Preset | Console Logs | File Logs | Plots | Solver Output | Use Case |
            |--------|-------------|-----------|-------|---------------|----------|
            | ``CONFIG.exploring()`` | INFO (colored) | No | Browser | Yes | Interactive exploration |
            | ``CONFIG.debug()`` | DEBUG (colored) | No | Default | Yes | Troubleshooting |
            | ``CONFIG.production('app.log')`` | No | INFO | No | No | Production deployments |
            | ``CONFIG.silent()`` | No | No | No | No | Silent operation |

            Examples:
                ```python
                CONFIG.exploring()  # Start exploring interactively
                CONFIG.debug()  # See everything for troubleshooting
                CONFIG.production('logs/prod.log')  # Production mode
                ```

        Direct Control - Logging Only:
            For fine-grained control of logging without affecting other settings:

            Methods:
                - ``enable_console(level='INFO', colored=True, stream=None)``
                - ``enable_file(level='INFO', path='flixopt.log', max_bytes=10MB, backup_count=5)``
                - ``disable()`` - Remove all handlers
                - ``set_colors(log_colors)`` - Customize level colors

            Log Levels:
                Standard levels plus custom SUCCESS level (between INFO and WARNING):
                - DEBUG (10): Detailed debugging information
                - INFO (20): General informational messages
                - SUCCESS (25): Success messages (custom level)
                - WARNING (30): Warning messages
                - ERROR (40): Error messages
                - CRITICAL (50): Critical error messages

            Examples:
                ```python
                import logging
                from flixopt.config import CONFIG, SUCCESS_LEVEL

                # Console and file logging
                CONFIG.Logging.enable_console('INFO')
                CONFIG.Logging.enable_file('DEBUG', 'debug.log')

                # Use SUCCESS level with logger.log()
                logger = logging.getLogger('flixopt')
                CONFIG.Logging.enable_console('SUCCESS')  # Shows SUCCESS, WARNING, ERROR, CRITICAL
                logger.log(SUCCESS_LEVEL, 'Operation completed successfully!')

                # Or use numeric level directly
                logger.log(25, 'Also works with numeric level')

                # Customize colors
                CONFIG.Logging.set_colors(
                    {
                        'INFO': 'bold_white',
                        'SUCCESS': 'bold_green,bg_black',
                        'CRITICAL': 'bold_white,bg_red',
                    }
                )

                # Non-colored output
                CONFIG.Logging.enable_console('INFO', colored=False)
                ```

        Advanced Customization:
            For full control, use Python's standard logging or create custom formatters:

            ```python
            # Custom formatter
            from flixopt.config import ColoredMultilineFormatter
            import colorlog, logging

            handler = colorlog.StreamHandler()
            handler.setFormatter(ColoredMultilineFormatter(...))
            logging.getLogger('flixopt').addHandler(handler)

            # Or standard Python logging
            import logging

            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
            ```

        Note:
            Default formatters (MultilineFormatter and ColoredMultilineFormatter)
            provide pretty output with box borders for multi-line messages.
        """

        @classmethod
        def enable_console(cls, level: str | int = 'INFO', colored: bool = True, stream: TextIO | None = None) -> None:
            """Enable colored console logging.

            Args:
                level: Log level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL or numeric level)
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
                if level.upper().strip() == 'SUCCESS':
                    level = SUCCESS_LEVEL
                else:
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
                            'INFO': '',  # No color - use default terminal color
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
            encoding: str = 'utf-8',
        ) -> None:
            """Enable file logging with rotation. Removes all existing file handlers!

            Args:
                level: Log level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL or numeric level)
                path: Path to log file (default: 'flixopt.log')
                max_bytes: Maximum file size before rotation in bytes (default: 10MB)
                backup_count: Number of backup files to keep (default: 5)
                encoding: File encoding (default: 'utf-8'). Use 'utf-8' for maximum compatibility.

            Note:
                For full control over formatting and handlers, use logging module directly.

            Examples:
                ```python
                # Basic file logging
                CONFIG.Logging.enable_file('INFO', 'app.log')

                # With custom rotation
                CONFIG.Logging.enable_file('DEBUG', 'debug.log', max_bytes=50 * 1024 * 1024, backup_count=10)

                # With explicit encoding
                CONFIG.Logging.enable_file('INFO', 'app.log', encoding='utf-8')
                ```
            """
            logger = logging.getLogger('flixopt')

            # Convert string level to logging constant
            if isinstance(level, str):
                if level.upper().strip() == 'SUCCESS':
                    level = SUCCESS_LEVEL
                else:
                    level = getattr(logging, level.upper())

            logger.setLevel(level)

            # Remove existing file handlers to avoid duplicates, keep all non-file handlers (including custom handlers)
            logger.handlers = [
                h for h in logger.handlers if not isinstance(h, (logging.FileHandler, RotatingFileHandler))
            ]

            # Create log directory if needed
            log_path = Path(path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding)
            handler.setFormatter(MultilineFormatter())

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

        @classmethod
        def set_colors(cls, log_colors: dict[str, str]) -> None:
            """Customize log level colors for console output.

            This updates the colors for the current console handler.
            If no console handler exists, this does nothing.

            Args:
                log_colors: Dictionary mapping log levels to color names.
                    Colors can be comma-separated for multiple attributes
                    (e.g., 'bold_red,bg_white').

            Available colors:
                - Basic: black, red, green, yellow, blue, purple, cyan, white
                - Bold: bold_red, bold_green, bold_yellow, bold_blue, etc.
                - Light: light_red, light_green, light_yellow, light_blue, etc.
                - Backgrounds: bg_red, bg_green, bg_light_red, etc.
                - Combined: 'bold_white,bg_red' for white text on red background

            Examples:
                ```python
                # Enable console first
                CONFIG.Logging.enable_console('INFO')

                # Then customize colors
                CONFIG.Logging.set_colors(
                    {
                        'DEBUG': 'cyan',
                        'INFO': 'bold_white',
                        'SUCCESS': 'bold_green',
                        'WARNING': 'bold_yellow,bg_black',  # Yellow on black
                        'ERROR': 'bold_red',
                        'CRITICAL': 'bold_white,bg_red',  # White on red
                    }
                )
                ```

            Note:
                Requires colorlog to be installed. Has no effect on file handlers.
            """
            if not COLORLOG_AVAILABLE:
                warnings.warn('colorlog is not installed. Colors cannot be customized.', stacklevel=2)
                return

            logger = logging.getLogger('flixopt')

            # Find and update ColoredMultilineFormatter
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    formatter = handler.formatter
                    if isinstance(formatter, ColoredMultilineFormatter):
                        formatter.log_colors = log_colors
                        return

            warnings.warn(
                'No ColoredMultilineFormatter found. Call CONFIG.Logging.enable_console() with colored=True first.',
                stacklevel=2,
            )

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
            compute_infeasibilities: Whether to compute infeasibility analysis when the model is infeasible.

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
        compute_infeasibilities: bool = _DEFAULTS['solving']['compute_infeasibilities']

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
        default_line_shape: str = _DEFAULTS['plotting']['default_line_shape']

    class Carriers:
        """Default carrier definitions for common energy types.

        Provides convenient defaults for carriers. Colors are from D3/Plotly palettes.

        Predefined: electricity, heat, gas, hydrogen, fuel, biomass

        Examples:
            ```python
            import flixopt as fx

            # Access predefined carriers
            fx.CONFIG.Carriers.electricity  # Carrier with color '#FECB52'
            fx.CONFIG.Carriers.heat.color  # '#D62728'

            # Use with buses
            bus = fx.Bus('Grid', carrier='electricity')
            ```
        """

        from .carrier import Carrier

        # Default carriers - colors from D3/Plotly palettes
        electricity: Carrier = Carrier('electricity', '#FECB52')  # Yellow
        heat: Carrier = Carrier('heat', '#D62728')  # Red
        gas: Carrier = Carrier('gas', '#1F77B4')  # Blue
        hydrogen: Carrier = Carrier('hydrogen', '#9467BD')  # Purple
        fuel: Carrier = Carrier('fuel', '#8C564B')  # Brown
        biomass: Carrier = Carrier('biomass', '#2CA02C')  # Green

    config_name: str = _DEFAULTS['config_name']

    @classmethod
    def reset(cls) -> None:
        """Reset all configuration values to defaults.

        This resets modeling, solving, and plotting settings to their default values,
        and disables all logging handlers (back to silent mode).

        Examples:
            ```python
            CONFIG.debug()  # Enable debug mode
            # ... do some work ...
            CONFIG.reset()  # Back to defaults (silent)
            ```
        """
        # Reset settings
        for key, value in _DEFAULTS['modeling'].items():
            setattr(cls.Modeling, key, value)

        for key, value in _DEFAULTS['solving'].items():
            setattr(cls.Solving, key, value)

        for key, value in _DEFAULTS['plotting'].items():
            setattr(cls.Plotting, key, value)

        # Reset Carriers to defaults
        from .carrier import Carrier

        cls.Carriers.electricity = Carrier('electricity', '#FECB52')
        cls.Carriers.heat = Carrier('heat', '#D62728')
        cls.Carriers.gas = Carrier('gas', '#1F77B4')
        cls.Carriers.hydrogen = Carrier('hydrogen', '#9467BD')
        cls.Carriers.fuel = Carrier('fuel', '#8C564B')
        cls.Carriers.biomass = Carrier('biomass', '#2CA02C')

        cls.config_name = _DEFAULTS['config_name']

        # Reset logging to default (silent)
        cls.Logging.disable()

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
                'compute_infeasibilities': cls.Solving.compute_infeasibilities,
            },
            'plotting': {
                'default_show': cls.Plotting.default_show,
                'default_engine': cls.Plotting.default_engine,
                'default_dpi': cls.Plotting.default_dpi,
                'default_facet_cols': cls.Plotting.default_facet_cols,
                'default_sequential_colorscale': cls.Plotting.default_sequential_colorscale,
                'default_qualitative_colorscale': cls.Plotting.default_qualitative_colorscale,
                'default_line_shape': cls.Plotting.default_line_shape,
            },
        }

    @classmethod
    def silent(cls) -> type[CONFIG]:
        """Configure for silent operation.

        Disables all logging, solver output, and result logging
        for clean production runs. Does not show plots.

        Examples:
            ```python
            CONFIG.silent()
            # Now run optimizations with no output
            result = optimization.solve()
            ```
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

        Examples:
            ```python
            CONFIG.debug()
            # See detailed DEBUG logs and full solver output
            optimization.solve()
            ```
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

        Examples:
            ```python
            CONFIG.exploring()
            # Perfect for interactive sessions
            optimization.solve()  # Shows INFO logs and solver output
            result.plot()  # Opens plots in browser
            ```
        """
        cls.Logging.enable_console('INFO')
        cls.Solving.log_to_console = True
        cls.Solving.log_main_results = True
        cls.browser_plotting()
        return cls

    @classmethod
    def production(cls, log_file: str | Path = 'flixopt.log') -> type[CONFIG]:
        """Configure for production use.

        Enables file logging only (no console output), disables plots,
        and disables solver console output for clean production runs.

        Args:
            log_file: Path to log file (default: 'flixopt.log')

        Examples:
            ```python
            CONFIG.production('production.log')
            # Logs to file, no console output
            optimization.solve()
            ```
        """
        cls.Logging.disable()  # Clear any console handlers
        cls.Logging.enable_file('INFO', log_file)
        cls.Plotting.default_show = False
        cls.Solving.log_to_console = False
        cls.Solving.log_main_results = False
        return cls

    @classmethod
    def browser_plotting(cls) -> type[CONFIG]:
        """Configure for interactive usage with plotly to open plots in browser.

        Sets plotly.io.renderers.default = 'browser'. Useful for running examples
        and viewing interactive plots. Does NOT modify CONFIG.Plotting settings.

        Respects FLIXOPT_CI environment variable if set.

        Examples:
            ```python
            CONFIG.browser_plotting()
            result.plot()  # Opens in browser instead of inline
            ```
        """
        cls.Plotting.default_show = True

        # Only set to True if environment variable hasn't overridden it
        if 'FLIXOPT_CI' not in os.environ:
            import plotly.io as pio

            pio.renderers.default = 'browser'

        return cls

    @classmethod
    def notebook(cls) -> type[CONFIG]:
        """Configure for Jupyter notebook environments.

        Optimizes settings for notebook usage:
        - Sets plotly renderer to 'notebook' for inline display
        - Disables automatic plot.show() calls (notebooks display via _repr_html_)
        - Enables SUCCESS-level console logging
        - Disables solver console output for cleaner notebook cells

        Examples:
            ```python
            # At the start of your notebook
            import flixopt as fx

            fx.CONFIG.notebook()

            # Now plots display inline automatically
            flow_system.statistics.plot.balance('Heat')  # Displays inline
            ```
        """
        import plotly.io as pio

        # Set plotly to render inline in notebooks (respect PLOTLY_RENDERER env var)
        if 'PLOTLY_RENDERER' not in os.environ:
            pio.renderers.default = 'notebook'
        pio.templates.default = 'plotly_white'

        # Disable default show since notebooks render via _repr_html_
        cls.Plotting.default_show = False

        # Light logging - SUCCESS level without too much noise
        cls.Logging.enable_console('SUCCESS')

        # Disable verbose solver output for cleaner notebook cells
        cls.Solving.log_to_console = False
        cls.Solving.log_main_results = False

        return cls

    @classmethod
    def load_from_file(cls, config_file: str | Path) -> type[CONFIG]:
        """Load configuration from YAML file and apply it.

        Args:
            config_file: Path to the YAML configuration file.

        Returns:
            The CONFIG class for method chaining.

        Raises:
            FileNotFoundError: If the config file does not exist.

        Examples:
            ```python
            CONFIG.load_from_file('my_config.yaml')
            ```

            Example YAML file:
            ```yaml
            config_name: my_project
            modeling:
                big: 10000000
                epsilon: 0.00001
            solving:
                mip_gap: 0.001
                time_limit_seconds: 600
            plotting:
                default_engine: matplotlib
                default_dpi: 600
            ```
        """
        # Import here to avoid circular import
        from . import io as fx_io

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_file}')

        config_dict = fx_io.load_yaml(config_path)
        cls._apply_config_dict(config_dict)

        return cls

    @classmethod
    def _apply_config_dict(cls, config_dict: dict) -> None:
        """Apply configuration dictionary to class attributes.

        Args:
            config_dict: Dictionary containing configuration values.
        """
        for key, value in config_dict.items():
            if key == 'modeling' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if hasattr(cls.Modeling, nested_key):
                        setattr(cls.Modeling, nested_key, nested_value)
            elif key == 'solving' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if hasattr(cls.Solving, nested_key):
                        setattr(cls.Solving, nested_key, nested_value)
            elif key == 'plotting' and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if hasattr(cls.Plotting, nested_key):
                        setattr(cls.Plotting, nested_key, nested_value)
            elif hasattr(cls, key) and key != 'logging':
                # Skip 'logging' as it requires special handling via CONFIG.Logging methods
                setattr(cls, key, value)
