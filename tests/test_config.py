"""Tests for the config module."""

import logging
import sys
from pathlib import Path

import pytest

from flixopt.config import _DEFAULTS, CONFIG, MultilineFormatter

logger = logging.getLogger('flixopt')


# All tests in this class will run in the same worker to prevent issues with global config altering
@pytest.mark.xdist_group(name='config_tests')
class TestConfigModule:
    """Test the CONFIG class and logging setup."""

    def setup_method(self):
        """Reset CONFIG to defaults before each test."""
        CONFIG.reset()

    def teardown_method(self):
        """Clean up after each test to prevent state leakage."""
        CONFIG.reset()

    def test_config_defaults(self):
        """Test that CONFIG has correct default values."""
        assert CONFIG.Modeling.big == 10_000_000
        assert CONFIG.Modeling.epsilon == 1e-5
        assert CONFIG.Modeling.big_binary_bound == 100_000
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True
        assert CONFIG.config_name == 'flixopt'

    def test_module_initialization(self, capfd):
        """Test that logging is silent by default on module import."""
        # By default, flixopt should be silent (WARNING level, no handlers)
        logger.info('test message')
        captured = capfd.readouterr()
        assert 'test message' not in captured.out
        assert 'test message' not in captured.err

    def test_enable_console_logging(self, capfd):
        """Test enabling console logging."""
        CONFIG.Logging.enable_console('DEBUG')

        test_message = 'test debug message 12345'
        logger.debug(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.out

    def test_enable_file_logging(self, tmp_path):
        """Test enabling file logging."""
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.enable_file('WARNING', str(log_file))

        test_message = 'test warning message 67890'
        logger.warning(test_message)

        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_enable_console_non_colored(self, capfd):
        """Test enabling console logging without colors."""
        CONFIG.Logging.enable_console('INFO', colored=False)

        test_message = 'test info message 11111'
        logger.info(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.out

    def test_enable_console_stderr(self, capfd):
        """Test enabling console logging to stderr."""
        CONFIG.Logging.enable_console('INFO', stream=sys.stderr)

        test_message = 'test info to stderr 22222'
        logger.info(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.err

    def test_logging_levels(self, capfd):
        """Test all valid logging levels."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        for level in levels:
            CONFIG.Logging.disable()  # Clean up
            CONFIG.Logging.enable_console(level)

            test_message = f'test message at {level} 55555'
            getattr(logger, level.lower())(test_message)
            captured = capfd.readouterr()
            output = captured.out + captured.err
            assert test_message in output, f'Expected {level} message to appear'

    def test_success_level(self, capfd):
        """Test custom SUCCESS log level."""
        CONFIG.Logging.enable_console('INFO')

        test_message = 'test success message 33333'
        logger.success(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.out

    def test_multiline_formatter(self):
        """Test that MultilineFormatter handles multi-line messages."""
        formatter = MultilineFormatter()
        record = logging.LogRecord(
            'test', logging.INFO, '', 1, 'Line 1\nLine 2\nLine 3', (), None
        )
        formatted = formatter.format(record)
        assert '┌─' in formatted
        assert '│' in formatted
        assert '└─' in formatted

    def test_disable_logging(self, capfd):
        """Test disabling logging."""
        CONFIG.Logging.enable_console('DEBUG')
        logger.debug('This should appear')
        captured = capfd.readouterr()
        assert 'This should appear' in captured.out

        CONFIG.Logging.disable()
        logger.debug('This should NOT appear')
        captured = capfd.readouterr()
        assert 'This should NOT appear' not in captured.out

    def test_console_and_file_logging(self, tmp_path, capfd):
        """Test logging to both console and file simultaneously."""
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.enable_console('INFO')
        CONFIG.Logging.enable_file('INFO', str(log_file))

        test_message = 'test both outputs 77777'
        logger.info(test_message)

        # Check console
        captured = capfd.readouterr()
        assert test_message in captured.out

        # Check file
        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_config_to_dict(self):
        """Test converting CONFIG to dictionary."""
        config_dict = CONFIG.to_dict()

        assert config_dict['config_name'] == 'flixopt'
        assert 'modeling' in config_dict
        assert config_dict['modeling']['big'] == 10_000_000
        assert 'solving' in config_dict
        assert config_dict['solving']['mip_gap'] == 0.01
        assert config_dict['solving']['time_limit_seconds'] == 300
        assert config_dict['solving']['log_to_console'] is True
        assert config_dict['solving']['log_main_results'] is True

    def test_config_attribute_modification(self):
        """Test that config attributes can be modified directly."""
        # Modify attributes
        CONFIG.Modeling.big = 12345678
        CONFIG.Modeling.epsilon = 1e-8
        CONFIG.Solving.mip_gap = 0.001

        # Verify modifications
        assert CONFIG.Modeling.big == 12345678
        assert CONFIG.Modeling.epsilon == 1e-8
        assert CONFIG.Solving.mip_gap == 0.001

    def test_config_reset(self):
        """Test that CONFIG.reset() restores all defaults."""
        # Modify values
        CONFIG.Modeling.big = 99999999
        CONFIG.Modeling.epsilon = 1e-8
        CONFIG.Solving.mip_gap = 0.0001
        CONFIG.Solving.time_limit_seconds = 1800
        CONFIG.config_name = 'test_config'

        # Enable logging
        CONFIG.Logging.enable_console('DEBUG')

        # Reset should restore all defaults and disable logging
        CONFIG.reset()

        # Verify values are back to defaults
        assert CONFIG.Modeling.big == 10_000_000
        assert CONFIG.Modeling.epsilon == 1e-5
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.config_name == 'flixopt'

        # Verify logging was disabled
        assert len(logger.handlers) == 0
        assert logger.level == logging.CRITICAL

    def test_preset_exploring(self, capfd):
        """Test CONFIG.exploring() preset."""
        CONFIG.exploring()

        # Should enable INFO level console logging
        logger.info('test exploring')
        captured = capfd.readouterr()
        assert 'test exploring' in captured.out

        # Should enable solver output
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True

        # Should enable plots
        assert CONFIG.Plotting.default_show is True

    def test_preset_debug(self, capfd):
        """Test CONFIG.debug() preset."""
        CONFIG.debug()

        # Should enable DEBUG level console logging
        logger.debug('test debug')
        captured = capfd.readouterr()
        assert 'test debug' in captured.out

        # Should enable solver output
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True

    def test_preset_notebook(self, capfd):
        """Test CONFIG.notebook() preset."""
        CONFIG.notebook()

        # Should enable INFO level console logging
        logger.info('test notebook')
        captured = capfd.readouterr()
        assert 'test notebook' in captured.out

        # Should enable plots
        assert CONFIG.Plotting.default_show is True

        # Should enable solver output
        assert CONFIG.Solving.log_to_console is True

    def test_preset_production(self, tmp_path):
        """Test CONFIG.production() preset."""
        log_file = tmp_path / 'prod.log'
        CONFIG.production(str(log_file))

        # Should enable file logging
        logger.info('test production')
        assert log_file.exists()
        log_content = log_file.read_text()
        assert 'test production' in log_content

        # Should disable plots
        assert CONFIG.Plotting.default_show is False

        # Should disable solver console output
        assert CONFIG.Solving.log_to_console is False
        assert CONFIG.Solving.log_main_results is False

    def test_preset_silent(self, capfd):
        """Test CONFIG.silent() preset."""
        CONFIG.silent()

        # Should not log anything
        logger.info('should not appear')
        captured = capfd.readouterr()
        assert 'should not appear' not in captured.out

        # Should disable plots
        assert CONFIG.Plotting.default_show is False

        # Should disable solver output
        assert CONFIG.Solving.log_to_console is False

    def test_change_logging_level_deprecated(self):
        """Test that change_logging_level is deprecated."""
        from flixopt import change_logging_level

        # Should emit deprecation warning
        with pytest.warns(DeprecationWarning, match='change_logging_level is deprecated'):
            change_logging_level('DEBUG')

    def test_solving_config_defaults(self):
        """Test that CONFIG.Solving has correct default values."""
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True

    def test_solving_config_modification(self):
        """Test that CONFIG.Solving attributes can be modified."""
        CONFIG.Solving.mip_gap = 0.005
        CONFIG.Solving.time_limit_seconds = 600
        CONFIG.Solving.log_main_results = False

        assert CONFIG.Solving.mip_gap == 0.005
        assert CONFIG.Solving.time_limit_seconds == 600
        assert CONFIG.Solving.log_main_results is False

    def test_solving_config_in_to_dict(self):
        """Test that CONFIG.Solving is included in to_dict()."""
        CONFIG.Solving.mip_gap = 0.003
        CONFIG.Solving.time_limit_seconds = 450

        config_dict = CONFIG.to_dict()

        assert 'solving' in config_dict
        assert config_dict['solving']['mip_gap'] == 0.003
        assert config_dict['solving']['time_limit_seconds'] == 450

    def test_plotting_config_defaults(self):
        """Test that CONFIG.Plotting has correct default values."""
        assert CONFIG.Plotting.default_show is True
        assert CONFIG.Plotting.default_engine == 'plotly'
        assert CONFIG.Plotting.default_dpi == 300

    def test_file_logging_rotation_params(self, tmp_path):
        """Test file logging with custom rotation parameters."""
        log_file = tmp_path / 'rotating.log'
        CONFIG.Logging.enable_file(
            'INFO', str(log_file), max_bytes=1024, backup_count=2
        )

        # Write some logs
        for i in range(10):
            logger.info(f'Log message {i}')

        # Verify file exists
        assert log_file.exists()

    def test_consistent_formatting_console_and_file(self, tmp_path, capfd):
        """Test that console and file use consistent formatting."""
        log_file = tmp_path / 'format_test.log'
        CONFIG.Logging.enable_console('INFO', colored=False)
        CONFIG.Logging.enable_file('INFO', str(log_file))

        test_message = 'Multi-line test\nLine 2\nLine 3'
        logger.info(test_message)

        # Get console output
        captured = capfd.readouterr()
        console_output = captured.out

        # Get file output
        file_output = log_file.read_text()

        # Both should have box borders
        assert '┌─' in console_output
        assert '┌─' in file_output
        assert '└─' in console_output
        assert '└─' in file_output

    def test_public_api_exports(self):
        """Test that expected items are exported from config module."""
        from flixopt import config

        # Should export these
        assert hasattr(config, 'CONFIG')
        assert hasattr(config, 'get_logger')
        assert hasattr(config, 'change_logging_level')
        assert hasattr(config, 'MultilineFormatter')

    def test_get_logger_function(self):
        """Test get_logger() function."""
        from flixopt.config import get_logger

        test_logger = get_logger()
        assert test_logger.name == 'flixopt'

        custom_logger = get_logger('flixopt.custom')
        assert custom_logger.name == 'flixopt.custom'
