"""Tests for the config module."""

import logging
import sys

import pytest

from flixopt.config import CONFIG, MultilineFormatter

logger = logging.getLogger('flixopt')


@pytest.mark.xdist_group(name='config_tests')
class TestConfigModule:
    """Test the CONFIG class and logging setup."""

    def setup_method(self):
        """Reset CONFIG to defaults before each test."""
        CONFIG.reset()

    def teardown_method(self):
        """Clean up after each test."""
        CONFIG.reset()

    def test_config_defaults(self):
        """Test that CONFIG has correct default values."""
        assert CONFIG.Modeling.big == 10_000_000
        assert CONFIG.Modeling.epsilon == 1e-5
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.config_name == 'flixopt'

    def test_silent_by_default(self, capfd):
        """Test that flixopt is silent by default."""
        logger.info('should not appear')
        captured = capfd.readouterr()
        assert 'should not appear' not in captured.out

    def test_enable_console_logging(self, capfd):
        """Test enabling console logging."""
        CONFIG.Logging.enable_console('INFO')
        logger.info('test message')
        captured = capfd.readouterr()
        assert 'test message' in captured.out

    def test_enable_file_logging(self, tmp_path):
        """Test enabling file logging."""
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.enable_file('INFO', str(log_file))
        logger.info('test file message')

        assert log_file.exists()
        assert 'test file message' in log_file.read_text()

    def test_console_and_file_together(self, tmp_path, capfd):
        """Test logging to both console and file."""
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.enable_console('INFO')
        CONFIG.Logging.enable_file('INFO', str(log_file))

        logger.info('test both')

        # Check both outputs
        assert 'test both' in capfd.readouterr().out
        assert 'test both' in log_file.read_text()

    def test_disable_logging(self, capfd):
        """Test disabling logging."""
        CONFIG.Logging.enable_console('INFO')
        CONFIG.Logging.disable()

        logger.info('should not appear')
        assert 'should not appear' not in capfd.readouterr().out

    def test_custom_success_level(self, capfd):
        """Test custom SUCCESS log level."""
        CONFIG.Logging.enable_console('INFO')
        logger.success('success message')
        assert 'success message' in capfd.readouterr().out

    def test_multiline_formatting(self):
        """Test that multi-line messages get box borders."""
        formatter = MultilineFormatter()
        record = logging.LogRecord('test', logging.INFO, '', 1, 'Line 1\nLine 2\nLine 3', (), None)
        formatted = formatter.format(record)
        assert '┌─' in formatted
        assert '└─' in formatted

    def test_console_stderr(self, capfd):
        """Test logging to stderr."""
        CONFIG.Logging.enable_console('INFO', stream=sys.stderr)
        logger.info('stderr test')
        assert 'stderr test' in capfd.readouterr().err

    def test_non_colored_output(self, capfd):
        """Test non-colored console output."""
        CONFIG.Logging.enable_console('INFO', colored=False)
        logger.info('plain text')
        assert 'plain text' in capfd.readouterr().out

    def test_preset_exploring(self, capfd):
        """Test exploring preset."""
        CONFIG.exploring()
        logger.info('exploring')
        assert 'exploring' in capfd.readouterr().out
        assert CONFIG.Solving.log_to_console is True

    def test_preset_debug(self, capfd):
        """Test debug preset."""
        CONFIG.debug()
        logger.debug('debug')
        assert 'debug' in capfd.readouterr().out

    def test_preset_production(self, tmp_path):
        """Test production preset."""
        log_file = tmp_path / 'prod.log'
        CONFIG.production(str(log_file))
        logger.info('production')

        assert log_file.exists()
        assert 'production' in log_file.read_text()
        assert CONFIG.Plotting.default_show is False

    def test_preset_silent(self, capfd):
        """Test silent preset."""
        CONFIG.silent()
        logger.info('should not appear')
        assert 'should not appear' not in capfd.readouterr().out

    def test_config_reset(self):
        """Test that reset() restores defaults and disables logging."""
        CONFIG.Modeling.big = 99999999
        CONFIG.Logging.enable_console('DEBUG')

        CONFIG.reset()

        assert CONFIG.Modeling.big == 10_000_000
        assert len(logger.handlers) == 0

    def test_config_to_dict(self):
        """Test converting CONFIG to dictionary."""
        config_dict = CONFIG.to_dict()
        assert config_dict['modeling']['big'] == 10_000_000
        assert config_dict['solving']['mip_gap'] == 0.01

    def test_attribute_modification(self):
        """Test modifying config attributes."""
        CONFIG.Modeling.big = 12345678
        CONFIG.Solving.mip_gap = 0.001

        assert CONFIG.Modeling.big == 12345678
        assert CONFIG.Solving.mip_gap == 0.001

    def test_change_logging_level_deprecated(self):
        """Test deprecated change_logging_level function."""
        from flixopt import change_logging_level

        with pytest.warns(DeprecationWarning, match='change_logging_level is deprecated'):
            change_logging_level('INFO')

    def test_exception_logging(self, capfd):
        """Test that exceptions are properly logged with tracebacks."""
        CONFIG.Logging.enable_console('INFO')

        try:
            raise ValueError('Test exception')
        except ValueError:
            logger.exception('An error occurred')

        captured = capfd.readouterr().out
        assert 'An error occurred' in captured
        assert 'ValueError' in captured
        assert 'Test exception' in captured
        assert 'Traceback' in captured

    def test_exception_logging_non_colored(self, capfd):
        """Test that exceptions are properly logged with tracebacks in non-colored mode."""
        CONFIG.Logging.enable_console('INFO', colored=False)

        try:
            raise ValueError('Test exception non-colored')
        except ValueError:
            logger.exception('An error occurred')

        captured = capfd.readouterr().out
        assert 'An error occurred' in captured
        assert 'ValueError: Test exception non-colored' in captured
        assert 'Traceback' in captured

    def test_enable_file_preserves_custom_handlers(self, tmp_path, capfd):
        """Test that enable_file preserves custom non-file handlers."""
        # Add a custom console handler first
        CONFIG.Logging.enable_console('INFO')
        logger.info('console test')
        assert 'console test' in capfd.readouterr().out

        # Now add file logging - should keep the console handler
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.enable_file('INFO', str(log_file))

        logger.info('both outputs')

        # Check console still works
        console_output = capfd.readouterr().out
        assert 'both outputs' in console_output

        # Check file was created and has the message
        assert log_file.exists()
        assert 'both outputs' in log_file.read_text()

    def test_enable_file_removes_duplicate_file_handlers(self, tmp_path):
        """Test that enable_file removes existing file handlers to avoid duplicates."""
        log_file = tmp_path / 'test.log'

        # Enable file logging twice
        CONFIG.Logging.enable_file('INFO', str(log_file))
        CONFIG.Logging.enable_file('INFO', str(log_file))

        logger.info('duplicate test')

        # Count file handlers - should only be 1
        from logging.handlers import RotatingFileHandler

        file_handlers = [h for h in logger.handlers if isinstance(h, (logging.FileHandler, RotatingFileHandler))]
        assert len(file_handlers) == 1

        # Message should appear only once in the file
        log_content = log_file.read_text()
        assert log_content.count('duplicate test') == 1
