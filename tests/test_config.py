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
        record = logging.LogRecord(
            'test', logging.INFO, '', 1, 'Line 1\nLine 2\nLine 3', (), None
        )
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

    def test_preset_notebook(self, capfd):
        """Test notebook preset."""
        CONFIG.notebook()
        logger.info('notebook')
        assert 'notebook' in capfd.readouterr().out
        assert CONFIG.Plotting.default_show is True

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

    def test_get_logger_function(self):
        """Test get_logger() function."""
        from flixopt.config import get_logger

        test_logger = get_logger()
        assert test_logger.name == 'flixopt'

        custom_logger = get_logger('flixopt.custom')
        assert custom_logger.name == 'flixopt.custom'
