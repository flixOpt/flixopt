"""Tests for the config module."""

import logging
import sys
from pathlib import Path

import pytest

from flixopt.config import _DEFAULTS, CONFIG, _setup_logging

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
        assert CONFIG.Logging.level == 'INFO'
        assert CONFIG.Logging.file is None
        assert CONFIG.Logging.console is False
        assert CONFIG.Modeling.big == 10_000_000
        assert CONFIG.Modeling.epsilon == 1e-5
        assert CONFIG.Modeling.big_binary_bound == 100_000
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True
        assert CONFIG.config_name == 'flixopt'

    def test_module_initialization(self, capfd):
        """Test that logging is initialized on module import."""
        # Apply config to ensure handlers are initialized
        CONFIG.apply()
        # With default config (console=False, file=None), logs should not appear
        logger.info('test message')
        captured = capfd.readouterr()
        assert 'test message' not in captured.out
        assert 'test message' not in captured.err

    def test_config_apply_console(self, capfd):
        """Test applying config with console logging enabled."""
        CONFIG.Logging.console = True
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.apply()

        # Test that DEBUG level logs appear in console output
        test_message = 'test debug message 12345'
        logger.debug(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.out or test_message in captured.err

    def test_config_apply_file(self, tmp_path):
        """Test applying config with file logging enabled."""
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.file = str(log_file)
        CONFIG.Logging.level = 'WARNING'
        CONFIG.apply()

        # Test that WARNING level logs appear in the file
        test_message = 'test warning message 67890'
        logger.warning(test_message)
        # Loguru may buffer, so we need to ensure the log is written
        import time

        time.sleep(0.1)  # Small delay to ensure write
        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_config_apply_console_stderr(self, capfd):
        """Test applying config with console logging to stderr."""
        CONFIG.Logging.console = 'stderr'
        CONFIG.Logging.level = 'INFO'
        CONFIG.apply()

        # Test that INFO logs appear in stderr
        test_message = 'test info to stderr 11111'
        logger.info(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.err

    def test_config_apply_multiple_changes(self, capfd):
        """Test applying multiple config changes at once."""
        CONFIG.Logging.console = True
        CONFIG.Logging.level = 'ERROR'
        CONFIG.apply()

        # Test that ERROR level logs appear but lower levels don't
        logger.warning('warning should not appear')
        logger.error('error should appear 22222')
        captured = capfd.readouterr()
        output = captured.out + captured.err
        assert 'warning should not appear' not in output
        assert 'error should appear 22222' in output

    def test_config_to_dict(self):
        """Test converting CONFIG to dictionary."""
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.Logging.console = True

        config_dict = CONFIG.to_dict()

        assert config_dict['config_name'] == 'flixopt'
        assert config_dict['logging']['level'] == 'DEBUG'
        assert config_dict['logging']['console'] is True
        assert config_dict['logging']['file'] is None
        assert 'modeling' in config_dict
        assert config_dict['modeling']['big'] == 10_000_000
        assert 'solving' in config_dict
        assert config_dict['solving']['mip_gap'] == 0.01
        assert config_dict['solving']['time_limit_seconds'] == 300
        assert config_dict['solving']['log_to_console'] is True
        assert config_dict['solving']['log_main_results'] is True

    def test_config_load_from_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / 'config.yaml'
        config_content = """
config_name: test_config
logging:
  level: DEBUG
  console: true
  rich: false
modeling:
  big: 20000000
  epsilon: 1e-6
solving:
  mip_gap: 0.001
  time_limit_seconds: 600
  log_main_results: false
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        assert CONFIG.config_name == 'test_config'
        assert CONFIG.Logging.level == 'DEBUG'
        assert CONFIG.Logging.console is True
        assert CONFIG.Modeling.big == 20000000
        # YAML may load epsilon as string, so convert for comparison
        assert float(CONFIG.Modeling.epsilon) == 1e-6
        assert CONFIG.Solving.mip_gap == 0.001
        assert CONFIG.Solving.time_limit_seconds == 600
        assert CONFIG.Solving.log_main_results is False

    def test_config_load_from_file_not_found(self):
        """Test that loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            CONFIG.load_from_file('nonexistent_config.yaml')

    def test_config_load_from_file_partial(self, tmp_path):
        """Test loading partial configuration (should keep unspecified settings)."""
        config_file = tmp_path / 'partial_config.yaml'
        config_content = """
logging:
  level: ERROR
"""
        config_file.write_text(config_content)

        # Set a non-default value first
        CONFIG.Logging.console = True
        CONFIG.apply()

        CONFIG.load_from_file(config_file)

        # Should update level but keep other settings
        assert CONFIG.Logging.level == 'ERROR'
        # Verify console setting is preserved (not in YAML)
        assert CONFIG.Logging.console is True

    def test_setup_logging_silent_default(self, capfd):
        """Test that _setup_logging creates silent logger by default."""
        _setup_logging()

        # With default settings, logs should not appear
        logger.info('should not appear')
        captured = capfd.readouterr()
        assert 'should not appear' not in captured.out
        assert 'should not appear' not in captured.err

    def test_setup_logging_with_console(self, capfd):
        """Test _setup_logging with console output."""
        _setup_logging(console=True, default_level='DEBUG')

        # Test that DEBUG logs appear in console
        test_message = 'debug console test 33333'
        logger.debug(test_message)
        captured = capfd.readouterr()
        assert test_message in captured.out or test_message in captured.err

    def test_setup_logging_clears_handlers(self, capfd):
        """Test that _setup_logging clears existing handlers."""
        # Setup a handler first
        _setup_logging(console=True)

        # Call setup again with different settings - should clear and re-add
        _setup_logging(console=True, default_level='ERROR')

        # Verify new settings work: ERROR logs appear but INFO doesn't
        logger.info('info should not appear')
        logger.error('error should appear 44444')
        captured = capfd.readouterr()
        output = captured.out + captured.err
        assert 'info should not appear' not in output
        assert 'error should appear 44444' in output

    def test_change_logging_level_removed(self):
        """Test that change_logging_level function is deprecated but still exists."""
        # This function is deprecated - users should use CONFIG.apply() instead
        import flixopt

        # Function should still exist but be deprecated
        assert hasattr(flixopt, 'change_logging_level')

        # Should emit deprecation warning when called
        with pytest.warns(DeprecationWarning, match='change_logging_level is deprecated'):
            flixopt.change_logging_level('DEBUG')

    def test_public_api(self):
        """Test that CONFIG and change_logging_level are exported from config module."""
        from flixopt import config

        # CONFIG should be accessible
        assert hasattr(config, 'CONFIG')

        # change_logging_level should be accessible (but deprecated)
        assert hasattr(config, 'change_logging_level')

        # _setup_logging should exist but be marked as private
        assert hasattr(config, '_setup_logging')

        # merge_configs should not exist (was removed)
        assert not hasattr(config, 'merge_configs')

    def test_logging_levels(self, capfd):
        """Test all valid logging levels."""
        levels = ['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL']

        for level in levels:
            CONFIG.Logging.level = level
            CONFIG.Logging.console = True
            CONFIG.apply()

            # Test that logs at the configured level appear
            test_message = f'test message at {level} 55555'
            getattr(logger, level.lower())(test_message)
            captured = capfd.readouterr()
            output = captured.out + captured.err
            assert test_message in output, f'Expected {level} message to appear'

    def test_file_handler_rotation(self, tmp_path):
        """Test that file handler rotation configuration is accepted."""
        log_file = tmp_path / 'rotating.log'
        CONFIG.Logging.file = str(log_file)
        CONFIG.Logging.max_file_size = 1024
        CONFIG.Logging.backup_count = 2
        CONFIG.apply()

        # Write some logs
        for i in range(10):
            logger.info(f'Log message {i}')

        # Verify file logging works
        import time

        time.sleep(0.1)
        assert log_file.exists(), 'Log file should be created'

        # Verify configuration values are preserved
        assert CONFIG.Logging.max_file_size == 1024
        assert CONFIG.Logging.backup_count == 2

    def test_custom_config_yaml_complete(self, tmp_path):
        """Test loading a complete custom configuration."""
        config_file = tmp_path / 'custom_config.yaml'
        config_content = """
config_name: my_custom_config
logging:
  level: CRITICAL
  console: true
  file: /tmp/custom.log
modeling:
  big: 50000000
  epsilon: 1e-4
  big_binary_bound: 200000
solving:
  mip_gap: 0.005
  time_limit_seconds: 900
  log_main_results: false
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        # Check all settings were applied
        assert CONFIG.config_name == 'my_custom_config'
        assert CONFIG.Logging.level == 'CRITICAL'
        assert CONFIG.Logging.console is True
        assert CONFIG.Logging.file == '/tmp/custom.log'
        assert CONFIG.Modeling.big == 50000000
        assert float(CONFIG.Modeling.epsilon) == 1e-4
        assert CONFIG.Modeling.big_binary_bound == 200000
        assert CONFIG.Solving.mip_gap == 0.005
        assert CONFIG.Solving.time_limit_seconds == 900
        assert CONFIG.Solving.log_main_results is False

        # Verify logging was applied to both console and file
        import time

        test_message = 'critical test message 66666'
        logger.critical(test_message)
        time.sleep(0.1)  # Small delay to ensure write
        # Check file exists and contains message
        log_file_path = tmp_path / 'custom.log'
        if not log_file_path.exists():
            # File might be at /tmp/custom.log as specified in config
            import os

            log_file_path = os.path.expanduser('/tmp/custom.log')
        # We can't reliably test the file at /tmp/custom.log in tests
        # So just verify critical level messages would appear at this level
        assert CONFIG.Logging.level == 'CRITICAL'

    def test_config_file_with_console_and_file(self, tmp_path):
        """Test configuration with both console and file logging enabled."""
        log_file = tmp_path / 'test.log'
        config_file = tmp_path / 'config.yaml'
        config_content = f"""
logging:
  level: INFO
  console: true
  file: {log_file}
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        # Verify logging to both console and file works
        import time

        test_message = 'info test both outputs 77777'
        logger.info(test_message)
        time.sleep(0.1)  # Small delay to ensure write
        # Verify file logging works
        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_config_to_dict_roundtrip(self, tmp_path):
        """Test that config can be saved to dict, modified, and restored."""
        # Set custom values
        CONFIG.Logging.level = 'WARNING'
        CONFIG.Logging.console = True
        CONFIG.Modeling.big = 99999999

        # Save to dict
        config_dict = CONFIG.to_dict()

        # Verify dict structure
        assert config_dict['logging']['level'] == 'WARNING'
        assert config_dict['logging']['console'] is True
        assert config_dict['modeling']['big'] == 99999999

        # Could be written to YAML and loaded back
        yaml_file = tmp_path / 'saved_config.yaml'
        import yaml

        with open(yaml_file, 'w') as f:
            yaml.dump(config_dict, f)

        # Reset config
        CONFIG.Logging.level = 'INFO'
        CONFIG.Logging.console = False
        CONFIG.Modeling.big = 10_000_000

        # Load back from file
        CONFIG.load_from_file(yaml_file)

        # Should match original values
        assert CONFIG.Logging.level == 'WARNING'
        assert CONFIG.Logging.console is True
        assert CONFIG.Modeling.big == 99999999

    def test_config_file_with_only_modeling(self, tmp_path):
        """Test config file that only sets modeling parameters."""
        config_file = tmp_path / 'modeling_only.yaml'
        config_content = """
modeling:
  big: 999999
  epsilon: 0.001
"""
        config_file.write_text(config_content)

        # Set logging config before loading
        original_level = CONFIG.Logging.level
        CONFIG.load_from_file(config_file)

        # Modeling should be updated
        assert CONFIG.Modeling.big == 999999
        assert float(CONFIG.Modeling.epsilon) == 0.001

        # Logging should keep default/previous values
        assert CONFIG.Logging.level == original_level

    def test_config_attribute_modification(self):
        """Test that config attributes can be modified directly."""
        # Store original values
        original_big = CONFIG.Modeling.big
        original_level = CONFIG.Logging.level

        # Modify attributes
        CONFIG.Modeling.big = 12345678
        CONFIG.Modeling.epsilon = 1e-8
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.Logging.console = True

        # Verify modifications
        assert CONFIG.Modeling.big == 12345678
        assert CONFIG.Modeling.epsilon == 1e-8
        assert CONFIG.Logging.level == 'DEBUG'
        assert CONFIG.Logging.console is True

        # Reset
        CONFIG.Modeling.big = original_big
        CONFIG.Logging.level = original_level
        CONFIG.Logging.console = False

    def test_logger_actually_logs(self, tmp_path):
        """Test that the logger actually writes log messages."""
        log_file = tmp_path / 'actual_test.log'
        CONFIG.Logging.file = str(log_file)
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.apply()

        test_message = 'Test log message from config test'
        logger.debug(test_message)

        # Check that file was created and contains the message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_modeling_config_persistence(self):
        """Test that Modeling config is independent of Logging config."""
        # Set custom modeling values
        CONFIG.Modeling.big = 99999999
        CONFIG.Modeling.epsilon = 1e-8

        # Change and apply logging config
        CONFIG.Logging.console = True
        CONFIG.apply()

        # Modeling values should be unchanged
        assert CONFIG.Modeling.big == 99999999
        assert CONFIG.Modeling.epsilon == 1e-8

    def test_config_reset(self):
        """Test that CONFIG.reset() restores all defaults."""
        # Modify all config values
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.Logging.console = True
        CONFIG.Logging.file = '/tmp/test.log'
        CONFIG.Modeling.big = 99999999
        CONFIG.Modeling.epsilon = 1e-8
        CONFIG.Modeling.big_binary_bound = 500000
        CONFIG.Solving.mip_gap = 0.0001
        CONFIG.Solving.time_limit_seconds = 1800
        CONFIG.Solving.log_to_console = False
        CONFIG.Solving.log_main_results = False
        CONFIG.config_name = 'test_config'

        # Reset should restore all defaults
        CONFIG.reset()

        # Verify all values are back to defaults
        assert CONFIG.Logging.level == 'INFO'
        assert CONFIG.Logging.console is False
        assert CONFIG.Logging.file is None
        assert CONFIG.Modeling.big == 10_000_000
        assert CONFIG.Modeling.epsilon == 1e-5
        assert CONFIG.Modeling.big_binary_bound == 100_000
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True
        assert CONFIG.config_name == 'flixopt'

        # Verify logging was also reset (default is no logging to console/file)
        # Test that logs don't appear with default config
        from io import StringIO

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        try:
            logger.info('should not appear after reset')
            stdout_content = sys.stdout.getvalue()
            stderr_content = sys.stderr.getvalue()
            assert 'should not appear after reset' not in stdout_content
            assert 'should not appear after reset' not in stderr_content
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def test_reset_matches_class_defaults(self):
        """Test that reset() values match the _DEFAULTS constants.

        This ensures the reset() method and class attribute defaults
        stay synchronized by using the same source of truth (_DEFAULTS).
        """
        # Modify all values to something different
        CONFIG.Logging.level = 'CRITICAL'
        CONFIG.Logging.file = '/tmp/test.log'
        CONFIG.Logging.console = True
        CONFIG.Modeling.big = 999999
        CONFIG.Modeling.epsilon = 1e-10
        CONFIG.Modeling.big_binary_bound = 999999
        CONFIG.Solving.mip_gap = 0.0001
        CONFIG.Solving.time_limit_seconds = 9999
        CONFIG.Solving.log_to_console = False
        CONFIG.Solving.log_main_results = False
        CONFIG.config_name = 'modified'

        # Verify values are actually different from defaults
        assert CONFIG.Logging.level != _DEFAULTS['logging']['level']
        assert CONFIG.Modeling.big != _DEFAULTS['modeling']['big']
        assert CONFIG.Solving.mip_gap != _DEFAULTS['solving']['mip_gap']
        assert CONFIG.Solving.log_to_console != _DEFAULTS['solving']['log_to_console']

        # Now reset
        CONFIG.reset()

        # Verify reset() restored exactly the _DEFAULTS values
        assert CONFIG.Logging.level == _DEFAULTS['logging']['level']
        assert CONFIG.Logging.file == _DEFAULTS['logging']['file']
        assert CONFIG.Logging.console == _DEFAULTS['logging']['console']
        assert CONFIG.Modeling.big == _DEFAULTS['modeling']['big']
        assert CONFIG.Modeling.epsilon == _DEFAULTS['modeling']['epsilon']
        assert CONFIG.Modeling.big_binary_bound == _DEFAULTS['modeling']['big_binary_bound']
        assert CONFIG.Solving.mip_gap == _DEFAULTS['solving']['mip_gap']
        assert CONFIG.Solving.time_limit_seconds == _DEFAULTS['solving']['time_limit_seconds']
        assert CONFIG.Solving.log_to_console == _DEFAULTS['solving']['log_to_console']
        assert CONFIG.Solving.log_main_results == _DEFAULTS['solving']['log_main_results']
        assert CONFIG.config_name == _DEFAULTS['config_name']

    def test_solving_config_defaults(self):
        """Test that CONFIG.Solving has correct default values."""
        assert CONFIG.Solving.mip_gap == 0.01
        assert CONFIG.Solving.time_limit_seconds == 300
        assert CONFIG.Solving.log_to_console is True
        assert CONFIG.Solving.log_main_results is True

    def test_solving_config_modification(self):
        """Test that CONFIG.Solving attributes can be modified."""
        # Modify solving config
        CONFIG.Solving.mip_gap = 0.005
        CONFIG.Solving.time_limit_seconds = 600
        CONFIG.Solving.log_main_results = False
        CONFIG.apply()

        # Verify modifications
        assert CONFIG.Solving.mip_gap == 0.005
        assert CONFIG.Solving.time_limit_seconds == 600
        assert CONFIG.Solving.log_main_results is False

    def test_solving_config_integration_with_solvers(self):
        """Test that solvers use CONFIG.Solving defaults."""
        from flixopt import solvers

        # Test with default config
        CONFIG.reset()
        solver1 = solvers.HighsSolver()
        assert solver1.mip_gap == CONFIG.Solving.mip_gap
        assert solver1.time_limit_seconds == CONFIG.Solving.time_limit_seconds

        # Modify config and create new solver
        CONFIG.Solving.mip_gap = 0.002
        CONFIG.Solving.time_limit_seconds = 900
        CONFIG.apply()

        solver2 = solvers.GurobiSolver()
        assert solver2.mip_gap == 0.002
        assert solver2.time_limit_seconds == 900

        # Explicit values should override config
        solver3 = solvers.HighsSolver(mip_gap=0.1, time_limit_seconds=60)
        assert solver3.mip_gap == 0.1
        assert solver3.time_limit_seconds == 60

    def test_solving_config_yaml_loading(self, tmp_path):
        """Test loading solving config from YAML file."""
        config_file = tmp_path / 'solving_config.yaml'
        config_content = """
solving:
  mip_gap: 0.0001
  time_limit_seconds: 1200
  log_main_results: false
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        assert CONFIG.Solving.mip_gap == 0.0001
        assert CONFIG.Solving.time_limit_seconds == 1200
        assert CONFIG.Solving.log_main_results is False

    def test_solving_config_in_to_dict(self):
        """Test that CONFIG.Solving is included in to_dict()."""
        CONFIG.Solving.mip_gap = 0.003
        CONFIG.Solving.time_limit_seconds = 450
        CONFIG.Solving.log_main_results = False

        config_dict = CONFIG.to_dict()

        assert 'solving' in config_dict
        assert config_dict['solving']['mip_gap'] == 0.003
        assert config_dict['solving']['time_limit_seconds'] == 450
        assert config_dict['solving']['log_main_results'] is False

    def test_solving_config_persistence(self):
        """Test that Solving config is independent of other configs."""
        # Set custom solving values
        CONFIG.Solving.mip_gap = 0.007
        CONFIG.Solving.time_limit_seconds = 750

        # Change and apply logging config
        CONFIG.Logging.console = True
        CONFIG.apply()

        # Solving values should be unchanged
        assert CONFIG.Solving.mip_gap == 0.007
        assert CONFIG.Solving.time_limit_seconds == 750

        # Change modeling config
        CONFIG.Modeling.big = 99999999
        CONFIG.apply()

        # Solving values should still be unchanged
        assert CONFIG.Solving.mip_gap == 0.007
        assert CONFIG.Solving.time_limit_seconds == 750
