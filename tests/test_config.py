"""Tests for the config module."""

import logging
import sys
from pathlib import Path

import pytest

from flixopt.config import CONFIG, _setup_logging


class TestConfigModule:
    """Test the CONFIG class and logging setup."""

    def setup_method(self):
        """Reset CONFIG to defaults before each test."""
        CONFIG.Logging.level = 'INFO'
        CONFIG.Logging.file = None
        CONFIG.Logging.rich = False
        CONFIG.Logging.console = False

        # Clear and reset logger completely
        logger = logging.getLogger('flixopt')
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Apply clean state
        CONFIG.apply()

    def test_config_defaults(self):
        """Test that CONFIG has correct default values."""
        assert CONFIG.Logging.level == 'INFO'
        assert CONFIG.Logging.file is None
        assert CONFIG.Logging.rich is False
        assert CONFIG.Logging.console is False
        assert CONFIG.Modeling.big == 10_000_000
        assert CONFIG.Modeling.epsilon == 1e-5
        assert CONFIG.Modeling.big_binary_bound == 100_000
        assert CONFIG.config_name == 'flixopt'

    def test_module_initialization(self):
        """Test that logging is initialized on module import."""
        # Apply config to ensure handlers are initialized
        CONFIG.apply()
        logger = logging.getLogger('flixopt')
        # Should have at least one handler (NullHandler by default)
        assert len(logger.handlers) >= 1
        # Should have NullHandler when no console/file output is configured
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_config_apply_console(self):
        """Test applying config with console logging enabled."""
        CONFIG.Logging.console = True
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.apply()

        logger = logging.getLogger('flixopt')
        assert logger.level == logging.DEBUG
        # Should have a StreamHandler for console output
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        # Should not have NullHandler when console is enabled
        assert not any(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_config_apply_file(self, tmp_path):
        """Test applying config with file logging enabled."""
        log_file = tmp_path / 'test.log'
        CONFIG.Logging.file = str(log_file)
        CONFIG.Logging.level = 'WARNING'
        CONFIG.apply()

        logger = logging.getLogger('flixopt')
        assert logger.level == logging.WARNING
        # Should have a RotatingFileHandler for file output
        from logging.handlers import RotatingFileHandler

        assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)

    def test_config_apply_rich(self):
        """Test applying config with rich logging enabled."""
        CONFIG.Logging.console = True
        CONFIG.Logging.rich = True
        CONFIG.apply()

        logger = logging.getLogger('flixopt')
        # Should have a RichHandler
        from rich.logging import RichHandler

        assert any(isinstance(h, RichHandler) for h in logger.handlers)

    def test_config_apply_multiple_changes(self):
        """Test applying multiple config changes at once."""
        CONFIG.Logging.console = True
        CONFIG.Logging.level = 'ERROR'
        CONFIG.apply()

        logger = logging.getLogger('flixopt')
        assert logger.level == logging.ERROR
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_config_to_dict(self):
        """Test converting CONFIG to dictionary."""
        CONFIG.Logging.level = 'DEBUG'
        CONFIG.Logging.console = True

        config_dict = CONFIG.to_dict()

        assert config_dict['config_name'] == 'flixopt'
        assert config_dict['logging']['level'] == 'DEBUG'
        assert config_dict['logging']['console'] is True
        assert config_dict['logging']['file'] is None
        assert config_dict['logging']['rich'] is False
        assert 'modeling' in config_dict
        assert config_dict['modeling']['big'] == 10_000_000

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
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        assert CONFIG.config_name == 'test_config'
        assert CONFIG.Logging.level == 'DEBUG'
        assert CONFIG.Logging.console is True
        assert CONFIG.Modeling.big == 20000000
        # YAML may load epsilon as string, so convert for comparison
        assert float(CONFIG.Modeling.epsilon) == 1e-6

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

    def test_setup_logging_silent_default(self):
        """Test that _setup_logging creates silent logger by default."""
        _setup_logging()

        logger = logging.getLogger('flixopt')
        # Should have NullHandler when console=False and log_file=None
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
        assert not logger.propagate

    def test_setup_logging_with_console(self):
        """Test _setup_logging with console output."""
        _setup_logging(console=True, default_level='DEBUG')

        logger = logging.getLogger('flixopt')
        assert logger.level == logging.DEBUG
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_setup_logging_clears_handlers(self):
        """Test that _setup_logging clears existing handlers."""
        logger = logging.getLogger('flixopt')

        # Add a dummy handler
        dummy_handler = logging.NullHandler()
        logger.addHandler(dummy_handler)
        _ = len(logger.handlers)

        _setup_logging(console=True)

        # Should have cleared old handlers and added new one
        assert dummy_handler not in logger.handlers

    def test_change_logging_level_removed(self):
        """Test that change_logging_level function no longer exists."""
        # This function was removed - users should use CONFIG.apply() instead
        import flixopt

        assert not hasattr(flixopt, 'change_logging_level')

    def test_public_api(self):
        """Test that only CONFIG is exported from config module."""
        from flixopt import config

        # CONFIG should be accessible
        assert hasattr(config, 'CONFIG')

        # _setup_logging should exist but be marked as private
        assert hasattr(config, '_setup_logging')

        # merge_configs should not exist (was removed)
        assert not hasattr(config, 'merge_configs')

    def test_logging_levels(self):
        """Test all valid logging levels."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        for level in levels:
            CONFIG.Logging.level = level
            CONFIG.Logging.console = True
            CONFIG.apply()

            logger = logging.getLogger('flixopt')
            assert logger.level == getattr(logging, level)

    def test_logger_propagate_disabled(self):
        """Test that logger propagation is disabled."""
        CONFIG.apply()
        logger = logging.getLogger('flixopt')
        assert not logger.propagate

    def test_file_handler_rotation(self, tmp_path):
        """Test that file handler uses rotation."""
        log_file = tmp_path / 'rotating.log'
        CONFIG.Logging.file = str(log_file)
        CONFIG.apply()

        logger = logging.getLogger('flixopt')
        from logging.handlers import RotatingFileHandler

        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1

        handler = file_handlers[0]
        # Check rotation settings
        assert handler.maxBytes == 10_485_760  # 10MB
        assert handler.backupCount == 5

    def test_custom_config_yaml_complete(self, tmp_path):
        """Test loading a complete custom configuration."""
        config_file = tmp_path / 'custom_config.yaml'
        config_content = """
config_name: my_custom_config
logging:
  level: CRITICAL
  console: true
  rich: true
  file: /tmp/custom.log
modeling:
  big: 50000000
  epsilon: 1e-4
  big_binary_bound: 200000
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        # Check all settings were applied
        assert CONFIG.config_name == 'my_custom_config'
        assert CONFIG.Logging.level == 'CRITICAL'
        assert CONFIG.Logging.console is True
        assert CONFIG.Logging.rich is True
        assert CONFIG.Logging.file == '/tmp/custom.log'
        assert CONFIG.Modeling.big == 50000000
        assert float(CONFIG.Modeling.epsilon) == 1e-4
        assert CONFIG.Modeling.big_binary_bound == 200000

        # Verify logging was applied
        logger = logging.getLogger('flixopt')
        assert logger.level == logging.CRITICAL

    def test_config_file_with_console_and_file(self, tmp_path):
        """Test configuration with both console and file logging enabled."""
        log_file = tmp_path / 'test.log'
        config_file = tmp_path / 'config.yaml'
        config_content = f"""
logging:
  level: INFO
  console: true
  rich: false
  file: {log_file}
"""
        config_file.write_text(config_content)

        CONFIG.load_from_file(config_file)

        logger = logging.getLogger('flixopt')
        # Should have both StreamHandler and RotatingFileHandler
        from logging.handlers import RotatingFileHandler

        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        # Should NOT have NullHandler when console/file are enabled
        assert not any(isinstance(h, logging.NullHandler) for h in logger.handlers)

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

        logger = logging.getLogger('flixopt')
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

        # Reset for other tests
        CONFIG.Modeling.big = 10_000_000
        CONFIG.Modeling.epsilon = 1e-5
