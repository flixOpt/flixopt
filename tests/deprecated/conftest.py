"""Fixtures and configuration for deprecated API tests.

This folder contains tests for the deprecated Optimization/Results API.
Delete this entire folder when deprecation cycle ends in v6.0.0.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Apply markers to all tests in this folder."""
    for item in items:
        # Only apply to tests in this folder
        if 'deprecated' in str(item.fspath):
            item.add_marker(pytest.mark.deprecated_api)
            item.add_marker(pytest.mark.filterwarnings('ignore::DeprecationWarning'))
