"""Configuration for superseded math tests.

Enable legacy solution access for backward compatibility.
"""

import pytest

import flixopt as fx


@pytest.fixture(autouse=True)
def _enable_legacy_access():
    """Enable legacy solution access for all superseded math tests, then restore."""
    original = fx.CONFIG.Legacy.solution_access
    fx.CONFIG.Legacy.solution_access = True
    yield
    fx.CONFIG.Legacy.solution_access = original
