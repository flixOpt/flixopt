"""
This module contains several utility functions used throughout the flixopt framework.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import xarray as xr

logger = logging.getLogger('flixopt')


@contextmanager
def suppress_output():
    """Redirect both Python and C-level stdout/stderr to os.devnull."""
    with open(os.devnull, 'w') as devnull:
        # Save original file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            # Flush any pending text
            sys.stdout.flush()
            sys.stderr.flush()
            # Redirect low-level fds to devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            # Restore fds
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
