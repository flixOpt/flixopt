"""
Run this script to run all tests with pytest
Alternatively, use 'python -m unittest discover -s tests' in the command line (using unittest)
or run testmodules individually
"""

import pytest

if __name__ == '__main__':
    pytest.main(['test_integration.py', '--disable-warnings'])
