"""Minimal test for deprecation warnings with v5.0.0 removal message."""

import warnings

import pandas as pd

import flixopt as fx
from flixopt.linear_converters import Boiler

warnings.simplefilter('always', DeprecationWarning)


def check_warning(func, expected_substr='Will be removed in v5.0.0'):
    """Helper to check if warning contains removal version."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        func()
        assert len(w) > 0, 'No warning raised!'
        assert expected_substr in str(w[0].message), f"Missing '{expected_substr}' in: {w[0].message}"
        return str(w[0].message)


# Test cases
ts = pd.date_range('2023-01-01', periods=24, freq='h')
fs = fx.FlowSystem(ts)
bus = fx.Bus('bus')
fs.add_elements(bus)

print('Testing deprecation warnings...\n')

# 1. _handle_deprecated_kwarg (via Source)
msg = check_warning(lambda: fx.Source('s', source=fx.Flow('out', bus='bus', size=10)))
print(f"✓ Source 'source' param: {msg[:60]}...")

# 2. Property (TimeSeriesData.agg_group)
data = fx.TimeSeriesData([1, 2, 3], aggregation_group=1)
msg = check_warning(lambda: data.agg_group)
print(f'✓ TimeSeriesData.agg_group: {msg[:60]}...')

# 3. Linear converter property (Boiler.eta)
boiler = Boiler('b', fuel_flow=fx.Flow('f', 'bus', 10), thermal_flow=fx.Flow('h', 'bus', 9), thermal_efficiency=0.9)
msg = check_warning(lambda: boiler.eta)
print(f'✓ Boiler.eta: {msg[:60]}...')

# 4. InvestParameters 'optional' param
msg = check_warning(lambda: fx.InvestParameters(minimum_size=10, optional=True))
print(f"✓ InvestParameters 'optional': {msg[:60]}...")

print("\n✓ All deprecation warnings include 'Will be removed in v5.0.0'")
