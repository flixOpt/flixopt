"""Smoke tests for plotting API robustness improvements."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt import plotting


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    time = np.arange(10)
    data = xr.Dataset(
        {
            'var1': (['time'], np.random.rand(10)),
            'var2': (['time'], np.random.rand(10)),
            'var3': (['time'], np.random.rand(10)),
        },
        coords={'time': time},
    )
    return data


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    time = np.arange(10)
    df = pd.DataFrame({'var1': np.random.rand(10), 'var2': np.random.rand(10), 'var3': np.random.rand(10)}, index=time)
    df.index.name = 'time'
    return df


def test_kwargs_passthrough_plotly(sample_dataset):
    """Test that backend-specific kwargs are passed through correctly."""
    fig = plotting.with_plotly(
        sample_dataset,
        mode='line',
        trace_kwargs={'line': {'width': 5}},
        layout_kwargs={'width': 1200, 'height': 600},
    )
    assert fig.layout.width == 1200
    assert fig.layout.height == 600


def test_dataframe_support_plotly(sample_dataframe):
    """Test that DataFrames are accepted by plotting functions."""
    fig = plotting.with_plotly(sample_dataframe, mode='line')
    assert fig is not None


def test_data_validation_non_numeric():
    """Test that validation catches non-numeric data."""
    data = xr.Dataset({'var1': (['time'], ['a', 'b', 'c'])}, coords={'time': [0, 1, 2]})

    with pytest.raises(TypeError, match='non-numeric dtype'):
        plotting.with_plotly(data)


def test_ensure_dataset_invalid_type():
    """Test that _ensure_dataset raises error for invalid types."""
    with pytest.raises(TypeError, match='must be xr.Dataset or pd.DataFrame'):
        plotting._ensure_dataset([1, 2, 3])
