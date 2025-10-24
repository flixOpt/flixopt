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
    assert all(getattr(t, 'line', None) and t.line.width == 5 for t in fig.data)


def test_dataframe_support_plotly(sample_dataframe):
    """Test that DataFrames are accepted by plotting functions."""
    fig = plotting.with_plotly(sample_dataframe, mode='line')
    assert fig is not None


def test_data_validation_non_numeric():
    """Test that validation catches non-numeric data."""
    data = xr.Dataset({'var1': (['time'], ['a', 'b', 'c'])}, coords={'time': [0, 1, 2]})

    with pytest.raises(TypeError, match='non-?numeric'):
        plotting.with_plotly(data)


def test_ensure_dataset_invalid_type():
    """Test that invalid types raise error via the public API."""
    with pytest.raises(TypeError, match='xr\\.Dataset|pd\\.DataFrame'):
        plotting.with_plotly([1, 2, 3], mode='line')


@pytest.mark.parametrize(
    'engine,mode',
    [
        ('plotly', 'stacked_bar'),
        ('plotly', 'line'),
        ('plotly', 'area'),
        ('plotly', 'grouped_bar'),
        ('matplotlib', 'stacked_bar'),
        ('matplotlib', 'line'),
    ],
)
def test_all_data_types_and_modes(engine, mode):
    """Test that Dataset, DataFrame, and Series work with all plotting modes."""
    time = pd.date_range('2020-01-01', periods=5, freq='h')

    # Create Dataset
    dataset = xr.Dataset({'A': (['time'], [1, 2, 3, 4, 5]), 'B': (['time'], [5, 4, 3, 2, 1])}, coords={'time': time})

    # Create DataFrame
    dataframe = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}, index=time)

    # Create Series
    series = pd.Series([1, 2, 3, 4, 5], index=time, name='A')

    # Test all three data types with the specified mode
    for data in [dataset, dataframe, series]:
        if engine == 'plotly':
            fig = plotting.with_plotly(data, mode=mode)
            assert fig is not None
            assert len(fig.data) > 0
        else:
            fig, ax = plotting.with_matplotlib(data, mode=mode)
            assert fig is not None
            assert ax is not None


@pytest.mark.parametrize('engine', ['plotly', 'matplotlib'])
def test_pie_plots_all_data_types(engine):
    """Test that dual pie charts work with Dataset, DataFrame, and Series."""
    # Create data for pie charts (summed values)
    dataset = xr.Dataset({'A': xr.DataArray(10), 'B': xr.DataArray(20), 'C': xr.DataArray(30)})
    dataframe = pd.DataFrame({'A': [10], 'B': [20], 'C': [30]})
    series = pd.Series({'A': 10, 'B': 20, 'C': 30})

    # Test all three data types
    for data_left, data_right in [(dataset, dataset), (dataframe, dataframe), (series, series)]:
        if engine == 'plotly':
            fig = plotting.dual_pie_with_plotly(data_left, data_right)
            assert fig is not None
            assert len(fig.data) >= 2  # At least 2 pie charts
        else:
            fig, axes = plotting.dual_pie_with_matplotlib(data_left, data_right)
            assert fig is not None
            assert len(axes) == 2  # Two pie charts side by side
