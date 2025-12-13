"""Smoke tests for plotting API robustness improvements."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt import plotting


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    rng = np.random.default_rng(0)
    time = np.arange(10)
    data = xr.Dataset(
        {
            'var1': (['time'], rng.random(10)),
            'var2': (['time'], rng.random(10)),
            'var3': (['time'], rng.random(10)),
        },
        coords={'time': time},
    )
    return data


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    rng = np.random.default_rng(1)
    time = np.arange(10)
    df = pd.DataFrame({'var1': rng.random(10), 'var2': rng.random(10), 'var3': rng.random(10)}, index=time)
    df.index.name = 'time'
    return df


def test_kwargs_passthrough_plotly(sample_dataset):
    """Test that px_kwargs are passed through and figure can be customized after creation."""
    # Test that px_kwargs are passed through
    fig = plotting.with_plotly(
        sample_dataset,
        mode='line',
        range_y=[0, 100],
    )
    assert list(fig.layout.yaxis.range) == [0, 100]

    # Test that figure can be customized after creation
    fig.update_traces(line={'width': 5})
    fig.update_layout(width=1200, height=600)
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
    'engine,mode,data_type',
    [
        *[
            (e, m, dt)
            for e in ['plotly', 'matplotlib']
            for m in ['stacked_bar', 'line', 'area', 'grouped_bar']
            for dt in ['dataset', 'dataframe', 'series']
            if not (e == 'matplotlib' and m in ['area', 'grouped_bar'])
        ],
    ],
)
def test_all_data_types_and_modes(engine, mode, data_type):
    """Test that Dataset, DataFrame, and Series work with all plotting modes."""
    time = pd.date_range('2020-01-01', periods=5, freq='h')

    data = {
        'dataset': xr.Dataset(
            {'A': (['time'], [1, 2, 3, 4, 5]), 'B': (['time'], [5, 4, 3, 2, 1])}, coords={'time': time}
        ),
        'dataframe': pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}, index=time),
        'series': pd.Series([1, 2, 3, 4, 5], index=time, name='A'),
    }[data_type]

    if engine == 'plotly':
        fig = plotting.with_plotly(data, mode=mode)
        assert fig is not None and len(fig.data) > 0
    else:
        fig, ax = plotting.with_matplotlib(data, mode=mode)
        assert fig is not None and ax is not None


@pytest.mark.parametrize(
    'engine,data_type', [(e, dt) for e in ['plotly', 'matplotlib'] for dt in ['dataset', 'dataframe', 'series']]
)
def test_pie_plots(engine, data_type):
    """Test pie charts with all data types, including automatic summing."""
    time = pd.date_range('2020-01-01', periods=5, freq='h')

    # Single-value data
    single_data = {
        'dataset': xr.Dataset({'A': xr.DataArray(10), 'B': xr.DataArray(20), 'C': xr.DataArray(30)}),
        'dataframe': pd.DataFrame({'A': [10], 'B': [20], 'C': [30]}),
        'series': pd.Series({'A': 10, 'B': 20, 'C': 30}),
    }[data_type]

    # Multi-dimensional data (for summing test)
    multi_data = {
        'dataset': xr.Dataset(
            {'A': (['time'], [1, 2, 3, 4, 5]), 'B': (['time'], [5, 5, 5, 5, 5])}, coords={'time': time}
        ),
        'dataframe': pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 5, 5, 5, 5]}, index=time),
        'series': pd.Series([1, 2, 3, 4, 5], index=time, name='A'),
    }[data_type]

    for data in [single_data, multi_data]:
        if engine == 'plotly':
            fig = plotting.dual_pie_with_plotly(data, data)
            assert fig is not None and len(fig.data) >= 2
            if data is multi_data and data_type != 'series':
                assert sum(fig.data[0].values) == pytest.approx(40)
        else:
            fig, axes = plotting.dual_pie_with_matplotlib(data, data)
            assert fig is not None and len(axes) == 2
