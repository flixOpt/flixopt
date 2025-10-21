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


@pytest.mark.parametrize('engine', ['plotly', 'matplotlib'])
def test_kwargs_passthrough(sample_dataset, engine):
    """Test that backend-specific kwargs are passed through correctly."""
    if engine == 'plotly':
        # Test with_plotly kwargs
        fig = plotting.with_plotly(
            sample_dataset,
            mode='line',
            trace_kwargs={'line': {'width': 5}},
            layout_kwargs={'width': 1200, 'height': 600},
        )
        assert fig.layout.width == 1200
        assert fig.layout.height == 600

    elif engine == 'matplotlib':
        # Test with_matplotlib kwargs
        fig, ax = plotting.with_matplotlib(sample_dataset, mode='line', plot_kwargs={'linewidth': 3, 'alpha': 0.7})
        # Verify that the plot was created (basic smoke test)
        assert fig is not None
        assert ax is not None


def test_dataframe_support_plotly(sample_dataframe):
    """Test that DataFrames are accepted by plotting functions."""
    # Should not raise an error
    fig = plotting.with_plotly(sample_dataframe, mode='line')
    assert fig is not None


def test_dataframe_support_matplotlib(sample_dataframe):
    """Test that DataFrames are accepted by matplotlib plotting functions."""
    # Should not raise an error
    fig, ax = plotting.with_matplotlib(sample_dataframe, mode='line')
    assert fig is not None
    assert ax is not None


def test_heatmap_vmin_vmax():
    """Test that vmin/vmax parameters work for heatmaps."""
    data = xr.DataArray(np.random.rand(10, 10), dims=['x', 'y'])

    fig, ax = plotting.heatmap_with_matplotlib(data, vmin=0.2, vmax=0.8)
    assert fig is not None
    assert ax is not None

    # Check that the image has the correct vmin/vmax
    images = [child for child in ax.get_children() if hasattr(child, 'get_clim')]
    if images:
        vmin, vmax = images[0].get_clim()
        assert vmin == 0.2
        assert vmax == 0.8


def test_heatmap_imshow_kwargs():
    """Test that imshow_kwargs are passed to imshow."""
    data = xr.DataArray(np.random.rand(10, 10), dims=['x', 'y'])

    fig, ax = plotting.heatmap_with_matplotlib(data, imshow_kwargs={'interpolation': 'nearest', 'aspect': 'equal'})
    assert fig is not None
    assert ax is not None


def test_pie_text_customization():
    """Test that pie chart text customization parameters work."""
    data = xr.Dataset({'var1': 10, 'var2': 20, 'var3': 30})

    fig = plotting.pie_with_plotly(
        data, text_info='percent', text_position='outside', hover_template='Custom: %{label} = %{value}'
    )
    assert fig is not None

    # Check that the trace has the correct parameters
    assert fig.data[0].textinfo == 'percent'
    assert fig.data[0].textposition == 'outside'
    assert fig.data[0].hovertemplate == 'Custom: %{label} = %{value}'


def test_data_validation_non_numeric():
    """Test that validation catches non-numeric data."""
    # Create dataset with non-numeric data
    data = xr.Dataset({'var1': (['time'], ['a', 'b', 'c'])}, coords={'time': [0, 1, 2]})

    with pytest.raises(TypeError, match='non-numeric dtype'):
        plotting.with_plotly(data)


def test_data_validation_nan_handling(sample_dataset):
    """Test that validation handles NaN values without raising an error."""
    # Add NaN to the dataset
    data = sample_dataset.copy()
    data['var1'].values[0] = np.nan

    # Should not raise an error (warning is logged but we can't easily test that)
    fig = plotting.with_plotly(data)
    assert fig is not None


def test_export_figure_dpi(sample_dataset, tmp_path):
    """Test that DPI parameter works for export_figure."""
    import matplotlib.pyplot as plt

    fig, ax = plotting.with_matplotlib(sample_dataset, mode='line')

    output_path = tmp_path / 'test_plot.png'
    plotting.export_figure((fig, ax), default_path=output_path, save=True, show=False, dpi=150)

    assert output_path.exists()
    plt.close(fig)


def test_ensure_dataset_invalid_type():
    """Test that _ensure_dataset raises error for invalid types."""
    with pytest.raises(TypeError, match='must be xr.Dataset or pd.DataFrame'):
        plotting._ensure_dataset([1, 2, 3])  # List is not valid


def test_validate_plotting_data_empty():
    """Test that validation handles empty datasets appropriately."""
    empty_data = xr.Dataset()

    # Should raise ValueError when allow_empty=False
    with pytest.raises(ValueError, match='Empty Dataset'):
        plotting._validate_plotting_data(empty_data, allow_empty=False)

    # Should not raise when allow_empty=True
    plotting._validate_plotting_data(empty_data, allow_empty=True)
