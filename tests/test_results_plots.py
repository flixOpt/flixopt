import matplotlib.pyplot as plt
import pytest

import flixopt as fx

from .conftest import create_calculation_and_solve, simple_flow_system


@pytest.fixture(params=[True, False])
def show(request):
    return request.param


@pytest.fixture(params=[simple_flow_system])
def flow_system(request):
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture(params=[True, False])
def save(request):
    return request.param


@pytest.fixture(params=['plotly', 'matplotlib'])
def plotting_engine(request):
    return request.param


@pytest.fixture(
    params=[
        'viridis',  # Test string colormap
        ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'],  # Test color list
        {
            'Boiler(Q_th)|flow_rate': '#ff0000',
            'Heat Demand(Q_th)|flow_rate': '#00ff00',
            'Speicher(Q_th_load)|flow_rate': '#0000ff',
        },  # Test color dict
    ]
)
def color_spec(request):
    return request.param

@pytest.mark.slow
def test_results_plots(flow_system, plotting_engine, show, save, color_spec):
    calculation = create_calculation_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 30), 'test_results_plots')
    results = calculation.results

    results['Boiler'].plot_node_balance(engine=plotting_engine, save=save, show=show, colors=color_spec)

    results.plot_heatmap(
        'Speicher(Q_th_load)|flow_rate',
        heatmap_timeframes='D',
        heatmap_timesteps_per_frame='h',
        color_map='viridis',  # Note: heatmap only accepts string colormap
        save=show,
        show=save,
        engine=plotting_engine,
    )

    results['Speicher'].plot_node_balance_pie(engine=plotting_engine, save=save, show=show, colors=color_spec)

    if plotting_engine == 'matplotlib':
        with pytest.raises(NotImplementedError):
            results['Speicher'].plot_charge_state(engine=plotting_engine)
    else:
        results['Speicher'].plot_charge_state(engine=plotting_engine)

    plt.close('all')

@pytest.mark.slow
def test_color_handling_edge_cases(flow_system, plotting_engine, show, save):
    """Test edge cases for color handling"""
    calculation = create_calculation_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 30), 'test_color_edge_cases')
    results = calculation.results

    # Test with empty color list (should fall back to default)
    results['Boiler'].plot_node_balance(engine=plotting_engine, save=save, show=show, colors=[])

    # Test with invalid colormap name (should use default and log warning)
    results['Boiler'].plot_node_balance(engine=plotting_engine, save=save, show=show, colors='nonexistent_colormap')

    # Test with insufficient colors for elements (should cycle colors)
    results['Boiler'].plot_node_balance(engine=plotting_engine, save=save, show=show, colors=['#ff0000', '#00ff00'])

    # Test with color dict missing some elements (should use default for missing)
    partial_color_dict = {'Boiler(Q_th)|flow_rate': '#ff0000'}  # Missing other elements
    results['Boiler'].plot_node_balance(engine=plotting_engine, save=save, show=show, colors=partial_color_dict)

    plt.close('all')
