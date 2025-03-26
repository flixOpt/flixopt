import matplotlib.pyplot as plt
import pytest

import flixOpt as fx

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


def test_results_plots_matplotlib(flow_system, show, save):
    calculation = create_calculation_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 30), 'test_results_plots')
    results = calculation.results

    results['Boiler'].plot_node_balance(engine='matplotlib', save=save, show=show)

    results.plot_heatmap('Speicher(Q_th_load)|flow_rate',
                         heatmap_timeframes='D',
                         heatmap_timesteps_per_frame='h',
                         color_map='viridis',
                         save=show,
                         show=save,
                         engine='matplotlib')

    with pytest.raises(NotImplementedError):
        results['Speicher'].plot_charge_state(engine='matplotlib')

    with pytest.raises(NotImplementedError):
        results['Speicher'].plot_node_balance_pie(engine='matplotlib', save=save, show=show)
    plt.close('all')


def test_results_plots_plotly(flow_system, save, show):
    calculation = create_calculation_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 30), 'test_results_plots')
    results = calculation.results

    results['Boiler'].plot_node_balance(engine='plotly', save=save, show=show)

    results.plot_heatmap('Speicher(Q_th_load)|flow_rate',
                         heatmap_timeframes='D',
                         heatmap_timesteps_per_frame='h',
                         color_map='viridis',
                         save=show,
                         show=save,
                         engine='matplotlib')

    results['Speicher'].plot_charge_state(engine='plotly')

    results['Speicher'].plot_node_balance_pie(engine='plotly', save=save, show=show)


