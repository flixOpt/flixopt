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


@pytest.fixture(params=['plotly', 'matplotlib'])
def plotting_engine(request):
    return request.param


def test_results_plots(flow_system, plotting_engine, show, save):
    calculation = create_calculation_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 30), 'test_results_plots')
    results = calculation.results

    results['Boiler'].plot_node_balance(engine=plotting_engine, save=save, show=show)

    results.plot_heatmap('Speicher(Q_th_load)|flow_rate',
                         heatmap_timeframes='D',
                         heatmap_timesteps_per_frame='h',
                         color_map='viridis',
                         save=show,
                         show=save,
                         engine=plotting_engine)

    results['Speicher'].plot_node_balance_pie(engine=plotting_engine, save=save, show=show)

    if plotting_engine == 'matplotlib':
        with pytest.raises(NotImplementedError):
            results['Speicher'].plot_charge_state(engine=plotting_engine)
    else:
        results['Speicher'].plot_charge_state(engine=plotting_engine)

    plt.close('all')
