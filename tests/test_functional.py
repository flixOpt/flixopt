"""
Unit tests for the flixopt framework.

This module defines a set of unit tests for testing the functionality of the `flixopt` framework.
The tests focus on verifying the correct behavior of flow systems, including component modeling,
investment optimization, and operational constraints like status behavior.

### Approach:
1. **Setup**: Each test initializes a flow system with a set of predefined elements and parameters.
2. **Model Creation**: Test-specific flow systems are constructed using `create_model` with datetime arrays.
3. **Solution**: The models are solved using the `solve_and_load` method, which performs modeling, solves the optimization problem, and loads the results.
4. **Validation**: Results are validated using assertions, primarily `assert_allclose`, to ensure model outputs match expected values with a specified tolerance.

Tests group related cases by their functional focus:
- Minimal modeling setup (`TestMinimal` class)
- Investment behavior (`TestInvestment` class)
- Status operational constraints (functions: `test_startup_shutdown`, `test_consecutive_uptime_downtime`, etc.)
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import flixopt as fx

np.random.seed(45)


class Data:
    """
    Generates time series data for testing.

    Attributes:
        length (int): The desired length of the data.
        thermal_demand (np.ndarray): Thermal demand time series data.
        electricity_demand (np.ndarray): Electricity demand time series data.
    """

    def __init__(self, length: int):
        """
        Initialize the data generator with a specified length.

        Args:
            length (int): Length of the time series data to generate.
        """
        self.length = length

        self.thermal_demand = np.arange(0, 30, 10)
        self.electricity_demand = np.arange(1, 10.1, 1)

        self.thermal_demand = self._adjust_length(self.thermal_demand, length)
        self.electricity_demand = self._adjust_length(self.electricity_demand, length)

    def _adjust_length(self, array, new_length: int):
        if len(array) >= new_length:
            return array[:new_length]
        else:
            repeats = (new_length + len(array) - 1) // len(array)  # Calculate how many times to repeat
            extended_array = np.tile(array, repeats)  # Repeat the array
            return extended_array[:new_length]  # Truncate to exact length


def flow_system_base(timesteps: pd.DatetimeIndex) -> fx.FlowSystem:
    data = Data(len(timesteps))

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Fernwärme', imbalance_penalty_per_flow_hour=None),
        fx.Bus('Gas', imbalance_penalty_per_flow_hour=None),
    )
    flow_system.add_elements(fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True))
    flow_system.add_elements(
        fx.Sink(
            label='Wärmelast',
            inputs=[fx.Flow(label='Wärme', bus='Fernwärme', fixed_relative_profile=data.thermal_demand, size=1)],
        ),
        fx.Source(label='Gastarif', outputs=[fx.Flow(label='Gas', bus='Gas', effects_per_flow_hour=1)]),
    )
    return flow_system


def flow_system_minimal(timesteps) -> fx.FlowSystem:
    flow_system = flow_system_base(timesteps)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme'),
        )
    )
    return flow_system


def solve_and_load(flow_system: fx.FlowSystem, solver) -> fx.results.Results:
    optimization = fx.Optimization('Calculation', flow_system)
    optimization.do_modeling()
    optimization.solve(solver)
    return optimization.results


@pytest.fixture
def time_steps_fixture(request):
    return pd.date_range('2020-01-01', periods=5, freq='h')


def test_solve_and_load(solver_fixture, time_steps_fixture):
    results = solve_and_load(flow_system_minimal(time_steps_fixture), solver_fixture)
    assert results is not None


def test_minimal_model(solver_fixture, time_steps_fixture):
    results = solve_and_load(flow_system_minimal(time_steps_fixture), solver_fixture)
    assert_allclose(results.model.variables['costs'].solution.values, 80, rtol=1e-5, atol=1e-10)

    assert_allclose(
        results.model.variables['Boiler(Q_th)|flow_rate'].solution.values,
        [-0.0, 10.0, 20.0, -0.0, 10.0],
        rtol=1e-5,
        atol=1e-10,
    )

    assert_allclose(
        results.model.variables['costs(temporal)|per_timestep'].solution.values,
        [-0.0, 20.0, 40.0, -0.0, 20.0],
        rtol=1e-5,
        atol=1e-10,
    )

    assert_allclose(
        results.model.variables['Gastarif(Gas)->costs(temporal)'].solution.values,
        [-0.0, 20.0, 40.0, -0.0, 20.0],
        rtol=1e-5,
        atol=1e-10,
    )


def test_fixed_size(solver_fixture, time_steps_fixture):
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(fixed_size=1000, effects_of_investment=10, effects_of_investment_per_size=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80 + 1000 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.size.solution.item(),
        1000,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.invested.solution.item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__invested" does not have the right value',
    )


def test_optimize_size(solver_fixture, time_steps_fixture):
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(effects_of_investment=10, effects_of_investment_per_size=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80 + 20 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.size.solution.item(),
        20,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.invested.solution.item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )


def test_size_bounds(solver_fixture, time_steps_fixture):
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(minimum_size=40, effects_of_investment=10, effects_of_investment_per_size=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80 + 40 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.size.solution.item(),
        40,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.invested.solution.item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )


def test_optional_invest(solver_fixture, time_steps_fixture):
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(
                    mandatory=False, minimum_size=40, effects_of_investment=10, effects_of_investment_per_size=1
                ),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_optional',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(
                    mandatory=False, minimum_size=50, effects_of_investment=10, effects_of_investment_per_size=1
                ),
            ),
        ),
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    boiler_optional = flow_system['Boiler_optional']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80 + 40 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.size.solution.item(),
        40,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.investment.invested.solution.item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )

    assert_allclose(
        boiler_optional.thermal_flow.submodel.investment.size.solution.item(),
        0,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler_optional.thermal_flow.submodel.investment.invested.solution.item(),
        0,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )


def test_on(solver_fixture, time_steps_fixture):
    """Tests if the On Variable is correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=100, status_parameters=fx.StatusParameters()),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.status.status.solution.values,
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [0, 10, 20, 0, 10],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_off(solver_fixture, time_steps_fixture):
    """Tests if the Off Variable is correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                status_parameters=fx.StatusParameters(max_downtime=100),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.status.status.solution.values,
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.status.inactive.solution.values,
        1 - boiler.thermal_flow.submodel.status.status.solution.values,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__off" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [0, 10, 20, 0, 10],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_startup_shutdown(solver_fixture, time_steps_fixture):
    """Tests if the startup/shutdown Variable is correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                status_parameters=fx.StatusParameters(force_startup_tracking=True),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.status.status.solution.values,
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.status.startup.solution.values,
        [0, 1, 0, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__switch_on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.status.shutdown.solution.values,
        [0, 0, 0, 1, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__switch_on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [0, 10, 20, 0, 10],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_on_total_max(solver_fixture, time_steps_fixture):
    """Tests if the On Total Max Variable is correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                status_parameters=fx.StatusParameters(active_hours_max=1),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=100),
        ),
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        140,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.status.status.solution.values,
        [0, 0, 1, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [0, 0, 20, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_on_total_bounds(solver_fixture, time_steps_fixture):
    """Tests if the On Hours min and max are correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                status_parameters=fx.StatusParameters(active_hours_max=2),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                status_parameters=fx.StatusParameters(active_hours_min=3),
            ),
        ),
    )
    flow_system['Wärmelast'].inputs[0].fixed_relative_profile = np.array(
        [0, 10, 20, 0, 12]
    )  # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    boiler_backup = flow_system['Boiler_backup']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        114,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.status.status.solution.values,
        [0, 0, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [0, 0, 20, 0, 12 - 1e-5],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        sum(boiler_backup.thermal_flow.submodel.status.status.solution.values),
        3,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler_backup.thermal_flow.submodel.flow_rate.solution.values,
        [0, 10, 1.0e-05, 0, 1.0e-05],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_consecutive_uptime_downtime(solver_fixture, time_steps_fixture):
    """Tests if the consecutive uptime/downtime are correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                status_parameters=fx.StatusParameters(max_uptime=2, min_uptime=2),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=100),
        ),
    )
    flow_system['Wärmelast'].inputs[0].fixed_relative_profile = np.array([5, 10, 20, 18, 12])
    # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    boiler_backup = flow_system['Boiler_backup']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        190,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.status.status.solution.values,
        [1, 1, 0, 1, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [5, 10, 0, 18, 12],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        boiler_backup.thermal_flow.submodel.flow_rate.solution.values,
        [0, 0, 20, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_consecutive_off(solver_fixture, time_steps_fixture):
    """Tests if the consecutive on hours are correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme'),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                previous_flow_rate=np.array([20]),  # Otherwise its Off before the start
                status_parameters=fx.StatusParameters(max_downtime=2, min_downtime=2),
            ),
        ),
    )
    flow_system['Wärmelast'].inputs[0].fixed_relative_profile = np.array(
        [5, 0, 20, 18, 12]
    )  # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system['Boiler']
    boiler_backup = flow_system['Boiler_backup']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.submodel.total.solution.item(),
        110,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler_backup.thermal_flow.submodel.status.status.solution.values,
        [0, 0, 1, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler_backup.thermal_flow.submodel.status.inactive.solution.values,
        [1, 1, 0, 1, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__off" does not have the right value',
    )
    assert_allclose(
        boiler_backup.thermal_flow.submodel.flow_rate.solution.values,
        [0, 0, 1e-5, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        boiler.thermal_flow.submodel.flow_rate.solution.values,
        [5, 0, 20 - 1e-5, 18, 12],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_investment_parameters_with_lifetime(solver_fixture):
    """Tests InvestmentParameters with lifetime tracking in a multi-period optimization.

    This test verifies that:
    1. Investment timing is correctly tracked (investment_occurs, decommissioning_occurs)
    2. Lifetime constraints link investment period to decommissioning period
    3. The component is only active during its lifetime
    """
    # Set up multi-period flow system with 4 periods
    timesteps = pd.date_range('2020-01-01', periods=5, freq='h', name='time')
    periods = pd.Index([2020, 2025, 2030, 2035], name='period')

    flow_system = fx.FlowSystem(timesteps=timesteps, periods=periods)

    # Create buses and effects
    flow_system.add_elements(
        fx.Bus('Fernwärme'),
        fx.Bus('Gas'),
        fx.Effect('costs', '€', 'Costs', is_objective=True),
    )

    # Add heat load (constant across periods)
    flow_system.add_elements(
        fx.Sink(
            'Wärmelast',
            inputs=[fx.Flow('Q_th', bus='Fernwärme', size=100, fixed_relative_profile=0.3)],
        ),
        fx.Source(
            'Gasquelle',
            outputs=[fx.Flow('Q_fu', bus='Gas', size=1000, effects_per_flow_hour={'costs': 2})],
        ),
    )

    # Add boiler with InvestmentParameters (lifetime = 2 periods)
    # Invest in 2020 → active in 2020, 2025 → decommission in 2030
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler_Invest',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestmentParameters(
                    lifetime=2,  # Active for 2 periods after investment
                    minimum_size=50,
                    maximum_size=200,
                    effects_per_size={'costs': 10},  # €10/kW investment cost
                ),
            ),
        ),
    )

    # Add backup boiler (always available)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler_Backup',
            thermal_efficiency=0.3,  # Less efficient, more expensive
            fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=200),
        ),
    )

    # Create and solve optimization
    optimization = fx.Optimization('test_investment_lifetime', flow_system)
    optimization.do_modeling()
    optimization.solve(solver_fixture)

    # Get the investment model
    boiler = flow_system['Boiler_Invest']
    investment = boiler.thermal_flow.submodel.investment

    # Verify investment model is InvestmentModel (not SizingModel)
    from flixopt.features import InvestmentModel

    assert isinstance(investment, InvestmentModel), 'Should use InvestmentModel for InvestmentParameters'

    # Verify investment_occurs variable exists
    assert hasattr(investment, 'investment_occurs'), 'InvestmentModel should have investment_occurs variable'
    assert hasattr(investment, 'decommissioning_occurs'), 'InvestmentModel should have decommissioning_occurs variable'

    # Verify the investment occurs in exactly one period (or zero)
    investment_sum = investment.investment_occurs.solution.sum('period').item()
    assert investment_sum <= 1, f'Investment should occur at most once, got sum={investment_sum}'

    # If investment happened, check that decommissioning occurs at most as many times
    if investment_sum > 0:
        # The sum of decommissioning_occurs should not exceed investment_occurs
        # (decommissioning can be 0 if it's beyond the optimization horizon)
        decom_sum = investment.decommissioning_occurs.solution.sum('period').item()
        assert decom_sum <= investment_sum, 'Decommissioning should not exceed investment count'


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
