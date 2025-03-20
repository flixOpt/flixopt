"""
Unit tests for the flixOpt framework.

This module defines a set of unit tests for testing the functionality of the `flixOpt` framework.
The tests focus on verifying the correct behavior of flow systems, including component modeling,
investment optimization, and operational constraints like on-off behavior.

### Approach:
1. **Setup**: Each test initializes a flow system with a set of predefined elements and parameters.
2. **Model Creation**: Test-specific flow systems are constructed using `create_model` with datetime arrays.
3. **Solution**: The models are solved using the `solve_and_load` method, which performs modeling, solves the optimization problem, and loads the results.
4. **Validation**: Results are validated using assertions, primarily `assert_allclose`, to ensure model outputs match expected values with a specified tolerance.

Classes group related test cases by their functional focus:
- Minimal modeling setup (`TestMinimal`)
- Investment behavior (`TestInvestment`)
- On-off operational constraints (`TestOnOff`).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import flixOpt as fx

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
        fx.Bus('Fernwärme', excess_penalty_per_flow_hour=None),
        fx.Bus('Gas', excess_penalty_per_flow_hour=None),
    )
    flow_system.add_elements(fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True))
    flow_system.add_elements(
        fx.Sink(
            label='Wärmelast',
            sink=fx.Flow(label='Wärme', bus='Fernwärme', fixed_relative_profile=data.thermal_demand, size=1),
        ),
        fx.Source(label='Gastarif', source=fx.Flow(label='Gas', bus='Gas', effects_per_flow_hour=1)),
    )
    return flow_system


def flow_system_minimal(timesteps) -> fx.FlowSystem:
    flow_system = flow_system_base(timesteps)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme'),
        )
    )
    return flow_system


def solve_and_load(
    flow_system: fx.FlowSystem, solver
) -> fx.results.CalculationResults:
    calculation = fx.FullCalculation('Calculation', flow_system)
    calculation.do_modeling()
    calculation.solve(solver)
    return calculation.results


@pytest.fixture
def time_steps_fixture(request):
    return pd.date_range('2020-01-01', periods=5, freq='h')


def test_solve_and_load(solver_fixture, time_steps_fixture):
    results = solve_and_load(flow_system_minimal(time_steps_fixture), solver_fixture)
    assert results is not None


def test_minimal_model(solver_fixture, time_steps_fixture):
    results = solve_and_load(flow_system_minimal(time_steps_fixture), solver_fixture)
    assert_allclose(results.model.variables['costs|total'].solution.values, 80, rtol=1e-5, atol=1e-10)

    assert_allclose(
        results.model.variables['Boiler(Q_th)|flow_rate'].solution.values,
        [-0.0, 10.0, 20.0, -0.0, 10.0],
        rtol=1e-5,
        atol=1e-10,
    )

    assert_allclose(
        results.model.variables['costs(operation)|total_per_timestep'].solution.values,
        [-0.0, 20.0, 40.0, -0.0, 20.0],
        rtol=1e-5,
        atol=1e-10,
    )

    assert_allclose(
        results.model.variables['Gastarif(Gas)->costs(operation)'].solution.values,
        [-0.0, 20.0, 40.0, -0.0, 20.0],
        rtol=1e-5,
        atol=1e-10,
    )


def test_fixed_size(solver_fixture, time_steps_fixture):
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(fixed_size=1000, fix_effects=10, specific_effects=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80 + 1000 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.size.solution.item(),
        1000,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.is_invested.solution.item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )


def test_optimize_size(solver_fixture, time_steps_fixture):
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(fix_effects=10, specific_effects=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80 + 20 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.size.solution.item(),
        20,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.is_invested.solution.item(),
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
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(minimum_size=40, fix_effects=10, specific_effects=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80 + 40 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.size.solution.item(),
        40,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.is_invested.solution.item(),
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
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(optional=True, minimum_size=40, fix_effects=10, specific_effects=1),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_optional',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=fx.InvestParameters(optional=True, minimum_size=50, fix_effects=10, specific_effects=1),
            ),
        ),
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    boiler_optional = flow_system.all_elements['Boiler_optional']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80 + 40 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.size.solution.item(),
        40,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model._investment.is_invested.solution.item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )

    assert_allclose(
        boiler_optional.Q_th.model._investment.size.solution.item(),
        0,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        boiler_optional.Q_th.model._investment.is_invested.solution.item(),
        0,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )

    def test_fixed_relative_profile(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=fx.InvestParameters(optional=True, minimum_size=40, fix_effects=10, specific_effects=1),
                ),
            ),
            fx.linear_converters.Boiler(
                'Boiler_optional',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=fx.InvestParameters(optional=True, minimum_size=50, fix_effects=10, specific_effects=1),
                ),
            ),
        )
        self.flow_system.add_elements(
            fx.Source(
                'Wärmequelle',
                source=fx.Flow('Q_th',
                               bus=self.get_element('Fernwärme'),
                               fixed_relative_profile=np.linspace(0, 5, len(self.datetime_array)),
                               size=fx.InvestParameters(optional=False, minimum_size=2, maximum_size=5),
                               )
            )
        )
        self.get_element('Fernwärme').excess_penalty_per_flow_hour = 1e5

        self.solve_and_load(self.flow_system)
        source = self.get_element('Wärmequelle')
        assert_allclose(
            source.source.model.flow_rate.result,
            np.linspace(0, 5, len(self.datetime_array)) * source.source.model._investment.size.result,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )
        assert_allclose(
            source.source.model._investment.size.result,
            2,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )




def test_on(solver_fixture, time_steps_fixture):
    """Tests if the On Variable is correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme', size=100, on_off_parameters=fx.OnOffParameters()
        ),
    ))

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.on_off.on.solution.values,
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
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
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                on_off_parameters=fx.OnOffParameters(consecutive_off_hours_max=100),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.on_off.on.solution.values,
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.on_off.off.solution.values,
        1 - boiler.Q_th.model.on_off.on.solution.values,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__off" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
        [0, 10, 20, 0, 10],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_switch_on_off(solver_fixture, time_steps_fixture):
    """Tests if the Switch On/Off Variable is correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                on_off_parameters=fx.OnOffParameters(force_switch_on=True),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.on_off.on.solution.values,
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.on_off.switch_on.solution.values,
        [0, 1, 0, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__switch_on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.on_off.switch_off.solution.values,
        [0, 0, 0, 1, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__switch_on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
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
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                on_off_parameters=fx.OnOffParameters(on_hours_total_max=1),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            0.2,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme', size=100),
        ),
    )

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        140,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.on_off.on.solution.values,
        [0, 0, 1, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
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
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                on_off_parameters=fx.OnOffParameters(on_hours_total_max=2),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            0.2,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                on_off_parameters=fx.OnOffParameters(on_hours_total_min=3),
            ),
        ),
    )
    flow_system.all_elements['Wärmelast'].sink.fixed_relative_profile = np.array([0, 10, 20, 0, 12])  # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    boiler_backup = flow_system.all_elements['Boiler_backup']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        114,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.on_off.on.solution.values,
        [0, 0, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
        [0, 0, 20, 0, 12 - 1e-5],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        sum(boiler_backup.Q_th.model.on_off.on.solution.values),
        3,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler_backup.Q_th.model.flow_rate.solution.values,
        [0, 10, 1.0e-05, 0, 1.0e-05],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


def test_consecutive_on_off(solver_fixture, time_steps_fixture):
    """Tests if the consecutive on/off hours are correctly created and calculated in a Flow"""
    flow_system = flow_system_base(time_steps_fixture)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                on_off_parameters=fx.OnOffParameters(consecutive_on_hours_max=2, consecutive_on_hours_min=2),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            0.2,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme', size=100),
        ),
    )
    flow_system.all_elements['Wärmelast'].sink.fixed_relative_profile = np.array([5, 10, 20, 18, 12])
    # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    boiler_backup = flow_system.all_elements['Boiler_backup']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        190,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.on_off.on.solution.values,
        [1, 1, 0, 1, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
        [5, 10, 0, 18, 12],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        boiler_backup.Q_th.model.flow_rate.solution.values,
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
            0.5,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme'),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            0.2,
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
            Q_th=fx.Flow(
                'Q_th',
                bus='Fernwärme',
                size=100,
                previous_flow_rate=np.array([20]),  # Otherwise its Off before the start
                on_off_parameters=fx.OnOffParameters(consecutive_off_hours_max=2, consecutive_off_hours_min=2),
            ),
        ),
    )
    flow_system.all_elements['Wärmelast'].sink.fixed_relative_profile = np.array([5, 0, 20, 18, 12])  # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    boiler = flow_system.all_elements['Boiler']
    boiler_backup = flow_system.all_elements['Boiler_backup']
    costs = flow_system.effects['costs']
    assert_allclose(
        costs.model.total.solution.item(),
        110,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        boiler_backup.Q_th.model.on_off.on.solution.values,
        [0, 0, 1, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__on" does not have the right value',
    )
    assert_allclose(
        boiler_backup.Q_th.model.on_off.off.solution.values,
        [1, 1, 0, 1, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__off" does not have the right value',
    )
    assert_allclose(
        boiler_backup.Q_th.model.flow_rate.solution.values,
        [0, 0, 1e-5, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        boiler.Q_th.model.flow_rate.solution.values,
        [5, 0, 20 - 1e-5, 18, 12],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
