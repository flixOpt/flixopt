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

import unittest

import numpy as np
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


class BaseTest(unittest.TestCase):
    """
    Base test class for setting up flow systems in flixOpt.

    Provides shared setup, utility methods, and common functionality for the other test cases.

    Methods:
        - setUp: Initializes logging and default parameters.
        - create_model: Creates a base flow system model with predefined buses and components.
        - solve_and_load: Solves the flow system model and loads the results.
        - get_element: Retrieves an element from the flow system by label.
        - solver: Configures and returns a solver instance.
    """

    def setUp(self):
        fx.change_logging_level('DEBUG')
        self.mip_gap = 0.0001
        self.datetime_array = fx.create_datetime_array('2020-01-01', 5, 'h')

    def create_model(self, datetime_array: np.ndarray[np.datetime64]) -> fx.FlowSystem:
        self.flow_system = fx.FlowSystem(datetime_array)
        self.buses = {
            'Fernwärme': fx.Bus('Fernwärme', excess_penalty_per_flow_hour=None),
            'Gas': fx.Bus('Gas', excess_penalty_per_flow_hour=None),
        }
        self.flow_system.add_elements(fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True))
        data = Data(len(datetime_array))
        self.flow_system.add_elements(
            fx.Sink(
                label='Wärmelast',
                sink=fx.Flow(
                    label='Wärme', bus=self.get_element('Fernwärme'), fixed_relative_profile=data.thermal_demand, size=1
                ),
            ),
            fx.Source(
                label='Gastarif', source=fx.Flow(label='Gas', bus=self.get_element('Gas'), effects_per_flow_hour=1)
            ),
        )
        return self.flow_system

    def solve_and_load(self, flow_system: fx.FlowSystem) -> fx.results.CalculationResults:
        calculation = fx.FullCalculation('Calculation', flow_system)
        calculation.do_modeling()
        calculation.solve(self.solver, True)
        results = fx.results.CalculationResults('Calculation', 'results')
        return results

    def get_element(self, label: str):
        return {**self.flow_system.all_elements, **self.buses}[label]

    @property
    def solver(self):
        """Returns a (new) solver instance with the specified parameters."""
        return fx.solvers.HighsSolver(mip_gap=self.mip_gap, time_limit_seconds=3600, solver_output_to_console=False)


class TestMinimal(BaseTest):
    """
    Tests a minimal setup of a flow system.

    Focuses on:
    - Adding basic components.
    - Verifying the correct setup and results for a small system with minimal complexity.
    """

    def create_model(self, datetime_array: np.ndarray[np.datetime64]) -> fx.FlowSystem:
        super().create_model(datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme')),
            )
        )
        return self.flow_system

    def test_solve_and_load(self):
        flow_system = self.create_model(self.datetime_array)
        self.solve_and_load(flow_system)

    def test_results(self):
        flow_system = self.create_model(self.datetime_array)
        results = self.solve_and_load(flow_system)

        assert_allclose(
            results.effect_results['costs'].all_results['all']['all_sum'], 80, rtol=self.mip_gap, atol=1e-10
        )

        assert_allclose(
            results.component_results['Boiler'].all_results['Q_th']['flow_rate'],
            [-0.0, 10.0, 20.0, -0.0, 10.0],
            rtol=self.mip_gap,
            atol=1e-10,
        )

        assert_allclose(
            results.effect_results['costs'].all_results['operation']['operation_sum_TS'],
            [-0.0, 20.0, 40.0, -0.0, 20.0],
            rtol=self.mip_gap,
            atol=1e-10,
        )

        assert_allclose(
            results.effect_results['costs'].all_results['operation']['Shares']['Gastarif__Gas__effects_per_flow_hour'],
            [-0.0, 20.0, 40.0, -0.0, 20.0],
            rtol=self.mip_gap,
            atol=1e-10,
        )


class TestInvestment(BaseTest):
    """
    Tests investment modeling and optimization in flow systems.

    Focuses on:
    - Fixed size investments.
    - Optimized sizing of components.
    - Investment constraints, including bounds and optional investments.
    - Validating cost calculations and investment decisions.
    """

    def test_fixed_size(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=fx.InvestParameters(fixed_size=1000, fix_effects=10, specific_effects=1),
                ),
            )
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80 + 1000 * 1 + 10,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )
        assert_allclose(
            boiler.model.all_variables['Boiler__Q_th__Investment_size'].result,
            1000,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
        )
        assert_allclose(
            boiler.model.all_variables['Boiler__Q_th__Investment_isInvested'].result,
            1,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
        )

    def test_optimize_size(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=fx.InvestParameters(fix_effects=10, specific_effects=1),
                ),
            )
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80 + 20 * 1 + 10,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._investment.size.result,
            20,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._investment.is_invested.result,
            1,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
        )

    def test_size_bounds(self):
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=fx.InvestParameters(minimum_size=40, fix_effects=10, specific_effects=1),
                ),
            )
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80 + 40 * 1 + 10,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._investment.size.result,
            40,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._investment.is_invested.result,
            1,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
        )

    def test_optional_invest(self):
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

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        boiler_optional = self.get_element('Boiler_optional')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80 + 40 * 1 + 10,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._investment.size.result,
            40,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._investment.is_invested.result,
            1,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
        )

        assert_allclose(
            boiler_optional.Q_th.model._investment.size.result,
            0,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
        )
        assert_allclose(
            boiler_optional.Q_th.model._investment.is_invested.result,
            0,
            rtol=self.mip_gap,
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




class TestOnOff(BaseTest):
    """
    Tests on-off operational constraints in flow systems.

    Focuses on:
    - Verifying the correct behavior of Flows that can toggle on or off.
    - Testing constraints like maximum consecutive off hours.
    - Validating flow rates and operational costs under on-off scenarios.
    """

    def test_on(self):
        """Tests if the On Variable is correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th', bus=self.get_element('Fernwärme'), size=100, on_off_parameters=fx.OnOffParameters()
                ),
            )
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [0, 1, 1, 0, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [0, 10, 20, 0, 10],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_off(self):
        """Tests if the Off Variable is correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(consecutive_off_hours_max=100),
                ),
            )
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [0, 1, 1, 0, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._on.off.result,
            1 - boiler.Q_th.model._on.on.result,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__off" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [0, 10, 20, 0, 10],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_switch_on_off(self):
        """Tests if the Switch On/Off Variable is correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(force_switch_on=True),
                ),
            )
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            80,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [0, 1, 1, 0, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._on.switch_on.result,
            [0, 1, 0, 0, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__switch_on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._on.switch_off.result,
            [0, 0, 0, 1, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__switch_on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [0, 10, 20, 0, 10],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_on_total_max(self):
        """Tests if the On Total Max Variable is correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(on_hours_total_max=1),
                ),
            ),
            fx.linear_converters.Boiler(
                'Boiler_backup',
                0.2,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'), size=100),
            ),
        )

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            140,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [0, 0, 1, 0, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [0, 0, 20, 0, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_on_total_bounds(self):
        """Tests if the On Hours min and max are correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(on_hours_total_max=2),
                ),
            ),
            fx.linear_converters.Boiler(
                'Boiler_backup',
                0.2,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(on_hours_total_min=3),
                ),
            ),
        )
        self.get_element('Wärmelast').sink.fixed_relative_profile = [0, 10, 20, 0, 12]  # Else its non deterministic

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        boiler_backup = self.get_element('Boiler_backup')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            114,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [0, 0, 1, 0, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [0, 0, 20, 0, 12 - 1e-5],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

        assert_allclose(
            sum(boiler_backup.Q_th.model._on.on.result),
            3,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler_backup__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler_backup.Q_th.model.flow_rate.result,
            [0, 10, 1.0e-05, 0, 1.0e-05],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_consecutive_on(self):
        """Tests if the consecutive on/off hours are correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(consecutive_on_hours_max=2, consecutive_on_hours_min=2),
                ),
            ),
            fx.linear_converters.Boiler(
                'Boiler_backup',
                0.2,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'), size=100),
            ),
        )
        self.get_element('Wärmelast').sink.fixed_relative_profile = [5, 10, 20, 18, 12]  # Else its non deterministic

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        boiler_backup = self.get_element('Boiler_backup')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            190,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [1, 1, 0, 1, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [5, 10, 0, 18, 12],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

        assert_allclose(
            boiler_backup.Q_th.model.flow_rate.result,
            [0, 0, 20, 0, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_consecutive_off(self):
        """Tests if the consecutive on hours are correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme')),
            ),
            fx.linear_converters.Boiler(
                'Boiler_backup',
                0.2,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    previous_flow_rate=np.array([20]),  # Otherwise its Off before the start
                    on_off_parameters=fx.OnOffParameters(consecutive_off_hours_max=2, consecutive_off_hours_min=2),
                ),
            ),
        )
        self.get_element('Wärmelast').sink.fixed_relative_profile = [5, 0, 20, 18, 12]  # Else its non deterministic

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        boiler_backup = self.get_element('Boiler_backup')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            110,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler_backup.Q_th.model._on.on.result,
            [0, 0, 1, 0, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler_backup__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler_backup.Q_th.model._on.off.result,
            [1, 1, 0, 1, 1],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler_backup__Q_th__off" does not have the right value',
        )
        assert_allclose(
            boiler_backup.Q_th.model.flow_rate.result,
            [0, 0, 1e-5, 0, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler_backup__Q_th__flow_rate" does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [5, 0, 20 - 1e-5, 18, 12],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

    def test_consecutive_on_off(self):
        """Tests if the consecutive on/off hours are correctly created and calculated in a Flow"""
        self.flow_system = self.create_model(self.datetime_array)
        self.flow_system.add_elements(
            fx.linear_converters.Boiler(
                'Boiler',
                0.5,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow(
                    'Q_th',
                    bus=self.get_element('Fernwärme'),
                    size=100,
                    on_off_parameters=fx.OnOffParameters(consecutive_on_hours_min=2,
                                                         consecutive_off_hours_min=2),
                    # Previous flow_rate is 0 by default
                ),
            ),
            fx.linear_converters.Boiler(
                'Boiler_backup',
                0.2,
                Q_fu=fx.Flow('Q_fu', bus=self.get_element('Gas')),
                Q_th=fx.Flow('Q_th', bus=self.get_element('Fernwärme'), size=100),
            ),
        )
        self.get_element('Wärmelast').sink.fixed_relative_profile = [5, 10, 20, 18, 12]  # Else its non deterministic

        self.solve_and_load(self.flow_system)
        boiler = self.get_element('Boiler')
        boiler_backup = self.get_element('Boiler_backup')
        costs = self.get_element('costs')
        assert_allclose(
            costs.model.all.sum.result,
            145,
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='The total costs does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.on.result,
            [0, 1, 1, 1, 1],  # Still of for second timestep!
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__on" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._on.off.result,
            [1, 0, 0, 0, 0],  # Still of for second timestep!
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__off" does not have the right value',
        )
        assert_allclose(
            boiler.Q_th.model._on.consecutive_on_hours.result,
            [0, 1, 2, 3, 4],  # Still of for second timestep!
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__consecutive_on_hours" does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model._on.consecutive_off_hours.result,
            [2, 0, 0, 0, 0],  # Still of for second timestep!
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__consecutive_off_hours" does not have the right value',
        )

        assert_allclose(
            boiler.Q_th.model.flow_rate.result,
            [0, 10, 20, 18, 12],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )

        assert_allclose(
            boiler_backup.Q_th.model.flow_rate.result,
            [5, 0, 0, 0, 0],
            rtol=self.mip_gap,
            atol=1e-10,
            err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
        )


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
