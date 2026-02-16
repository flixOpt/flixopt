"""
Unit tests for the flixopt framework.

.. deprecated::
    Superseded — These tests are superseded by tests/test_math/ which provides more thorough,
    analytically verified coverage with sensitivity documentation. Specifically:
    - Investment tests → test_math/test_flow_invest.py (9 tests + 3 invest+status combo tests)
    - Status tests → test_math/test_flow_status.py (9 tests + 6 previous_flow_rate tests)
    - Efficiency tests → test_math/test_conversion.py (3 tests)
    - Effect tests → test_math/test_effects.py (11 tests)
    Each test_math test runs in 3 modes (solve, save→reload→solve, solve→save→reload),
    making the IO roundtrip tests here redundant as well.
    Kept temporarily for reference. Safe to delete.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import flixopt as fx

np.random.seed(45)

pytestmark = pytest.mark.skip(reason='Superseded by tests/test_math/ — see module docstring')


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
            'Wärmelast',
            inputs=[fx.Flow(bus='Fernwärme', flow_id='Wärme', fixed_relative_profile=data.thermal_demand, size=1)],
        ),
        fx.Source('Gastarif', outputs=[fx.Flow(bus='Gas', flow_id='Gas', effects_per_flow_hour=1)]),
    )
    return flow_system


def flow_system_minimal(timesteps) -> fx.FlowSystem:
    flow_system = flow_system_base(timesteps)
    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(bus='Fernwärme', flow_id='Q_th'),
        )
    )
    return flow_system


def solve_and_load(flow_system: fx.FlowSystem, solver) -> fx.FlowSystem:
    """Optimize the flow system and return it with the solution."""
    flow_system.optimize(solver)
    return flow_system


@pytest.fixture
def time_steps_fixture(request):
    return pd.date_range('2020-01-01', periods=5, freq='h')


def test_solve_and_load(solver_fixture, time_steps_fixture):
    flow_system = solve_and_load(flow_system_minimal(time_steps_fixture), solver_fixture)
    assert flow_system.solution is not None


def test_minimal_model(solver_fixture, time_steps_fixture):
    flow_system = solve_and_load(flow_system_minimal(time_steps_fixture), solver_fixture)

    assert_allclose(flow_system.solution['effect|total'].sel(effect='costs').values, 80, rtol=1e-5, atol=1e-10)

    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
        [-0.0, 10.0, 20.0, -0.0, 10.0],
        rtol=1e-5,
        atol=1e-10,
    )

    assert_allclose(
        flow_system.solution['effect|per_timestep'].sel(effect='costs').values[:-1],
        [-0.0, 20.0, 40.0, -0.0, 20.0],
        rtol=1e-5,
        atol=1e-10,
    )

    assert_allclose(
        flow_system.solution['share|temporal'].sel(effect='costs', contributor='Gastarif(Gas)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=fx.InvestParameters(fixed_size=1000, effects_of_investment=10, effects_of_investment_per_size=1),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80 + 1000 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|size'].sel(flow='Boiler(Q_th)').item(),
        1000,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|invested'].sel(flow='Boiler(Q_th)').item(),
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=fx.InvestParameters(effects_of_investment=10, effects_of_investment_per_size=1, maximum_size=100),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80 + 20 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|size'].sel(flow='Boiler(Q_th)').item(),
        20,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|invested'].sel(flow='Boiler(Q_th)').item(),
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=fx.InvestParameters(
                    minimum_size=40, maximum_size=100, effects_of_investment=10, effects_of_investment_per_size=1
                ),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80 + 40 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|size'].sel(flow='Boiler(Q_th)').item(),
        40,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|invested'].sel(flow='Boiler(Q_th)').item(),
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=fx.InvestParameters(
                    mandatory=False,
                    minimum_size=40,
                    maximum_size=100,
                    effects_of_investment=10,
                    effects_of_investment_per_size=1,
                ),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_optional',
            thermal_efficiency=0.5,
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=fx.InvestParameters(
                    mandatory=False,
                    minimum_size=50,
                    maximum_size=100,
                    effects_of_investment=10,
                    effects_of_investment_per_size=1,
                ),
            ),
        ),
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80 + 40 * 1 + 10,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|size'].sel(flow='Boiler(Q_th)').item(),
        40,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|invested'].sel(flow='Boiler(Q_th)').item(),
        1,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__IsInvested" does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|size'].sel(flow='Boiler_optional(Q_th)').item(),
        0,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__Investment_size" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|invested'].sel(flow='Boiler_optional(Q_th)').item(),
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(bus='Fernwärme', flow_id='Q_th', size=100, status_parameters=fx.StatusParameters()),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=100,
                status_parameters=fx.StatusParameters(max_downtime=100),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|inactive'].sel(flow='Boiler(Q_th)').values[:-1],
        1 - flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__off" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=100,
                status_parameters=fx.StatusParameters(force_startup_tracking=True),
            ),
        )
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        80,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 1, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|startup'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 1, 0, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__switch_on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|shutdown'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 0, 0, 1, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__switch_on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=100,
                status_parameters=fx.StatusParameters(active_hours_max=1),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(bus='Fernwärme', flow_id='Q_th', size=100),
        ),
    )

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        140,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 0, 1, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=100,
                status_parameters=fx.StatusParameters(active_hours_max=2),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=100,
                status_parameters=fx.StatusParameters(active_hours_min=3),
            ),
        ),
    )
    flow_system['Wärmelast'].inputs[0].fixed_relative_profile = np.array(
        [0, 10, 20, 0, 12]
    )  # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        114,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 0, 1, 0, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
        [0, 0, 20, 0, 12 - 1e-5],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        sum(flow_system.solution['flow|status'].sel(flow='Boiler_backup(Q_th)').values[:-1]),
        3,
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler_backup(Q_th)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
                size=100,
                previous_flow_rate=0,  # Required for initial uptime constraint
                status_parameters=fx.StatusParameters(max_uptime=2, min_uptime=2),
            ),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(bus='Fernwärme', flow_id='Q_th', size=100),
        ),
    )
    flow_system['Wärmelast'].inputs[0].fixed_relative_profile = np.array([5, 10, 20, 18, 12])
    # Else its non deterministic

    solve_and_load(flow_system, solver_fixture)
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        190,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler(Q_th)').values[:-1],
        [1, 1, 0, 1, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
        [5, 10, 0, 18, 12],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler_backup(Q_th)').values[:-1],
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
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(bus='Fernwärme', flow_id='Q_th'),
        ),
        fx.linear_converters.Boiler(
            'Boiler_backup',
            thermal_efficiency=0.2,
            fuel_flow=fx.Flow(bus='Gas', flow_id='Q_fu'),
            thermal_flow=fx.Flow(
                bus='Fernwärme',
                flow_id='Q_th',
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
    assert_allclose(
        flow_system.solution['effect|total'].sel(effect='costs').item(),
        110,
        rtol=1e-5,
        atol=1e-10,
        err_msg='The total costs does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|status'].sel(flow='Boiler_backup(Q_th)').values[:-1],
        [0, 0, 1, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__on" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|inactive'].sel(flow='Boiler_backup(Q_th)').values[:-1],
        [1, 1, 0, 1, 1],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__off" does not have the right value',
    )
    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler_backup(Q_th)').values[:-1],
        [0, 0, 1e-5, 0, 0],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler_backup__Q_th__flow_rate" does not have the right value',
    )

    assert_allclose(
        flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values[:-1],
        [5, 0, 20 - 1e-5, 18, 12],
        rtol=1e-5,
        atol=1e-10,
        err_msg='"Boiler__Q_th__flow_rate" does not have the right value',
    )


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
