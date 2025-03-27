"""
The conftest.py file is used by pytest to define shared fixtures, hooks, and configuration
that apply to multiple test files without needing explicit imports.
It helps avoid redundancy and centralizes reusable test logic.
"""

import os

import numpy as np
import pandas as pd
import pytest

import flixOpt as fx


@pytest.fixture()
def highs_solver():
    return fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=300)


@pytest.fixture()
def gurobi_solver():
    return fx.solvers.GurobiSolver(mip_gap=0, time_limit_seconds=300)


@pytest.fixture(params=[highs_solver, gurobi_solver])
def solver_fixture(request):
    return request.getfixturevalue(request.param.__name__)


# Custom assertion function
def assert_almost_equal_numeric(
        actual,
        desired,
        err_msg,
        relative_error_range_in_percent=0.011,
        absolute_tolerance=1e-9
):
    """
    Custom assertion function for comparing numeric values with relative and absolute tolerances
    """
    relative_tol = relative_error_range_in_percent / 100

    if isinstance(desired, (int, float)):
        delta = abs(relative_tol * desired) if desired != 0 else absolute_tolerance
        assert np.isclose(actual, desired, atol=delta), err_msg
    else:
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=relative_tol,
            atol=absolute_tolerance,
            err_msg=err_msg
        )


@pytest.fixture
def simple_flow_system() -> fx.FlowSystem:
    """
    Create a simple energy system for testing
    """
    base_thermal_load = np.array([30.0, 0.0, 90.0, 110, 110, 20, 20, 20, 20])
    base_electrical_price = 1 / 1000 * np.array([80.0, 80.0, 80.0, 80, 80, 80, 80, 80, 80])
    base_timesteps = pd.date_range('2020-01-01', periods=9, freq='h', name='time')
    # Define effects
    costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
    co2 = fx.Effect(
        'CO2',
        'kg',
        'CO2_e-Emissionen',
        specific_share_to_other_effects_operation={costs.label: 0.2},
        maximum_operation_per_hour=1000,
    )

    # Create components
    boiler = fx.linear_converters.Boiler(
        'Boiler',
        eta=0.5,
        Q_th=fx.Flow(
            'Q_th',
            bus='Fernwärme',
            size=50,
            relative_minimum=5 / 50,
            relative_maximum=1,
            on_off_parameters=fx.OnOffParameters(),
        ),
        Q_fu=fx.Flow('Q_fu', bus='Gas'),
    )

    chp = fx.linear_converters.CHP(
        'CHP_unit',
        eta_th=0.5,
        eta_el=0.4,
        P_el=fx.Flow('P_el', bus='Strom', size=60, relative_minimum=5 / 60, on_off_parameters=fx.OnOffParameters()),
        Q_th=fx.Flow('Q_th', bus='Fernwärme'),
        Q_fu=fx.Flow('Q_fu', bus='Gas'),
    )

    storage = fx.Storage(
        'Speicher',
        charging=fx.Flow('Q_th_load', bus='Fernwärme', size=1e4),
        discharging=fx.Flow('Q_th_unload', bus='Fernwärme', size=1e4),
        capacity_in_flow_hours=fx.InvestParameters(fix_effects=20, fixed_size=30, optional=False),
        initial_charge_state=0,
        relative_maximum_charge_state=1 / 100 * np.array([80.0, 70.0, 80.0, 80, 80, 80, 80, 80, 80, 80]),
        eta_charge=0.9,
        eta_discharge=1,
        relative_loss_per_hour=0.08,
        prevent_simultaneous_charge_and_discharge=True,
    )

    heat_load = fx.Sink(
        'Wärmelast',
        sink=fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=base_thermal_load)
    )

    gas_tariff = fx.Source(
        'Gastarif',
        source=fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': 0.04, 'CO2': 0.3})
    )

    electricity_feed_in = fx.Sink(
        'Einspeisung',
        sink=fx.Flow('P_el', bus='Strom', effects_per_flow_hour=-1 * base_electrical_price)
    )

    # Create flow system
    flow_system = fx.FlowSystem(base_timesteps)
    flow_system.add_elements(
        fx.Bus('Strom'),
        fx.Bus('Fernwärme'),
        fx.Bus('Gas')
    )
    flow_system.add_elements(storage, costs, co2, boiler, heat_load, gas_tariff, electricity_feed_in, chp)

    return flow_system


@pytest.fixture
def basic_flow_system() -> fx.FlowSystem:
    """Create basic elements for component testing"""
    flow_system = fx.FlowSystem(pd.date_range('2020-01-01', periods=10, freq='h', name='time'))
    thermal_load = np.array([np.random.random() for _ in range(10)]) * 180
    p_el = (np.array([np.random.random() for _ in range(10)]) + 0.5) / 1.5 * 50

    flow_system.add_elements(
        fx.Bus('Strom'),
        fx.Bus('Fernwärme'),
        fx.Bus('Gas'),
        fx.Effect('Costs', '€', 'Kosten', is_standard=True, is_objective=True),
        fx.Sink('Wärmelast', sink=fx.Flow('Q_th_Last', 'Fernwärme', size=1, fixed_relative_profile=thermal_load)),
        fx.Source('Gastarif', source=fx.Flow('Q_Gas', 'Gas', size=1000, effects_per_flow_hour=0.04)),
        fx.Sink('Einspeisung', sink=fx.Flow('P_el', 'Strom', effects_per_flow_hour=-1 * p_el)),
    )

    return flow_system


@pytest.fixture
def flow_system_complex() -> fx.FlowSystem:
    """
    Helper method to create a base model with configurable parameters
    """
    thermal_load = np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])
    electrical_load = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40])
    flow_system = fx.FlowSystem(pd.date_range('2020-01-01', periods=9, freq='h', name='time'))
    # Define the components and flow_system
    flow_system.add_elements(
        fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={'costs': 0.2}),
        fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3),
        fx.Bus('Strom'),
        fx.Bus('Fernwärme'),
        fx.Bus('Gas'),
        fx.Sink('Wärmelast', sink=fx.Flow('Q_th_Last', 'Fernwärme', size=1, fixed_relative_profile=thermal_load)),
        fx.Source('Gastarif', source=fx.Flow('Q_Gas', 'Gas', size=1000, effects_per_flow_hour={'costs': 0.04, 'CO2': 0.3})),
        fx.Sink('Einspeisung', sink=fx.Flow('P_el', 'Strom', effects_per_flow_hour=-1 * electrical_load)),
    )

    boiler = fx.linear_converters.Boiler(
        'Kessel',
        eta=0.5,
        on_off_parameters=fx.OnOffParameters(effects_per_running_hour={'costs': 0, 'CO2': 1000}),
        Q_th=fx.Flow(
            'Q_th',
            bus='Fernwärme',
            load_factor_max=1.0,
            load_factor_min=0.1,
            relative_minimum=5 / 50,
            relative_maximum=1,
            previous_flow_rate=50,
            size=fx.InvestParameters(
                fix_effects=1000, fixed_size=50, optional=False, specific_effects={'costs': 10, 'PE': 2}
            ),
            on_off_parameters=fx.OnOffParameters(
                on_hours_total_min=0,
                on_hours_total_max=1000,
                consecutive_on_hours_max=10,
                consecutive_on_hours_min=1,
                consecutive_off_hours_max=10,
                effects_per_switch_on=0.01,
                switch_on_total_max=1000,
            ),
            flow_hours_total_max=1e6,
        ),
        Q_fu=fx.Flow('Q_fu', bus='Gas', size=200, relative_minimum=0, relative_maximum=1),
    )

    invest_speicher = fx.InvestParameters(
        fix_effects=0,
        effects_in_segments=([(5, 25), (25, 100)],
                             {'costs': [(50, 250), (250, 800)], 'PE': [(5, 25), (25, 100)]}
                             ),
        optional=False,
        specific_effects={'costs': 0.01, 'CO2': 0.01},
        minimum_size=0,
        maximum_size=1000,
    )
    speicher = fx.Storage(
        'Speicher',
        charging=fx.Flow('Q_th_load', bus='Fernwärme', size=1e4),
        discharging=fx.Flow('Q_th_unload', bus='Fernwärme', size=1e4),
        capacity_in_flow_hours=invest_speicher,
        initial_charge_state=0,
        maximal_final_charge_state=10,
        eta_charge=0.9,
        eta_discharge=1,
        relative_loss_per_hour=0.08,
        prevent_simultaneous_charge_and_discharge=True,
    )

    flow_system.add_elements(boiler, speicher)

    return flow_system


@pytest.fixture
def flow_system_base(flow_system_complex) -> fx.FlowSystem:
    """
    Helper method to create a base model with configurable parameters
    """
    flow_system = flow_system_complex

    flow_system.add_elements(fx.linear_converters.CHP(
        'KWK',
        eta_th=0.5,
        eta_el=0.4,
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
        P_el=fx.Flow('P_el', bus='Strom', size=60, relative_minimum=5 / 60, previous_flow_rate=10),
        Q_th=fx.Flow('Q_th', bus='Fernwärme', size=1e3),
        Q_fu=fx.Flow('Q_fu', bus='Gas', size=1e3),
    ))

    return flow_system


@pytest.fixture
def flow_system_segments_of_flows(flow_system_complex) -> fx.FlowSystem:
    flow_system = flow_system_complex

    flow_system.add_elements(fx.LinearConverter(
        'KWK',
        inputs=[fx.Flow('Q_fu', bus='Gas')],
        outputs=[fx.Flow('P_el', bus='Strom', size=60, relative_maximum=55, previous_flow_rate=10),
                 fx.Flow('Q_th', bus='Fernwärme')],
        segmented_conversion_factors={
            'P_el': [(5, 30), (40, 60)],
            'Q_th': [(6, 35), (45, 100)],
            'Q_fu': [(12, 70), (90, 200)],
        },
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
    ))

    return flow_system


@pytest.fixture
def flow_system_long():
    """
    Fixture to create and return the flow system with loaded data
    """
    # Load data
    filename = os.path.join(os.path.dirname(__file__), 'ressources', 'Zeitreihen2020.csv')
    ts_raw = pd.read_csv(filename, index_col=0).sort_index()
    data = ts_raw['2020-01-01 00:00:00':'2020-12-31 23:45:00']['2020-01-01':'2020-01-03 23:45:00']

    # Extract data columns
    electrical_load = data['P_Netz/MW'].values
    thermal_load = data['Q_Netz/MW'].values
    p_el = data['Strompr.€/MWh'].values
    gas_price = data['Gaspr.€/MWh'].values

    thermal_load_ts, electrical_load_ts = fx.TimeSeriesData(thermal_load), fx.TimeSeriesData(electrical_load,
                                                                                             agg_weight=0.7)
    p_feed_in, p_sell = (
        fx.TimeSeriesData(-(p_el - 0.5), agg_group='p_el'),
        fx.TimeSeriesData(p_el + 0.5, agg_group='p_el'),
    )

    flow_system = fx.FlowSystem(pd.DatetimeIndex(data.index))
    flow_system.add_elements(
        fx.Bus('Strom'), fx.Bus('Fernwärme'),  fx.Bus('Gas'), fx.Bus('Kohle'),

        fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2_e-Emissionen'),
        fx.Effect('PE', 'kWh_PE', 'Primärenergie'),

        fx.Sink('Wärmelast', sink=fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=thermal_load_ts)),
        fx.Sink('Stromlast', sink=fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=electrical_load_ts)),
        fx.Source('Kohletarif',source=fx.Flow('Q_Kohle', bus='Kohle', size=1000, effects_per_flow_hour={'costs': 4.6, 'CO2': 0.3})),
        fx.Source('Gastarif', source=fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': gas_price, 'CO2': 0.3})),
        fx.Sink('Einspeisung', sink=fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour=p_feed_in)),
        fx.Source('Stromtarif', source=fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour={'costs': p_sell, 'CO2': 0.3}))
    )

    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Kessel',
            eta=0.85,
            Q_th=fx.Flow(label='Q_th', bus='Fernwärme'),
            Q_fu=fx.Flow(
                label='Q_fu',
                bus='Gas',
                size=95,
                relative_minimum=12 / 95,
                previous_flow_rate=0,
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=1000),
            ),
        ),
        fx.linear_converters.CHP(
            'BHKW2',
            eta_th=0.58,
            eta_el=0.22,
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=24000),
            P_el=fx.Flow('P_el', bus='Strom'),
            Q_th=fx.Flow('Q_th', bus='Fernwärme'),
            Q_fu=fx.Flow('Q_fu', bus='Kohle', size=288, relative_minimum=87 / 288),
        ),
        fx.Storage(
            'Speicher',
            charging=fx.Flow('Q_th_load', size=137, bus='Fernwärme'),
            discharging=fx.Flow('Q_th_unload', size=158, bus='Fernwärme'),
            capacity_in_flow_hours=684,
            initial_charge_state=137,
            minimal_final_charge_state=137,
            maximal_final_charge_state=158,
            eta_charge=1,
            eta_discharge=1,
            relative_loss_per_hour=0.001,
            prevent_simultaneous_charge_and_discharge=True,
        ),
    )

    # Return all the necessary data
    return flow_system, {
            'thermal_load_ts': thermal_load_ts,
            'electrical_load_ts': electrical_load_ts,
        }


def create_calculation_and_solve(flow_system: fx.FlowSystem, solver, name: str) -> fx.FullCalculation:
    calculation = fx.FullCalculation(name, flow_system)
    calculation.do_modeling()
    calculation.solve(solver)
    return calculation
