"""
The conftest.py file is used by pytest to define shared fixtures, hooks, and configuration
that apply to multiple test files without needing explicit imports.
It helps avoid redundancy and centralizes reusable test logic.
"""

import os
import warnings
from collections.abc import Iterable

import linopy.testing
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
from flixopt.structure import FlowSystemModel

# ============================================================================
# SKIP DEPRECATED TESTS
# ============================================================================
# The deprecated folder contains tests for the old per-element submodel API
# which is not supported in v7's batched architecture.


def pytest_collection_modifyitems(items, config):
    """Skip all tests in the deprecated folder."""
    skip_marker = pytest.mark.skip(
        reason='Deprecated tests use per-element submodel API not supported in v7 batched architecture'
    )
    for item in items:
        if '/deprecated/' in str(item.fspath) or '\\deprecated\\' in str(item.fspath):
            item.add_marker(skip_marker)


# ============================================================================
# SOLVER FIXTURES
# ============================================================================


@pytest.fixture()
def highs_solver():
    return fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=300)


@pytest.fixture()
def gurobi_solver():
    pytest.importorskip('gurobipy', reason='Gurobi not available in this environment')
    return fx.solvers.GurobiSolver(mip_gap=0, time_limit_seconds=300)


@pytest.fixture(params=[highs_solver, gurobi_solver], ids=['highs', 'gurobi'])
def solver_fixture(request):
    return request.getfixturevalue(request.param.__name__)


# =================================
# COORDINATE CONFIGURATION FIXTURES
# =================================


@pytest.fixture(
    params=[
        {
            'timesteps': pd.date_range('2020-01-01', periods=10, freq='h', name='time'),
            'periods': None,
            'scenarios': None,
        },
        {
            'timesteps': pd.date_range('2020-01-01', periods=10, freq='h', name='time'),
            'periods': None,
            'scenarios': pd.Index(['A', 'B'], name='scenario'),
        },
        {
            'timesteps': pd.date_range('2020-01-01', periods=10, freq='h', name='time'),
            'periods': pd.Index([2020, 2030, 2040], name='period'),
            'scenarios': None,
        },
        {
            'timesteps': pd.date_range('2020-01-01', periods=10, freq='h', name='time'),
            'periods': pd.Index([2020, 2030, 2040], name='period'),
            'scenarios': pd.Index(['A', 'B'], name='scenario'),
        },
    ],
    ids=['time_only', 'time+scenarios', 'time+periods', 'time+periods+scenarios'],
)
def coords_config(request):
    """Coordinate configurations for parametrized testing."""
    return request.param


# ============================================================================
# HIERARCHICAL ELEMENT LIBRARY
# ============================================================================


class Buses:
    """Standard buses used across flow systems"""

    @staticmethod
    def electricity():
        return fx.Bus('Strom')

    @staticmethod
    def heat():
        return fx.Bus('Fernwärme')

    @staticmethod
    def gas():
        return fx.Bus('Gas')

    @staticmethod
    def coal():
        return fx.Bus('Kohle')

    @staticmethod
    def defaults():
        """Get all standard buses at once"""
        return [Buses.electricity(), Buses.heat(), Buses.gas()]


class Effects:
    """Standard effects used across flow systems"""

    @staticmethod
    def costs():
        return fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)

    @staticmethod
    def costs_with_co2_share():
        return fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True, share_from_temporal={'CO2': 0.2})

    @staticmethod
    def co2():
        return fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')

    @staticmethod
    def primary_energy():
        return fx.Effect('PE', 'kWh_PE', 'Primärenergie')


class Converters:
    """Energy conversion components"""

    class Boilers:
        @staticmethod
        def simple():
            """Simple boiler from simple_flow_system"""
            return fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.5,
                thermal_flow=fx.Flow(
                    'Q_th',
                    bus='Fernwärme',
                    size=50,
                    relative_minimum=5 / 50,
                    relative_maximum=1,
                    status_parameters=fx.StatusParameters(),
                ),
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            )

        @staticmethod
        def complex():
            """Complex boiler with investment parameters from flow_system_complex"""
            return fx.linear_converters.Boiler(
                'Kessel',
                thermal_efficiency=0.5,
                status_parameters=fx.StatusParameters(effects_per_active_hour={'costs': 0, 'CO2': 1000}),
                thermal_flow=fx.Flow(
                    'Q_th',
                    bus='Fernwärme',
                    load_factor_max=1.0,
                    load_factor_min=0.1,
                    relative_minimum=5 / 50,
                    relative_maximum=1,
                    previous_flow_rate=50,
                    size=fx.InvestParameters(
                        effects_of_investment=1000,
                        fixed_size=50,
                        mandatory=True,
                        effects_of_investment_per_size={'costs': 10, 'PE': 2},
                    ),
                    status_parameters=fx.StatusParameters(
                        active_hours_min=0,
                        active_hours_max=1000,
                        max_uptime=10,
                        min_uptime=1,
                        max_downtime=10,
                        effects_per_startup=0.01,
                        startup_limit=1000,
                    ),
                    flow_hours_max=1e6,
                ),
                fuel_flow=fx.Flow('Q_fu', bus='Gas', size=200, relative_minimum=0, relative_maximum=1),
            )

    class CHPs:
        @staticmethod
        def simple():
            """Simple CHP from simple_flow_system"""
            return fx.linear_converters.CHP(
                'CHP_unit',
                thermal_efficiency=0.5,
                electrical_efficiency=0.4,
                electrical_flow=fx.Flow(
                    'P_el', bus='Strom', size=60, relative_minimum=5 / 60, status_parameters=fx.StatusParameters()
                ),
                thermal_flow=fx.Flow('Q_th', bus='Fernwärme'),
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            )

        @staticmethod
        def base():
            """CHP from flow_system_base"""
            return fx.linear_converters.CHP(
                'KWK',
                thermal_efficiency=0.5,
                electrical_efficiency=0.4,
                status_parameters=fx.StatusParameters(effects_per_startup=0.01),
                electrical_flow=fx.Flow('P_el', bus='Strom', size=60, relative_minimum=5 / 60, previous_flow_rate=10),
                thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=1e3),
                fuel_flow=fx.Flow('Q_fu', bus='Gas', size=1e3),
            )

    class LinearConverters:
        @staticmethod
        def piecewise():
            """Piecewise converter from flow_system_piecewise_conversion"""
            return fx.LinearConverter(
                'KWK',
                inputs=[fx.Flow('Q_fu', bus='Gas', size=200)],
                outputs=[
                    fx.Flow('P_el', bus='Strom', size=60, relative_maximum=55, previous_flow_rate=10),
                    fx.Flow('Q_th', bus='Fernwärme', size=100),
                ],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'P_el': fx.Piecewise([fx.Piece(5, 30), fx.Piece(40, 60)]),
                        'Q_th': fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
                        'Q_fu': fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
                    }
                ),
                status_parameters=fx.StatusParameters(effects_per_startup=0.01),
            )

        @staticmethod
        def segments(timesteps_length):
            """Segments converter with time-varying piecewise conversion"""
            return fx.LinearConverter(
                'KWK',
                inputs=[fx.Flow('Q_fu', bus='Gas', size=200)],
                outputs=[
                    fx.Flow('P_el', bus='Strom', size=60, relative_maximum=55, previous_flow_rate=10),
                    fx.Flow('Q_th', bus='Fernwärme', size=100),
                ],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'P_el': fx.Piecewise(
                            [
                                fx.Piece(np.linspace(5, 6, timesteps_length), 30),
                                fx.Piece(40, np.linspace(60, 70, timesteps_length)),
                            ]
                        ),
                        'Q_th': fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
                        'Q_fu': fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
                    }
                ),
                status_parameters=fx.StatusParameters(effects_per_startup=0.01),
            )


class Storage:
    """Energy storage components"""

    @staticmethod
    def simple(timesteps_length=9):
        """Simple storage from simple_flow_system"""
        # Create pattern [80.0, 70.0, 80.0] and repeat/slice to match timesteps_length
        pattern = [80.0, 70.0, 80.0, 80, 80, 80, 80, 80, 80]
        charge_state_values = (pattern * ((timesteps_length // len(pattern)) + 1))[:timesteps_length]

        return fx.Storage(
            'Speicher',
            charging=fx.Flow(
                'Q_th_load',
                bus='Fernwärme',
                size=fx.InvestParameters(fixed_size=1e4, mandatory=True),  # Investment for testing sizes
            ),
            discharging=fx.Flow('Q_th_unload', bus='Fernwärme', size=1e4),
            capacity_in_flow_hours=fx.InvestParameters(effects_of_investment=20, fixed_size=30, mandatory=True),
            initial_charge_state=0,
            relative_maximum_charge_state=1 / 100 * np.array(charge_state_values),
            relative_maximum_final_charge_state=0.8,
            eta_charge=0.9,
            eta_discharge=1,
            relative_loss_per_hour=0.08,
            prevent_simultaneous_charge_and_discharge=True,
        )

    @staticmethod
    def complex():
        """Complex storage with piecewise investment from flow_system_complex"""
        invest_speicher = fx.InvestParameters(
            effects_of_investment=0,
            piecewise_effects_of_investment=fx.PiecewiseEffects(
                piecewise_origin=fx.Piecewise([fx.Piece(5, 25), fx.Piece(25, 100)]),
                piecewise_shares={
                    'costs': fx.Piecewise([fx.Piece(50, 250), fx.Piece(250, 800)]),
                    'PE': fx.Piecewise([fx.Piece(5, 25), fx.Piece(25, 100)]),
                },
            ),
            mandatory=True,
            effects_of_investment_per_size={'costs': 0.01, 'CO2': 0.01},
            minimum_size=0,
            maximum_size=1000,
        )
        return fx.Storage(
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


class LoadProfiles:
    """Standard load and price profiles"""

    @staticmethod
    def thermal_simple(timesteps_length=9):
        # Create pattern and repeat/slice to match timesteps_length
        pattern = [30.0, 0.0, 90.0, 110, 110, 20, 20, 20, 20]
        values = (pattern * ((timesteps_length // len(pattern)) + 1))[:timesteps_length]
        return np.array(values)

    @staticmethod
    def thermal_complex():
        return np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])

    @staticmethod
    def electrical_simple(timesteps_length=9):
        # Create array of 80.0 repeated to match timesteps_length
        return np.array([80.0 / 1000] * timesteps_length)

    @staticmethod
    def electrical_scenario():
        return np.array([0.08, 0.1, 0.15])

    @staticmethod
    def electrical_complex(timesteps_length=9):
        # Create array of 40 repeated to match timesteps_length
        return np.array([40] * timesteps_length)

    @staticmethod
    def random_thermal(length=10, seed=42):
        np.random.seed(seed)
        return np.array([np.random.random() for _ in range(length)]) * 180

    @staticmethod
    def random_electrical(length=10, seed=42):
        np.random.seed(seed)
        return (np.array([np.random.random() for _ in range(length)]) + 0.5) / 1.5 * 50


class Sinks:
    """Energy sinks (loads)"""

    @staticmethod
    def heat_load(thermal_profile):
        """Create thermal heat load sink"""
        return fx.Sink(
            'Wärmelast', inputs=[fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=thermal_profile)]
        )

    @staticmethod
    def electricity_feed_in(electrical_price_profile):
        """Create electricity feed-in sink"""
        return fx.Sink(
            'Einspeisung', inputs=[fx.Flow('P_el', bus='Strom', effects_per_flow_hour=-1 * electrical_price_profile)]
        )

    @staticmethod
    def electricity_load(electrical_profile):
        """Create electrical load sink (for flow_system_long)"""
        return fx.Sink(
            'Stromlast', inputs=[fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=electrical_profile)]
        )


class Sources:
    """Energy sources"""

    @staticmethod
    def gas_with_costs_and_co2():
        """Standard gas tariff with CO2 emissions"""
        source = Sources.gas_with_costs()
        source.outputs[0].effects_per_flow_hour = {'costs': 0.04, 'CO2': 0.3}
        return source

    @staticmethod
    def gas_with_costs():
        """Simple gas tariff without CO2"""
        return fx.Source(
            'Gastarif', outputs=[fx.Flow(label='Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': 0.04})]
        )


# ============================================================================
# RECREATED FIXTURES USING HIERARCHICAL LIBRARY
# ============================================================================


def build_simple_flow_system() -> fx.FlowSystem:
    """Create a simple energy system for testing (factory function)."""
    base_timesteps = pd.date_range('2020-01-01', periods=9, freq='h', name='time')
    timesteps_length = len(base_timesteps)
    base_thermal_load = LoadProfiles.thermal_simple(timesteps_length)
    base_electrical_price = LoadProfiles.electrical_simple(timesteps_length)

    # Define effects
    costs = Effects.costs_with_co2_share()
    co2 = Effects.co2()
    co2.maximum_per_hour = 1000

    # Create components
    boiler = Converters.Boilers.simple()
    chp = Converters.CHPs.simple()
    storage = Storage.simple(timesteps_length)
    heat_load = Sinks.heat_load(base_thermal_load)
    gas_tariff = Sources.gas_with_costs_and_co2()
    electricity_feed_in = Sinks.electricity_feed_in(base_electrical_price)

    # Create flow system
    flow_system = fx.FlowSystem(base_timesteps)
    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(storage, costs, co2, boiler, heat_load, gas_tariff, electricity_feed_in, chp)

    return flow_system


@pytest.fixture
def simple_flow_system() -> fx.FlowSystem:
    """Create a simple energy system for testing."""
    return build_simple_flow_system()


@pytest.fixture
def simple_flow_system_scenarios() -> fx.FlowSystem:
    """
    Create a simple energy system for testing
    """
    base_timesteps = pd.date_range('2020-01-01', periods=9, freq='h', name='time')
    timesteps_length = len(base_timesteps)
    base_thermal_load = LoadProfiles.thermal_simple(timesteps_length)
    base_electrical_price = LoadProfiles.electrical_scenario()

    # Define effects
    costs = Effects.costs_with_co2_share()
    co2 = Effects.co2()
    co2.maximum_per_hour = 1000

    # Create components
    boiler = Converters.Boilers.simple()
    chp = Converters.CHPs.simple()
    storage = Storage.simple(timesteps_length)
    heat_load = Sinks.heat_load(base_thermal_load)
    gas_tariff = Sources.gas_with_costs_and_co2()
    electricity_feed_in = Sinks.electricity_feed_in(base_electrical_price)

    # Create flow system
    flow_system = fx.FlowSystem(
        base_timesteps, scenarios=pd.Index(['A', 'B', 'C']), scenario_weights=np.array([0.5, 0.25, 0.25])
    )
    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(storage, costs, co2, boiler, heat_load, gas_tariff, electricity_feed_in, chp)

    return flow_system


@pytest.fixture
def basic_flow_system() -> fx.FlowSystem:
    """Create basic elements for component testing"""
    flow_system = fx.FlowSystem(pd.date_range('2020-01-01', periods=10, freq='h', name='time'))

    thermal_load = LoadProfiles.random_thermal(10)
    p_el = LoadProfiles.random_electrical(10)

    costs = Effects.costs()
    heat_load = Sinks.heat_load(thermal_load)
    gas_source = Sources.gas_with_costs()
    electricity_sink = Sinks.electricity_feed_in(p_el)

    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(costs, heat_load, gas_source, electricity_sink)

    return flow_system


@pytest.fixture
def flow_system_complex() -> fx.FlowSystem:
    """
    Helper method to create a base model with configurable parameters
    """
    thermal_load = LoadProfiles.thermal_complex()
    electrical_load = LoadProfiles.electrical_complex()
    flow_system = fx.FlowSystem(pd.date_range('2020-01-01', periods=9, freq='h', name='time'))

    # Define the components and flow_system
    costs = Effects.costs()
    co2 = Effects.co2()
    costs.share_from_temporal = {'CO2': 0.2}
    pe = Effects.primary_energy()
    pe.maximum_total = 3.5e3

    heat_load = Sinks.heat_load(thermal_load)
    gas_tariff = Sources.gas_with_costs_and_co2()
    electricity_feed_in = Sinks.electricity_feed_in(electrical_load)

    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(costs, co2, pe, heat_load, gas_tariff, electricity_feed_in)

    boiler = Converters.Boilers.complex()
    speicher = Storage.complex()

    flow_system.add_elements(boiler, speicher)

    return flow_system


@pytest.fixture
def flow_system_base(flow_system_complex) -> fx.FlowSystem:
    """
    Helper method to create a base model with configurable parameters
    """
    flow_system = flow_system_complex
    chp = Converters.CHPs.base()
    flow_system.add_elements(chp)
    return flow_system


@pytest.fixture
def flow_system_piecewise_conversion(flow_system_complex) -> fx.FlowSystem:
    flow_system = flow_system_complex
    converter = Converters.LinearConverters.piecewise()
    flow_system.add_elements(converter)
    return flow_system


@pytest.fixture
def flow_system_segments_of_flows_2(flow_system_complex) -> fx.FlowSystem:
    """
    Use segments/Piecewise with numeric data
    """
    flow_system = flow_system_complex
    converter = Converters.LinearConverters.segments(len(flow_system.timesteps))
    flow_system.add_elements(converter)
    return flow_system


@pytest.fixture
def flow_system_long():
    """
    Special fixture with CSV data loading - kept separate for backward compatibility
    Uses library components where possible, but has special elements inline
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

    thermal_load_ts, electrical_load_ts = (
        fx.TimeSeriesData(thermal_load),
        fx.TimeSeriesData(electrical_load, clustering_weight=0.7),
    )
    p_feed_in, p_sell = (
        fx.TimeSeriesData(-(p_el - 0.5), clustering_group='p_el'),
        fx.TimeSeriesData(p_el + 0.5, clustering_group='p_el'),
    )

    flow_system = fx.FlowSystem(pd.DatetimeIndex(data.index))
    flow_system.add_elements(
        *Buses.defaults(),
        Buses.coal(),
        Effects.costs(),
        Effects.co2(),
        Effects.primary_energy(),
        fx.Sink(
            'Wärmelast', inputs=[fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=thermal_load_ts)]
        ),
        fx.Sink(
            'Stromlast', inputs=[fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=electrical_load_ts)]
        ),
        fx.Source(
            'Kohletarif',
            outputs=[fx.Flow('Q_Kohle', bus='Kohle', size=1000, effects_per_flow_hour={'costs': 4.6, 'CO2': 0.3})],
        ),
        fx.Source(
            'Gastarif',
            outputs=[fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': gas_price, 'CO2': 0.3})],
        ),
        fx.Sink('Einspeisung', inputs=[fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour=p_feed_in)]),
        fx.Source(
            'Stromtarif',
            outputs=[fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour={'costs': p_sell, 'CO2': 0.3})],
        ),
    )

    flow_system.add_elements(
        fx.linear_converters.Boiler(
            'Kessel',
            thermal_efficiency=0.85,
            thermal_flow=fx.Flow(label='Q_th', bus='Fernwärme'),
            fuel_flow=fx.Flow(
                label='Q_fu',
                bus='Gas',
                size=95,
                relative_minimum=12 / 95,
                previous_flow_rate=0,
                status_parameters=fx.StatusParameters(effects_per_startup=1000),
            ),
        ),
        fx.linear_converters.CHP(
            'BHKW2',
            thermal_efficiency=(eta_th := 0.58),
            electrical_efficiency=(eta_el := 0.22),
            status_parameters=fx.StatusParameters(effects_per_startup=24000),
            fuel_flow=fx.Flow('Q_fu', bus='Kohle', size=(fuel_size := 288), relative_minimum=87 / fuel_size),
            electrical_flow=fx.Flow('P_el', bus='Strom', size=fuel_size * eta_el),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=fuel_size * eta_th),
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


@pytest.fixture(params=['h', '3h'], ids=['hourly', '3-hourly'])
def timesteps_linopy(request):
    return pd.date_range('2020-01-01', periods=10, freq=request.param, name='time')


@pytest.fixture
def basic_flow_system_linopy(timesteps_linopy) -> fx.FlowSystem:
    """Create basic elements for component testing"""
    flow_system = fx.FlowSystem(timesteps_linopy)

    n = len(flow_system.timesteps)
    thermal_load = LoadProfiles.random_thermal(n)
    p_el = LoadProfiles.random_electrical(n)

    costs = Effects.costs()
    heat_load = Sinks.heat_load(thermal_load)
    gas_source = Sources.gas_with_costs()
    electricity_sink = Sinks.electricity_feed_in(p_el)

    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(costs, heat_load, gas_source, electricity_sink)

    return flow_system


@pytest.fixture
def basic_flow_system_linopy_coords(coords_config) -> fx.FlowSystem:
    """Create basic elements for component testing with coordinate parametrization."""
    flow_system = fx.FlowSystem(**coords_config)

    thermal_load = LoadProfiles.random_thermal(10)
    p_el = LoadProfiles.random_electrical(10)

    costs = Effects.costs()
    heat_load = Sinks.heat_load(thermal_load)
    gas_source = Sources.gas_with_costs()
    electricity_sink = Sinks.electricity_feed_in(p_el)

    flow_system.add_elements(*Buses.defaults())
    flow_system.add_elements(costs, heat_load, gas_source, electricity_sink)

    return flow_system


# ============================================================================
# UTILITY FUNCTIONS (kept for backward compatibility)
# ============================================================================


# Custom assertion function
def assert_almost_equal_numeric(
    actual, desired, err_msg, relative_error_range_in_percent=0.011, absolute_tolerance=1e-7
):
    """
    Custom assertion function for comparing numeric values with relative and absolute tolerances.

    Handles the extra timestep in solutions by trimming actual arrays to match desired length
    when the extra values are NaN (from storage charge_state variables using extra_timestep).
    """
    relative_tol = relative_error_range_in_percent / 100

    if isinstance(desired, (int, float)):
        delta = abs(relative_tol * desired) if desired != 0 else absolute_tolerance
        assert np.isclose(actual, desired, atol=delta), err_msg
    else:
        actual = np.asarray(actual)
        desired = np.asarray(desired)
        # Handle extra timestep: trim actual to desired length if extra values are NaN
        if actual.shape != desired.shape and actual.ndim == 1 and desired.ndim == 1:
            if len(actual) > len(desired):
                extra = actual[len(desired) :]
                if np.all(np.isnan(extra)):
                    # Warn if trimming more than the expected single extra timestep
                    if len(extra) > 1:
                        warnings.warn(
                            f'Trimming {len(extra)} NaN values from actual array (expected 1)',
                            stacklevel=2,
                        )
                    actual = actual[: len(desired)]
        np.testing.assert_allclose(actual, desired, rtol=relative_tol, atol=absolute_tolerance, err_msg=err_msg)


def create_optimization_and_solve(
    flow_system: fx.FlowSystem, solver, name: str, allow_infeasible: bool = False
) -> fx.Optimization:
    optimization = fx.Optimization(name, flow_system)
    optimization.do_modeling()
    try:
        optimization.solve(solver)
    except RuntimeError:
        if not allow_infeasible:
            raise
    return optimization


def create_linopy_model(flow_system: fx.FlowSystem) -> FlowSystemModel:
    """
    Create a FlowSystemModel from a FlowSystem by performing the modeling phase.

    Args:
        flow_system: The FlowSystem to build the model from.

    Returns:
        FlowSystemModel: The built model from FlowSystem.build_model().
    """
    flow_system.build_model()
    return flow_system.model


def assert_conequal(actual: linopy.Constraint, desired: linopy.Constraint):
    """Assert that two constraints are equal with detailed error messages."""

    try:
        linopy.testing.assert_linequal(actual.lhs, desired.lhs)
    except AssertionError as e:
        raise AssertionError(f"{actual.name} left-hand sides don't match:\n{e}") from e

    try:
        xr.testing.assert_equal(actual.sign, desired.sign)
    except AssertionError as e:
        raise AssertionError(f"{actual.name} signs don't match:\n{e}") from e

    try:
        xr.testing.assert_equal(actual.rhs, desired.rhs)
    except AssertionError as e:
        raise AssertionError(f"{actual.name} right-hand sides don't match:\n{e}") from e


def assert_var_equal(actual: linopy.Variable, desired: linopy.Variable):
    """Assert that two variables are equal with detailed error messages.

    Drops scalar coordinates (non-dimension coords) before comparison to handle
    batched model slices that carry element coordinates.
    """
    name = actual.name

    def drop_scalar_coords(arr: xr.DataArray) -> xr.DataArray:
        """Drop coordinates that are not dimensions (scalar coords from .sel())."""
        scalar_coords = [c for c in arr.coords if c not in arr.dims]
        return arr.drop_vars(scalar_coords) if scalar_coords else arr

    try:
        xr.testing.assert_equal(drop_scalar_coords(actual.lower), drop_scalar_coords(desired.lower))
    except AssertionError as e:
        raise AssertionError(
            f"{name} lower bounds don't match:\nActual: {actual.lower}\nExpected: {desired.lower}"
        ) from e

    try:
        xr.testing.assert_equal(drop_scalar_coords(actual.upper), drop_scalar_coords(desired.upper))
    except AssertionError as e:
        raise AssertionError(
            f"{name} upper bounds don't match:\nActual: {actual.upper}\nExpected: {desired.upper}"
        ) from e

    if actual.type != desired.type:
        raise AssertionError(f"{name} types don't match: {actual.type} != {desired.type}")

    if actual.size != desired.size:
        raise AssertionError(f"{name} sizes don't match: {actual.size} != {desired.size}")

    if actual.shape != desired.shape:
        raise AssertionError(f"{name} shapes don't match: {actual.shape} != {desired.shape}")

    # Compare only dimension coordinates (drop scalar coords from batched model slices)
    actual_dim_coords = {k: v for k, v in actual.coords.items() if k in actual.dims}
    desired_dim_coords = {k: v for k, v in desired.coords.items() if k in desired.dims}
    try:
        xr.testing.assert_equal(xr.Coordinates(actual_dim_coords), xr.Coordinates(desired_dim_coords))
    except AssertionError as e:
        raise AssertionError(
            f"{name} dimension coordinates don't match:\nActual: {actual_dim_coords}\nExpected: {desired_dim_coords}"
        ) from e

    # Compare dims (the tuple of dimension names)
    if actual.dims != desired.dims:
        raise AssertionError(f"{name} dimensions don't match: {actual.dims} != {desired.dims}")


def assert_sets_equal(set1: Iterable, set2: Iterable, msg=''):
    """Assert two sets are equal with custom error message."""
    set1, set2 = set(set1), set(set2)

    extra = set1 - set2
    missing = set2 - set1

    if extra or missing:
        parts = []
        if extra:
            parts.append(f'Extra: {sorted(extra, key=repr)}')
        if missing:
            parts.append(f'Missing: {sorted(missing, key=repr)}')

        error_msg = ', '.join(parts)
        if msg:
            error_msg = f'{msg}: {error_msg}'

        raise AssertionError(error_msg)


def assert_dims_compatible(data: xr.DataArray, model_coords: tuple[str, ...], msg: str = ''):
    """Assert that data dimensions are a subset of model coordinates (compatible with broadcasting).

    Parameters in flixopt now stay in minimal form (scalar, 1D, etc.) and are broadcast
    at the linopy interface. This helper verifies that data dims are valid for the model.

    Args:
        data: DataArray to check
        model_coords: Tuple of model coordinate names (from model.get_coords())
        msg: Optional message for assertion error
    """
    extra_dims = set(data.dims) - set(model_coords)
    if extra_dims:
        error = f'Data has dimensions {extra_dims} not in model coordinates {model_coords}'
        if msg:
            error = f'{msg}: {error}'
        raise AssertionError(error)


# ============================================================================
# PLOTTING CLEANUP FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_figures():
    """
    Cleanup matplotlib figures after each test.

    This fixture runs automatically after every test to:
    - Close all matplotlib figures to prevent memory leaks
    """
    yield
    # Close all matplotlib figures
    import matplotlib.pyplot as plt

    plt.close('all')


@pytest.fixture(scope='session', autouse=True)
def set_test_environment():
    """
    Configure plotting for test environment.

    This fixture runs once per test session to:
    - Set matplotlib to use non-interactive 'Agg' backend
    - Set plotly to use non-interactive 'json' renderer
    - Prevent GUI windows from opening during tests
    """
    import matplotlib

    matplotlib.use('Agg')  # Use non-interactive backend

    import plotly.io as pio

    pio.renderers.default = 'json'  # Use non-interactive renderer

    fx.CONFIG.Plotting.default_show = False

    yield
