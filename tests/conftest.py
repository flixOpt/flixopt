"""
The conftest.py file is used by pytest to define shared fixtures, hooks, and configuration
that apply to multiple test files without needing explicit imports.
It helps avoid redundancy and centralizes reusable test logic.
"""

import os
from collections.abc import Iterable

import linopy.testing
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
from flixopt.structure import FlowSystemModel

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

        @staticmethod
        def complex():
            """Complex boiler with investment parameters from flow_system_complex"""
            return fx.linear_converters.Boiler(
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
                        effects_of_investment=1000,
                        fixed_size=50,
                        mandatory=True,
                        effects_of_investment_per_size={'costs': 10, 'PE': 2},
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

    class CHPs:
        @staticmethod
        def simple():
            """Simple CHP from simple_flow_system"""
            return fx.linear_converters.CHP(
                'CHP_unit',
                eta_th=0.5,
                eta_el=0.4,
                P_el=fx.Flow(
                    'P_el', bus='Strom', size=60, relative_minimum=5 / 60, on_off_parameters=fx.OnOffParameters()
                ),
                Q_th=fx.Flow('Q_th', bus='Fernwärme'),
                Q_fu=fx.Flow('Q_fu', bus='Gas'),
            )

        @staticmethod
        def base():
            """CHP from flow_system_base"""
            return fx.linear_converters.CHP(
                'KWK',
                eta_th=0.5,
                eta_el=0.4,
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
                P_el=fx.Flow('P_el', bus='Strom', size=60, relative_minimum=5 / 60, previous_flow_rate=10),
                Q_th=fx.Flow('Q_th', bus='Fernwärme', size=1e3),
                Q_fu=fx.Flow('Q_fu', bus='Gas', size=1e3),
            )

    class LinearConverters:
        @staticmethod
        def piecewise():
            """Piecewise converter from flow_system_piecewise_conversion"""
            return fx.LinearConverter(
                'KWK',
                inputs=[fx.Flow('Q_fu', bus='Gas')],
                outputs=[
                    fx.Flow('P_el', bus='Strom', size=60, relative_maximum=55, previous_flow_rate=10),
                    fx.Flow('Q_th', bus='Fernwärme'),
                ],
                piecewise_conversion=fx.PiecewiseConversion(
                    {
                        'P_el': fx.Piecewise([fx.Piece(5, 30), fx.Piece(40, 60)]),
                        'Q_th': fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
                        'Q_fu': fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
                    }
                ),
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
            )

        @staticmethod
        def segments(timesteps_length):
            """Segments converter with time-varying piecewise conversion"""
            return fx.LinearConverter(
                'KWK',
                inputs=[fx.Flow('Q_fu', bus='Gas')],
                outputs=[
                    fx.Flow('P_el', bus='Strom', size=60, relative_maximum=55, previous_flow_rate=10),
                    fx.Flow('Q_th', bus='Fernwärme'),
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
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
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
            charging=fx.Flow('Q_th_load', bus='Fernwärme', size=1e4),
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


class Flows:
    """Common Flow patterns to reduce inline instantiation"""

    @staticmethod
    def thermal(label='Q_th', bus='Fernwärme', size=50, **kwargs):
        """Create a standard thermal flow with customizable parameters"""
        return fx.Flow(label, bus=bus, size=size, **kwargs)

    @staticmethod
    def electrical(label='P_el', bus='Strom', size=30, **kwargs):
        """Create a standard electrical flow with customizable parameters"""
        return fx.Flow(label, bus=bus, size=size, **kwargs)

    @staticmethod
    def fuel(label='Q_fu', bus='Gas', size=100, **kwargs):
        """Create a standard fuel flow with customizable parameters"""
        return fx.Flow(label, bus=bus, size=size, **kwargs)

    @staticmethod
    def with_investment(label, bus, invest_params=None, **flow_kwargs):
        """
        Create a Flow with investment parameters.

        Args:
            label: Flow label
            bus: Bus name
            invest_params: Dict of InvestParameters kwargs or InvestParameters instance
            **flow_kwargs: Additional Flow parameters
        """
        if invest_params is None:
            invest_params = {}

        if isinstance(invest_params, dict):
            size = fx.InvestParameters(**invest_params)
        else:
            size = invest_params

        return fx.Flow(label, bus=bus, size=size, **flow_kwargs)

    @staticmethod
    def with_onoff(label, bus, size=50, onoff_params=None, **flow_kwargs):
        """
        Create a Flow with OnOff parameters.

        Args:
            label: Flow label
            bus: Bus name
            size: Flow size
            onoff_params: Dict of OnOffParameters kwargs or OnOffParameters instance
            **flow_kwargs: Additional Flow parameters
        """
        if onoff_params is None:
            onoff_params = {}

        if isinstance(onoff_params, dict):
            on_off_parameters = fx.OnOffParameters(**onoff_params)
        else:
            on_off_parameters = onoff_params

        return fx.Flow(label, bus=bus, size=size, on_off_parameters=on_off_parameters, **flow_kwargs)


# ============================================================================
# COMPONENT FACTORY EXTENSIONS
# ============================================================================


class BoilerFactory:
    """Factory methods for creating Boilers with common test configurations"""

    @staticmethod
    def with_investment(label='Boiler', eta=0.5, invest_params=None, q_th_label='Q_th', q_fu_label='Q_fu', **kwargs):
        """
        Create a Boiler with investment parameters on Q_th flow.

        Args:
            label: Boiler label
            eta: Efficiency
            invest_params: Dict of InvestParameters kwargs or InvestParameters instance
            q_th_label: Thermal flow label
            q_fu_label: Fuel flow label
            **kwargs: Additional Boiler parameters (e.g., on_off_parameters)
        """
        if invest_params is None:
            invest_params = {'minimum_size': 20, 'maximum_size': 100, 'mandatory': False}

        q_th = Flows.with_investment(q_th_label, 'Fernwärme', invest_params)
        q_fu = Flows.fuel(q_fu_label)

        return fx.linear_converters.Boiler(label, eta=eta, Q_th=q_th, Q_fu=q_fu, **kwargs)

    @staticmethod
    def with_onoff(label='Boiler', eta=0.5, size=50, onoff_params=None, q_th_label='Q_th', q_fu_label='Q_fu', **kwargs):
        """
        Create a Boiler with OnOff parameters.

        Args:
            label: Boiler label
            eta: Efficiency
            size: Q_th flow size
            onoff_params: Dict of OnOffParameters kwargs or OnOffParameters instance
            q_th_label: Thermal flow label
            q_fu_label: Fuel flow label
            **kwargs: Additional Boiler parameters
        """
        if onoff_params is None:
            onoff_params = {}

        q_th = Flows.with_onoff(q_th_label, 'Fernwärme', size, onoff_params)
        q_fu = Flows.fuel(q_fu_label)

        return fx.linear_converters.Boiler(label, eta=eta, Q_th=q_th, Q_fu=q_fu, **kwargs)

    @staticmethod
    def minimal(label='Boiler', eta=0.5, **kwargs):
        """
        Create minimal Boiler for basic testing.

        Args:
            label: Boiler label
            eta: Thermal efficiency
            **kwargs: Additional Boiler parameters
        """
        return fx.linear_converters.Boiler(
            label,
            eta=eta,
            Q_th=Flows.thermal(),
            Q_fu=Flows.fuel(),
            **kwargs,
        )


class CHPFactory:
    """Factory methods for creating CHPs with common test configurations"""

    @staticmethod
    def with_investment(label='CHP', eta_th=0.5, eta_el=0.4, invest_params=None, **kwargs):
        """
        Create a CHP with investment parameters on P_el flow.

        Args:
            label: CHP label
            eta_th: Thermal efficiency
            eta_el: Electrical efficiency
            invest_params: Dict of InvestParameters kwargs or InvestParameters instance
            **kwargs: Additional CHP parameters
        """
        if invest_params is None:
            invest_params = {'minimum_size': 20, 'maximum_size': 100, 'mandatory': False}

        p_el = Flows.with_investment('P_el', 'Strom', invest_params)
        q_th = Flows.thermal()
        q_fu = Flows.fuel()

        return fx.linear_converters.CHP(label, eta_th=eta_th, eta_el=eta_el, P_el=p_el, Q_th=q_th, Q_fu=q_fu, **kwargs)

    @staticmethod
    def with_onoff(label='CHP', eta_th=0.5, eta_el=0.4, size=60, onoff_params=None, **kwargs):
        """
        Create a CHP with OnOff parameters.

        Args:
            label: CHP label
            eta_th: Thermal efficiency
            eta_el: Electrical efficiency
            size: P_el flow size
            onoff_params: Dict of OnOffParameters kwargs or OnOffParameters instance
            **kwargs: Additional CHP parameters
        """
        if onoff_params is None:
            onoff_params = {}

        p_el = Flows.with_onoff('P_el', 'Strom', size, onoff_params)
        q_th = Flows.thermal()
        q_fu = Flows.fuel()

        return fx.linear_converters.CHP(label, eta_th=eta_th, eta_el=eta_el, P_el=p_el, Q_th=q_th, Q_fu=q_fu, **kwargs)

    @staticmethod
    def minimal(label='CHP', eta_th=0.5, eta_el=0.4, **kwargs):
        """
        Create minimal CHP for basic testing.

        Args:
            label: CHP label
            eta_th: Thermal efficiency
            eta_el: Electrical efficiency
            **kwargs: Additional CHP parameters
        """
        return fx.linear_converters.CHP(
            label,
            eta_th=eta_th,
            eta_el=eta_el,
            P_el=Flows.electrical(),
            Q_th=Flows.thermal(),
            Q_fu=Flows.fuel(),
            **kwargs,
        )


class StorageFactory:
    """Factory methods for creating Storage with common test configurations"""

    @staticmethod
    def with_investment(label='Storage', invest_params=None, charging_size=20, discharging_size=20, **kwargs):
        """
        Create a Storage with investment parameters on capacity.

        Args:
            label: Storage label
            invest_params: Dict of InvestParameters kwargs or InvestParameters instance
            charging_size: Size of charging flow
            discharging_size: Size of discharging flow
            **kwargs: Additional Storage parameters (e.g., eta_charge, prevent_simultaneous_charge_and_discharge)
        """
        if invest_params is None:
            invest_params = {'minimum_size': 20, 'maximum_size': 100, 'mandatory': False}

        if isinstance(invest_params, dict):
            capacity = fx.InvestParameters(**invest_params)
        else:
            capacity = invest_params

        # Set defaults for common parameters if not provided
        kwargs.setdefault('initial_charge_state', 0)
        kwargs.setdefault('prevent_simultaneous_charge_and_discharge', True)

        return fx.Storage(
            label,
            charging=Flows.thermal('Q_th_in', size=charging_size),
            discharging=Flows.thermal('Q_th_out', size=discharging_size),
            capacity_in_flow_hours=capacity,
            **kwargs,
        )

    @staticmethod
    def with_onoff(label='Storage', charging_size=20, discharging_size=20, capacity=30, onoff_params=None, **kwargs):
        """
        Create a Storage with OnOff parameters on charging/discharging flows.

        Args:
            label: Storage label
            charging_size: Size of charging flow
            discharging_size: Size of discharging flow
            capacity: Storage capacity in flow hours
            onoff_params: Dict of OnOffParameters kwargs or OnOffParameters instance (applied to charging flow)
            **kwargs: Additional Storage parameters
        """
        if onoff_params is None:
            onoff_params = {}

        # Set defaults
        kwargs.setdefault('initial_charge_state', 0)
        kwargs.setdefault('prevent_simultaneous_charge_and_discharge', True)

        charging = Flows.with_onoff('Q_th_in', 'Fernwärme', charging_size, onoff_params)
        discharging = Flows.thermal('Q_th_out', size=discharging_size)

        return fx.Storage(label, charging=charging, discharging=discharging, capacity_in_flow_hours=capacity, **kwargs)

    @staticmethod
    def minimal(label='Storage', capacity=30, **kwargs):
        """
        Create minimal Storage for basic testing.

        Args:
            label: Storage label
            capacity: Storage capacity in flow hours
            **kwargs: Additional Storage parameters (e.g., initial_charge_state, eta_charge,
                     relative_maximum_charge_state, relative_minimum_charge_state, etc.)
        """
        # Set defaults only if not provided in kwargs
        kwargs.setdefault('initial_charge_state', 0)
        kwargs.setdefault('prevent_simultaneous_charge_and_discharge', True)

        return fx.Storage(
            label,
            charging=Flows.thermal('Q_th_in', size=20),
            discharging=Flows.thermal('Q_th_out', size=20),
            capacity_in_flow_hours=capacity,
            **kwargs,
        )


# ============================================================================
# COMPONENT COLLECTIONS FOR PARAMETRIZED TESTING
# ============================================================================


def get_investable_components():
    """
    Get all component types that support investment parameters.
    Returns a list of tuples (component_name, factory_function).
    """
    return [
        ('Boiler', BoilerFactory.with_investment),
        ('CHP', CHPFactory.with_investment),
        ('Storage', StorageFactory.with_investment),
    ]


def get_onoff_components():
    """
    Get all component types that support OnOff parameters.
    Returns a list of tuples (component_name, factory_function).
    """
    return [
        ('Boiler', BoilerFactory.with_onoff),
        ('CHP', CHPFactory.with_onoff),
        ('Storage', StorageFactory.with_onoff),
    ]


# ============================================================================
# RECREATED FIXTURES USING HIERARCHICAL LIBRARY
# ============================================================================


@pytest.fixture
def simple_flow_system() -> fx.FlowSystem:
    """
    Create a simple energy system for testing
    """
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
        base_timesteps, scenarios=pd.Index(['A', 'B', 'C']), weights=np.array([0.5, 0.25, 0.25])
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
        fx.TimeSeriesData(electrical_load, aggregation_weight=0.7),
    )
    p_feed_in, p_sell = (
        fx.TimeSeriesData(-(p_el - 0.5), aggregation_group='p_el'),
        fx.TimeSeriesData(p_el + 0.5, aggregation_group='p_el'),
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
# COMPONENT TEST HELPERS
# ============================================================================


def verify_investment_variables(component, mandatory=False):
    """
    Verify that a component with investment parameters has the expected variables.

    Args:
        component: Component with investment (Flow, Storage, etc.)
        mandatory: Whether investment is mandatory

    Returns:
        set: Set of investment-related variable names found
    """
    var_names = set(component.submodel.variables)

    # All investable components should have 'size' variable
    assert 'size' in [v.split('|')[-1] for v in var_names], f"Component {component.label} should have 'size' variable"

    # Optional investment should have 'invested' binary variable
    if not mandatory:
        assert 'invested' in [v.split('|')[-1] for v in var_names], (
            f"Component {component.label} with optional investment should have 'invested' variable"
        )

    return var_names


def verify_onoff_variables(component):
    """
    Verify that a component with OnOff parameters has the expected variables.

    Args:
        component: Component with OnOff parameters

    Returns:
        set: Set of OnOff-related variable names found
    """
    # For components with OnOff, check the flow's submodel
    # CHP has OnOff on P_el, Boiler on Q_th, Storage on charging
    if hasattr(component, 'P_el'):  # CHP
        flow_submodel = component.P_el.submodel
    elif hasattr(component, 'Q_th'):  # Boiler
        flow_submodel = component.Q_th.submodel
    elif hasattr(component, 'charging'):  # Storage
        flow_submodel = component.charging.submodel
    else:
        raise ValueError(f'Cannot determine OnOff flow for component type {type(component)}')

    var_names = set(flow_submodel.variables)

    # Should have 'on' variable for OnOff parameters
    on_vars = [v for v in var_names if 'on' in v.split('|')[-1].lower()]
    assert len(on_vars) > 0, f"Component {component.label} with OnOff should have 'on' related variables"

    return var_names


def get_component_label_prefix(component):
    """
    Get the label prefix used in variable/constraint names for a component.

    Args:
        component: flixopt component

    Returns:
        str: Label prefix (e.g., 'Boiler(Q_th)', 'Storage')
    """
    # Different component types use different labeling schemes
    if hasattr(component, 'Q_th') and hasattr(component, 'Q_fu'):  # Boiler/CHP
        return f'{component.label}({component.Q_th.label})'
    elif hasattr(component, 'charging'):  # Storage
        return component.label
    else:
        return component.label


# ============================================================================
# UTILITY FUNCTIONS (kept for backward compatibility)
# ============================================================================


# Custom assertion function
def assert_almost_equal_numeric(
    actual, desired, err_msg, relative_error_range_in_percent=0.011, absolute_tolerance=1e-7
):
    """
    Custom assertion function for comparing numeric values with relative and absolute tolerances
    """
    relative_tol = relative_error_range_in_percent / 100

    if isinstance(desired, (int, float)):
        delta = abs(relative_tol * desired) if desired != 0 else absolute_tolerance
        assert np.isclose(actual, desired, atol=delta), err_msg
    else:
        np.testing.assert_allclose(actual, desired, rtol=relative_tol, atol=absolute_tolerance, err_msg=err_msg)


def create_calculation_and_solve(
    flow_system: fx.FlowSystem, solver, name: str, allow_infeasible: bool = False
) -> fx.FullCalculation:
    calculation = fx.FullCalculation(name, flow_system)
    calculation.do_modeling()
    try:
        calculation.solve(solver)
    except RuntimeError as e:
        if allow_infeasible:
            pass
        else:
            raise RuntimeError from e
    return calculation


def create_linopy_model(flow_system: fx.FlowSystem) -> FlowSystemModel:
    """
    Create a FlowSystemModel from a FlowSystem by performing the modeling phase.

    Args:
        flow_system: The FlowSystem to build the model from.

    Returns:
        FlowSystemModel: The built model from FullCalculation.do_modeling().
    """
    calculation = fx.FullCalculation('GenericName', flow_system)
    calculation.do_modeling()
    return calculation.model


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
    """Assert that two variables are equal with detailed error messages."""
    name = actual.name
    try:
        xr.testing.assert_equal(actual.lower, desired.lower)
    except AssertionError as e:
        raise AssertionError(
            f"{name} lower bounds don't match:\nActual: {actual.lower}\nExpected: {desired.lower}"
        ) from e

    try:
        xr.testing.assert_equal(actual.upper, desired.upper)
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

    try:
        xr.testing.assert_equal(actual.coords, desired.coords)
    except AssertionError as e:
        raise AssertionError(
            f"{name} coordinates don't match:\nActual: {actual.coords}\nExpected: {desired.coords}"
        ) from e

    if actual.coord_dims != desired.coord_dims:
        raise AssertionError(f"{name} coordinate dimensions don't match: {actual.coord_dims} != {desired.coord_dims}")


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
