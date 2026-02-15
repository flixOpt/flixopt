import importlib.util

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
from flixopt import Effect, InvestParameters, Sink, Source, Storage
from flixopt.elements import Bus, Flow
from flixopt.flow_system import FlowSystem

from .conftest import create_linopy_model

GUROBI_AVAILABLE = importlib.util.find_spec('gurobipy') is not None


@pytest.fixture
def test_system():
    """Create a basic test system with scenarios."""
    # Create a two-day time index with hourly resolution
    timesteps = pd.date_range('2023-01-01', periods=48, freq='h', name='time')

    # Create two scenarios
    scenarios = pd.Index(['Scenario A', 'Scenario B'], name='scenario')

    # Create scenario weights
    scenario_weights = np.array([0.7, 0.3])

    # Create a flow system with scenarios
    flow_system = FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_weights=scenario_weights,
    )

    # Create demand profiles that differ between scenarios
    # Scenario A: Higher demand in first day, lower in second day
    # Scenario B: Lower demand in first day, higher in second day
    demand_profile_a = np.concatenate(
        [
            np.sin(np.linspace(0, 2 * np.pi, 24)) * 5 + 10,  # Day 1, max ~15
            np.sin(np.linspace(0, 2 * np.pi, 24)) * 2 + 5,  # Day 2, max ~7
        ]
    )

    demand_profile_b = np.concatenate(
        [
            np.sin(np.linspace(0, 2 * np.pi, 24)) * 2 + 5,  # Day 1, max ~7
            np.sin(np.linspace(0, 2 * np.pi, 24)) * 5 + 10,  # Day 2, max ~15
        ]
    )

    # Stack the profiles into a 2D array (time, scenario)
    demand_profiles = np.column_stack([demand_profile_a, demand_profile_b])

    # Create the necessary model elements
    # Create buses
    electricity_bus = Bus('Electricity')

    # Create a demand sink with scenario-dependent profiles
    demand = Flow(electricity_bus.label_full, flow_id='Demand', fixed_relative_profile=demand_profiles)
    demand_sink = Sink('Demand', inputs=[demand])

    # Create a power source with investment option
    power_gen = Flow(
        electricity_bus.label_full,
        flow_id='Generation',
        size=InvestParameters(
            minimum_size=0,
            maximum_size=20,
            effects_of_investment_per_size={'costs': 100},  # €/kW
        ),
        effects_per_flow_hour={'costs': 20},  # €/MWh
    )
    generator = Source('Generator', outputs=[power_gen])

    # Create a storage for electricity
    storage_charge = Flow(electricity_bus.label_full, flow_id='Charge', size=10)
    storage_discharge = Flow(electricity_bus.label_full, flow_id='Discharge', size=10)
    storage = Storage(
        'Battery',
        charging=storage_charge,
        discharging=storage_discharge,
        capacity_in_flow_hours=InvestParameters(
            minimum_size=0,
            maximum_size=50,
            effects_of_investment_per_size={'costs': 50},  # €/kWh
        ),
        eta_charge=0.95,
        eta_discharge=0.95,
        initial_charge_state='equals_final',
    )

    # Create effects and objective
    cost_effect = Effect('costs', unit='€', description='Total costs', is_standard=True, is_objective=True)

    # Add all elements to the flow system
    flow_system.add_elements(electricity_bus, generator, demand_sink, storage, cost_effect)

    # Return the created system and its components
    return {
        'flow_system': flow_system,
        'timesteps': timesteps,
        'scenarios': scenarios,
        'electricity_bus': electricity_bus,
        'demand': demand,
        'demand_sink': demand_sink,
        'generator': generator,
        'power_gen': power_gen,
        'storage': storage,
        'storage_charge': storage_charge,
        'storage_discharge': storage_discharge,
        'cost_effect': cost_effect,
    }


@pytest.fixture
def flow_system_complex_scenarios() -> fx.FlowSystem:
    """
    Helper method to create a base model with configurable parameters
    """
    thermal_load = np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])
    electrical_load = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40])
    flow_system = fx.FlowSystem(
        pd.date_range('2020-01-01', periods=9, freq='h', name='time'),
        scenarios=pd.Index(['A', 'B', 'C'], name='scenario'),
    )
    # Define the components and flow_system
    flow_system.add_elements(
        fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True, share_from_temporal={'CO2': 0.2}),
        fx.Effect('CO2', 'kg', 'CO2_e-Emissionen'),
        fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3),
        fx.Bus('Strom'),
        fx.Bus('Fernwärme'),
        fx.Bus('Gas'),
        fx.Sink(
            'Wärmelast', inputs=[fx.Flow('Fernwärme', flow_id='Q_th_Last', size=1, fixed_relative_profile=thermal_load)]
        ),
        fx.Source(
            'Gastarif',
            outputs=[fx.Flow('Gas', flow_id='Q_Gas', size=1000, effects_per_flow_hour={'costs': 0.04, 'CO2': 0.3})],
        ),
        fx.Sink('Einspeisung', inputs=[fx.Flow('Strom', flow_id='P_el', effects_per_flow_hour=-1 * electrical_load)]),
    )

    boiler = fx.linear_converters.Boiler(
        'Kessel',
        thermal_efficiency=0.5,
        status_parameters=fx.StatusParameters(effects_per_active_hour={'costs': 0, 'CO2': 1000}),
        thermal_flow=fx.Flow(
            'Fernwärme',
            flow_id='Q_th',
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
        fuel_flow=fx.Flow('Gas', flow_id='Q_fu', size=200, relative_minimum=0, relative_maximum=1),
    )

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
    speicher = fx.Storage(
        'Speicher',
        charging=fx.Flow('Fernwärme', flow_id='Q_th_load', size=1e4),
        discharging=fx.Flow('Fernwärme', flow_id='Q_th_unload', size=1e4),
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
def flow_system_piecewise_conversion_scenarios(flow_system_complex_scenarios) -> fx.FlowSystem:
    """
    Use segments/Piecewise with numeric data
    """
    flow_system = flow_system_complex_scenarios

    flow_system.add_elements(
        fx.LinearConverter(
            'KWK',
            inputs=[fx.Flow('Gas', flow_id='Q_fu', size=200)],
            outputs=[
                fx.Flow('Strom', flow_id='P_el', size=60, relative_maximum=55, previous_flow_rate=10),
                fx.Flow('Fernwärme', flow_id='Q_th', size=100),
            ],
            piecewise_conversion=fx.PiecewiseConversion(
                {
                    'P_el': fx.Piecewise(
                        [
                            fx.Piece(np.linspace(5, 6, len(flow_system.timesteps)), 30),
                            fx.Piece(40, np.linspace(60, 70, len(flow_system.timesteps))),
                        ]
                    ),
                    'Q_th': fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
                    'Q_fu': fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
                }
            ),
            status_parameters=fx.StatusParameters(effects_per_startup=0.01),
        )
    )

    return flow_system


def test_weights(flow_system_piecewise_conversion_scenarios):
    """Test that scenario weights are correctly used in the model."""
    scenarios = flow_system_piecewise_conversion_scenarios.scenarios
    scenario_weights = np.linspace(0.5, 1, len(scenarios))
    scenario_weights_da = xr.DataArray(
        scenario_weights,
        dims=['scenario'],
        coords={'scenario': scenarios},
    )
    flow_system_piecewise_conversion_scenarios.scenario_weights = scenario_weights_da
    model = create_linopy_model(flow_system_piecewise_conversion_scenarios)
    normalized_weights = scenario_weights / sum(scenario_weights)
    np.testing.assert_allclose(model.objective_weights.values, normalized_weights)
    # Effects are now batched as 'effect|total' with an 'effect' dimension
    assert 'effect|total' in model.variables
    effect_total = model.variables['effect|total']
    assert 'effect' in effect_total.dims
    assert 'costs' in effect_total.coords['effect'].values
    assert 'Penalty' in effect_total.coords['effect'].values
    # Verify objective weights are normalized
    assert np.isclose(model.objective_weights.sum().item(), 1)


def test_weights_io(flow_system_piecewise_conversion_scenarios):
    """Test that scenario weights are correctly used in the model."""
    scenarios = flow_system_piecewise_conversion_scenarios.scenarios
    scenario_weights = np.linspace(0.5, 1, len(scenarios))
    scenario_weights_da = xr.DataArray(
        scenario_weights,
        dims=['scenario'],
        coords={'scenario': scenarios},
    )
    normalized_scenario_weights_da = scenario_weights_da / scenario_weights_da.sum()
    flow_system_piecewise_conversion_scenarios.scenario_weights = scenario_weights_da

    model = create_linopy_model(flow_system_piecewise_conversion_scenarios)
    np.testing.assert_allclose(model.objective_weights.values, normalized_scenario_weights_da)
    # Effects are now batched as 'effect|total' with an 'effect' dimension
    assert 'effect|total' in model.variables
    effect_total = model.variables['effect|total']
    assert 'effect' in effect_total.dims
    assert 'costs' in effect_total.coords['effect'].values
    assert 'Penalty' in effect_total.coords['effect'].values
    # Verify objective weights are normalized
    assert np.isclose(model.objective_weights.sum().item(), 1.0)


def test_scenario_dimensions_in_variables(flow_system_piecewise_conversion_scenarios):
    """Test that all variables have the scenario dimension where appropriate."""
    model = create_linopy_model(flow_system_piecewise_conversion_scenarios)
    # Variables can have various dimension combinations with scenarios
    # Batched variables now have element dimensions (flow, storage, effect, etc.)
    for var_name in model.variables:
        var = model.variables[var_name]
        # If it has time dim, it should also have scenario (or be time-only which happens during model building)
        # For batched variables, allow additional dimensions like 'flow', 'storage', 'effect', etc.
        allowed_dims_with_scenario = {
            ('time', 'scenario'),
            ('scenario',),
            (),
            # Batched variable dimensions
            ('flow', 'time', 'scenario'),
            ('storage', 'time', 'scenario'),
            ('effect', 'scenario'),
            ('effect', 'time', 'scenario'),
            ('bus', 'time', 'scenario'),
            ('flow', 'scenario'),
            ('storage', 'scenario'),
            ('converter', 'segment', 'time', 'scenario'),
            ('flow', 'effect', 'time', 'scenario'),
            ('component', 'time', 'scenario'),
        }
        # Check that scenario is present if time is present (or variable is scalar)
        if 'scenario' in var.dims or var.ndim == 0 or var.dims in allowed_dims_with_scenario:
            pass  # OK
        else:
            # Allow any dimension combination that includes scenario when expected
            assert 'scenario' in var.dims or var.ndim == 0, (
                f'Variable {var_name} missing scenario dimension: {var.dims}'
            )


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason='Gurobi solver not installed')
def test_full_scenario_optimization(flow_system_piecewise_conversion_scenarios):
    """Test a full optimization with scenarios and verify results."""
    scenarios = flow_system_piecewise_conversion_scenarios.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights

    # Optimize using new API
    flow_system_piecewise_conversion_scenarios.optimize(fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60))

    # Verify solution exists and has scenario dimension
    assert flow_system_piecewise_conversion_scenarios.solution is not None
    assert 'scenario' in flow_system_piecewise_conversion_scenarios.solution.dims


@pytest.mark.skip(reason='This test is taking too long with highs and is too big for gurobipy free')
def test_io_persistence(flow_system_piecewise_conversion_scenarios, tmp_path):
    """Test a full optimization with scenarios and verify results."""
    scenarios = flow_system_piecewise_conversion_scenarios.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights

    # Optimize using new API
    flow_system_piecewise_conversion_scenarios.optimize(fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=60))
    original_objective = flow_system_piecewise_conversion_scenarios.solution['objective'].item()

    # Save and restore
    filepath = tmp_path / 'flow_system_scenarios.nc4'
    flow_system_piecewise_conversion_scenarios.to_netcdf(filepath)
    flow_system_2 = fx.FlowSystem.from_netcdf(filepath)

    # Re-optimize restored flow system
    flow_system_2.optimize(fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=60))

    np.testing.assert_allclose(original_objective, flow_system_2.solution['objective'].item(), rtol=0.001)


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason='Gurobi solver not installed')
def test_scenarios_selection(flow_system_piecewise_conversion_scenarios):
    """Test scenario selection/subsetting functionality."""
    flow_system_full = flow_system_piecewise_conversion_scenarios
    scenarios = flow_system_full.scenarios
    scenario_weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_full.scenario_weights = scenario_weights
    flow_system = flow_system_full.sel(scenario=scenarios[0:2])

    assert flow_system.scenarios.equals(flow_system_full.scenarios[0:2])

    # Scenario weights are always normalized - subset is re-normalized to sum to 1
    subset_weights = flow_system_full.scenario_weights[0:2]
    expected_normalized = subset_weights / subset_weights.sum()
    np.testing.assert_allclose(flow_system.scenario_weights.values, expected_normalized.values)

    # Optimize using new API
    flow_system.optimize(
        fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60),
    )

    # Penalty has same structure as other effects: 'Penalty' is the total, 'Penalty(temporal)' and 'Penalty(periodic)' are components
    np.testing.assert_allclose(
        flow_system.solution['objective'].item(),
        (
            (flow_system.solution['effect|total'].sel(effect='costs') * flow_system.scenario_weights).sum()
            + (flow_system.solution['effect|total'].sel(effect='Penalty') * flow_system.scenario_weights).sum()
        ).item(),
    )  ## Account for rounding errors

    assert flow_system.solution.indexes['scenario'].equals(flow_system_full.scenarios[0:2])


def test_sizes_per_scenario_default():
    """Test that scenario_independent_sizes defaults to True (sizes equalized) and flow_rates to False (vary)."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios)

    assert fs.scenario_independent_sizes is True
    assert fs.scenario_independent_flow_rates is False


def test_sizes_per_scenario_bool():
    """Test scenario_independent_sizes with boolean values."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    # Test False (vary per scenario)
    fs1 = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios, scenario_independent_sizes=False)
    assert fs1.scenario_independent_sizes is False

    # Test True (equalized across scenarios)
    fs2 = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios, scenario_independent_sizes=True)
    assert fs2.scenario_independent_sizes is True


def test_sizes_per_scenario_list():
    """Test scenario_independent_sizes with list of element labels."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_independent_sizes=['solar->grid', 'battery->grid'],
    )

    assert fs.scenario_independent_sizes == ['solar->grid', 'battery->grid']


def test_flow_rates_per_scenario_default():
    """Test that scenario_independent_flow_rates defaults to False (flow rates vary by scenario)."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios)

    assert fs.scenario_independent_flow_rates is False


def test_flow_rates_per_scenario_bool():
    """Test scenario_independent_flow_rates with boolean values."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    # Test False (vary per scenario)
    fs1 = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios, scenario_independent_flow_rates=False)
    assert fs1.scenario_independent_flow_rates is False

    # Test True (equalized across scenarios)
    fs2 = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios, scenario_independent_flow_rates=True)
    assert fs2.scenario_independent_flow_rates is True


def test_scenario_parameters_property_setters():
    """Test that scenario parameters can be changed via property setters."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios)

    # Change scenario_independent_sizes
    fs.scenario_independent_sizes = True
    assert fs.scenario_independent_sizes is True

    fs.scenario_independent_sizes = ['component1', 'component2']
    assert fs.scenario_independent_sizes == ['component1', 'component2']

    # Change scenario_independent_flow_rates
    fs.scenario_independent_flow_rates = True
    assert fs.scenario_independent_flow_rates is True

    fs.scenario_independent_flow_rates = ['flow1', 'flow2']
    assert fs.scenario_independent_flow_rates == ['flow1', 'flow2']


def test_scenario_parameters_validation():
    """Test that scenario parameters are validated correctly."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(timesteps=timesteps, scenarios=scenarios)

    # Test invalid type
    with pytest.raises(TypeError, match='must be bool or list'):
        fs.scenario_independent_sizes = 'invalid'

    # Test invalid list content
    with pytest.raises(ValueError, match='must contain only strings'):
        fs.scenario_independent_sizes = [1, 2, 3]


def test_size_equality_constraints():
    """Test that size equality constraints are created when scenario_independent_sizes=True."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_independent_sizes=True,  # Sizes should be equalized
        scenario_independent_flow_rates=False,  # Flow rates can vary
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=fx.InvestParameters(
                    minimum_size=10,
                    maximum_size=100,
                    effects_of_investment_per_size={'cost': 100},
                ),
            )
        ],
    )

    fs.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    fs.build_model()

    # Check that size equality constraint exists
    constraint_names = [str(c) for c in fs.model.constraints]
    size_constraints = [c for c in constraint_names if 'scenario_independent' in c and 'size' in c]

    assert len(size_constraints) > 0, 'Size equality constraint should exist'


def test_flow_rate_equality_constraints():
    """Test that flow_rate equality constraints are created when scenario_independent_flow_rates=True."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_independent_sizes=False,  # Sizes can vary
        scenario_independent_flow_rates=True,  # Flow rates should be equalized
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=fx.InvestParameters(
                    minimum_size=10,
                    maximum_size=100,
                    effects_of_investment_per_size={'cost': 100},
                ),
            )
        ],
    )

    fs.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    fs.build_model()

    # Check that flow_rate equality constraint exists
    constraint_names = [str(c) for c in fs.model.constraints]
    flow_rate_constraints = [c for c in constraint_names if 'scenario_independent' in c and 'flow_rate' in c]

    assert len(flow_rate_constraints) > 0, 'Flow rate equality constraint should exist'


def test_selective_scenario_independence():
    """Test selective scenario independence with specific element lists."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_independent_sizes=['solar(out)'],  # Only solar size is equalized
        scenario_independent_flow_rates=['demand(in)'],  # Only demand flow_rate is equalized
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=fx.InvestParameters(
                    minimum_size=10, maximum_size=100, effects_of_investment_per_size={'cost': 100}
                ),
            )
        ],
    )
    sink = fx.Sink(
        'demand',
        inputs=[fx.Flow('grid', flow_id='in', size=50)],
    )

    fs.add_elements(bus, source, sink, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    fs.build_model()

    constraint_names = [str(c) for c in fs.model.constraints]

    # Solar SHOULD have size constraints (it's in the list, so equalized)
    solar_size_constraints = [c for c in constraint_names if 'solar(out)|size' in c and 'scenario_independent' in c]
    assert len(solar_size_constraints) > 0

    # Solar should NOT have flow_rate constraints (not in the list, so varies per scenario)
    solar_flow_constraints = [
        c for c in constraint_names if 'solar(out)|flow_rate' in c and 'scenario_independent' in c
    ]
    assert len(solar_flow_constraints) == 0

    # Demand should NOT have size constraints (no InvestParameters, size is fixed)
    demand_size_constraints = [c for c in constraint_names if 'demand(in)|size' in c and 'scenario_independent' in c]
    assert len(demand_size_constraints) == 0

    # Demand SHOULD have flow_rate constraints (it's in the list, so equalized)
    demand_flow_constraints = [
        c for c in constraint_names if 'demand(in)|flow_rate' in c and 'scenario_independent' in c
    ]
    assert len(demand_flow_constraints) > 0


def test_scenario_parameters_io_persistence():
    """Test that scenario_independent_sizes and scenario_independent_flow_rates persist through IO operations."""

    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    # Create FlowSystem with custom scenario parameters
    fs_original = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_independent_sizes=['solar(out)'],
        scenario_independent_flow_rates=True,
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=fx.InvestParameters(
                    minimum_size=10, maximum_size=100, effects_of_investment_per_size={'cost': 100}
                ),
            )
        ],
    )

    fs_original.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    # Save to dataset
    fs_original.connect_and_transform()
    ds = fs_original.to_dataset()

    # Load from dataset
    fs_loaded = fx.FlowSystem.from_dataset(ds)

    # Verify parameters persisted
    assert fs_loaded.scenario_independent_sizes == fs_original.scenario_independent_sizes
    assert fs_loaded.scenario_independent_flow_rates == fs_original.scenario_independent_flow_rates


def test_scenario_parameters_io_with_calculation(tmp_path):
    """Test that scenario parameters persist through full calculation IO."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'high'], name='scenario')

    fs = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_independent_sizes=True,
        scenario_independent_flow_rates=['demand(in)'],
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=fx.InvestParameters(
                    minimum_size=10, maximum_size=100, effects_of_investment_per_size={'cost': 100}
                ),
            )
        ],
    )
    sink = fx.Sink(
        'demand',
        inputs=[fx.Flow('grid', flow_id='in', size=50)],
    )

    fs.add_elements(bus, source, sink, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    # Solve using new API
    fs.optimize(fx.solvers.HighsSolver(mip_gap=0.01, time_limit_seconds=60))
    original_model = fs.model

    # Save and restore
    filepath = tmp_path / 'flow_system_scenarios.nc4'
    fs.to_netcdf(filepath)
    fs_loaded = fx.FlowSystem.from_netcdf(filepath)

    # Verify parameters persisted
    assert fs_loaded.scenario_independent_sizes == fs.scenario_independent_sizes
    assert fs_loaded.scenario_independent_flow_rates == fs.scenario_independent_flow_rates

    # Verify constraints are recreated correctly when building model
    fs_loaded.build_model()

    constraint_names1 = [str(c) for c in original_model.constraints]
    constraint_names2 = [str(c) for c in fs_loaded.model.constraints]

    size_constraints1 = [c for c in constraint_names1 if 'scenario_independent' in c and 'size' in c]
    size_constraints2 = [c for c in constraint_names2 if 'scenario_independent' in c and 'size' in c]

    assert len(size_constraints1) == len(size_constraints2)


def test_weights_io_persistence():
    """Test that weights persist through IO operations (to_dataset/from_dataset)."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'mid', 'high'], name='scenario')
    custom_scenario_weights = np.array([0.3, 0.5, 0.2])

    # Create FlowSystem with custom scenario weights
    fs_original = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_weights=custom_scenario_weights,
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=fx.InvestParameters(
                    minimum_size=10, maximum_size=100, effects_of_investment_per_size={'cost': 100}
                ),
            )
        ],
    )

    fs_original.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    # Save to dataset
    fs_original.connect_and_transform()
    ds = fs_original.to_dataset()

    # Load from dataset
    fs_loaded = fx.FlowSystem.from_dataset(ds)

    # Verify weights persisted correctly
    np.testing.assert_allclose(fs_loaded.scenario_weights.values, fs_original.scenario_weights.values)
    assert fs_loaded.scenario_weights.dims == fs_original.scenario_weights.dims


def test_weights_selection():
    """Test that weights are correctly sliced when using FlowSystem.sel()."""
    timesteps = pd.date_range('2023-01-01', periods=24, freq='h')
    scenarios = pd.Index(['base', 'mid', 'high'], name='scenario')
    custom_scenario_weights = np.array([0.3, 0.5, 0.2])

    # Create FlowSystem with custom scenario weights
    fs_full = fx.FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_weights=custom_scenario_weights,
    )

    bus = fx.Bus('grid')
    source = fx.Source(
        'solar',
        outputs=[
            fx.Flow(
                'grid',
                flow_id='out',
                size=10,
            )
        ],
    )

    fs_full.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    # Select a subset of scenarios
    fs_subset = fs_full.sel(scenario=['base', 'high'])

    # Verify weights are correctly sliced
    assert fs_subset.scenarios.equals(pd.Index(['base', 'high'], name='scenario'))
    # Scenario weights are always normalized - subset is re-normalized to sum to 1
    subset_weights = np.array([0.3, 0.2])  # Original weights for selected scenarios
    expected_normalized = subset_weights / subset_weights.sum()
    np.testing.assert_allclose(fs_subset.scenario_weights.values, expected_normalized)

    # Verify weights are 1D with just scenario dimension (no period dimension)
    assert fs_subset.scenario_weights.dims == ('scenario',)
