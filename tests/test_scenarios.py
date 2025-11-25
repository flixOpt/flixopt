import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from linopy.testing import assert_linequal

import flixopt as fx
from flixopt import Effect, InvestParameters, Sink, Source, Storage
from flixopt.elements import Bus, Flow
from flixopt.flow_system import FlowSystem

from .conftest import create_linopy_model, create_optimization_and_solve


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
    demand = Flow(label='Demand', bus=electricity_bus.label_full, fixed_relative_profile=demand_profiles)
    demand_sink = Sink('Demand', inputs=[demand])

    # Create a power source with investment option
    power_gen = Flow(
        label='Generation',
        bus=electricity_bus.label_full,
        size=InvestParameters(
            minimum_size=0,
            maximum_size=20,
            effects_of_investment_per_size={'costs': 100},  # €/kW
        ),
        effects_per_flow_hour={'costs': 20},  # €/MWh
    )
    generator = Source('Generator', outputs=[power_gen])

    # Create a storage for electricity
    storage_charge = Flow(label='Charge', bus=electricity_bus.label_full, size=10)
    storage_discharge = Flow(label='Discharge', bus=electricity_bus.label_full, size=10)
    storage = Storage(
        label='Battery',
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
    cost_effect = Effect(label='costs', unit='€', description='Total costs', is_standard=True, is_objective=True)

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
        fx.Sink('Wärmelast', inputs=[fx.Flow('Q_th_Last', 'Fernwärme', size=1, fixed_relative_profile=thermal_load)]),
        fx.Source(
            'Gastarif', outputs=[fx.Flow('Q_Gas', 'Gas', size=1000, effects_per_flow_hour={'costs': 0.04, 'CO2': 0.3})]
        ),
        fx.Sink('Einspeisung', inputs=[fx.Flow('P_el', 'Strom', effects_per_flow_hour=-1 * electrical_load)]),
    )

    boiler = fx.linear_converters.Boiler(
        'Kessel',
        thermal_efficiency=0.5,
        on_off_parameters=fx.OnOffParameters(effects_per_running_hour={'costs': 0, 'CO2': 1000}),
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
            on_off_parameters=fx.OnOffParameters(
                on_hours_min=0,
                on_hours_max=1000,
                consecutive_on_hours_max=10,
                consecutive_on_hours_min=1,
                consecutive_off_hours_max=10,
                effects_per_switch_on=0.01,
                switch_on_max=1000,
            ),
            flow_hours_max=1e6,
        ),
        fuel_flow=fx.Flow('Q_fu', bus='Gas', size=200, relative_minimum=0, relative_maximum=1),
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
def flow_system_piecewise_conversion_scenarios(flow_system_complex_scenarios) -> fx.FlowSystem:
    """
    Use segments/Piecewise with numeric data
    """
    flow_system = flow_system_complex_scenarios

    flow_system.add_elements(
        fx.LinearConverter(
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
                            fx.Piece(np.linspace(5, 6, len(flow_system.timesteps)), 30),
                            fx.Piece(40, np.linspace(60, 70, len(flow_system.timesteps))),
                        ]
                    ),
                    'Q_th': fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
                    'Q_fu': fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
                }
            ),
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
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
    # Penalty is now an effect with temporal and periodic components
    penalty_total = flow_system_piecewise_conversion_scenarios.effects.penalty_effect.submodel.total
    assert_linequal(
        model.objective.expression,
        (model.variables['costs'] * normalized_weights).sum() + (penalty_total * normalized_weights).sum(),
    )
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
    # Penalty is now an effect with temporal and periodic components
    penalty_total = flow_system_piecewise_conversion_scenarios.effects.penalty_effect.submodel.total
    assert_linequal(
        model.objective.expression,
        (model.variables['costs'] * normalized_scenario_weights_da).sum()
        + (penalty_total * normalized_scenario_weights_da).sum(),
    )
    assert np.isclose(model.objective_weights.sum().item(), 1.0)


def test_scenario_dimensions_in_variables(flow_system_piecewise_conversion_scenarios):
    """Test that all time variables are correctly broadcasted to scenario dimensions."""
    model = create_linopy_model(flow_system_piecewise_conversion_scenarios)
    for var in model.variables:
        assert model.variables[var].dims in [('time', 'scenario'), ('scenario',), ()]


def test_full_scenario_optimization(flow_system_piecewise_conversion_scenarios):
    """Test a full optimization with scenarios and verify results."""
    scenarios = flow_system_piecewise_conversion_scenarios.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights
    calc = create_optimization_and_solve(
        flow_system_piecewise_conversion_scenarios,
        solver=fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60),
        name='test_full_scenario',
    )
    calc.results.to_file()

    res = fx.results.Results.from_file('results', 'test_full_scenario')
    fx.FlowSystem.from_dataset(res.flow_system_data)
    _ = create_optimization_and_solve(
        flow_system_piecewise_conversion_scenarios,
        solver=fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60),
        name='test_full_scenario_2',
    )


@pytest.mark.skip(reason='This test is taking too long with highs and is too big for gurobipy free')
def test_io_persistence(flow_system_piecewise_conversion_scenarios):
    """Test a full optimization with scenarios and verify results."""
    scenarios = flow_system_piecewise_conversion_scenarios.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights
    calc = create_optimization_and_solve(
        flow_system_piecewise_conversion_scenarios,
        solver=fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=60),
        name='test_io_persistence',
    )
    calc.results.to_file()

    res = fx.results.Results.from_file('results', 'test_io_persistence')
    flow_system_2 = fx.FlowSystem.from_dataset(res.flow_system_data)
    calc_2 = create_optimization_and_solve(
        flow_system_2,
        solver=fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=60),
        name='test_io_persistence_2',
    )

    np.testing.assert_allclose(calc.results.objective, calc_2.results.objective, rtol=0.001)


def test_scenarios_selection(flow_system_piecewise_conversion_scenarios):
    flow_system_full = flow_system_piecewise_conversion_scenarios
    scenarios = flow_system_full.scenarios
    scenario_weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_full.scenario_weights = scenario_weights
    flow_system = flow_system_full.sel(scenario=scenarios[0:2])

    assert flow_system.scenarios.equals(flow_system_full.scenarios[0:2])

    np.testing.assert_allclose(flow_system.weights.values, flow_system_full.weights[0:2])

    calc = fx.Optimization(flow_system=flow_system, name='test_scenarios_selection', normalize_weights=False)
    calc.do_modeling()
    calc.solve(fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60))

    calc.results.to_file()

    # Penalty has same structure as other effects: 'Penalty' is the total, 'Penalty(temporal)' and 'Penalty(periodic)' are components
    np.testing.assert_allclose(
        calc.results.objective,
        (
            (calc.results.solution['costs'] * flow_system.weights).sum()
            + (calc.results.solution['Penalty'] * flow_system.weights).sum()
        ).item(),
    )  ## Account for rounding errors

    assert calc.results.solution.indexes['scenario'].equals(flow_system_full.scenarios[0:2])


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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
                size=fx.InvestParameters(
                    minimum_size=10,
                    maximum_size=100,
                    effects_of_investment_per_size={'cost': 100},
                ),
            )
        ],
    )

    fs.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    calc = fx.Optimization('test', fs)
    calc.do_modeling()

    # Check that size equality constraint exists
    constraint_names = [str(c) for c in calc.model.constraints]
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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
                size=fx.InvestParameters(
                    minimum_size=10,
                    maximum_size=100,
                    effects_of_investment_per_size={'cost': 100},
                ),
            )
        ],
    )

    fs.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    calc = fx.Optimization('test', fs)
    calc.do_modeling()

    # Check that flow_rate equality constraint exists
    constraint_names = [str(c) for c in calc.model.constraints]
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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
                size=fx.InvestParameters(
                    minimum_size=10, maximum_size=100, effects_of_investment_per_size={'cost': 100}
                ),
            )
        ],
    )
    sink = fx.Sink(
        label='demand',
        inputs=[fx.Flow(label='in', bus='grid', size=50)],
    )

    fs.add_elements(bus, source, sink, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    calc = fx.Optimization('test', fs)
    calc.do_modeling()

    constraint_names = [str(c) for c in calc.model.constraints]

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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
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


def test_scenario_parameters_io_with_calculation():
    """Test that scenario parameters persist through full calculation IO."""
    import shutil

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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
                size=fx.InvestParameters(
                    minimum_size=10, maximum_size=100, effects_of_investment_per_size={'cost': 100}
                ),
            )
        ],
    )
    sink = fx.Sink(
        label='demand',
        inputs=[fx.Flow(label='in', bus='grid', size=50)],
    )

    fs.add_elements(bus, source, sink, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    # Create temp directory for results
    temp_dir = tempfile.mkdtemp()

    try:
        # Solve and save
        calc = fx.Optimization('test_io', fs, folder=temp_dir)
        calc.do_modeling()
        calc.solve(fx.solvers.HighsSolver(mip_gap=0.01, time_limit_seconds=60))
        calc.results.to_file()

        # Load results
        results = fx.results.Results.from_file(temp_dir, 'test_io')
        fs_loaded = fx.FlowSystem.from_dataset(results.flow_system_data)

        # Verify parameters persisted
        assert fs_loaded.scenario_independent_sizes == fs.scenario_independent_sizes
        assert fs_loaded.scenario_independent_flow_rates == fs.scenario_independent_flow_rates

        # Verify constraints are recreated correctly
        calc2 = fx.Optimization('test_io_2', fs_loaded, folder=temp_dir)
        calc2.do_modeling()

        constraint_names1 = [str(c) for c in calc.model.constraints]
        constraint_names2 = [str(c) for c in calc2.model.constraints]

        size_constraints1 = [c for c in constraint_names1 if 'scenario_independent' in c and 'size' in c]
        size_constraints2 = [c for c in constraint_names2 if 'scenario_independent' in c and 'size' in c]

        assert len(size_constraints1) == len(size_constraints2)

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
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
    np.testing.assert_allclose(fs_loaded.weights.values, fs_original.weights.values)
    assert fs_loaded.weights.dims == fs_original.weights.dims


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
        label='solar',
        outputs=[
            fx.Flow(
                label='out',
                bus='grid',
                size=10,
            )
        ],
    )

    fs_full.add_elements(bus, source, fx.Effect('cost', 'Total cost', '€', is_objective=True))

    # Select a subset of scenarios
    fs_subset = fs_full.sel(scenario=['base', 'high'])

    # Verify weights are correctly sliced
    assert fs_subset.scenarios.equals(pd.Index(['base', 'high'], name='scenario'))
    np.testing.assert_allclose(fs_subset.weights.values, custom_scenario_weights[[0, 2]])

    # Verify weights are 1D with just scenario dimension (no period dimension)
    assert fs_subset.weights.dims == ('scenario',)
