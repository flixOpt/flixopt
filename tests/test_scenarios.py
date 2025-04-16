import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from linopy.testing import assert_linequal

import flixopt as fx
from flixopt.commons import Effect, FullCalculation, InvestParameters, Sink, Source, Storage, TimeSeriesData, solvers
from flixopt.elements import Bus, Flow
from flixopt.flow_system import FlowSystem

from .conftest import create_calculation_and_solve, create_linopy_model


@pytest.fixture
def test_system():
    """Create a basic test system with scenarios."""
    # Create a two-day time index with hourly resolution
    timesteps = pd.date_range(
        "2023-01-01", periods=48, freq="h", name="time"
    )

    # Create two scenarios
    scenarios = pd.Index(["Scenario A", "Scenario B"], name="scenario")

    # Create scenario weights as TimeSeriesData
    # Using TimeSeriesData to avoid conversion issues
    scenario_weights = TimeSeriesData(np.array([0.7, 0.3]))

    # Create a flow system with scenarios
    flow_system = FlowSystem(
        timesteps=timesteps,
        scenarios=scenarios,
        scenario_weights=scenario_weights  # Use TimeSeriesData for weights
    )

    # Create demand profiles that differ between scenarios
    # Scenario A: Higher demand in first day, lower in second day
    # Scenario B: Lower demand in first day, higher in second day
    demand_profile_a = np.concatenate([
        np.sin(np.linspace(0, 2*np.pi, 24)) * 5 + 10,  # Day 1, max ~15
        np.sin(np.linspace(0, 2*np.pi, 24)) * 2 + 5    # Day 2, max ~7
    ])

    demand_profile_b = np.concatenate([
        np.sin(np.linspace(0, 2*np.pi, 24)) * 2 + 5,   # Day 1, max ~7
        np.sin(np.linspace(0, 2*np.pi, 24)) * 5 + 10   # Day 2, max ~15
    ])

    # Stack the profiles into a 2D array (time, scenario)
    demand_profiles = np.column_stack([demand_profile_a, demand_profile_b])

    # Create the necessary model elements
    # Create buses
    electricity_bus = Bus("Electricity")

    # Create a demand sink with scenario-dependent profiles
    demand = Flow(
        label="Demand",
        bus=electricity_bus.label_full,
        fixed_relative_profile=demand_profiles
    )
    demand_sink = Sink("Demand", sink=demand)

    # Create a power source with investment option
    power_gen = Flow(
        label="Generation",
        bus=electricity_bus.label_full,
        size=InvestParameters(
            minimum_size=0,
            maximum_size=20,
            specific_effects={"Costs": 100}  # €/kW
        ),
        effects_per_flow_hour={"Costs": 20}  # €/MWh
    )
    generator = Source("Generator", source=power_gen)

    # Create a storage for electricity
    storage_charge = Flow(
        label="Charge",
        bus=electricity_bus.label_full,
        size=10
    )
    storage_discharge = Flow(
        label="Discharge",
        bus=electricity_bus.label_full,
        size=10
    )
    storage = Storage(
        label="Battery",
        charging=storage_charge,
        discharging=storage_discharge,
        capacity_in_flow_hours=InvestParameters(
            minimum_size=0,
            maximum_size=50,
            specific_effects={"Costs": 50}  # €/kWh
        ),
        eta_charge=0.95,
        eta_discharge=0.95,
        initial_charge_state="lastValueOfSim"
    )

    # Create effects and objective
    cost_effect = Effect(
        label="Costs",
        unit="€",
        description="Total costs",
        is_standard=True,
        is_objective=True
    )

    # Add all elements to the flow system
    flow_system.add_elements(
        electricity_bus,
        generator,
        demand_sink,
        storage,
        cost_effect
    )

    # Return the created system and its components
    return {
        "flow_system": flow_system,
        "timesteps": timesteps,
        "scenarios": scenarios,
        "electricity_bus": electricity_bus,
        "demand": demand,
        "demand_sink": demand_sink,
        "generator": generator,
        "power_gen": power_gen,
        "storage": storage,
        "storage_charge": storage_charge,
        "storage_discharge": storage_discharge,
        "cost_effect": cost_effect
    }

@pytest.fixture
def flow_system_complex_scenarios() -> fx.FlowSystem:
    """
    Helper method to create a base model with configurable parameters
    """
    thermal_load = np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])
    electrical_load = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40])
    flow_system = fx.FlowSystem(pd.date_range('2020-01-01', periods=9, freq='h', name='time'),
                                pd.Index(['A', 'B', 'C'], name='scenario'))
    # Define the components and flow_system
    flow_system.add_elements(
        fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={'costs': 0.2}),
        fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3),
        fx.Bus('Strom'),
        fx.Bus('Fernwärme'),
        fx.Bus('Gas'),
        fx.Sink('Wärmelast', sink=fx.Flow('Q_th_Last', 'Fernwärme', size=1, fixed_relative_profile=thermal_load)),
        fx.Source(
            'Gastarif', source=fx.Flow('Q_Gas', 'Gas', size=1000, effects_per_flow_hour={'costs': 0.04, 'CO2': 0.3})
        ),
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
        piecewise_effects=fx.PiecewiseEffects(
            piecewise_origin=fx.Piecewise([fx.Piece(5, 25), fx.Piece(25, 100)]),
            piecewise_shares={
                'costs': fx.Piecewise([fx.Piece(50, 250), fx.Piece(250, 800)]),
                'PE': fx.Piecewise([fx.Piece(5, 25), fx.Piece(25, 100)]),
            },
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
                            fx.Piece(np.linspace(5, 6, len(flow_system.time_series_collection.timesteps)), 30),
                            fx.Piece(40, np.linspace(60, 70, len(flow_system.time_series_collection.timesteps))),
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


def test_scenario_weights(flow_system_piecewise_conversion_scenarios):
    """Test that scenario weights are correctly used in the model."""
    scenarios = flow_system_piecewise_conversion_scenarios.time_series_collection.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights
    model = create_linopy_model(flow_system_piecewise_conversion_scenarios)
    np.testing.assert_allclose(model.scenario_weights.values, weights)
    assert_linequal(model.objective.expression,
                    (model.variables['costs|total'] * weights).sum() + model.variables['Penalty|total'])
    assert np.isclose(model.scenario_weights.sum().item(), 1.0)

def test_scenario_dimensions_in_variables(flow_system_piecewise_conversion_scenarios):
    """Test that all time variables are correctly broadcasted to scenario dimensions."""
    model = create_linopy_model(flow_system_piecewise_conversion_scenarios)
    for var in model.variables:
        assert  model.variables[var].dims in [('time', 'scenario'), ('scenario',), ()]

def test_full_scenario_optimization(flow_system_piecewise_conversion_scenarios):
    """Test a full optimization with scenarios and verify results."""
    scenarios = flow_system_piecewise_conversion_scenarios.time_series_collection.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights
    calc = create_calculation_and_solve(flow_system_piecewise_conversion_scenarios,
                                        solver=fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60),
                                        name='test_full_scenario')
    calc.results.to_file()

    res = fx.results.CalculationResults.from_file('results', 'test_full_scenario')
    fx.FlowSystem.from_dataset(res.flow_system)
    calc = create_calculation_and_solve(
        flow_system_piecewise_conversion_scenarios,
        solver=fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60),
        name='test_full_scenario',
    )

@pytest.mark.skip(reason="This test is taking too long with highs and is too big for gurobipy free")
def test_io_persistance(flow_system_piecewise_conversion_scenarios):
    """Test a full optimization with scenarios and verify results."""
    scenarios = flow_system_piecewise_conversion_scenarios.time_series_collection.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights
    calc = create_calculation_and_solve(flow_system_piecewise_conversion_scenarios,
                                        solver=fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=60),
                                        name='test_full_scenario')
    calc.results.to_file()

    res = fx.results.CalculationResults.from_file('results', 'test_full_scenario')
    flow_system_2 = fx.FlowSystem.from_dataset(res.flow_system)
    calc_2 = create_calculation_and_solve(
        flow_system_2,
        solver=fx.solvers.HighsSolver(mip_gap=0.001, time_limit_seconds=60),
        name='test_full_scenario_2',
    )

    np.testing.assert_allclose(calc.results.objective, calc_2.results.objective, rtol=0.001)


def test_scenarios_selection(flow_system_piecewise_conversion_scenarios):
    flow_system = flow_system_piecewise_conversion_scenarios
    scenarios = flow_system_piecewise_conversion_scenarios.time_series_collection.scenarios
    weights = np.linspace(0.5, 1, len(scenarios)) / np.sum(np.linspace(0.5, 1, len(scenarios)))
    flow_system_piecewise_conversion_scenarios.scenario_weights = weights
    calc = fx.FullCalculation(flow_system=flow_system_piecewise_conversion_scenarios,
                              selected_scenarios=flow_system.time_series_collection.scenarios[0:2],
                              name='test_full_scenario')
    calc.do_modeling()
    calc.solve(fx.solvers.GurobiSolver(mip_gap=0.01, time_limit_seconds=60))

    calc.results.to_file()
    flow_system_2 = fx.FlowSystem.from_dataset(calc.results.flow_system)

    assert calc.results.solution.indexes['scenario'].equals(flow_system.time_series_collection.scenarios[0:2])

    assert flow_system_2.time_series_collection.scenarios.equals(flow_system.time_series_collection.scenarios[0:2])

    np.testing.assert_allclose(flow_system_2.scenario_weights.selected_data.values, weights[0:2])
