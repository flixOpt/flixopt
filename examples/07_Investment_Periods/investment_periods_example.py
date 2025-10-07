"""
This example demonstrates how to use the period dimension with InvestParameters.

The period dimension allows modeling investment decisions across multiple time periods,
enabling multi-year planning and investment timing optimization.

This example shows:
1. Basic InvestParameters with periods - different investment costs per period
2. Using linked_periods to link investment decisions across periods
3. Period-specific investment constraints
"""

import numpy as np
import pandas as pd

import flixopt as fx

if __name__ == '__main__':
    # --- Create Time Series Data ---
    # Define timesteps for a single representative day
    timesteps = pd.date_range('2020-01-01', periods=24, freq='h')

    # Define multiple periods (e.g., years 2020, 2025, 2030)
    periods = pd.Index([2020, 2025, 2030], name='period')

    # Heat demand profile (kW) - same pattern for each period
    heat_demand_per_h = np.array(
        [30, 25, 20, 20, 25, 40, 60, 80, 90, 100, 95, 90, 85, 80, 75, 80, 85, 90, 80, 70, 60, 50, 40, 35]
    )

    # Power prices varying by period (€/kWh) - increasing over time
    power_prices_per_period = np.array([0.08, 0.10, 0.12])  # 2020, 2025, 2030

    # Create flow system with periods
    flow_system = fx.FlowSystem(timesteps=timesteps, periods=periods)

    # --- Define Energy Buses ---
    flow_system.add_elements(fx.Bus(label='Electricity'), fx.Bus(label='Heat'), fx.Bus(label='Gas'))

    # --- Define Effects ---
    costs = fx.Effect(
        label='costs',
        unit='€',
        description='Total costs',
        is_standard=True,
        is_objective=True,
    )

    CO2 = fx.Effect(
        label='CO2',
        unit='kg',
        description='CO2 emissions',
    )

    # --- Example 1: Basic Investment with Period-Specific Costs ---
    # Solar panels with decreasing costs over time (technology learning curve)
    solar_panels = fx.Source(
        label='Solar',
        outputs=[
            fx.Flow(
                label='P_solar',
                bus='Electricity',
                size=fx.InvestParameters(
                    minimum_size=0,
                    maximum_size=100,  # kW
                    optional=True,
                    fix_effects={
                        'costs': np.array([10000, 8000, 6000]),  # Fixed costs decrease over periods
                    },
                    specific_effects={
                        'costs': np.array([1200, 1000, 800]),  # €/kW decreases due to technology improvement
                        'CO2': np.array([-500, -500, -500]),  # Avoided emissions per kW (constant)
                    },
                ),
            )
        ],
    )

    # --- Example 2: Investment with Linked Periods ---
    # Battery storage - once invested in period 1, it's available in subsequent periods
    # linked_periods controls this behavior
    battery = fx.Storage(
        label='Battery',
        charging=fx.Flow('P_charge', bus='Electricity', size=50),
        discharging=fx.Flow('P_discharge', bus='Electricity', size=50),
        capacity_in_flow_hours=fx.InvestParameters(
            minimum_size=10,  # kWh
            maximum_size=200,
            optional=True,
            fix_effects={
                'costs': 5000,  # Grid connection costs (same for all periods)
            },
            specific_effects={
                'costs': np.array([800, 650, 500]),  # €/kWh decreases over time
            },
            # linked_periods: Once invested in an early period, available in later periods
            # This creates a binary investment variable that is shared across periods
            linked_periods=(2020, 1),  # Links all periods together (1D array with single link group)
        ),
        initial_charge_state=0,
        eta_charge=0.95,
        eta_discharge=0.95,
        relative_loss_per_hour=0.001,
        prevent_simultaneous_charge_and_discharge=True,
    )

    # --- Example 3: CHP with Period-Specific Maximum Size ---
    # CHP can be expanded over time (different maximum in each period)
    chp = fx.linear_converters.CHP(
        label='CHP',
        eta_th=0.5,
        eta_el=0.4,
        P_el=fx.Flow(
            'P_el',
            bus='Electricity',
            size=fx.InvestParameters(
                minimum_size=0,
                maximum_size=np.array([50, 75, 100]),  # Maximum capacity increases per period
                optional=True,
                fix_effects={
                    'costs': 15000,
                },
                specific_effects={
                    'costs': 1500,  # €/kW
                    'CO2': 200,  # kg CO2 per kW (lifecycle)
                },
                # No linked_periods - can invest independently in each period
            ),
        ),
        Q_th=fx.Flow('Q_th', bus='Heat'),
        Q_fu=fx.Flow('Q_fu', bus='Gas'),
    )

    # --- Supporting Components ---
    # Heat demand
    heat_sink = fx.Sink(
        label='Heat Demand',
        inputs=[fx.Flow(label='Q_th_demand', bus='Heat', size=1, fixed_relative_profile=heat_demand_per_h)],
    )

    # Gas source
    gas_source = fx.Source(
        label='Gas Supply',
        outputs=[fx.Flow(label='Q_gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': 0.06, 'CO2': 0.2})],
    )

    # Grid electricity (with period-varying prices)
    grid_import = fx.Source(
        label='Grid Import',
        outputs=[
            fx.Flow(
                label='P_import', bus='Electricity', size=200, effects_per_flow_hour={'costs': power_prices_per_period}
            )
        ],
    )

    # Grid export
    grid_export = fx.Sink(
        label='Grid Export',
        inputs=[
            fx.Flow(
                label='P_export',
                bus='Electricity',
                size=200,
                effects_per_flow_hour={'costs': -0.9 * power_prices_per_period},  # 90% of import price
            )
        ],
    )

    # --- Build Flow System ---
    flow_system.add_elements(costs, CO2, solar_panels, battery, chp, heat_sink, gas_source, grid_import, grid_export)

    # --- Visualize and Solve ---
    flow_system.plot_network(show=True)

    calculation = fx.FullCalculation(name='InvestmentPeriods', flow_system=flow_system)
    calculation.do_modeling()
    calculation.solve(fx.solvers.HighsSolver(mip_gap=0.01, time_limit_seconds=60))

    # --- Analyze Results ---
    # The investment decisions are automatically printed in the calculation summary above
    print('\n=== Additional Analysis ===')

    # Access investment variables directly from the solution
    solar_var = 'Solar(P_solar)|size'
    battery_var = 'Battery|capacity_in_flow_hours|size'
    chp_var = 'CHP(P_el)|size'

    if solar_var in calculation.results.solution.data_vars:
        print(f'\nSolar capacity per period: {calculation.results.solution[solar_var].values}')

    if battery_var in calculation.results.solution.data_vars:
        print(f'\nBattery capacity (linked): {calculation.results.solution[battery_var].values}')

    if chp_var in calculation.results.solution.data_vars:
        print(f'\nCHP capacity per period: {calculation.results.solution[chp_var].values}')

    # Plot results
    calculation.results['Heat'].plot_node_balance()
    calculation.results['Electricity'].plot_node_balance()

    # Save results
    calculation.results.to_file()
