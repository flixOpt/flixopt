"""
Advanced example demonstrating the linked_periods parameter of InvestParameters.

The linked_periods parameter controls how investment decisions are linked across periods:
- None: Independent investment decision for each period
- (0, 1): All periods linked - once invested, available in all periods
- Custom array: Fine-grained control over which periods are linked

This example shows various linked_periods configurations and their use cases.
"""

import numpy as np
import pandas as pd

import flixopt as fx

if __name__ == '__main__':
    # --- Setup ---
    timesteps = pd.date_range('2020-01-01', periods=24, freq='h')
    periods = pd.Index([2020, 2025, 2030, 2035, 2040], name='period')

    heat_demand = np.array(
        [40, 35, 30, 30, 35, 50, 70, 90, 100, 110, 105, 100, 95, 90, 85, 90, 95, 100, 90, 80, 70, 60, 50, 45]
    )

    flow_system = fx.FlowSystem(timesteps=timesteps, periods=periods)

    flow_system.add_elements(fx.Bus(label='Electricity'), fx.Bus(label='Heat'), fx.Bus(label='Gas'))

    costs = fx.Effect(label='costs', unit='€', description='Total costs', is_standard=True, is_objective=True)

    # --- Example 1: No Linking (Independent Decisions) ---
    # Heat pump can be invested in independently for each period
    # Use case: Technology can be installed/uninstalled between periods
    heat_pump_independent = fx.linear_converters.HeatPump(
        label='HeatPump_Independent',
        COP=3.5,
        P_el=fx.Flow(
            'P_el',
            bus='Electricity',
            size=fx.InvestParameters(
                minimum_size=0,
                maximum_size=50,
                optional=True,
                fix_effects={'costs': 5000},
                specific_effects={'costs': 1000},
                linked_periods=None,  # No linking - independent per period
            ),
        ),
        Q_th=fx.Flow('Q_th', bus='Heat'),
    )

    # --- Example 2: Full Linking (All Periods) ---
    # Storage battery - once built in any period, it exists in all periods
    # Use case: Long-lived infrastructure that persists across all periods
    battery_fully_linked = fx.Storage(
        label='Battery_FullyLinked',
        charging=fx.Flow('P_charge', bus='Electricity', size=40),
        discharging=fx.Flow('P_discharge', bus='Electricity', size=40),
        capacity_in_flow_hours=fx.InvestParameters(
            minimum_size=10,
            maximum_size=150,
            optional=True,
            fix_effects={'costs': 8000},
            specific_effects={'costs': np.array([700, 600, 500, 450, 400])},  # Cost reduction over time
            linked_periods=(0, 1),  # All periods linked - single investment decision
        ),
        initial_charge_state=0,
        eta_charge=0.93,
        eta_discharge=0.93,
        prevent_simultaneous_charge_and_discharge=True,
    )

    # --- Example 3: Custom Linking Pattern ---
    # Solar panels with phased rollout
    # First deployment period (2020-2030), second deployment (2030-2040)
    # linked_periods array: [group_id for each period]
    # Same group_id means periods are linked
    solar_phased = fx.Source(
        label='Solar_Phased',
        outputs=[
            fx.Flow(
                label='P_solar',
                bus='Electricity',
                size=fx.InvestParameters(
                    minimum_size=0,
                    maximum_size=80,
                    optional=True,
                    fix_effects={'costs': np.array([12000, 11000, 10000, 9000, 8000])},
                    specific_effects={'costs': np.array([1100, 950, 800, 700, 600])},
                    # Phase 1: 2020-2025-2030 linked (group 1)
                    # Phase 2: 2035-2040 linked (group 2)
                    linked_periods=np.array([1, 1, 1, 2, 2]),
                ),
            )
        ],
    )

    # --- Example 4: Incremental Upgrades ---
    # Boiler with potential upgrades/expansions in later periods
    # Early periods linked, then separate decision for final period
    boiler_upgradeable = fx.linear_converters.Boiler(
        label='Boiler_Upgradeable',
        eta=0.92,
        Q_th=fx.Flow(
            label='Q_th',
            bus='Heat',
            size=fx.InvestParameters(
                minimum_size=0,
                maximum_size=np.array([60, 60, 80, 80, 100]),  # Increasing max over time
                optional=True,
                fix_effects={'costs': 10000},
                specific_effects={'costs': 800},
                # Periods 0-1 linked (group 1), 2-3 linked (group 2), 4 independent (group 3)
                linked_periods=np.array([1, 1, 2, 2, 3]),
            ),
        ),
        Q_fu=fx.Flow('Q_fu', bus='Gas'),
    )

    # --- Example 5: Sequential Periods (No Group Overlap) ---
    # CHP with replacement cycles - each period represents a separate lifecycle
    # Once a CHP is installed in one "generation", it doesn't carry to the next
    chp_sequential = fx.linear_converters.CHP(
        label='CHP_Sequential',
        eta_th=0.55,
        eta_el=0.38,
        P_el=fx.Flow(
            'P_el',
            bus='Electricity',
            size=fx.InvestParameters(
                minimum_size=0,
                maximum_size=70,
                optional=True,
                fix_effects={'costs': 20000},
                specific_effects={'costs': 1800},
                # Each period is its own group - completely independent
                linked_periods=np.array([1, 2, 3, 4, 5]),
            ),
        ),
        Q_th=fx.Flow('Q_th', bus='Heat'),
        Q_fu=fx.Flow('Q_fu', bus='Gas'),
    )

    # --- Supporting Components ---
    heat_sink = fx.Sink(
        label='Heat Demand',
        inputs=[fx.Flow(label='Q_demand', bus='Heat', size=1, fixed_relative_profile=heat_demand)],
    )

    gas_source = fx.Source(
        label='Gas Supply',
        outputs=[fx.Flow(label='Q_gas', bus='Gas', size=500, effects_per_flow_hour={'costs': 0.05})],
    )

    grid = fx.Source(
        label='Grid',
        outputs=[fx.Flow(label='P_grid', bus='Electricity', size=150, effects_per_flow_hour={'costs': 0.15})],
    )

    # --- Build and Solve ---
    flow_system.add_elements(
        costs,
        heat_pump_independent,
        battery_fully_linked,
        solar_phased,
        boiler_upgradeable,
        chp_sequential,
        heat_sink,
        gas_source,
        grid,
    )

    flow_system.plot_network(show=True)

    calculation = fx.FullCalculation(name='LinkedPeriodsAdvanced', flow_system=flow_system)
    calculation.do_modeling()
    calculation.solve(fx.solvers.HighsSolver(mip_gap=0.02, time_limit_seconds=120))

    # --- Analyze Results ---
    print('\n' + '=' * 60)
    print('INVESTMENT DECISIONS ACROSS PERIODS')
    print('=' * 60)
    print('\nThe investment decisions are shown in the calculation summary above.')
    print('Key observations from the results:')

    # Access size variables
    hp_var = 'HeatPump_Independent(P_el)|size'
    bat_var = 'Battery_FullyLinked|capacity_in_flow_hours|size'
    sol_var = 'Solar_Phased(P_solar)|size'
    boil_var = 'Boiler_Upgradeable(Q_th)|size'
    chp_var = 'CHP_Sequential(P_el)|size'

    print('\n1. Independent Heat Pump (no linking):')
    if hp_var in calculation.results.solution.data_vars:
        values = calculation.results.solution[hp_var].values
        print(f'   Sizes per period: {values}')

    print('\n2. Fully Linked Battery (linked_periods=(0,1)):')
    if bat_var in calculation.results.solution.data_vars:
        values = calculation.results.solution[bat_var].values
        print(f'   Sizes (should be same): {values}')

    print('\n3. Phased Solar (groups [1,1,1,2,2]):')
    if sol_var in calculation.results.solution.data_vars:
        values = calculation.results.solution[sol_var].values
        print(f'   Sizes: {values}')
        print('   Periods 0-2 should match, periods 3-4 should match')

    print('\n4. Upgradeable Boiler (groups [1,1,2,2,3]):')
    if boil_var in calculation.results.solution.data_vars:
        values = calculation.results.solution[boil_var].values
        print(f'   Sizes: {values}')
        print('   Periods 0-1 match, 2-3 match, 4 independent')

    print('\n5. Sequential CHP (groups [1,2,3,4,5]):')
    if chp_var in calculation.results.solution.data_vars:
        values = calculation.results.solution[chp_var].values
        print(f'   Sizes (all independent): {values}')

    print('\n' + '=' * 60)
    print('\nKey Insights:')
    print('- linked_periods=None: Maximum flexibility, but may be unrealistic')
    print('- linked_periods=(0,1): Single decision, realistic for long-lived assets')
    print('- Custom arrays: Model technology generations, phased rollouts, upgrades')
    print('=' * 60)

    # Save results
    calculation.results.to_file()
