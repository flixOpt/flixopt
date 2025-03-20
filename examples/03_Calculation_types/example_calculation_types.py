"""
This script demonstrates how to use the different calcualtion types in the flixOPt framework
to model the same energy system. THe Results will be compared to each other.
"""

import pathlib
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
from rich.pretty import pprint  # Used for pretty printing

import flixOpt as fx

if __name__ == '__main__':
    # Calculation Types
    full, segmented, aggregated = True, True, True

    # Segmented Properties
    segment_length, overlap_length = 96, 1

    # Aggregated Properties
    aggregation_parameters = fx.AggregationParameters(
        hours_per_period=6,
        nr_of_periods=4,
        fix_storage_flows=False,
        aggregate_data_and_fix_non_binary_vars=True,
        percentage_of_period_freedom=0,
        penalty_of_period_freedom=0,
    )
    keep_extreme_periods = True
    excess_penalty = 1e5  # or set to None if not needed

    # Data Import
    data_import = pd.read_csv(pathlib.Path('Zeitreihen2020.csv'), index_col=0).sort_index()
    filtered_data = data_import['2020-01-01':'2020-01-2 23:45:00']
    # filtered_data = data_import[0:500]  # Alternatively filter by index

    filtered_data.index = pd.to_datetime(filtered_data.index)
    timesteps = filtered_data.index

    # Access specific columns and convert to 1D-numpy array
    electricity_demand = filtered_data['P_Netz/MW'].to_numpy()
    heat_demand = filtered_data['Q_Netz/MW'].to_numpy()
    electricity_price = filtered_data['Strompr.€/MWh'].to_numpy()
    gas_price = filtered_data['Gaspr.€/MWh'].to_numpy()

    # TimeSeriesData objects
    TS_heat_demand = fx.TimeSeriesData(heat_demand)
    TS_electricity_demand = fx.TimeSeriesData(electricity_demand, agg_weight=0.7)
    TS_electricity_price_sell = fx.TimeSeriesData(-(electricity_demand - 0.5), agg_group='p_el')
    TS_electricity_price_buy = fx.TimeSeriesData(electricity_price + 0.5, agg_group='p_el')

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Strom', excess_penalty_per_flow_hour=excess_penalty),
        fx.Bus('Fernwärme', excess_penalty_per_flow_hour=excess_penalty),
        fx.Bus('Gas', excess_penalty_per_flow_hour=excess_penalty),
        fx.Bus('Kohle', excess_penalty_per_flow_hour=excess_penalty),
    )

    # Effects
    costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
    CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
    PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie')

    # Component Definitions

    # 1. Boiler
    a_gaskessel = fx.linear_converters.Boiler(
        'Kessel',
        eta=0.85,
        Q_th=fx.Flow(label='Q_th', bus='Fernwärme'),
        Q_fu=fx.Flow(
            label='Q_fu',
            bus='Gas',
            size=95,
            relative_minimum=12 / 95,
            previous_flow_rate=20,
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=1000),
        ),
    )

    # 2. CHP
    a_kwk = fx.linear_converters.CHP(
        'BHKW2',
        eta_th=0.58,
        eta_el=0.22,
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=24000),
        P_el=fx.Flow('P_el', bus='Strom', size=200),
        Q_th=fx.Flow('Q_th', bus='Fernwärme', size=200),
        Q_fu=fx.Flow('Q_fu', bus='Kohle', size=288, relative_minimum=87 / 288, previous_flow_rate=100),
    )

    # 3. Storage
    a_speicher = fx.Storage(
        'Speicher',
        capacity_in_flow_hours=684,
        initial_charge_state=137,
        minimal_final_charge_state=137,
        maximal_final_charge_state=158,
        eta_charge=1,
        eta_discharge=1,
        relative_loss_per_hour=0.001,
        prevent_simultaneous_charge_and_discharge=True,
        charging=fx.Flow('Q_th_load', size=137, bus='Fernwärme'),
        discharging=fx.Flow('Q_th_unload', size=158, bus='Fernwärme'),
    )

    # 4. Sinks and Sources
    # Heat Load Profile
    a_waermelast = fx.Sink(
        'Wärmelast', sink=fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=TS_heat_demand)
    )

    # Electricity Feed-in
    a_strom_last = fx.Sink(
        'Stromlast', sink=fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=TS_electricity_demand)
    )

    # Gas Tariff
    a_gas_tarif = fx.Source(
        'Gastarif', source=fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={costs.label: gas_price, CO2.label: 0.3})
    )

    # Coal Tariff
    a_kohle_tarif = fx.Source(
        'Kohletarif', source=fx.Flow('Q_Kohle', bus='Kohle', size=1000, effects_per_flow_hour={costs.label: 4.6, CO2.label: 0.3})
    )

    # Electricity Tariff and Feed-in
    a_strom_einspeisung = fx.Sink(
        'Einspeisung', sink=fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour=TS_electricity_price_sell)
    )

    a_strom_tarif = fx.Source(
        'Stromtarif',
        source=fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour={costs.label: TS_electricity_price_buy, CO2: 0.3}),
    )

    # Flow System Setup
    flow_system.add_elements(costs, CO2, PE)
    flow_system.add_elements(
        a_gaskessel,
        a_waermelast,
        a_strom_last,
        a_gas_tarif,
        a_kohle_tarif,
        a_strom_einspeisung,
        a_strom_tarif,
        a_kwk,
        a_speicher,
    )
    flow_system.plot_network(controls=False, show=True)

    # Calculations
    calculations: List[Union[fx.FullCalculation, fx.AggregatedCalculation, fx.SegmentedCalculation]] = []

    if full:
        calculation = fx.FullCalculation('Full', flow_system)
        calculation.do_modeling()
        calculation.solve(fx.solvers.HighsSolver(0, 60))
        calculations.append(calculation)

    if segmented:
        calculation = fx.SegmentedCalculation('Segmented', flow_system, segment_length, overlap_length)
        calculation.do_modeling_and_solve(fx.solvers.HighsSolver(0, 60))
        calculations.append(calculation)

    if aggregated:
        if keep_extreme_periods:
            aggregation_parameters.time_series_for_high_peaks = [TS_heat_demand]
            aggregation_parameters.time_series_for_low_peaks = [TS_electricity_demand, TS_heat_demand]
        calculation = fx.AggregatedCalculation('Aggregated', flow_system, aggregation_parameters)
        calculation.do_modeling()
        calculation.solve(fx.solvers.HighsSolver(0, 60))
        calculations.append(calculation)

    # Get solutions for plotting for different calculations
    def get_solutions(calcs: List, variable: str) -> xr.Dataset:
        dataarrays = []
        for calc in calcs:
            if calc.name == 'Segmented':
                dataarrays.append(calc.results.solution_without_overlap(variable).rename(calc.name))
            else:
                dataarrays.append(calc.results.model.variables[variable].solution.rename(calc.name))
        return xr.merge(dataarrays)

    # --- Plotting for comparison ---
    fx.plotting.with_plotly(
        get_solutions(calculations, 'Speicher|charge_state').to_dataframe(),
        mode='line', title='Charge State Comparison', ylabel='Charge state', path='results/Charge State.html', save=True
    )

    fx.plotting.with_plotly(
        get_solutions(calculations, 'BHKW2(Q_th)|flow_rate').to_dataframe(),
        mode='line', title='BHKW2(Q_th) Flow Rate Comparison', ylabel='Flow rate', path='results/BHKW2 Thermal Power.html', save=True
    )

    fx.plotting.with_plotly(
        get_solutions(calculations, 'costs(operation)|total_per_timestep').to_dataframe(),
        mode='line', title='Operation Cost Comparison', ylabel='Costs [€]', path='results/Operation Costs.html', save=True
    )

    fx.plotting.with_plotly(
        pd.DataFrame(get_solutions(calculations, 'costs(operation)|total_per_timestep').to_dataframe().sum()).T,
        mode='bar', title='Total Cost Comparison', ylabel='Costs [€]'
    ).update_layout(barmode='group').write_html('results/Total Costs.html')

    fx.plotting.with_plotly(
        pd.DataFrame([calc.durations for calc in calculations], index=[calc.name for calc in calculations]), 'bar'
    ).update_layout(
        title='Duration Comparison', xaxis_title='Calculation type', yaxis_title='Time (s)'
    ).write_html('results/Speed Comparison.html')
