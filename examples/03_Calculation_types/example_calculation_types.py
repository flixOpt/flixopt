"""
This script demonstrates how to use the different calcualtion types in the flixOPt framework
to model the same energy system. THe Results will be compared to each other.
"""

import pathlib
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from rich.pretty import pprint  # Used for pretty printing

import flixopt as fx

if __name__ == '__main__':
    # Calculation Types
    full, segmented, aggregated = True, True, True

    # Segmented Properties
    segment_length, overlap_length = 96, 1

    # Aggregated Properties
    aggregation_parameters = fx.AggregationParameters(
        hours_per_period=6,
        nr_of_periods=4,
        fix_storage_flows=True,
        aggregate_data_and_fix_non_binary_vars=True,
        percentage_of_period_freedom=0,
        penalty_of_period_freedom=0,
    )
    keep_extreme_periods = True

    # Data Import
    data_import = pd.read_csv(pathlib.Path('Zeitreihen2020.csv'), index_col=0).sort_index()
    filtered_data = data_import['2020-01-01':'2020-01-2 23:45:00']
    # filtered_data = data_import[0:500]  # Alternatively filter by index

    filtered_data.index = pd.to_datetime(filtered_data.index)
    datetime_series = np.array(filtered_data.index).astype('datetime64')

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

    # Bus Definitions
    excess_penalty = 1e5  # or set to None if not needed
    Strom = fx.Bus('Strom', excess_penalty_per_flow_hour=excess_penalty)
    Fernwaerme = fx.Bus('Fernwärme', excess_penalty_per_flow_hour=excess_penalty)
    Gas = fx.Bus('Gas', excess_penalty_per_flow_hour=excess_penalty)
    Kohle = fx.Bus('Kohle', excess_penalty_per_flow_hour=excess_penalty)

    # Effects
    costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
    CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
    PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie')

    # Component Definitions

    # 1. Boiler
    a_gaskessel = fx.linear_converters.Boiler(
        'Kessel',
        eta=0.85,
        Q_th=fx.Flow(label='Q_th', bus=Fernwaerme),
        Q_fu=fx.Flow(
            label='Q_fu',
            bus=Gas,
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
        P_el=fx.Flow('P_el', bus=Strom, size=200),
        Q_th=fx.Flow('Q_th', bus=Fernwaerme, size=200),
        Q_fu=fx.Flow('Q_fu', bus=Kohle, size=288, relative_minimum=87 / 288, previous_flow_rate=100),
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
        charging=fx.Flow('Q_th_load', size=137, bus=Fernwaerme),
        discharging=fx.Flow('Q_th_unload', size=158, bus=Fernwaerme),
    )

    # 4. Sinks and Sources
    # Heat Load Profile
    a_waermelast = fx.Sink(
        'Wärmelast', sink=fx.Flow('Q_th_Last', bus=Fernwaerme, size=1, fixed_relative_profile=TS_heat_demand)
    )

    # Electricity Feed-in
    a_strom_last = fx.Sink(
        'Stromlast', sink=fx.Flow('P_el_Last', bus=Strom, size=1, fixed_relative_profile=TS_electricity_demand)
    )

    # Gas Tariff
    a_gas_tarif = fx.Source(
        'Gastarif', source=fx.Flow('Q_Gas', bus=Gas, size=1000, effects_per_flow_hour={costs: gas_price, CO2: 0.3})
    )

    # Coal Tariff
    a_kohle_tarif = fx.Source(
        'Kohletarif', source=fx.Flow('Q_Kohle', bus=Kohle, size=1000, effects_per_flow_hour={costs: 4.6, CO2: 0.3})
    )

    # Electricity Tariff and Feed-in
    a_strom_einspeisung = fx.Sink(
        'Einspeisung', sink=fx.Flow('P_el', bus=Strom, size=1000, effects_per_flow_hour=TS_electricity_price_sell)
    )

    a_strom_tarif = fx.Source(
        'Stromtarif',
        source=fx.Flow('P_el', bus=Strom, size=1000, effects_per_flow_hour={costs: TS_electricity_price_buy, CO2: 0.3}),
    )

    # Flow System Setup
    flow_system = fx.FlowSystem(datetime_series)
    flow_system.add_effects(costs, CO2, PE)
    flow_system.add_components(
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
    flow_system.visualize_network(controls=False)
    # Calculations
    kinds = ['Full', 'Segmented', 'Aggregated']
    calculations: dict = {key: None for key in kinds}
    results: dict = {key: None for key in kinds}

    if full:
        calculation = fx.FullCalculation('fullModel', flow_system, 'pyomo')
        calculation.do_modeling()
        calculation.solve(fx.solvers.HighsSolver())
        calculations['Full'] = calculation
        results['Full'] = calculations['Full'].results()

    if segmented:
        calculation = fx.SegmentedCalculation('segModel', flow_system, segment_length, overlap_length)
        calculation.do_modeling_and_solve(fx.solvers.HighsSolver())
        calculations['Segmented'] = calculation
        results['Segmented'] = calculations['Segmented'].results(combined_arrays=True)

    if aggregated:
        if keep_extreme_periods:
            aggregation_parameters.time_series_for_high_peaks = [TS_heat_demand]
            aggregation_parameters.time_series_for_low_peaks = [TS_electricity_demand, TS_heat_demand]
        calculation = fx.AggregatedCalculation('aggModel', flow_system, aggregation_parameters)
        calculation.do_modeling()
        calculation.solve(fx.solvers.HighsSolver())
        calculations['Aggregated'] = calculation
        results['Aggregated'] = calculations['Aggregated'].results()
    pprint(results)

    def extract_result(results_data: dict[str, dict], keys: List[str]) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Function to retrieve values from a nested dictionary.
        Tries to get the wanted value for eachnkey in the first layer of the dict.
        Returns a dict with one key value pair for each dict it found a value in.
        """

        def get_nested_value(d, ks):
            for k in ks:
                if isinstance(d, dict):
                    d = d.get(k, None)
                else:
                    return None
            return d

        return {kind: get_nested_value(results_data.get(kind, {}), keys) for kind in results_data.keys()}

    if calculations['Full'] is not None:
        time_series_used = calculations['Full'].system_model.time_series
        time_series_used_w_end = calculations['Full'].system_model.time_series_with_end
    else:
        time_series_used = calculations['Aggregated'].system_model.time_series
        time_series_used_w_end = calculations['Aggregated'].system_model.time_series_with_end

    data = pd.DataFrame(
        extract_result(results, ['Components', 'Speicher', 'charge_state']), index=time_series_used_w_end
    )
    fig = fx.plotting.with_plotly(data, 'line')
    fig.update_layout(title='Charge State Comparison', xaxis_title='Time', yaxis_title='Charge state')
    fig.write_html('results/Charge State.html')

    data = pd.DataFrame(extract_result(results, ['Components', 'BHKW2', 'Q_th', 'flow_rate']), index=time_series_used)
    fig = fx.plotting.with_plotly(data, 'line')
    fig.update_layout(title='BHKW2 Q_th Flow Rate Comparison', xaxis_title='Time', yaxis_title='Flow rate')
    fig.write_html('results/BHKW2 Thermal Power.html')

    data = pd.DataFrame(
        extract_result(results, ['Effects', 'costs', 'operation', 'operation_sum_TS']),
        index=calculations['Full'].system_model.time_series,
    )
    fig = fx.plotting.with_plotly(data, 'line')
    fig.update_layout(title='Cost Comparison', xaxis_title='Time', yaxis_title='Costs (€)')
    fig.write_html('results/Operation Costs.html')

    data = pd.DataFrame(
        extract_result(results, ['Effects', 'costs', 'operation', 'operation_sum_TS']), index=time_series_used
    )
    data = pd.DataFrame(data.sum()).T
    fig = fx.plotting.with_plotly(data, 'bar')
    fig.update_layout(title='Total Cost Comparison', yaxis_title='Costs (€)', barmode='group')
    fig.write_html('results/Total Costs.html')

    duration_data = pd.DataFrame(
        {
            'Full': [calculations['Full'].durations.get(key, 0) for key in calculations['Aggregated'].durations],
            'Aggregated': [
                calculations['Aggregated'].durations.get(key, 0) for key in calculations['Aggregated'].durations
            ],
            'Segmented': [
                calculations['Segmented'].durations.get(key, 0) for key in calculations['Aggregated'].durations
            ],
        },
        index=list(calculations['Aggregated'].durations.keys()),
    ).T
    fig = fx.plotting.with_plotly(duration_data, 'bar')
    fig.update_layout(title='Duration Comparison', xaxis_title='Calculation type', yaxis_title='Time (s)')
    fig.write_html('results/Speed Comparison.html')
