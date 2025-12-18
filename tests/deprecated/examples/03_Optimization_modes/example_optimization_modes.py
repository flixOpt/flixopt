"""
This script demonstrates how to use the different calculation types in the flixopt framework
to model the same energy system. The results will be compared to each other.
"""

import pathlib

import pandas as pd
import xarray as xr

import flixopt as fx


# Get solutions for plotting for different optimizations
def get_solutions(optimizations: list, variable: str) -> xr.Dataset:
    dataarrays = []
    for optimization in optimizations:
        if optimization.name == 'Segmented':
            # SegmentedOptimization requires special handling to remove overlaps
            dataarrays.append(optimization.results.solution_without_overlap(variable).rename(optimization.name))
        else:
            # For Full and Clustered, access solution from the flow_system
            dataarrays.append(optimization.flow_system.solution[variable].rename(optimization.name))
    return xr.merge(dataarrays, join='outer')


if __name__ == '__main__':
    fx.CONFIG.exploring()

    # Calculation Types
    full, segmented, aggregated = True, True, True

    # Segmented Properties
    segment_length, overlap_length = 96, 1

    # Clustering Properties
    n_clusters = 4
    cluster_duration = '6h'
    include_storage = False
    keep_extreme_periods = True
    imbalance_penalty = 1e5  # or set to None if not needed

    # Data Import
    data_import = pd.read_csv(
        pathlib.Path(__file__).parents[4] / 'docs' / 'notebooks' / 'data' / 'Zeitreihen2020.csv', index_col=0
    ).sort_index()
    filtered_data = data_import['2020-01-01':'2020-01-07 23:45:00']
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
    TS_electricity_demand = fx.TimeSeriesData(electricity_demand, clustering_weight=0.7)
    TS_electricity_price_sell = fx.TimeSeriesData(-(electricity_price - 0.5), clustering_group='p_el')
    TS_electricity_price_buy = fx.TimeSeriesData(electricity_price + 0.5, clustering_group='p_el')

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Strom', carrier='electricity', imbalance_penalty_per_flow_hour=imbalance_penalty),
        fx.Bus('Fernwärme', carrier='heat', imbalance_penalty_per_flow_hour=imbalance_penalty),
        fx.Bus('Gas', carrier='gas', imbalance_penalty_per_flow_hour=imbalance_penalty),
        fx.Bus('Kohle', carrier='fuel', imbalance_penalty_per_flow_hour=imbalance_penalty),
    )

    # Effects
    costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
    CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
    PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie')

    # Component Definitions

    # 1. Boiler
    a_gaskessel = fx.linear_converters.Boiler(
        'Kessel',
        thermal_efficiency=0.85,
        thermal_flow=fx.Flow(label='Q_th', bus='Fernwärme'),
        fuel_flow=fx.Flow(
            label='Q_fu',
            bus='Gas',
            size=95,
            relative_minimum=12 / 95,
            previous_flow_rate=20,
            status_parameters=fx.StatusParameters(effects_per_startup=1000),
        ),
    )

    # 2. CHP
    a_kwk = fx.linear_converters.CHP(
        'BHKW2',
        thermal_efficiency=0.58,
        electrical_efficiency=0.22,
        status_parameters=fx.StatusParameters(effects_per_startup=24000),
        electrical_flow=fx.Flow('P_el', bus='Strom', size=200),
        thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=200),
        fuel_flow=fx.Flow('Q_fu', bus='Kohle', size=288, relative_minimum=87 / 288, previous_flow_rate=100),
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
        'Wärmelast', inputs=[fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=TS_heat_demand)]
    )

    # Electricity Feed-in
    a_strom_last = fx.Sink(
        'Stromlast', inputs=[fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=TS_electricity_demand)]
    )

    # Gas Tariff
    a_gas_tarif = fx.Source(
        'Gastarif',
        outputs=[
            fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={costs.label: gas_price, CO2.label: 0.3})
        ],
    )

    # Coal Tariff
    a_kohle_tarif = fx.Source(
        'Kohletarif',
        outputs=[fx.Flow('Q_Kohle', bus='Kohle', size=1000, effects_per_flow_hour={costs.label: 4.6, CO2.label: 0.3})],
    )

    # Electricity Tariff and Feed-in
    a_strom_einspeisung = fx.Sink(
        'Einspeisung', inputs=[fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour=TS_electricity_price_sell)]
    )

    a_strom_tarif = fx.Source(
        'Stromtarif',
        outputs=[
            fx.Flow(
                'P_el',
                bus='Strom',
                size=1000,
                effects_per_flow_hour={costs.label: TS_electricity_price_buy, CO2.label: 0.3},
            )
        ],
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
    flow_system.topology.plot()

    # Optimizations
    optimizations: list[fx.Optimization | fx.ClusteredOptimization | fx.SegmentedOptimization] = []

    if full:
        optimization = fx.Optimization('Full', flow_system.copy())
        optimization.do_modeling()
        optimization.solve(fx.solvers.HighsSolver(0.01 / 100, 60))
        optimizations.append(optimization)

    if segmented:
        optimization = fx.SegmentedOptimization('Segmented', flow_system.copy(), segment_length, overlap_length)
        optimization.do_modeling_and_solve(fx.solvers.HighsSolver(0.01 / 100, 60))
        optimizations.append(optimization)

    if aggregated:
        # Use the new transform.cluster() API
        time_series_for_high_peaks = [TS_heat_demand] if keep_extreme_periods else None
        time_series_for_low_peaks = [TS_electricity_demand, TS_heat_demand] if keep_extreme_periods else None

        clustered_fs = flow_system.copy().transform.cluster(
            n_clusters=n_clusters,
            cluster_duration=cluster_duration,
            include_storage=include_storage,
            time_series_for_high_peaks=time_series_for_high_peaks,
            time_series_for_low_peaks=time_series_for_low_peaks,
        )
        clustered_fs.optimize(fx.solvers.HighsSolver(0.01 / 100, 60))

        # Wrap in a simple object for compatibility with comparison code
        class ClusteredResult:
            def __init__(self, name, fs):
                self.name = name
                self.flow_system = fs
                self.durations = {'total': 0}  # Placeholder

        optimization = ClusteredResult('Clustered', clustered_fs)
        optimizations.append(optimization)

    # --- Plotting for comparison ---
    fx.plotting.with_plotly(
        get_solutions(optimizations, 'Speicher|charge_state'),
        mode='line',
        title='Charge State Comparison',
        ylabel='Charge state',
        xlabel='Time in h',
    ).write_html('results/Charge State.html')

    fx.plotting.with_plotly(
        get_solutions(optimizations, 'BHKW2(Q_th)|flow_rate'),
        mode='line',
        title='BHKW2(Q_th) Flow Rate Comparison',
        ylabel='Flow rate',
        xlabel='Time in h',
    ).write_html('results/BHKW2 Thermal Power.html')

    fx.plotting.with_plotly(
        get_solutions(optimizations, 'costs(temporal)|per_timestep'),
        mode='line',
        title='Operation Cost Comparison',
        ylabel='Costs [€]',
        xlabel='Time in h',
    ).write_html('results/Operation Costs.html')

    fx.plotting.with_plotly(
        get_solutions(optimizations, 'costs(temporal)|per_timestep').sum('time'),
        mode='stacked_bar',
        title='Total Cost Comparison',
        ylabel='Costs [€]',
    ).update_layout(barmode='group').write_html('results/Total Costs.html')

    fx.plotting.with_plotly(
        pd.DataFrame(
            [calc.durations for calc in optimizations], index=[calc.name for calc in optimizations]
        ).to_xarray(),
        mode='stacked_bar',
    ).update_layout(title='Duration Comparison', xaxis_title='Optimization type', yaxis_title='Time (s)').write_html(
        'results/Speed Comparison.html'
    )
