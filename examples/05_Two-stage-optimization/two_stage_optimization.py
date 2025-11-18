"""
This script demonstrates how to use downsampling of a FlowSystem to effectively reduce the size of a model.
This can be very useful when working with large models or during development,
as it can drastically reduce the computational time.
This leads to faster results and easier debugging.
A common use case is to optimize the investments of a model with a downsampled version of the original model, and then fix the computed sizes when calculating the actual dispatch.
While the final optimum might differ from the global optimum, the solving will be much faster.
"""

import pathlib
import timeit

import pandas as pd
import xarray as xr
from loguru import logger

import flixopt as fx

if __name__ == '__main__':
    fx.CONFIG.exploring()

    # Data Import
    data_import = pd.read_csv(
        pathlib.Path(__file__).parent.parent / 'resources' / 'Zeitreihen2020.csv', index_col=0
    ).sort_index()
    filtered_data = data_import[:500]

    filtered_data.index = pd.to_datetime(filtered_data.index)
    timesteps = filtered_data.index

    # Access specific columns and convert to 1D-numpy array
    electricity_demand = filtered_data['P_Netz/MW'].to_numpy()
    heat_demand = filtered_data['Q_Netz/MW'].to_numpy()
    electricity_price = filtered_data['Strompr.€/MWh'].to_numpy()
    gas_price = filtered_data['Gaspr.€/MWh'].to_numpy()

    flow_system = fx.FlowSystem(timesteps)
    flow_system.add_elements(
        fx.Bus('Strom'),
        fx.Bus('Fernwärme'),
        fx.Bus('Gas'),
        fx.Bus('Kohle'),
        fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True),
        fx.Effect('CO2', 'kg', 'CO2_e-Emissionen'),
        fx.Effect('PE', 'kWh_PE', 'Primärenergie'),
        fx.linear_converters.Boiler(
            'Kessel',
            thermal_efficiency=0.85,
            thermal_flow=fx.Flow(label='Q_th', bus='Fernwärme'),
            fuel_flow=fx.Flow(
                label='Q_fu',
                bus='Gas',
                size=fx.InvestParameters(
                    effects_of_investment_per_size={'costs': 1_000}, minimum_size=10, maximum_size=500
                ),
                relative_minimum=0.2,
                previous_flow_rate=20,
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=300),
            ),
        ),
        fx.linear_converters.CHP(
            'BHKW2',
            thermal_efficiency=0.58,
            electrical_efficiency=0.22,
            on_off_parameters=fx.OnOffParameters(
                effects_per_switch_on=1_000, consecutive_on_hours_min=10, consecutive_off_hours_min=10
            ),
            electrical_flow=fx.Flow('P_el', bus='Strom'),
            thermal_flow=fx.Flow('Q_th', bus='Fernwärme'),
            fuel_flow=fx.Flow(
                'Q_fu',
                bus='Kohle',
                size=fx.InvestParameters(
                    effects_of_investment_per_size={'costs': 3_000}, minimum_size=10, maximum_size=500
                ),
                relative_minimum=0.3,
                previous_flow_rate=100,
            ),
        ),
        fx.Storage(
            'Speicher',
            capacity_in_flow_hours=fx.InvestParameters(
                minimum_size=10, maximum_size=1000, effects_of_investment_per_size={'costs': 60}
            ),
            initial_charge_state='equals_final',
            eta_charge=1,
            eta_discharge=1,
            relative_loss_per_hour=0.001,
            prevent_simultaneous_charge_and_discharge=True,
            charging=fx.Flow('Q_th_load', size=137, bus='Fernwärme'),
            discharging=fx.Flow('Q_th_unload', size=158, bus='Fernwärme'),
        ),
        fx.Sink(
            'Wärmelast', inputs=[fx.Flow('Q_th_Last', bus='Fernwärme', size=1, fixed_relative_profile=heat_demand)]
        ),
        fx.Source(
            'Gastarif',
            outputs=[fx.Flow('Q_Gas', bus='Gas', size=1000, effects_per_flow_hour={'costs': gas_price, 'CO2': 0.3})],
        ),
        fx.Source(
            'Kohletarif',
            outputs=[fx.Flow('Q_Kohle', bus='Kohle', size=1000, effects_per_flow_hour={'costs': 4.6, 'CO2': 0.3})],
        ),
        fx.Source(
            'Einspeisung',
            outputs=[
                fx.Flow(
                    'P_el', bus='Strom', size=1000, effects_per_flow_hour={'costs': electricity_price + 0.5, 'CO2': 0.3}
                )
            ],
        ),
        fx.Sink(
            'Stromlast',
            inputs=[fx.Flow('P_el_Last', bus='Strom', size=1, fixed_relative_profile=electricity_demand)],
        ),
        fx.Source(
            'Stromtarif',
            outputs=[
                fx.Flow('P_el', bus='Strom', size=1000, effects_per_flow_hour={'costs': electricity_price, 'CO2': 0.3})
            ],
        ),
    )

    # Separate optimization of flow sizes and dispatch
    start = timeit.default_timer()
    calculation_sizing = fx.FullCalculation('Sizing', flow_system.resample('2h'))
    calculation_sizing.do_modeling()
    calculation_sizing.solve(fx.solvers.HighsSolver(0.1 / 100, 60))
    timer_sizing = timeit.default_timer() - start

    start = timeit.default_timer()
    calculation_dispatch = fx.FullCalculation('Dispatch', flow_system)
    calculation_dispatch.do_modeling()
    calculation_dispatch.fix_sizes(calculation_sizing.results.solution)
    calculation_dispatch.solve(fx.solvers.HighsSolver(0.1 / 100, 60))
    timer_dispatch = timeit.default_timer() - start

    if (calculation_dispatch.results.sizes().round(5) == calculation_sizing.results.sizes().round(5)).all().item():
        logger.info('Sizes were correctly equalized')
    else:
        raise RuntimeError('Sizes were not correctly equalized')

    # Optimization of both flow sizes and dispatch together
    start = timeit.default_timer()
    calculation_combined = fx.FullCalculation('Combined', flow_system)
    calculation_combined.do_modeling()
    calculation_combined.solve(fx.solvers.HighsSolver(0.1 / 100, 600))
    timer_combined = timeit.default_timer() - start

    # Comparison of results
    comparison = xr.concat(
        [calculation_combined.results.solution, calculation_dispatch.results.solution], dim='mode'
    ).assign_coords(mode=['Combined', 'Two-stage'])
    comparison['Duration [s]'] = xr.DataArray([timer_combined, timer_sizing + timer_dispatch], dims='mode')

    comparison_main = comparison[
        [
            'Duration [s]',
            'costs',
            'costs(periodic)',
            'costs(temporal)',
            'BHKW2(Q_fu)|size',
            'Kessel(Q_fu)|size',
            'Speicher|size',
        ]
    ]
    comparison_main = xr.concat(
        [
            comparison_main,
            (
                (comparison_main.sel(mode='Two-stage') - comparison_main.sel(mode='Combined'))
                / comparison_main.sel(mode='Combined')
                * 100
            ).assign_coords(mode='Diff [%]'),
        ],
        dim='mode',
    )

    print(comparison_main.to_pandas().T.round(2))
