"""
This script shows how to use the flixopt framework to model a more complex energy system.
"""

import numpy as np
import pandas as pd

import flixopt as fx

if __name__ == '__main__':
    fx.CONFIG.exploring()

    # --- Experiment Options ---
    # Configure options for testing various parameters and behaviors
    check_penalty = False
    excess_penalty = 1e5
    use_chp_with_piecewise_conversion = True
    time_indices = None  # Define specific time steps for custom calculations, or use the entire series

    # --- Define Demand and Price Profiles ---
    # Input data for electricity and heat demands, as well as electricity price
    electricity_demand = np.array([70, 80, 90, 90, 90, 90, 90, 90, 90])
    heat_demand = (
        np.array([30, 0, 90, 110, 2000, 20, 20, 20, 20])
        if check_penalty
        else np.array([30, 0, 90, 110, 110, 20, 20, 20, 20])
    )
    electricity_price = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40])

    # --- Define the Flow System, that will hold all elements, and the time steps you want to model ---
    timesteps = pd.date_range('2020-01-01', periods=len(heat_demand), freq='h')
    flow_system = fx.FlowSystem(timesteps)  # Create FlowSystem

    # --- Define Energy Buses ---
    # Represent node balances (inputs=outputs) for the different energy carriers (electricity, heat, gas) in the system
    flow_system.add_elements(
        fx.Bus('Strom', excess_penalty_per_flow_hour=excess_penalty),
        fx.Bus('Fernwärme', excess_penalty_per_flow_hour=excess_penalty),
        fx.Bus('Gas', excess_penalty_per_flow_hour=excess_penalty),
    )

    # --- Define Effects ---
    # Specify effects related to costs, CO2 emissions, and primary energy consumption
    Costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True, share_from_temporal={'CO2': 0.2})
    CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen')
    PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3)

    # --- Define Components ---
    # 1. Define Boiler Component
    # A gas boiler that converts fuel into thermal output, with investment and on-off parameters
    Gaskessel = fx.linear_converters.Boiler(
        'Kessel',
        thermal_efficiency=0.5,  # Efficiency ratio
        on_off_parameters=fx.OnOffParameters(
            effects_per_running_hour={Costs.label: 0, CO2.label: 1000}
        ),  # CO2 emissions per hour
        thermal_flow=fx.Flow(
            label='Q_th',  # Thermal output
            bus='Fernwärme',  # Linked bus
            size=fx.InvestParameters(
                effects_of_investment=1000,  # Fixed investment costs
                fixed_size=50,  # Fixed size
                mandatory=True,  # Forced investment
                effects_of_investment_per_size={Costs.label: 10, PE.label: 2},  # Specific costs
            ),
            load_factor_max=1.0,  # Maximum load factor (50 kW)
            load_factor_min=0.1,  # Minimum load factor (5 kW)
            relative_minimum=5 / 50,  # Minimum part load
            relative_maximum=1,  # Maximum part load
            previous_flow_rate=50,  # Previous flow rate
            flow_hours_max=1e6,  # Total energy flow limit
            on_off_parameters=fx.OnOffParameters(
                on_hours_min=0,  # Minimum operating hours
                on_hours_max=1000,  # Maximum operating hours
                consecutive_on_hours_max=10,  # Max consecutive operating hours
                consecutive_on_hours_min=np.array([1, 1, 1, 1, 1, 2, 2, 2, 2]),  # min consecutive operation hours
                consecutive_off_hours_max=10,  # Max consecutive off hours
                effects_per_switch_on=0.01,  # Cost per switch-on
                switch_on_max=1000,  # Max number of starts
            ),
        ),
        fuel_flow=fx.Flow(label='Q_fu', bus='Gas', size=200),
    )

    # 2. Define CHP Unit
    # Combined Heat and Power unit that generates both electricity and heat from fuel
    bhkw = fx.linear_converters.CHP(
        'BHKW2',
        thermal_efficiency=0.5,
        electrical_efficiency=0.4,
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
        electrical_flow=fx.Flow('P_el', bus='Strom', size=60, relative_minimum=5 / 60),
        thermal_flow=fx.Flow('Q_th', bus='Fernwärme', size=1e3),
        fuel_flow=fx.Flow('Q_fu', bus='Gas', size=1e3, previous_flow_rate=20),  # The CHP was ON previously
    )

    # 3. Define CHP with Piecewise Conversion
    # This CHP unit uses piecewise conversion for more dynamic behavior over time
    P_el = fx.Flow('P_el', bus='Strom', size=60, previous_flow_rate=20)
    Q_th = fx.Flow('Q_th', bus='Fernwärme')
    Q_fu = fx.Flow('Q_fu', bus='Gas')
    piecewise_conversion = fx.PiecewiseConversion(
        {
            P_el.label: fx.Piecewise([fx.Piece(5, 30), fx.Piece(40, 60)]),
            Q_th.label: fx.Piecewise([fx.Piece(6, 35), fx.Piece(45, 100)]),
            Q_fu.label: fx.Piecewise([fx.Piece(12, 70), fx.Piece(90, 200)]),
        }
    )

    bhkw_2 = fx.LinearConverter(
        'BHKW2',
        inputs=[Q_fu],
        outputs=[P_el, Q_th],
        piecewise_conversion=piecewise_conversion,
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
    )

    # 4. Define Storage Component
    # Storage with variable size and piecewise investment effects
    segmented_investment_effects = fx.PiecewiseEffects(
        piecewise_origin=fx.Piecewise([fx.Piece(5, 25), fx.Piece(25, 100)]),
        piecewise_shares={
            Costs.label: fx.Piecewise([fx.Piece(50, 250), fx.Piece(250, 800)]),
            PE.label: fx.Piecewise([fx.Piece(5, 25), fx.Piece(25, 100)]),
        },
    )

    speicher = fx.Storage(
        'Speicher',
        charging=fx.Flow('Q_th_load', bus='Fernwärme', size=1e4),
        discharging=fx.Flow('Q_th_unload', bus='Fernwärme', size=1e4),
        capacity_in_flow_hours=fx.InvestParameters(
            piecewise_effects_of_investment=segmented_investment_effects,  # Investment effects
            mandatory=True,  # Forced investment
            minimum_size=0,
            maximum_size=1000,  # Optimizing between 0 and 1000 kWh
        ),
        initial_charge_state=0,  # Initial charge state
        maximal_final_charge_state=10,  # Maximum final charge state
        eta_charge=0.9,
        eta_discharge=1,  # Charge/discharge efficiency
        relative_loss_per_hour=0.08,  # Energy loss per hour, relative to current charge state
        prevent_simultaneous_charge_and_discharge=True,  # Prevent simultaneous charge/discharge
    )

    # 5. Define Sinks and Sources
    # 5.a) Heat demand profile
    Waermelast = fx.Sink(
        'Wärmelast',
        inputs=[
            fx.Flow(
                'Q_th_Last',  # Heat sink
                bus='Fernwärme',  # Linked bus
                size=1,
                fixed_relative_profile=heat_demand,  # Fixed demand profile
            )
        ],
    )

    # 5.b) Gas tariff
    Gasbezug = fx.Source(
        'Gastarif',
        outputs=[
            fx.Flow(
                'Q_Gas',
                bus='Gas',  # Gas source
                size=1000,  # Nominal size
                effects_per_flow_hour={Costs.label: 0.04, CO2.label: 0.3},
            )
        ],
    )

    # 5.c) Feed-in of electricity
    Stromverkauf = fx.Sink(
        'Einspeisung',
        inputs=[
            fx.Flow(
                'P_el',
                bus='Strom',  # Feed-in tariff for electricity
                effects_per_flow_hour=-1 * electricity_price,  # Negative price for feed-in
            )
        ],
    )

    # --- Build FlowSystem ---
    # Select components to be included in the flow system
    flow_system.add_elements(Costs, CO2, PE, Gaskessel, Waermelast, Gasbezug, Stromverkauf, speicher)
    flow_system.add_elements(bhkw_2) if use_chp_with_piecewise_conversion else flow_system.add_elements(bhkw)

    print(flow_system)  # Get a string representation of the FlowSystem
    try:
        flow_system.start_network_app()  # Start the network app
    except ImportError as e:
        print(f'Network app requires extra dependencies: {e}')

    # --- Solve FlowSystem ---
    calculation = fx.FullCalculation('complex example', flow_system, time_indices)
    calculation.do_modeling()

    calculation.solve(fx.solvers.HighsSolver(0.01, 60))

    # --- Results ---
    # You can analyze results directly or save them to file and reload them later.
    calculation.results.to_file()

    # But let's plot some results anyway
    calculation.results.plot_heatmap('BHKW2(Q_th)|flow_rate')
    calculation.results['BHKW2'].plot_node_balance()
    calculation.results['Speicher'].plot_charge_state()
    calculation.results['Fernwärme'].plot_node_balance_pie()
