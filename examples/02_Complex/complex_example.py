"""
This script shows how to use the flixOpt framework to model a more complex energy system.
"""

import numpy as np
import pandas as pd
from rich.pretty import pprint  # Used for pretty printing

import flixOpt as fx

if __name__ == '__main__':
    # --- Experiment Options ---
    # Configure options for testing various parameters and behaviors
    check_penalty = False
    excess_penalty = 1e5
    use_chp_with_segments = False
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

    time_series = fx.create_datetime_array('2020-01-01', len(heat_demand), freq='h')

    # --- Define Energy Buses ---
    # Represent different energy carriers (electricity, heat, gas) in the system
    Strom = fx.Bus(
        'Strom',
        excess_penalty_per_flow_hour=excess_penalty,
        medium=fx.Medium('electricity', 'kWh', categories=[fx.MediumCategories.electricity]),
    )
    Fernwaerme = fx.Bus(
        'Fernwärme',
        excess_penalty_per_flow_hour=excess_penalty,
        medium=fx.Medium('heat', 'kWh', categories=[fx.MediumCategories.heat]),
    )
    Gas = fx.Bus(
        'Gas',
        excess_penalty_per_flow_hour=excess_penalty,
        medium=fx.Medium('gas', 'kWh', categories=[fx.MediumCategories.fuel]),
    )

    # --- Define Effects ---
    # Specify effects related to costs, CO2 emissions, and primary energy consumption
    Costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
    CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={Costs: 0.2})
    PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3)

    # --- Define Components ---
    # 1. Define Boiler Component
    # A gas boiler that converts fuel into thermal output, with investment and on-off parameters
    Gaskessel = fx.linear_converters.Boiler(
        'Kessel',
        eta=0.5,  # Efficiency ratio
        on_off_parameters=fx.OnOffParameters(effects_per_running_hour={Costs: 0, CO2: 1000}),  # CO2 emissions per hour
        Q_th=fx.Flow(
            label='Q_th',  # Thermal output
            bus=Fernwaerme,  # Linked bus
            size=fx.InvestParameters(
                fix_effects=1000,  # Fixed investment costs
                fixed_size=50,  # Fixed size
                optional=False,  # Forced investment
                specific_effects={Costs: 10, PE: 2},  # Specific costs
            ),
            load_factor_max=1.0,  # Maximum load factor (50 kW)
            load_factor_min=0.1,  # Minimum load factor (5 kW)
            relative_minimum=5 / 50,  # Minimum part load
            relative_maximum=1,  # Maximum part load
            previous_flow_rate=50,  # Previous flow rate
            flow_hours_total_max=1e6,  # Total energy flow limit
            on_off_parameters=fx.OnOffParameters(
                on_hours_total_min=0,  # Minimum operating hours
                on_hours_total_max=1000,  # Maximum operating hours
                consecutive_on_hours_max=10,  # Max consecutive operating hours
                consecutive_on_hours_min=np.array(
                    [1, 1, 1, 1, 1, 2, 2, 2, 2]
                ),  # min consecutive operation hoursconsecutive_off_hours_max=10,  # Max consecutive off hours
                effects_per_switch_on=0.01,  # Cost per switch-on
                switch_on_total_max=1000,  # Max number of starts
            ),
        ),
        Q_fu=fx.Flow(label='Q_fu', bus=Gas, size=200),
    )

    # 2. Define CHP Unit
    # Combined Heat and Power unit that generates both electricity and heat from fuel
    bhkw = fx.linear_converters.CHP(
        'BHKW2',
        eta_th=0.5,
        eta_el=0.4,
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
        P_el=fx.Flow('P_el', bus=Strom, size=60, relative_minimum=5 / 60),
        Q_th=fx.Flow('Q_th', bus=Fernwaerme, size=1e3),
        Q_fu=fx.Flow('Q_fu', bus=Gas, size=1e3, previous_flow_rate=20),  # The CHP was ON previously
    )

    # 3. Define CHP with Linear Segments
    # This CHP unit uses linear segments for more dynamic behavior over time
    P_el = fx.Flow('P_el', bus=Strom, size=60, previous_flow_rate=20)
    Q_th = fx.Flow('Q_th', bus=Fernwaerme)
    Q_fu = fx.Flow('Q_fu', bus=Gas)
    segmented_conversion_factors = {
        P_el: [(5, 30), (40, 60)],  # Similar to eta_th, each factor here can be an array
        Q_th: [(6, 35), (45, 100)],
        Q_fu: [(12, 70), (90, 200)],
    }

    bhkw_2 = fx.LinearConverter(
        'BHKW2',
        inputs=[Q_fu],
        outputs=[P_el, Q_th],
        segmented_conversion_factors=segmented_conversion_factors,
        on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
    )

    # 4. Define Storage Component
    # Storage with variable size and segmented investment effects
    segmented_investment_effects = (
        [(5, 25), (25, 100)],  # Investment size
        {
            Costs: [(50, 250), (250, 800)],  # Investment costs
            PE: [(5, 25), (25, 100)],  # Primary energy costs
        },
    )

    speicher = fx.Storage(
        'Speicher',
        charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1e4),
        discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1e4),
        capacity_in_flow_hours=fx.InvestParameters(
            effects_in_segments=segmented_investment_effects,  # Investment effects
            optional=False,  # Forced investment
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
        sink=fx.Flow(
            'Q_th_Last',  # Heat sink
            bus=Fernwaerme,  # Linked bus
            size=1,
            fixed_relative_profile=heat_demand,  # Fixed demand profile
        ),
    )

    # 5.b) Gas tariff
    Gasbezug = fx.Source(
        'Gastarif',
        source=fx.Flow(
            'Q_Gas',
            bus=Gas,  # Gas source
            size=1000,  # Nominal size
            effects_per_flow_hour={Costs: 0.04, CO2: 0.3},
        ),
    )

    # 5.c) Feed-in of electricity
    Stromverkauf = fx.Sink(
        'Einspeisung',
        sink=fx.Flow(
            'P_el',
            bus=Strom,  # Feed-in tariff for electricity
            effects_per_flow_hour=-1 * electricity_price,  # Negative price for feed-in
        ),
    )

    # --- Build FlowSystem ---
    # Select components to be included in the final system model
    flow_system = fx.FlowSystem(time_series, last_time_step_hours=None)  # Create FlowSystem

    flow_system.add_elements(Costs, CO2, PE, Gaskessel, Waermelast, Gasbezug, Stromverkauf, speicher)
    flow_system.add_elements(bhkw_2) if use_chp_with_segments else flow_system.add_components(bhkw)

    pprint(flow_system)  # Get a string representation of the FlowSystem

    # --- Solve FlowSystem ---
    calculation = fx.FullCalculation('Sim1', flow_system, 'pyomo', time_indices)
    calculation.do_modeling()

    # Show variables as str (else, you can find them in the results.yaml file
    pprint(calculation.system_model.description_of_constraints())
    pprint(calculation.system_model.description_of_variables())

    calculation.solve(
        fx.solvers.HighsSolver(
            mip_gap=0.005, time_limit_seconds=30
        ),  # Specify which solver you want to use and specify parameters
        save_results='results',  # If and where to save results
    )

    # --- Results ---
    # You can analyze results directly. But it's better to save them to a file and start from there,
    # letting you continue at any time
    # See complex_example_evaluation.py
    used_time_series = time_series[time_indices] if time_indices else time_series
    # Analyze results directly
    fig = fx.plotting.with_plotly(
        data=pd.DataFrame(Gaskessel.Q_th.model.flow_rate.result, index=used_time_series), mode='bar', show=True
    )
