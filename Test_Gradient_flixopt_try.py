import numpy as np
import pandas as pd
import flixopt as fx

if __name__ == '__main__':

    elec_demand_per_h = np.array([50, 50, 50, 50, 50, 50, 50, 50,50,50,50,50,50,50,
                                  0, 0,0,0,0,0,50,50])
                                  #50, 50, 50, 50, 40, 30, 50, 50, 50, 50])

    timesteps = pd.date_range('2020-01-01', periods=len(elec_demand_per_h), freq='h',name='time')
    flow_system = fx.FlowSystem(timesteps=timesteps)

    flow_system.add_elements(fx.Bus(label='Strom'))

    costs = fx.Effect(
        label='costs',
        unit='€',
        description='Kosten',
        is_standard=True,
        is_objective=True,
    )

    elec_source_1 = fx.Source(
        label='Stromtarif_1',
        outputs=[fx.Flow(label='Netzbezug_Strom', bus='Strom',effects_per_flow_hour=10)]
    )

    elec_source_2 = fx.Source(
        label='Stromtarif_2',
        outputs=[fx.Flow(label='Netzbezug_Strom_Gradient',
                         bus='Strom',
                         size = 45,
                         #size = fx.InvestParameters(minimum_size=10,maximum_size=45),
                         relative_minimum=15/45,
                         status_parameters=fx.StatusParameters(force_startup_tracking=True),
                         max_increasing_gradient_abs=10,
                         max_decreasing_gradient_abs=10,
                         relax_on_startup=False,
                         relax_on_shutdown=True,
                         relax_on_startup_to_min_rate=True,
                         relax_on_shutdown_to_min_rate=False,
                         effects_per_flow_hour=1,
                         previous_flow_rate=0)
                 ]
    )

    power_sink = fx.Sink(
        label='Strombedarf', inputs=[fx.Flow(label='P_el', bus='Strom', size=1, fixed_relative_profile=elec_demand_per_h  )]
    )

    power_sink_help = fx.Sink(
        label='Strombedarf_help', inputs=[fx.Flow(label='P_el', bus='Strom', size=100  )]
    )


    flow_system.add_elements(costs,elec_source_1,elec_source_2,power_sink)
    flow_system.build_model()
    flow_system.solve(fx.solvers.GurobiSolver(mip_gap=0, time_limit_seconds=30))


    flow_system.statistics.plot.balance('Strom',show=True)
    flow_system.statistics.plot.sankey.flows()
