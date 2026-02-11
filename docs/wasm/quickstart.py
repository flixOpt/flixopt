import marimo

__generated_with = '0.19.9'
app = marimo.App(app_title='flixopt Quickstart')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _(mo):
    import sys

    if 'pyodide' in sys.modules:
        with mo.status.spinner('Installing packages...'):
            import micropip  # noqa: I001

            # highspy is a pre-built Pyodide package (C extension) — install it
            # from Pyodide's package index before flixopt tries to find it on PyPI
            await micropip.install('highspy')
            # keep_going=True skips unavailable C-extension deps (e.g. netcdf4)
            await micropip.install('flixopt', keep_going=True)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quickstart

    Heat a small workshop with a gas boiler - the minimal working example.

    This notebook introduces the **core concepts** of flixopt:

    - **FlowSystem**: The container for your energy system model
    - **Bus**: Balance nodes where energy flows meet
    - **Effect**: Quantities to track and optimize (costs, emissions)
    - **Components**: Equipment like boilers, sources, and sinks
    - **Flow**: Connections between components and buses

    > **Tip**: This notebook is fully editable! Modify any cell and press
    > `Shift+Enter` to re-run it. Changes propagate automatically.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import plotly.express as px
    import xarray as xr

    import flixopt as fx

    return fx, pd, px, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the Time Horizon

    Every optimization needs a time horizon. Here we model a simple 4-hour period:
    """)
    return


@app.cell
def _(pd):
    timesteps = pd.date_range('2024-01-15 08:00', periods=4, freq='h')
    print(f'Optimizing from {timesteps[0]} to {timesteps[-1]}')
    return (timesteps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the Heat Demand

    The workshop has varying heat demand throughout the morning:
    """)
    return


@app.cell
def _(px, timesteps, xr):
    # Heat demand in kW for each hour - using xarray
    heat_demand = xr.DataArray(
        [30, 50, 45, 25],
        dims=['time'],
        coords={'time': timesteps},
        name='Heat Demand [kW]',
    )

    # Visualize the demand with plotly
    fig = px.bar(x=heat_demand.time.values, y=heat_demand.values, labels={'x': 'Time', 'y': 'Heat Demand [kW]'})
    fig  # noqa: B018 — marimo displays the last expression as cell output
    return (heat_demand,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the Energy System Model

    Now we create the FlowSystem and add all components:

    ```
      Gas Supply ──► [Gas Bus] ──► Boiler ──► [Heat Bus] ──► Workshop
           €                         η=90%                    Demand
    ```
    """)
    return


@app.cell
def _(fx, heat_demand, timesteps):
    # Create the FlowSystem container
    flow_system = fx.FlowSystem(timesteps)

    flow_system.add_elements(
        # === Buses: Balance nodes for energy carriers ===
        fx.Bus('Gas'),  # Natural gas network connection
        fx.Bus('Heat'),  # Heat distribution within workshop
        # === Effect: What we want to minimize ===
        fx.Effect('costs', '€', 'Total Costs', is_standard=True, is_objective=True),
        # === Gas Supply: Unlimited gas at 0.08 €/kWh ===
        fx.Source(
            'GasGrid',
            outputs=[fx.Flow('Gas', bus='Gas', size=1000, effects_per_flow_hour=0.08)],
        ),
        # === Boiler: Converts gas to heat at 90% efficiency ===
        fx.linear_converters.Boiler(
            'Boiler',
            thermal_efficiency=0.9,
            thermal_flow=fx.Flow('Heat', bus='Heat', size=100),  # 100 kW capacity
            fuel_flow=fx.Flow('Gas', bus='Gas'),
        ),
        # === Workshop: Heat demand that must be met ===
        fx.Sink(
            'Workshop',
            inputs=[fx.Flow('Heat', bus='Heat', size=1, fixed_relative_profile=heat_demand.values)],
        ),
    )
    return (flow_system,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the Optimization

    Now we solve the model using the HiGHS solver (open-source, included with flixopt):
    """)
    return


@app.cell
def _(flow_system, fx):
    flow_system.optimize(fx.solvers.HighsSolver())
    solved = flow_system
    return (solved,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analyze Results

    ### Heat Balance

    The `statistics.plot.balance()` method shows how each bus is balanced:
    """)
    return


@app.cell
def _(solved):
    solved.stats.plot.balance('Heat')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Total Costs

    Access the optimized objective value:
    """)
    return


@app.cell
def _(heat_demand, mo, solved):
    total_costs = solved.solution['costs'].item()
    total_heat = float(heat_demand.sum())
    gas_consumed = total_heat / 0.9  # Account for boiler efficiency

    mo.md(
        f"""
        | Metric | Value |
        |--------|-------|
        | Total heat demand | {total_heat:.1f} kWh |
        | Gas consumed | {gas_consumed:.1f} kWh |
        | Total costs | {total_costs:.2f} € |
        | Average cost | {total_costs / total_heat:.3f} €/kWh_heat |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flow Rates Over Time

    Visualize all flow rates using the built-in plotting accessor:
    """)
    return


@app.cell
def _(solved):
    solved.stats.plot.flows()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Energy Flow Sankey

    A Sankey diagram visualizes the total energy flows through the system:
    """)
    return


@app.cell
def _(solved):
    solved.stats.plot.sankey.flows()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    In this quickstart, you learned the **basic workflow**:

    1. **Create** a `FlowSystem` with timesteps
    2. **Add** buses, effects, and components
    3. **Optimize** with `flow_system.optimize(solver)`
    4. **Analyze** results via `flow_system.stats`

    ### Next Steps

    Explore the full [flixopt documentation](https://flixopt.github.io/flixopt/)
    for more advanced examples including thermal storage, investment optimization,
    and multi-carrier systems.
    """)
    return


if __name__ == '__main__':
    app.run()
