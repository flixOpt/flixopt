import marimo

__generated_with = "0.15.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo):
    mo.md(
        """
    # Energy System Modeling with Flixopt

    This notebook demonstrates how to use the flixopt framework to model a minimalistic energy system
    with a boiler, heat demand, and natural gas supply.
    """
    )
    return


@app.cell
def _(mo):
    """
    ## System Configuration

    Define the basic parameters for our energy system simulation.
    """

    mo.md("""
    **Configure your energy system parameters:**
    """)
    return


@app.cell
def _(mo, solve_button, solve_status):
    mo.hstack([solve_button, solve_status])
    return


@app.cell
def _(boiler_efficiency, gas_price, mo, num_timesteps):

    mo.hstack([num_timesteps, boiler_efficiency, gas_price], justify="space-around")
    return


@app.cell
def _(calculation, mo):
    if calculation is not None:
        mo.hstack([
            mo.vstack([
                    mo.md("### 💰 Cost Analysis"),
                    mo.callout(
                        mo.md(f"**Total System Cost: {calculation.results.solution['costs|total'].item():.2f} €**"),
                        kind="info"
                    ),
            ]),
            calculation.results.solution['costs(operation)|total_per_timestep'].round(3).to_pandas()[:-1],
        ])
    return


@app.cell
def _(mo, np, num_timesteps, pd):
    """
    ## Time Series and Load Profile Setup
    """


    # Define timesteps
    timesteps = pd.date_range('2020-01-01', periods=num_timesteps.value, freq='h')

    # Define thermal load profile (extend or truncate based on timesteps)
    base_profile = np.array([30, 0, 20, 25, 15, 35, 10, 40])  # Extended base profile
    if len(timesteps) <= len(base_profile):
        thermal_load_profile = base_profile[:len(timesteps)]
    else:
        # Repeat pattern if we need more timesteps
        thermal_load_profile = np.tile(base_profile, (len(timesteps) // len(base_profile)) + 1)[:len(timesteps)]

    mo.md(f"""Modeling {len(timesteps)} timesteps     """
          f"""Thermal load profile: {thermal_load_profile}""")
    return thermal_load_profile, timesteps


@app.cell
def _(calculation, mo):
    """
    ## District Heating Balance Analysis
    """

    if calculation is None:
        dh_analysis = mo.callout(
            mo.md("No calculation available for district heating analysis"),
            kind="warn"
        )
        dh_results = None
    else:
        try:
            # Get district heating results
            dh_results = calculation.results['District Heating'].node_balance().to_dataframe()

            dh_analysis = mo.vstack([
                mo.md("### 🔥 District Heating Balance"),
                dh_results,
                calculation.results['District Heating'].plot_node_balance(show=False)
            ])

        except Exception as e:
            dh_analysis = mo.callout(
                mo.md(f"Error retrieving district heating results: {str(e)}"),
                kind="danger"
            )
            dh_results = None

    dh_analysis
    return (dh_results,)


@app.cell
def _(mo):
    num_timesteps = mo.ui.slider(
        start=3,
        stop=24,
        value=3,
        label="Number of timesteps",
        show_value=True
    )
    return (num_timesteps,)


@app.cell
def _(mo):
    # Interactive parameter controls using latest Marimo UI elements

    boiler_efficiency = mo.ui.slider(
        start=0.3,
        stop=0.9,
        value=0.5,
        step=0.05,
        label="Boiler efficiency",
        show_value=True
    )

    gas_price = mo.ui.slider(
        start=0.02,
        stop=0.08,
        value=0.04,
        step=0.01,
        label="Gas price (€/kWh)",
        show_value=True
    )
    return boiler_efficiency, gas_price


@app.cell
def _(boiler_efficiency, gas_price, mo, num_timesteps):
    # Display current parameter values in a callout
    mo.callout(
        mo.md(f"""
        **Current Configuration:**
        - Timesteps: {num_timesteps.value}
        - Boiler efficiency: {boiler_efficiency.value:.1%}
        - Gas price: {gas_price.value:.3f} €/kWh
        """),
        kind="info"
    )
    return


@app.cell
def _(
    boiler_efficiency,
    calculations,
    fx,
    gas_price,
    mo,
    solve_button,
    thermal_load_profile,
    timesteps,
):
    """
    ## Create and solve FLowSystem
    """
    solve_status = mo.md('')
    flow_system_status = mo.md('Click the button to build the FLow System and solve the Optimization')

    if not hasattr(solve_button, '_last_processed'):
        solve_button._last_processed = 0

    # Only run if button value has changed (new click)
    if solve_button.value > solve_button._last_processed:
        try:
            solve_button._last_processed = solve_button.value

            _fs = fx.FlowSystem(timesteps)

            _fs.add_elements(
                fx.Effect('costs', '€', 'Cost', is_standard=True, is_objective=True),
                fx.Source(
                    'Natural Gas Tariff',
                    source=fx.Flow(
                        label='Gas Flow',
                        bus='Natural Gas',
                        size=1000,
                        effects_per_flow_hour=gas_price.value
                    ),
                ),
                fx.Sink(
                    'Heat Demand',
                    sink=fx.Flow(
                        label='Thermal Load',
                        bus='District Heating',
                        size=1,
                        fixed_relative_profile=thermal_load_profile
                    ),
                ),
                fx.linear_converters.Boiler(
                    'Boiler',
                    eta=boiler_efficiency.value,
                    Q_th=fx.Flow(label='Thermal Output', bus='District Heating', size=50),
                    Q_fu=fx.Flow(label='Fuel Input', bus='Natural Gas'),
                ),
                fx.Bus('District Heating'),
                fx.Bus('Natural Gas'),
            )

            # Create and solve the calculation
            _current_calc = fx.FullCalculation('Simulation1', _fs)
            calculations.append(_current_calc)
            _current_calc.do_modeling()
            _current_calc.solve(fx.solvers.HighsSolver(0.01, 60))
            solve_status = mo.callout(
                mo.md(f"✅ **Optimization completed successfully!** (Run #{solve_button.value})"),
                kind="success"
            )
            flow_system_status = mo.md(f"""
                ✓ Flow system built successfully
                - Total elements: {len(_fs.all_elements)}
                """)
        except Exception as e:
            solve_status = mo.callout(
                mo.md(f"❌ **Optimization failed:** {str(e)}"),
                kind="danger"
            )
    return (solve_status,)


@app.cell
def _(mo):
    solve_button = mo.ui.button(
        label="🚀 Run Optimization",
        on_click=lambda value: value + 1,
        value = 0,
        kind="success",
    )
    return (solve_button,)


@app.cell
def _(calculations):
    calculation = calculations[-1] if len(calculations) > 0 else None
    results = calculation.results
    print(f'{calculation == calculations[-1]=}, {len(calculations)=}')
    return (calculation,)


@app.cell
def _(calculation, mo):
    mo.md(f"""Nr of timesteps = {str(len(calculation.flow_system.time_series_collection.timesteps))}""")
    return


@app.cell
def _(solve_button):
    print(solve_button.value)
    return


@app.cell
def _(solve_button):
    print(solve_button.value)
    return


@app.cell
def _(dh_results, mo):
    """
    Display district heating balance table
    """

    if dh_results is not None:
        mo.ui.table(
            data=dh_results.round(3),
            pagination=False,
            selection=None
        )
    else:
        mo.callout(
            mo.md("District heating results not available"),
            kind="warn"
        )
    return


@app.cell(column=1)
def _():
    # Import required libraries
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from rich.pretty import pprint

    import flixopt as fx

    calculations = []
    return calculations, fx, mo, np, pd


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
