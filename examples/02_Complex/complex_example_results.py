"""
This script shows how load results of a prior calcualtion and how to analyze them.
"""

import pandas as pd
import plotly.offline

import flixOpt as fx

use_custom_colors = True

if __name__ == '__main__':
    # --- Load Results ---
    try:
        results = fx.results.CalculationResults('Sim1', folder='results')
    except FileNotFoundError as e:
        raise FileNotFoundError('Results file was not found. Did you run complex_example.py already?') from e

    # --- You can change colors for plotting ---
    if use_custom_colors:
        results.change_colors({'Kessel': '#C6563E', 'BHKW2': '#C69A3E', 'Speicher': 'royalblue', 'Wärmelast': 'gray'})

    # --- Basic overview ---
    results.visualize_network()
    results.plot_operation('Fernwärme')
    results.plot_operation('Fernwärme', 'bar')
    results.plot_operation('Fernwärme', 'bar', engine='matplotlib')

    # --- Detailed Plots ---
    # In depth plot for individual flow rates ('__' is used as the delimiter between Component and Flow
    results.plot_operation('Wärmelast__Q_th_Last', 'heatmap')

    for flow in results.component_results['BHKW2'].flows:
        fig = results.plot_operation(flow.label_full, 'heatmap', heatmap_steps_per_period='h', heatmap_periods='D')

    # --- Plotting internal variables manually ---
    on_data = pd.DataFrame(
        {
            'BHKW2 On': results.component_results['BHKW2'].variables['Q_th']['OnOff']['on'],
            'Kessel On': results.component_results['Kessel'].variables['Q_th']['OnOff']['on'],
        },
        index=results.time,
    )
    fig = fx.plotting.with_plotly(on_data, 'line')
    fig.write_html('results/on.html')  # Writing to file

    fig = fx.plotting.with_plotly(on_data, 'bar')
    fig.update_layout(barmode='group', bargap=0.1)  # Applying custom layout
    plotly.offline.plot(fig)
