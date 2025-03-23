"""
This script shows how load results of a prior calcualtion and how to analyze them.
"""

import pandas as pd
import plotly.offline

import flixOpt as fx

if __name__ == '__main__':
    # --- Load Results ---
    try:
        results = fx.results.CalculationResults.from_file('results', 'complex example')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Results file not found in the specified directory ('results'). "
            f"Please ensure that the file is generated by running 'complex_example.py'. "
            f'Original error: {e}'
        ) from e

    # --- Basic overview ---
    fx.plotting.plot_network(*results.network_infos, show=True)
    results['Fernwärme'].plot_node_balance()

    # --- Detailed Plots ---
    # In depth plot for individual flow rates ('__' is used as the delimiter between Component and Flow
    results.plot_heatmap('Wärmelast(Q_th_Last)|flow_rate')
    for flow_rate in results['BHKW2'].inputs + results['BHKW2'].outputs:
        results.plot_heatmap(flow_rate)

    # --- Plotting internal variables manually ---
    results.plot_heatmap('BHKW2(Q_th)|on')
    results.plot_heatmap('Kessel(Q_th)|on')
