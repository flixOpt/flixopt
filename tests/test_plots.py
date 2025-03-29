# -*- coding: utf-8 -*-
"""
Manual test script for plots
"""

import unittest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import pytest

from flixopt import plotting


class TestPlots(unittest.TestCase):
    def setUp(self):
        np.random.seed(72)

    @staticmethod
    def get_sample_data(
        nr_of_columns: int = 7,
        nr_of_periods: int = 10,
        time_steps_per_period: int = 24,
        drop_fraction_of_indices: Optional[float] = None,
        only_pos_or_neg: bool = True,
        column_prefix: str = '',
    ):
        columns = [f'Region {i + 1}{column_prefix}' for i in range(nr_of_columns)]  # More realistic column labels
        values_per_column = nr_of_periods * time_steps_per_period
        if only_pos_or_neg:
            positive_data = np.abs(np.random.rand(values_per_column, nr_of_columns) * 100)
            negative_data = -np.abs(np.random.rand(values_per_column, nr_of_columns) * 100)
            data = pd.DataFrame(
                np.concatenate([positive_data, negative_data], axis=1),
                columns=[f'Region {i + 1}' for i in range(nr_of_columns)]
                + [f'Region {i + 1} Negative' for i in range(nr_of_columns)],
            )
        else:
            data = pd.DataFrame(
                np.random.randn(values_per_column, nr_of_columns) * 50 + 20, columns=columns
            )  # Random data with both positive and negative values
        data.index = pd.date_range('2023-01-01', periods=values_per_column, freq='h')

        if drop_fraction_of_indices:
            # Randomly drop a percentage of rows to create irregular intervals
            drop_indices = np.random.choice(data.index, int(len(data) * drop_fraction_of_indices), replace=False)
            data = data.drop(drop_indices)
        return data

    def test_bar_plots(self):
        data = self.get_sample_data(nr_of_columns=10, nr_of_periods=1, time_steps_per_period=24)
        plotly.offline.plot(plotting.with_plotly(data, 'bar'))
        plotting.with_matplotlib(data, 'bar')
        plt.show()

        data = self.get_sample_data(
            nr_of_columns=10, nr_of_periods=5, time_steps_per_period=24, drop_fraction_of_indices=0.3
        )
        plotly.offline.plot(plotting.with_plotly(data, 'bar'))
        plotting.with_matplotlib(data, 'bar')
        plt.show()

    def test_line_plots(self):
        data = self.get_sample_data(nr_of_columns=10, nr_of_periods=1, time_steps_per_period=24)
        plotly.offline.plot(plotting.with_plotly(data, 'line'))
        plotting.with_matplotlib(data, 'line')
        plt.show()

        data = self.get_sample_data(
            nr_of_columns=10, nr_of_periods=5, time_steps_per_period=24, drop_fraction_of_indices=0.3
        )
        plotly.offline.plot(plotting.with_plotly(data, 'line'))
        plotting.with_matplotlib(data, 'line')
        plt.show()

    def test_stacked_line_plots(self):
        data = self.get_sample_data(nr_of_columns=10, nr_of_periods=1, time_steps_per_period=24)
        plotly.offline.plot(plotting.with_plotly(data, 'area'))

        data = self.get_sample_data(
            nr_of_columns=10, nr_of_periods=5, time_steps_per_period=24, drop_fraction_of_indices=0.3
        )
        plotly.offline.plot(plotting.with_plotly(data, 'area'))

    def test_heat_map_plots(self):
        # Generate single-column data with datetime index for heatmap
        data = self.get_sample_data(nr_of_columns=1, nr_of_periods=10, time_steps_per_period=24, only_pos_or_neg=False)

        # Convert data for heatmap plotting using 'day' as period and 'hour' steps
        heatmap_data = plotting.reshape_to_2d(data.iloc[:, 0].values.flatten(), 24)
        # Plotting heatmaps with Plotly and Matplotlib
        plotly.offline.plot(plotting.heat_map_plotly(pd.DataFrame(heatmap_data)))
        plotting.heat_map_matplotlib(pd.DataFrame(pd.DataFrame(heatmap_data)))
        plt.show()

    def test_heat_map_plots_resampling(self):
        date_range = pd.date_range(start='2023-01-01', end='2023-03-21', freq='5min')

        # Generate random data for the DataFrame, simulating some metric (e.g., energy consumption, temperature)
        data = np.random.rand(len(date_range))

        # Create the DataFrame with a datetime index
        df = pd.DataFrame(data, index=date_range, columns=['value'])

        # Randomly drop a percentage of rows to create irregular intervals
        drop_fraction = 0.3  # Fraction of data points to drop (30% in this case)
        drop_indices = np.random.choice(df.index, int(len(df) * drop_fraction), replace=False)
        df_irregular = df.drop(drop_indices)

        # Generate single-column data with datetime index for heatmap
        data = df_irregular
        # Convert data for heatmap plotting using 'day' as period and 'hour' steps
        heatmap_data = plotting.heat_map_data_from_df(data, 'MS', 'D')
        plotly.offline.plot(plotting.heat_map_plotly(heatmap_data))
        plotting.heat_map_matplotlib(pd.DataFrame(heatmap_data))
        plt.show()

        heatmap_data = plotting.heat_map_data_from_df(data, 'W', 'h', fill='ffill')
        # Plotting heatmaps with Plotly and Matplotlib
        plotly.offline.plot(plotting.heat_map_plotly(pd.DataFrame(heatmap_data)))
        plotting.heat_map_matplotlib(pd.DataFrame(pd.DataFrame(heatmap_data)))
        plt.show()

        heatmap_data = plotting.heat_map_data_from_df(data, 'D', 'h', fill='ffill')
        # Plotting heatmaps with Plotly and Matplotlib
        plotly.offline.plot(plotting.heat_map_plotly(pd.DataFrame(heatmap_data)))
        plotting.heat_map_matplotlib(pd.DataFrame(pd.DataFrame(heatmap_data)))
        plt.show()


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
