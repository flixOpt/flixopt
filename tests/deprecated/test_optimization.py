"""Tests for deprecated Optimization classes.

This module tests the deprecated Optimization, SegmentedOptimization, and
ClusteredOptimization classes. These tests will be removed in v6.0.0.

For new tests, use FlowSystem.optimize(solver) instead.
"""

import pytest

import flixopt as fx

from ..conftest import (
    assert_almost_equal_numeric,
    create_optimization_and_solve,
)


class TestResultsPersistence:
    """Test deprecated Results.to_file() and Results.from_file() API."""

    def test_results_persistence(self, simple_flow_system, highs_solver):
        """
        Test saving and loading results (tests deprecated Results API)
        """
        # Save results to file
        optimization = create_optimization_and_solve(simple_flow_system, highs_solver, 'test_model_components')

        optimization.results.to_file(overwrite=True)

        # Load results from file
        results = fx.results.Results.from_file(optimization.folder, optimization.name)

        # Verify key variables from loaded results
        assert_almost_equal_numeric(
            results.solution['costs'].values,
            81.88394666666667,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(results.solution['CO2'].values, 255.09184, 'CO2 doesnt match expected value')


@pytest.mark.slow
class TestModelingTypes:
    """Tests for deprecated Optimization classes (Optimization, SegmentedOptimization, ClusteredOptimization)."""

    @pytest.fixture(params=['full', 'segmented', 'aggregated'])
    def modeling_calculation(self, request, flow_system_long, highs_solver):
        """
        Fixture to run optimizations with different modeling types
        """
        # Extract flow system and data from the fixture
        flow_system = flow_system_long[0]
        thermal_load_ts = flow_system_long[1]['thermal_load_ts']
        electrical_load_ts = flow_system_long[1]['electrical_load_ts']

        # Create calculation based on modeling type
        modeling_type = request.param
        if modeling_type == 'full':
            calc = fx.Optimization('fullModel', flow_system)
            calc.do_modeling()
            calc.solve(highs_solver)
        elif modeling_type == 'segmented':
            calc = fx.SegmentedOptimization('segModel', flow_system, timesteps_per_segment=96, overlap_timesteps=1)
            calc.do_modeling_and_solve(highs_solver)
        elif modeling_type == 'aggregated':
            calc = fx.ClusteredOptimization(
                'aggModel',
                flow_system,
                fx.ClusteringParameters(
                    hours_per_period=6,
                    nr_of_periods=4,
                    fix_storage_flows=False,
                    aggregate_data_and_fix_non_binary_vars=True,
                    percentage_of_period_freedom=0,
                    penalty_of_period_freedom=0,
                    time_series_for_low_peaks=[electrical_load_ts, thermal_load_ts],
                    time_series_for_high_peaks=[thermal_load_ts],
                ),
            )
            calc.do_modeling()
            calc.solve(highs_solver)

        return calc, modeling_type

    def test_modeling_types_costs(self, modeling_calculation):
        """
        Test total costs for different modeling types
        """
        calc, modeling_type = modeling_calculation

        expected_costs = {
            'full': 343613,
            'segmented': 343613,  # Approximate value
            'aggregated': 342967.0,
        }

        if modeling_type in ['full', 'aggregated']:
            assert_almost_equal_numeric(
                calc.results.model['costs'].solution.item(),
                expected_costs[modeling_type],
                f'costs do not match for {modeling_type} modeling type',
            )
        else:
            assert_almost_equal_numeric(
                calc.results.solution_without_overlap('costs(temporal)|per_timestep').sum(),
                expected_costs[modeling_type],
                f'costs do not match for {modeling_type} modeling type',
            )

    def test_segmented_io(self, modeling_calculation):
        calc, modeling_type = modeling_calculation
        if modeling_type == 'segmented':
            calc.results.to_file(overwrite=True)
            _ = fx.results.SegmentedResults.from_file(calc.folder, calc.name)
