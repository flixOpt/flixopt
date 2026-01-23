"""Tests for deprecated Optimization/Results API - ported from feature/v5.

This module contains the original integration tests from feature/v5 that use the
deprecated Optimization class. These tests will be removed in v6.0.0.

For new tests, use FlowSystem.optimize(solver) instead.
"""

import pytest

import flixopt as fx

from ..conftest import (
    assert_almost_equal_numeric,
    create_optimization_and_solve,
)


class TestFlowSystem:
    def test_simple_flow_system(self, simple_flow_system, highs_solver):
        """
        Test the effects of the simple energy system model
        """
        optimization = create_optimization_and_solve(simple_flow_system, highs_solver, 'test_simple_flow_system')

        effects = optimization.flow_system.effects

        # Cost assertions
        assert_almost_equal_numeric(
            effects['costs'].submodel.total.solution.item(), 81.88394666666667, 'costs doesnt match expected value'
        )

        # CO2 assertions
        assert_almost_equal_numeric(
            effects['CO2'].submodel.total.solution.item(), 255.09184, 'CO2 doesnt match expected value'
        )

    def test_model_components(self, simple_flow_system, highs_solver):
        """
        Test the component flows of the simple energy system model
        """
        optimization = create_optimization_and_solve(simple_flow_system, highs_solver, 'test_model_components')
        comps = optimization.flow_system.components

        # Boiler assertions
        assert_almost_equal_numeric(
            comps['Boiler'].thermal_flow.submodel.flow_rate.solution.values,
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )

        # CHP unit assertions
        assert_almost_equal_numeric(
            comps['CHP_unit'].thermal_flow.submodel.flow_rate.solution.values,
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )

    def test_results_persistence(self, simple_flow_system, highs_solver):
        """
        Test saving and loading results
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


class TestComplex:
    def test_basic_flow_system(self, flow_system_base, highs_solver):
        optimization = create_optimization_and_solve(flow_system_base, highs_solver, 'test_basic_flow_system')

        # Assertions
        assert_almost_equal_numeric(
            optimization.results.model['costs'].solution.item(),
            -11597.873624489237,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['costs(temporal)|per_timestep'].solution.values,
            [
                -2.38500000e03,
                -2.21681333e03,
                -2.38500000e03,
                -2.17599000e03,
                -2.35107029e03,
                -2.38500000e03,
                0.00000000e00,
                -1.68897826e-10,
                -2.16914486e-12,
            ],
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            sum(optimization.results.model['CO2(temporal)->costs(temporal)'].solution.values),
            258.63729669618675,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(optimization.results.model['Kessel(Q_th)->costs(temporal)'].solution.values),
            0.01,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(optimization.results.model['Kessel->costs(temporal)'].solution.values),
            -0.0,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(optimization.results.model['Gastarif(Q_Gas)->costs(temporal)'].solution.values),
            39.09153113079115,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(optimization.results.model['Einspeisung(P_el)->costs(temporal)'].solution.values),
            -14196.61245231646,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(optimization.results.model['KWK->costs(temporal)'].solution.values),
            0.0,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['Kessel(Q_th)->costs(periodic)'].solution.values,
            1000 + 500,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['Speicher->costs(periodic)'].solution.values,
            800 + 1,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['CO2(temporal)'].solution.values,
            1293.1864834809337,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            optimization.results.model['CO2(periodic)'].solution.values,
            0.9999999999999994,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            optimization.results.model['Kessel(Q_th)|flow_rate'].solution.values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['KWK(Q_th)|flow_rate'].solution.values,
            [
                7.50000000e01,
                6.97111111e01,
                7.50000000e01,
                7.50000000e01,
                7.39330280e01,
                7.50000000e01,
                0.00000000e00,
                3.12638804e-14,
                3.83693077e-14,
            ],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            optimization.results.model['KWK(P_el)|flow_rate'].solution.values,
            [
                6.00000000e01,
                5.57688889e01,
                6.00000000e01,
                6.00000000e01,
                5.91464224e01,
                6.00000000e01,
                0.00000000e00,
                2.50111043e-14,
                3.06954462e-14,
            ],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['Speicher|netto_discharge'].solution.values,
            [-45.0, -69.71111111, 15.0, -10.0, 36.06697198, -55.0, 20.0, 20.0, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )
        assert_almost_equal_numeric(
            optimization.results.model['Speicher|charge_state'].solution.values,
            [0.0, 40.5, 100.0, 77.0, 79.84, 37.38582802, 83.89496178, 57.18336484, 32.60869565, 10.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            optimization.results.model['Speicher|PiecewiseEffects|costs'].solution.values,
            800,
            'Speicher|PiecewiseEffects|costs doesnt match expected value',
        )

    def test_piecewise_conversion(self, flow_system_piecewise_conversion, highs_solver):
        optimization = create_optimization_and_solve(
            flow_system_piecewise_conversion, highs_solver, 'test_piecewise_conversion'
        )

        effects = optimization.flow_system.effects
        comps = optimization.flow_system.components

        # Compare expected values with actual values
        assert_almost_equal_numeric(
            effects['costs'].submodel.total.solution.item(), -10710.997365760755, 'costs doesnt match expected value'
        )
        assert_almost_equal_numeric(
            effects['CO2'].submodel.total.solution.item(), 1278.7939026086956, 'CO2 doesnt match expected value'
        )
        assert_almost_equal_numeric(
            comps['Kessel'].thermal_flow.submodel.flow_rate.solution.values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )
        kwk_flows = {flow.label: flow for flow in (comps['KWK'].inputs + comps['KWK'].outputs).values()}
        assert_almost_equal_numeric(
            kwk_flows['Q_th'].submodel.flow_rate.solution.values,
            [45.0, 45.0, 64.5962087, 100.0, 61.3136, 45.0, 45.0, 12.86469565, 0.0],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            kwk_flows['P_el'].submodel.flow_rate.solution.values,
            [40.0, 40.0, 47.12589407, 60.0, 45.93221818, 40.0, 40.0, 10.91784108, -0.0],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            comps['Speicher'].submodel.netto_discharge.solution.values,
            [-15.0, -45.0, 25.4037913, -35.0, 48.6864, -25.0, -25.0, 7.13530435, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            comps['Speicher'].submodel.variables['Speicher|PiecewiseEffects|costs'].solution.values,
            454.74666666666667,
            'Speicher investcosts_segmented_costs doesnt match expected value',
        )


@pytest.mark.slow
class TestModelingTypes:
    # Note: 'aggregated' case removed - ClusteredOptimization has been replaced by
    # FlowSystem.transform.cluster(). See tests/test_clustering/ for new clustering tests.
    @pytest.fixture(params=['full', 'segmented'])
    def modeling_calculation(self, request, flow_system_long, highs_solver):
        """
        Fixture to run optimizations with different modeling types
        """
        # Extract flow system and data from the fixture
        flow_system = flow_system_long[0]

        # Create calculation based on modeling type
        modeling_type = request.param
        if modeling_type == 'full':
            calc = fx.Optimization('fullModel', flow_system)
            calc.do_modeling()
            calc.solve(highs_solver)
        elif modeling_type == 'segmented':
            calc = fx.SegmentedOptimization('segModel', flow_system, timesteps_per_segment=96, overlap_timesteps=1)
            calc.do_modeling_and_solve(highs_solver)

        return calc, modeling_type

    def test_modeling_types_costs(self, modeling_calculation):
        """
        Test total costs for different modeling types
        """
        calc, modeling_type = modeling_calculation

        expected_costs = {
            'full': 343613,
            'segmented': 343613,  # Approximate value
        }

        if modeling_type == 'full':
            assert_almost_equal_numeric(
                calc.results.model['costs'].solution.item(),
                expected_costs[modeling_type],
                f'costs do not match for {modeling_type} modeling type',
            )
        elif modeling_type == 'segmented':
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


if __name__ == '__main__':
    pytest.main(['-v'])
