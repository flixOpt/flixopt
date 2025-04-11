import os

import numpy as np
import pandas as pd
import pytest

import flixopt as fx

from .conftest import (
    assert_almost_equal_numeric,
    create_calculation_and_solve,
)


class TestFlowSystem:
    def test_simple_flow_system(self, simple_flow_system, highs_solver):
        """
        Test the effects of the simple energy system model
        """
        calculation = create_calculation_and_solve(simple_flow_system, highs_solver, 'test_simple_flow_system')

        effects = calculation.flow_system.effects

        # Cost assertions
        assert_almost_equal_numeric(
            effects['costs'].model.total.solution.item(), 81.88394666666667, 'costs doesnt match expected value'
        )

        # CO2 assertions
        assert_almost_equal_numeric(
            effects['CO2'].model.total.solution.item(), 255.09184, 'CO2 doesnt match expected value'
        )

    def test_model_components(self, simple_flow_system, highs_solver):
        """
        Test the component flows of the simple energy system model
        """
        calculation = create_calculation_and_solve(simple_flow_system, highs_solver, 'test_model_components')
        comps = calculation.flow_system.components

        # Boiler assertions
        assert_almost_equal_numeric(
            comps['Boiler'].Q_th.model.flow_rate.solution.values,
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )

        # CHP unit assertions
        assert_almost_equal_numeric(
            comps['CHP_unit'].Q_th.model.flow_rate.solution.values,
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )

    def test_results_persistence(self, simple_flow_system, highs_solver):
        """
        Test saving and loading results
        """
        # Save results to file
        calculation = create_calculation_and_solve(simple_flow_system, highs_solver, 'test_model_components')

        calculation.results.to_file()

        # Load results from file
        results = fx.results.CalculationResults.from_file(calculation.folder, calculation.name)

        # Verify key variables from loaded results
        assert_almost_equal_numeric(
            results.solution['costs|total'].values,
            81.88394666666667,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(results.solution['CO2|total'].values, 255.09184, 'CO2 doesnt match expected value')


class TestComponents:
    def test_transmission_basic(self, basic_flow_system, highs_solver):
        """Test basic transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler', eta=0.5, Q_th=fx.Flow('Q_th', bus='Wärme lokal'), Q_fu=fx.Flow('Q_fu', bus='Gas')
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1', 'Wärme lokal', size=fx.InvestParameters(specific_effects=5, maximum_size=1e6)),
            out1=fx.Flow('Rohr2', 'Fernwärme', size=1000),
        )

        flow_system.add_elements(transmission, boiler)

        _ = create_calculation_and_solve(flow_system, highs_solver, 'test_transmission_basic')

        # Assertions
        assert_almost_equal_numeric(
            transmission.in1.model.on_off.on.solution.values,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'On does not work properly',
        )

        assert_almost_equal_numeric(
            transmission.in1.model.flow_rate.solution.values * 0.8 - 20,
            transmission.out1.model.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

    def test_transmission_advanced(self, basic_flow_system, highs_solver):
        """Test advanced transmission functionality"""
        flow_system = basic_flow_system
        flow_system.add_elements(fx.Bus('Wärme lokal'))

        boiler = fx.linear_converters.Boiler(
            'Boiler_Standard',
            eta=0.9,
            Q_th=fx.Flow('Q_th', bus='Fernwärme', relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])),
            Q_fu=fx.Flow('Q_fu', bus='Gas'),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup', eta=0.4, Q_th=fx.Flow('Q_th', bus='Wärme lokal'), Q_fu=fx.Flow('Q_fu', bus='Gas')
        )

        last2 = fx.Sink(
            'Wärmelast2',
            sink=fx.Flow(
                'Q_th_Last',
                bus='Wärme lokal',
                size=1,
                fixed_relative_profile=flow_system.components['Wärmelast'].sink.fixed_relative_profile
                * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            ),
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1a', bus='Wärme lokal', size=fx.InvestParameters(specific_effects=5, maximum_size=1000)),
            out1=fx.Flow('Rohr1b', 'Fernwärme', size=1000),
            in2=fx.Flow('Rohr2a', 'Fernwärme', size=1000),
            out2=fx.Flow('Rohr2b', bus='Wärme lokal', size=1000),
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        calculation = create_calculation_and_solve(flow_system, highs_solver, 'test_transmission_advanced')

        # Assertions
        assert_almost_equal_numeric(
            transmission.in1.model.on_off.on.solution.values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'On does not work properly',
        )

        assert_almost_equal_numeric(
            calculation.results.model.variables['Rohr(Rohr1b)|flow_rate'].solution.values,
            transmission.out1.model.flow_rate.solution.values,
            'Flow rate of Rohr__Rohr1b is not correct',
        )

        assert_almost_equal_numeric(
            transmission.in1.model.flow_rate.solution.values * 0.8
            - np.array([20 if val > 0.1 else 0 for val in transmission.in1.model.flow_rate.solution.values]),
            transmission.out1.model.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

        assert_almost_equal_numeric(
            transmission.in1.model._investment.size.solution.item(),
            transmission.in2.model._investment.size.solution.item(),
            'The Investments are not equated correctly',
        )


class TestComplex:
    def test_basic_flow_system(self, flow_system_base, highs_solver):
        calculation = create_calculation_and_solve(flow_system_base, highs_solver, 'test_basic_flow_system')

        # Assertions
        assert_almost_equal_numeric(
            calculation.results.model['costs|total'].solution.item(),
            -11597.873624489237,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['costs(operation)|total_per_timestep'].solution.values,
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
            sum(calculation.results.model['CO2(operation)->costs(operation)'].solution.values),
            258.63729669618675,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Kessel(Q_th)->costs(operation)'].solution.values),
            0.01,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Kessel->costs(operation)'].solution.values),
            -0.0,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Gastarif(Q_Gas)->costs(operation)'].solution.values),
            39.09153113079115,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['Einspeisung(P_el)->costs(operation)'].solution.values),
            -14196.61245231646,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sum(calculation.results.model['KWK->costs(operation)'].solution.values),
            0.0,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Kessel(Q_th)->costs(invest)'].solution.values,
            1000 + 500,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Speicher->costs(invest)'].solution.values,
            800 + 1,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['CO2(operation)|total'].solution.values,
            1293.1864834809337,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            calculation.results.model['CO2(invest)|total'].solution.values,
            0.9999999999999994,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            calculation.results.model['Kessel(Q_th)|flow_rate'].solution.values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['KWK(Q_th)|flow_rate'].solution.values,
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
            calculation.results.model['KWK(P_el)|flow_rate'].solution.values,
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
            calculation.results.model['Speicher|netto_discharge'].solution.values,
            [-45.0, -69.71111111, 15.0, -10.0, 36.06697198, -55.0, 20.0, 20.0, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )
        assert_almost_equal_numeric(
            calculation.results.model['Speicher|charge_state'].solution.values,
            [0.0, 40.5, 100.0, 77.0, 79.84, 37.38582802, 83.89496178, 57.18336484, 32.60869565, 10.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            calculation.results.model['Speicher|PiecewiseEffects|costs'].solution.values,
            800,
            'Speicher|PiecewiseEffects|costs doesnt match expected value',
        )

    def test_piecewise_conversion(self, flow_system_piecewise_conversion, highs_solver):
        calculation = create_calculation_and_solve(
            flow_system_piecewise_conversion, highs_solver, 'test_piecewise_conversion'
        )

        effects = calculation.flow_system.effects
        comps = calculation.flow_system.components

        # Compare expected values with actual values
        assert_almost_equal_numeric(
            effects['costs'].model.total.solution.item(), -10710.997365760755, 'costs doesnt match expected value'
        )
        assert_almost_equal_numeric(
            effects['CO2'].model.total.solution.item(), 1278.7939026086956, 'CO2 doesnt match expected value'
        )
        assert_almost_equal_numeric(
            comps['Kessel'].Q_th.model.flow_rate.solution.values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )
        kwk_flows = {flow.label: flow for flow in comps['KWK'].inputs + comps['KWK'].outputs}
        assert_almost_equal_numeric(
            kwk_flows['Q_th'].model.flow_rate.solution.values,
            [45.0, 45.0, 64.5962087, 100.0, 61.3136, 45.0, 45.0, 12.86469565, 0.0],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            kwk_flows['P_el'].model.flow_rate.solution.values,
            [40.0, 40.0, 47.12589407, 60.0, 45.93221818, 40.0, 40.0, 10.91784108, -0.0],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            comps['Speicher'].model.netto_discharge.solution.values,
            [-15.0, -45.0, 25.4037913, -35.0, 48.6864, -25.0, -25.0, 7.13530435, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            comps['Speicher'].model.variables['Speicher|PiecewiseEffects|costs'].solution.values,
            454.74666666666667,
            'Speicher investCosts_segmented_costs doesnt match expected value',
        )

@pytest.mark.slow
class TestModelingTypes:
    @pytest.fixture(params=['full', 'segmented', 'aggregated'])
    def modeling_calculation(self, request, flow_system_long, highs_solver):
        """
        Fixture to run calculations with different modeling types
        """
        # Extract flow system and data from the fixture
        flow_system = flow_system_long[0]
        thermal_load_ts = flow_system_long[1]['thermal_load_ts']
        electrical_load_ts = flow_system_long[1]['electrical_load_ts']

        # Create calculation based on modeling type
        modeling_type = request.param
        if modeling_type == 'full':
            calc = fx.FullCalculation('fullModel', flow_system)
            calc.do_modeling()
            calc.solve(highs_solver)
        elif modeling_type == 'segmented':
            calc = fx.SegmentedCalculation('segModel', flow_system, timesteps_per_segment=96, overlap_timesteps=1)
            calc.do_modeling_and_solve(highs_solver)
        elif modeling_type == 'aggregated':
            calc = fx.AggregatedCalculation(
                'aggModel',
                flow_system,
                fx.AggregationParameters(
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
                calc.results.model['costs|total'].solution.item(),
                expected_costs[modeling_type],
                f'Costs do not match for {modeling_type} modeling type',
            )
        else:
            assert_almost_equal_numeric(
                calc.results.solution_without_overlap('costs(operation)|total_per_timestep').sum(),
                expected_costs[modeling_type],
                f'Costs do not match for {modeling_type} modeling type',
            )


if __name__ == '__main__':
    pytest.main(['-v'])
