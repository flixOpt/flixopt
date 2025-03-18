import datetime
import os
import unittest
from typing import Literal

import numpy as np
import pandas as pd
import pytest

import flixopt as fx

np.random.seed(45)


class BaseTest(unittest.TestCase):
    def setUp(self):
        fx.change_logging_level('DEBUG')

    def get_solver(self):
        return fx.solvers.HighsSolver(mip_gap=0.0001, time_limit_seconds=3600, solver_output_to_console=False)

    def assert_almost_equal_numeric(
        self, actual, desired, err_msg, relative_error_range_in_percent=0.011, absolute_tolerance=1e-9
    ):  # error_range etwas höher als mip_gap, weil unterschiedl. Bezugswerte
        """
        Asserts that actual is almost equal to desired.
        Designed for comparing float and ndarrays. Whith respect to tolerances
        """
        relative_tol = relative_error_range_in_percent / 100
        if isinstance(desired, (int, float)):
            delta = abs(relative_tol * desired) if desired != 0 else absolute_tolerance
            self.assertAlmostEqual(actual, desired, msg=err_msg, delta=delta)
        else:
            np.testing.assert_allclose(actual, desired, rtol=relative_tol, atol=absolute_tolerance)


class TestSimple(BaseTest):
    def setUp(self):
        super().setUp()

        self.Q_th_Last = np.array([30.0, 0.0, 90.0, 110, 110, 20, 20, 20, 20])
        self.p_el = 1 / 1000 * np.array([80.0, 80.0, 80.0, 80, 80, 80, 80, 80, 80])
        self.aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(hours=1)
        self.aTimeSeries = self.aTimeSeries.astype('datetime64')

    def test_model(self):
        calculation = self.model()
        effects = calculation.flow_system.effect_collection.effects
        comps = calculation.flow_system.components

        # Compare expected values with actual values
        self.assert_almost_equal_numeric(
            effects['costs'].model.all.sum.result, 81.88394666666667, 'costs doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            effects['CO2'].model.all.sum.result, 255.09184, 'CO2 doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            comps['Boiler'].Q_th.model.flow_rate.result,
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            comps['CHP_unit'].Q_th.model.flow_rate.result,
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )

    def test_from_results(self):
        calculation = self.model(save_results=True)

        results = fx.results.CalculationResults(calculation.name, 'results')

        # test effect results
        self.assert_almost_equal_numeric(
            results.effect_results['costs'].all_results['all']['all_sum'],
            81.88394666666667,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            results.effect_results['CO2'].all_results['all']['all_sum'], 255.09184, 'CO2 doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            results.component_results['Boiler'].variables_flat['Q_th__flow_rate'],
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            results.component_results['CHP_unit'].variables_flat['Q_th__flow_rate'],
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )

        df = results.to_dataframe('Fernwärme', with_last_time_step=False)
        comps = calculation.flow_system.components
        self.assert_almost_equal_numeric(
            comps['Wärmelast'].sink.model.flow_rate.result,
            df['Wärmelast__Q_th_Last'],
            'Loaded Results and directly used results dont match, or loading didnt work properly',
        )

    def model(self, save_results=False) -> fx.FullCalculation:
        # Define the components and flow_system
        Strom = fx.Bus('Strom')
        Fernwaerme = fx.Bus('Fernwärme')
        Gas = fx.Bus('Gas')

        costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
        CO2 = fx.Effect(
            'CO2',
            'kg',
            'CO2_e-Emissionen',
            specific_share_to_other_effects_operation={costs: 0.2},
            maximum_operation_per_hour=1000,
        )

        aBoiler = fx.linear_converters.Boiler(
            'Boiler',
            eta=0.5,
            Q_th=fx.Flow(
                'Q_th',
                bus=Fernwaerme,
                size=50,
                relative_minimum=5 / 50,
                relative_maximum=1,
                on_off_parameters=fx.OnOffParameters(),
            ),
            Q_fu=fx.Flow('Q_fu', bus=Gas),
        )
        aKWK = fx.linear_converters.CHP(
            'CHP_unit',
            eta_th=0.5,
            eta_el=0.4,
            P_el=fx.Flow('P_el', bus=Strom, size=60, relative_minimum=5 / 60, on_off_parameters=fx.OnOffParameters()),
            Q_th=fx.Flow('Q_th', bus=Fernwaerme),
            Q_fu=fx.Flow('Q_fu', bus=Gas),
        )
        aSpeicher = fx.Storage(
            'Speicher',
            charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1e4),
            discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1e4),
            capacity_in_flow_hours=fx.InvestParameters(fix_effects=20, fixed_size=30, optional=False),
            initial_charge_state=0,
            relative_maximum_charge_state=1 / 100 * np.array([80.0, 70.0, 80.0, 80, 80, 80, 80, 80, 80, 80]),
            eta_charge=0.9,
            eta_discharge=1,
            relative_loss_per_hour=0.08,
            prevent_simultaneous_charge_and_discharge=True,
        )
        aWaermeLast = fx.Sink(
            'Wärmelast', sink=fx.Flow('Q_th_Last', bus=Fernwaerme, size=1, fixed_relative_profile=self.Q_th_Last)
        )
        aGasTarif = fx.Source(
            'Gastarif', source=fx.Flow('Q_Gas', bus=Gas, size=1000, effects_per_flow_hour={costs: 0.04, CO2: 0.3})
        )
        aStromEinspeisung = fx.Sink(
            'Einspeisung', sink=fx.Flow('P_el', bus=Strom, effects_per_flow_hour=-1 * self.p_el)
        )

        es = fx.FlowSystem(self.aTimeSeries, last_time_step_hours=None)
        es.add_components(aSpeicher)
        es.add_effects(costs, CO2)
        es.add_components(aBoiler, aWaermeLast, aGasTarif)
        es.add_components(aStromEinspeisung)
        es.add_components(aKWK)

        time_indices = None

        print(es)
        es.visualize_network()

        aCalc = fx.FullCalculation('Test_Sim', es, 'pyomo', time_indices)
        aCalc.do_modeling()

        aCalc.solve(self.get_solver(), save_results=save_results)

        return aCalc


class TestComponents(BaseTest):
    def setUp(self):
        super().setUp()
        self.Q_th_Last = np.array([np.random.random() for _ in range(10)]) * 180
        self.p_el = (np.array([np.random.random() for _ in range(10)]) + 0.5) / 1.5 * 50
        self.datetime_array = datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(
            hours=1
        )
        self.datetime_array = self.datetime_array.astype('datetime64')

    def create_basic_elements(self):
        self.busses = {label: fx.Bus(label) for label in ['Strom', 'Fernwärme', 'Gas']}
        self.effects = {'Costs': fx.Effect('Costs', '€', 'Kosten', is_standard=True, is_objective=True)}
        self.components = {
            'Wärmelast': fx.Sink(
                'Wärmelast',
                sink=fx.Flow('Q_th_Last', bus=self.busses['Fernwärme'], size=1, fixed_relative_profile=self.Q_th_Last),
            ),
            'Gastarif': fx.Source(
                'Gastarif', source=fx.Flow('Q_Gas', bus=self.busses['Gas'], size=1000, effects_per_flow_hour=0.04)
            ),
            'Einspeisung': fx.Sink(
                'Einspeisung', sink=fx.Flow('P_el', bus=self.busses['Strom'], effects_per_flow_hour=-1 * self.p_el)
            ),
        }

    def test_transmission_basic(self):
        self.create_basic_elements()
        flow_system = fx.FlowSystem(self.datetime_array, last_time_step_hours=None)
        flow_system.add_elements(*(list(self.effects.values()) + list(self.components.values())))
        extra_bus = fx.Bus('Wärme lokal')
        boiler = fx.linear_converters.Boiler(
            'Boiler', eta=0.5, Q_th=fx.Flow('Q_th', bus=extra_bus), Q_fu=fx.Flow('Q_fu', bus=self.busses['Gas'])
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1', extra_bus, size=fx.InvestParameters(specific_effects=5, maximum_size=1e6)),
            out1=fx.Flow('Rohr2', self.busses['Fernwärme'], size=1000),
        )

        flow_system.add_elements(transmission, boiler)
        calculation = fx.FullCalculation('Test_Sim', flow_system)
        calculation.do_modeling()
        calculation.solve(self.get_solver())
        print(calculation.results())
        self.assert_almost_equal_numeric(
            transmission.in1.model._on.on.result, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'On does not work properly'
        )

        self.assert_almost_equal_numeric(
            transmission.in1.model.flow_rate.result * 0.8 - 20,
            transmission.out1.model.flow_rate.result,
            'Losses are not computed correctly',
        )

    def test_transmission_advanced(self):
        self.create_basic_elements()
        flow_system = fx.FlowSystem(self.datetime_array, last_time_step_hours=None)
        flow_system.add_elements(*(list(self.effects.values()) + list(self.components.values())))
        extra_bus = fx.Bus('Wärme lokal')

        boiler = fx.linear_converters.Boiler(
            'Boiler_Standard',
            eta=0.9,
            Q_th=fx.Flow(
                'Q_th', bus=self.busses['Fernwärme'], relative_maximum=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
            ),
            Q_fu=fx.Flow('Q_fu', bus=self.busses['Gas']),
        )

        boiler2 = fx.linear_converters.Boiler(
            'Boiler_backup', eta=0.4, Q_th=fx.Flow('Q_th', bus=extra_bus), Q_fu=fx.Flow('Q_fu', bus=self.busses['Gas'])
        )

        last2 = fx.Sink(
            'Wärmelast2',
            sink=fx.Flow(
                'Q_th_Last',
                bus=extra_bus,
                size=1,
                fixed_relative_profile=self.Q_th_Last * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
            ),
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1a', bus=extra_bus, size=fx.InvestParameters(specific_effects=5, maximum_size=1000)),
            out1=fx.Flow('Rohr1b', self.busses['Fernwärme'], size=1000),
            in2=fx.Flow('Rohr2a', self.busses['Fernwärme'], size=1000),
            out2=fx.Flow('Rohr2b', bus=extra_bus, size=1000),
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)
        calculation = fx.FullCalculation('Test_Transmission', flow_system)
        calculation.do_modeling()
        calculation.solve(self.get_solver(), save_results=True)
        results = fx.results.CalculationResults(calculation.name, 'results')

        self.assert_almost_equal_numeric(
            transmission.in1.model._on.on.result, np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), 'On does not work properly'
        )

        self.assert_almost_equal_numeric(
            results.to_dataframe('Rohr', with_last_time_step=False)['Rohr__Rohr1b'].values,
            transmission.out1.model.flow_rate.result,
            'Flow rate of Rohr__Rohr1b is not correct',
        )

        self.assert_almost_equal_numeric(
            transmission.in1.model.flow_rate.result * 0.8
            - np.array([20 if val > 0.1 else 0 for val in transmission.in1.model.flow_rate.result]),
            transmission.out1.model.flow_rate.result,
            'Losses are not computed correctly',
        )

        self.assert_almost_equal_numeric(
            transmission.in1.model._investment.size.result,
            transmission.in2.model._investment.size.result,
            'THe Investments are not equated correctly',
        )

    def tearDown(self):
        self.busses = None
        self.effects = None
        self.components = None
        self.datetime_array = None
        self.Q_th_Last = None
        self.p_el = None
        self.datetime_array = None


class TestComplex(BaseTest):
    def setUp(self):
        super().setUp()
        self.Q_th_Last = np.array([30.0, 0.0, 90.0, 110, 110, 20, 20, 20, 20])
        self.P_el_Last = np.array([40.0, 40.0, 40.0, 40, 40, 40, 40, 40, 40])
        self.aTimeSeries = datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(hours=1)
        self.aTimeSeries = self.aTimeSeries.astype('datetime64')
        self.excessCosts = None
        self.useCHPwithLinearSegments = False

    def test_basic(self):
        calculation = self.basic_model()
        effects = calculation.flow_system.effect_collection.effects
        comps = calculation.flow_system.components

        # Compare expected values with actual values
        self.assert_almost_equal_numeric(
            effects['costs'].model.all.sum.result, -11597.873624489237, 'costs doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            effects['costs'].model.operation.sum_TS.result,
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

        self.assert_almost_equal_numeric(
            sum(effects['costs'].model.operation.shares['CO2_operation'].result),
            258.63729669618675,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            sum(effects['costs'].model.operation.shares['Kessel__Q_th__switch_on_effects'].result),
            0.01,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            sum(effects['costs'].model.operation.shares['Kessel__running_hour_effects'].result),
            -0.0,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            sum(effects['costs'].model.operation.shares['Gastarif__Q_Gas__effects_per_flow_hour'].result),
            39.09153113079115,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            sum(effects['costs'].model.operation.shares['Einspeisung__P_el__effects_per_flow_hour'].result),
            -14196.61245231646,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            sum(effects['costs'].model.operation.shares['KWK__switch_on_effects'].result),
            0.0,
            'costs doesnt match expected value',
        )

        self.assert_almost_equal_numeric(
            effects['costs'].model.invest.shares['Kessel__Q_th__fix_effects'].result,
            1000,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            effects['costs'].model.invest.shares['Kessel__Q_th__specific_effects'].result,
            500,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            effects['costs'].model.invest.shares['Speicher__specific_effects'].result,
            1,
            'costs doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            effects['costs'].model.invest.shares['Speicher__segmented_effects'].result,
            800,
            'costs doesnt match expected value',
        )

        self.assert_almost_equal_numeric(
            effects['CO2'].model.all.shares['operation'].result, 1293.1864834809337, 'CO2 doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            effects['CO2'].model.all.shares['invest'].result, 0.9999999999999994, 'CO2 doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            comps['Kessel'].Q_th.model.flow_rate.result,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )

        self.assert_almost_equal_numeric(
            comps['KWK'].Q_th.model.flow_rate.result,
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
        self.assert_almost_equal_numeric(
            comps['KWK'].P_el.model.flow_rate.result,
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

        self.assert_almost_equal_numeric(
            comps['Speicher'].model.netto_discharge.result,
            [-45.0, -69.71111111, 15.0, -10.0, 36.06697198, -55.0, 20.0, 20.0, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            comps['Speicher'].model.charge_state.result,
            [0.0, 40.5, 100.0, 77.0, 79.84, 37.38582802, 83.89496178, 57.18336484, 32.60869565, 10.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        self.assert_almost_equal_numeric(
            comps['Speicher'].model.results()['Investment']['SegmentedShares']['costs_segmented'],
            800,
            'Speicher investCosts_segmented_costs doesnt match expected value',
        )

    def test_segments_of_flows(self):
        calculation = self.segments_of_flows_model()
        effects = calculation.flow_system.effect_collection.effects
        comps = calculation.flow_system.components

        # Compare expected values with actual values
        self.assert_almost_equal_numeric(
            effects['costs'].model.all.sum.result, -10710.997365760755, 'costs doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            effects['CO2'].model.all.sum.result, 1278.7939026086956, 'CO2 doesnt match expected value'
        )
        self.assert_almost_equal_numeric(
            comps['Kessel'].Q_th.model.flow_rate.result,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )
        kwk_flows = {flow.label: flow for flow in comps['KWK'].inputs + comps['KWK'].outputs}
        self.assert_almost_equal_numeric(
            kwk_flows['Q_th'].model.flow_rate.result,
            [45.0, 45.0, 64.5962087, 100.0, 61.3136, 45.0, 45.0, 12.86469565, 0.0],
            'KWK Q_th doesnt match expected value',
        )
        self.assert_almost_equal_numeric(
            kwk_flows['P_el'].model.flow_rate.result,
            [40.0, 40.0, 47.12589407, 60.0, 45.93221818, 40.0, 40.0, 10.91784108, -0.0],
            'KWK P_el doesnt match expected value',
        )

        self.assert_almost_equal_numeric(
            comps['Speicher'].model.netto_discharge.result,
            [-15.0, -45.0, 25.4037913, -35.0, 48.6864, -25.0, -25.0, 7.13530435, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        self.assert_almost_equal_numeric(
            comps['Speicher'].model.results()['Investment']['SegmentedShares']['costs_segmented'],
            454.74666666666667,
            'Speicher investCosts_segmented_costs doesnt match expected value',
        )

    def basic_model(self) -> fx.FullCalculation:
        # Define the components and flow_system
        Strom = fx.Bus('Strom', excess_penalty_per_flow_hour=self.excessCosts)
        Fernwaerme = fx.Bus('Fernwärme', excess_penalty_per_flow_hour=self.excessCosts)
        Gas = fx.Bus('Gas', excess_penalty_per_flow_hour=self.excessCosts)

        costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
        CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={costs: 0.2})
        PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3)

        aGaskessel = fx.linear_converters.Boiler(
            'Kessel',
            eta=0.5,
            on_off_parameters=fx.OnOffParameters(effects_per_running_hour={costs: 0, CO2: 1000}),
            Q_th=fx.Flow(
                'Q_th',
                bus=Fernwaerme,
                load_factor_max=1.0,
                load_factor_min=0.1,
                relative_minimum=5 / 50,
                relative_maximum=1,
                previous_flow_rate=50,
                size=fx.InvestParameters(
                    fix_effects=1000, fixed_size=50, optional=False, specific_effects={costs: 10, PE: 2}
                ),
                on_off_parameters=fx.OnOffParameters(
                    on_hours_total_min=0,
                    on_hours_total_max=1000,
                    consecutive_on_hours_max=10,
                    consecutive_on_hours_min=1,
                    consecutive_off_hours_max=10,
                    effects_per_switch_on=0.01,
                    switch_on_total_max=1000,
                ),
                flow_hours_total_max=1e6,
            ),
            Q_fu=fx.Flow('Q_fu', bus=Gas, size=200, relative_minimum=0, relative_maximum=1),
        )

        aKWK = fx.linear_converters.CHP(
            'KWK',
            eta_th=0.5,
            eta_el=0.4,
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
            P_el=fx.Flow('P_el', bus=Strom, size=60, relative_minimum=5 / 60, previous_flow_rate=10),
            Q_th=fx.Flow('Q_th', bus=Fernwaerme, size=1e3),
            Q_fu=fx.Flow('Q_fu', bus=Gas, size=1e3),
        )

        costsInvestsizeSegments = ([(5, 25), (25, 100)], {costs: [(50, 250), (250, 800)], PE: [(5, 25), (25, 100)]})
        invest_Speicher = fx.InvestParameters(
            fix_effects=0,
            effects_in_segments=costsInvestsizeSegments,
            optional=False,
            specific_effects={costs: 0.01, CO2: 0.01},
            minimum_size=0,
            maximum_size=1000,
        )
        aSpeicher = fx.Storage(
            'Speicher',
            charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1e4),
            discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1e4),
            capacity_in_flow_hours=invest_Speicher,
            initial_charge_state=0,
            maximal_final_charge_state=10,
            eta_charge=0.9,
            eta_discharge=1,
            relative_loss_per_hour=0.08,
            prevent_simultaneous_charge_and_discharge=True,
        )

        aWaermeLast = fx.Sink(
            'Wärmelast',
            sink=fx.Flow(
                'Q_th_Last', bus=Fernwaerme, size=1, relative_minimum=0, fixed_relative_profile=self.Q_th_Last
            ),
        )
        aGasTarif = fx.Source(
            'Gastarif', source=fx.Flow('Q_Gas', bus=Gas, size=1000, effects_per_flow_hour={costs: 0.04, CO2: 0.3})
        )
        aStromEinspeisung = fx.Sink(
            'Einspeisung', sink=fx.Flow('P_el', bus=Strom, effects_per_flow_hour=-1 * np.array(self.P_el_Last))
        )

        es = fx.FlowSystem(self.aTimeSeries, last_time_step_hours=None)
        es.add_effects(costs, CO2, PE)
        es.add_components(aGaskessel, aWaermeLast, aGasTarif, aStromEinspeisung, aKWK, aSpeicher)

        print(es)
        es.visualize_network()

        aCalc = fx.FullCalculation('Sim1', es, 'pyomo', None)
        aCalc.do_modeling()

        aCalc.solve(self.get_solver())

        return aCalc

    def segments_of_flows_model(self):
        # Define the components and flow_system
        Strom = fx.Bus('Strom', excess_penalty_per_flow_hour=self.excessCosts)
        Fernwaerme = fx.Bus('Fernwärme', excess_penalty_per_flow_hour=self.excessCosts)
        Gas = fx.Bus('Gas', excess_penalty_per_flow_hour=self.excessCosts)

        costs = fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True)
        CO2 = fx.Effect('CO2', 'kg', 'CO2_e-Emissionen', specific_share_to_other_effects_operation={costs: 0.2})
        PE = fx.Effect('PE', 'kWh_PE', 'Primärenergie', maximum_total=3.5e3)

        invest_Gaskessel = fx.InvestParameters(
            fix_effects=1000, fixed_size=50, optional=False, specific_effects={costs: 10, PE: 2}
        )
        aGaskessel = fx.linear_converters.Boiler(
            'Kessel',
            eta=0.5,
            on_off_parameters=fx.OnOffParameters(effects_per_running_hour={costs: 0, CO2: 1000}),
            Q_th=fx.Flow(
                'Q_th',
                bus=Fernwaerme,
                size=invest_Gaskessel,
                load_factor_max=1.0,
                load_factor_min=0.1,
                relative_minimum=5 / 50,
                relative_maximum=1,
                on_off_parameters=fx.OnOffParameters(
                    on_hours_total_min=0,
                    on_hours_total_max=1000,
                    consecutive_on_hours_max=10,
                    consecutive_off_hours_max=10,
                    effects_per_switch_on=0.01,
                    switch_on_total_max=1000,
                ),
                previous_flow_rate=50,
                flow_hours_total_max=1e6,
            ),
            Q_fu=fx.Flow('Q_fu', bus=Gas, size=200, relative_minimum=0, relative_maximum=1),
        )

        P_el = fx.Flow('P_el', bus=Strom, size=60, relative_maximum=55, previous_flow_rate=10)
        Q_th = fx.Flow('Q_th', bus=Fernwaerme)
        Q_fu = fx.Flow('Q_fu', bus=Gas)
        segmented_conversion_factors = {
            P_el: [(5, 30), (40, 60)],
            Q_th: [(6, 35), (45, 100)],
            Q_fu: [(12, 70), (90, 200)],
        }
        aKWK = fx.LinearConverter(
            'KWK',
            inputs=[Q_fu],
            outputs=[P_el, Q_th],
            segmented_conversion_factors=segmented_conversion_factors,
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=0.01),
        )

        costsInvestsizeSegments = ([(5, 25), (25, 100)], {costs: [(50, 250), (250, 800)], PE: [(5, 25), (25, 100)]})
        invest_Speicher = fx.InvestParameters(
            fix_effects=0,
            effects_in_segments=costsInvestsizeSegments,
            optional=False,
            specific_effects={costs: 0.01, CO2: 0.01},
            minimum_size=0,
            maximum_size=1000,
        )
        aSpeicher = fx.Storage(
            'Speicher',
            charging=fx.Flow('Q_th_load', bus=Fernwaerme, size=1e4),
            discharging=fx.Flow('Q_th_unload', bus=Fernwaerme, size=1e4),
            capacity_in_flow_hours=invest_Speicher,
            initial_charge_state=0,
            maximal_final_charge_state=10,
            eta_charge=0.9,
            eta_discharge=1,
            relative_loss_per_hour=0.08,
            prevent_simultaneous_charge_and_discharge=True,
        )

        aWaermeLast = fx.Sink(
            'Wärmelast',
            sink=fx.Flow(
                'Q_th_Last', bus=Fernwaerme, size=1, relative_minimum=0, fixed_relative_profile=self.Q_th_Last
            ),
        )
        aGasTarif = fx.Source(
            'Gastarif', source=fx.Flow('Q_Gas', bus=Gas, size=1000, effects_per_flow_hour={costs: 0.04, CO2: 0.3})
        )
        aStromEinspeisung = fx.Sink(
            'Einspeisung', sink=fx.Flow('P_el', bus=Strom, effects_per_flow_hour=-1 * np.array(self.P_el_Last))
        )

        es = fx.FlowSystem(self.aTimeSeries, last_time_step_hours=None)
        es.add_effects(costs, CO2, PE)
        es.add_components(aGaskessel, aWaermeLast, aGasTarif, aStromEinspeisung, aKWK)
        es.add_components(aSpeicher)
        print(es)
        es.visualize_network()

        aCalc = fx.FullCalculation('Sim1', es, 'pyomo', None)
        aCalc.do_modeling()

        aCalc.solve(self.get_solver())

        return aCalc


class TestModelingTypes(BaseTest):
    def setUp(self):
        super().setUp()
        self.Q_th_Last = np.array([30.0, 0.0, 90.0, 110, 110, 20, 20, 20, 20])
        self.p_el = 1 / 1000 * np.array([80.0, 80.0, 80.0, 80, 80, 80, 80, 80, 80])
        self.aTimeSeries = (
            datetime.datetime(2020, 1, 1) + np.arange(len(self.Q_th_Last)) * datetime.timedelta(hours=1)
        ).astype('datetime64')
        self.max_emissions_per_hour = 1000

    def test_full(self):
        calculation = self.calculate('full')
        effects = calculation.flow_system.effect_collection.effects
        self.assert_almost_equal_numeric(
            effects['costs'].model.all.sum.result, 343613, 'costs doesnt match expected value'
        )

    def test_aggregated(self):
        calculation = self.calculate('aggregated')
        effects = calculation.flow_system.effect_collection.effects
        self.assert_almost_equal_numeric(
            effects['costs'].model.all.sum.result, 342967.0, 'costs doesnt match expected value'
        )

    def test_segmented(self):
        calculation = self.calculate('segmented')
        self.assert_almost_equal_numeric(
            sum(calculation.results(combined_arrays=True)['Effects']['costs']['operation']['operation_sum_TS']),
            343613,
            'costs doesnt match expected value',
        )

    def calculate(self, modeling_type: Literal['full', 'segmented', 'aggregated']):
        doFullCalc, doSegmentedCalc, doAggregatedCalc = (
            modeling_type == 'full',
            modeling_type == 'segmented',
            modeling_type == 'aggregated',
        )
        if not any([doFullCalc, doSegmentedCalc, doAggregatedCalc]):
            raise Exception('Unknown modeling type')

        filename = os.path.join(os.path.dirname(__file__), 'ressources', 'Zeitreihen2020.csv')
        ts_raw = pd.read_csv(filename, index_col=0).sort_index()
        data = ts_raw['2020-01-01 00:00:00':'2020-12-31 23:45:00']['2020-01-01':'2020-01-03 23:45:00']
        P_el_Last, Q_th_Last, p_el, gP = (
            data['P_Netz/MW'].values,
            data['Q_Netz/MW'].values,
            data['Strompr.€/MWh'].values,
            data['Gaspr.€/MWh'].values,
        )
        aTimeSeries = (
            datetime.datetime(2020, 1, 1) + np.arange(len(P_el_Last)) * datetime.timedelta(hours=0.25)
        ).astype('datetime64')

        Strom, Fernwaerme, Gas, Kohle = fx.Bus('Strom'), fx.Bus('Fernwärme'), fx.Bus('Gas'), fx.Bus('Kohle')
        costs, CO2, PE = (
            fx.Effect('costs', '€', 'Kosten', is_standard=True, is_objective=True),
            fx.Effect('CO2', 'kg', 'CO2_e-Emissionen'),
            fx.Effect('PE', 'kWh_PE', 'Primärenergie'),
        )

        aGaskessel = fx.linear_converters.Boiler(
            'Kessel',
            eta=0.85,
            Q_th=fx.Flow(label='Q_th', bus=Fernwaerme),
            Q_fu=fx.Flow(
                label='Q_fu',
                bus=Gas,
                size=95,
                relative_minimum=12 / 95,
                previous_flow_rate=0,
                on_off_parameters=fx.OnOffParameters(effects_per_switch_on=1000),
            ),
        )
        aKWK = fx.linear_converters.CHP(
            'BHKW2',
            eta_th=0.58,
            eta_el=0.22,
            on_off_parameters=fx.OnOffParameters(effects_per_switch_on=24000),
            P_el=fx.Flow('P_el', bus=Strom),
            Q_th=fx.Flow('Q_th', bus=Fernwaerme),
            Q_fu=fx.Flow('Q_fu', bus=Kohle, size=288, relative_minimum=87 / 288),
        )
        aSpeicher = fx.Storage(
            'Speicher',
            charging=fx.Flow('Q_th_load', size=137, bus=Fernwaerme),
            discharging=fx.Flow('Q_th_unload', size=158, bus=Fernwaerme),
            capacity_in_flow_hours=684,
            initial_charge_state=137,
            minimal_final_charge_state=137,
            maximal_final_charge_state=158,
            eta_charge=1,
            eta_discharge=1,
            relative_loss_per_hour=0.001,
            prevent_simultaneous_charge_and_discharge=True,
        )

        TS_Q_th_Last, TS_P_el_Last = fx.TimeSeriesData(Q_th_Last), fx.TimeSeriesData(P_el_Last, agg_weight=0.7)
        aWaermeLast, aStromLast = (
            fx.Sink(
                'Wärmelast', sink=fx.Flow('Q_th_Last', bus=Fernwaerme, size=1, fixed_relative_profile=TS_Q_th_Last)
            ),
            fx.Sink('Stromlast', sink=fx.Flow('P_el_Last', bus=Strom, size=1, fixed_relative_profile=TS_P_el_Last)),
        )
        aKohleTarif, aGasTarif = (
            fx.Source(
                'Kohletarif',
                source=fx.Flow('Q_Kohle', bus=Kohle, size=1000, effects_per_flow_hour={costs: 4.6, CO2: 0.3}),
            ),
            fx.Source(
                'Gastarif', source=fx.Flow('Q_Gas', bus=Gas, size=1000, effects_per_flow_hour={costs: gP, CO2: 0.3})
            ),
        )

        p_feed_in, p_sell = (
            fx.TimeSeriesData(-(p_el - 0.5), agg_group='p_el'),
            fx.TimeSeriesData(p_el + 0.5, agg_group='p_el'),
        )
        aStromEinspeisung, aStromTarif = (
            fx.Sink('Einspeisung', sink=fx.Flow('P_el', bus=Strom, size=1000, effects_per_flow_hour=p_feed_in)),
            fx.Source(
                'Stromtarif',
                source=fx.Flow('P_el', bus=Strom, size=1000, effects_per_flow_hour={costs: p_sell, CO2: 0.3}),
            ),
        )

        es = fx.FlowSystem(aTimeSeries, last_time_step_hours=None)
        es.add_effects(costs, CO2, PE)
        es.add_components(
            aGaskessel, aWaermeLast, aStromLast, aGasTarif, aKohleTarif, aStromEinspeisung, aStromTarif, aKWK, aSpeicher
        )

        print(es)
        es.visualize_network()

        if doFullCalc:
            calc = fx.FullCalculation('fullModel', es, 'pyomo')
            calc.do_modeling()
            calc.solve(self.get_solver(), save_results=True)
        elif doSegmentedCalc:
            calc = fx.SegmentedCalculation(
                'segModel', es, segment_length=96, overlap_length=1, modeling_language='pyomo'
            )
            calc.do_modeling_and_solve(self.get_solver(), save_results=True)
        elif doAggregatedCalc:
            calc = fx.AggregatedCalculation(
                'aggModel',
                es,
                fx.AggregationParameters(
                    hours_per_period=6,
                    nr_of_periods=4,
                    fix_storage_flows=False,
                    aggregate_data_and_fix_non_binary_vars=True,
                    percentage_of_period_freedom=0,
                    penalty_of_period_freedom=0,
                    time_series_for_low_peaks=[TS_P_el_Last, TS_Q_th_Last],
                    time_series_for_high_peaks=[TS_Q_th_Last],
                ),
            )
            calc.do_modeling()
            print(es)
            es.visualize_network()
            calc.solve(self.get_solver(), save_results=True)
        else:
            raise Exception('Wrong Modeling Type')

        return calc


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings'])
