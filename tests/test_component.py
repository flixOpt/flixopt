import numpy as np
import pytest

import flixopt as fx
import flixopt.elements

from .conftest import (
    assert_almost_equal_numeric,
    assert_conequal,
    assert_sets_equal,
    assert_var_equal,
    create_calculation_and_solve,
    create_linopy_model,
)


class TestComponentModel:
    def test_flow_label_check(self):
        """Test that flow model constraints are correctly generated."""
        inputs = [
            fx.Flow('Q_th_Last', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
            fx.Flow('Q_Gas', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
        ]
        outputs = [
            fx.Flow('Q_th_Last', 'Gas', relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Q_Gas', 'Gas', relative_minimum=np.ones(10) * 0.01),
        ]
        with pytest.raises(ValueError, match='Flow names must be unique!'):
            _ = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)

    def test_component(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
            fx.Flow('In2', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Out2', 'Gas', relative_minimum=np.ones(10) * 0.01),
        ]
        comp = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)
        flow_system.add_elements(comp)
        _ = create_linopy_model(flow_system)

        assert_sets_equal(
            set(comp.submodel.variables),
            {
                'TestComponent(In1)|flow_rate',
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In2)|flow_rate',
                'TestComponent(In2)|total_flow_hours',
                'TestComponent(Out1)|flow_rate',
                'TestComponent(Out1)|total_flow_hours',
                'TestComponent(Out2)|flow_rate',
                'TestComponent(Out2)|total_flow_hours',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(comp.submodel.constraints),
            {
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In2)|total_flow_hours',
                'TestComponent(Out1)|total_flow_hours',
                'TestComponent(Out2)|total_flow_hours',
            },
            msg='Incorrect constraints',
        )

    def test_on_with_multiple_flows(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200),
            fx.Flow('Out2', 'Gas', relative_minimum=np.ones(10) * 0.3, relative_maximum=ub_out2, size=300),
        ]
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, on_off_parameters=fx.OnOffParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(comp.submodel.variables),
            {
                'TestComponent(In1)|flow_rate',
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In1)|on',
                'TestComponent(In1)|on_hours_total',
                'TestComponent(Out1)|flow_rate',
                'TestComponent(Out1)|total_flow_hours',
                'TestComponent(Out1)|on',
                'TestComponent(Out1)|on_hours_total',
                'TestComponent(Out2)|flow_rate',
                'TestComponent(Out2)|total_flow_hours',
                'TestComponent(Out2)|on',
                'TestComponent(Out2)|on_hours_total',
                'TestComponent|on',
                'TestComponent|on_hours_total',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(comp.submodel.constraints),
            {
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In1)|flow_rate|lb',
                'TestComponent(In1)|flow_rate|ub',
                'TestComponent(In1)|on_hours_total',
                'TestComponent(Out1)|total_flow_hours',
                'TestComponent(Out1)|flow_rate|lb',
                'TestComponent(Out1)|flow_rate|ub',
                'TestComponent(Out1)|on_hours_total',
                'TestComponent(Out2)|total_flow_hours',
                'TestComponent(Out2)|flow_rate|lb',
                'TestComponent(Out2)|flow_rate|ub',
                'TestComponent(Out2)|on_hours_total',
                'TestComponent|on|lb',
                'TestComponent|on|ub',
                'TestComponent|on_hours_total',
            },
            msg='Incorrect constraints',
        )

        upper_bound_flow_rate = outputs[1].relative_maximum

        assert upper_bound_flow_rate.dims == tuple(model.get_coords())

        assert_var_equal(
            model['TestComponent(Out2)|flow_rate'],
            model.add_variables(lower=0, upper=300 * upper_bound_flow_rate, coords=model.get_coords()),
        )
        assert_var_equal(model['TestComponent|on'], model.add_variables(binary=True, coords=model.get_coords()))
        assert_var_equal(model['TestComponent(Out2)|on'], model.add_variables(binary=True, coords=model.get_coords()))

        assert_conequal(
            model.constraints['TestComponent(Out2)|flow_rate|lb'],
            model.variables['TestComponent(Out2)|flow_rate'] >= model.variables['TestComponent(Out2)|on'] * 0.3 * 300,
        )
        assert_conequal(
            model.constraints['TestComponent(Out2)|flow_rate|ub'],
            model.variables['TestComponent(Out2)|flow_rate']
            <= model.variables['TestComponent(Out2)|on'] * 300 * upper_bound_flow_rate,
        )

        assert_conequal(
            model.constraints['TestComponent|on|lb'],
            model.variables['TestComponent|on']
            >= (
                model.variables['TestComponent(In1)|on']
                + model.variables['TestComponent(Out1)|on']
                + model.variables['TestComponent(Out2)|on']
            )
            / (3 + 1e-5),
        )
        assert_conequal(
            model.constraints['TestComponent|on|ub'],
            model.variables['TestComponent|on']
            <= (
                model.variables['TestComponent(In1)|on']
                + model.variables['TestComponent(Out1)|on']
                + model.variables['TestComponent(Out2)|on']
            )
            + 1e-5,
        )

    def test_on_with_single_flow(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = []
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, on_off_parameters=fx.OnOffParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(comp.submodel.variables),
            {
                'TestComponent(In1)|flow_rate',
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In1)|on',
                'TestComponent(In1)|on_hours_total',
                'TestComponent|on',
                'TestComponent|on_hours_total',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(comp.submodel.constraints),
            {
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In1)|flow_rate|lb',
                'TestComponent(In1)|flow_rate|ub',
                'TestComponent(In1)|on_hours_total',
                'TestComponent|on',
                'TestComponent|on_hours_total',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model['TestComponent(In1)|flow_rate'], model.add_variables(lower=0, upper=100, coords=model.get_coords())
        )
        assert_var_equal(model['TestComponent|on'], model.add_variables(binary=True, coords=model.get_coords()))
        assert_var_equal(model['TestComponent(In1)|on'], model.add_variables(binary=True, coords=model.get_coords()))

        assert_conequal(
            model.constraints['TestComponent(In1)|flow_rate|lb'],
            model.variables['TestComponent(In1)|flow_rate'] >= model.variables['TestComponent(In1)|on'] * 0.1 * 100,
        )
        assert_conequal(
            model.constraints['TestComponent(In1)|flow_rate|ub'],
            model.variables['TestComponent(In1)|flow_rate'] <= model.variables['TestComponent(In1)|on'] * 100,
        )

        assert_conequal(
            model.constraints['TestComponent|on'],
            model.variables['TestComponent|on'] == model.variables['TestComponent(In1)|on'],
        )

    def test_previous_states_with_multiple_flows(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(
                'In1',
                'Fernwärme',
                relative_minimum=np.ones(10) * 0.1,
                size=100,
                previous_flow_rate=np.array([0, 0, 1e-6, 1e-5, 1e-4, 3, 4]),
            ),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200, previous_flow_rate=[3, 4, 5]),
            fx.Flow(
                'Out2',
                'Gas',
                relative_minimum=np.ones(10) * 0.3,
                relative_maximum=ub_out2,
                size=300,
                previous_flow_rate=20,
            ),
        ]
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, on_off_parameters=fx.OnOffParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(comp.submodel.variables),
            {
                'TestComponent(In1)|flow_rate',
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In1)|on',
                'TestComponent(In1)|on_hours_total',
                'TestComponent(Out1)|flow_rate',
                'TestComponent(Out1)|total_flow_hours',
                'TestComponent(Out1)|on',
                'TestComponent(Out1)|on_hours_total',
                'TestComponent(Out2)|flow_rate',
                'TestComponent(Out2)|total_flow_hours',
                'TestComponent(Out2)|on',
                'TestComponent(Out2)|on_hours_total',
                'TestComponent|on',
                'TestComponent|on_hours_total',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(comp.submodel.constraints),
            {
                'TestComponent(In1)|total_flow_hours',
                'TestComponent(In1)|flow_rate|lb',
                'TestComponent(In1)|flow_rate|ub',
                'TestComponent(In1)|on_hours_total',
                'TestComponent(Out1)|total_flow_hours',
                'TestComponent(Out1)|flow_rate|lb',
                'TestComponent(Out1)|flow_rate|ub',
                'TestComponent(Out1)|on_hours_total',
                'TestComponent(Out2)|total_flow_hours',
                'TestComponent(Out2)|flow_rate|lb',
                'TestComponent(Out2)|flow_rate|ub',
                'TestComponent(Out2)|on_hours_total',
                'TestComponent|on|lb',
                'TestComponent|on|ub',
                'TestComponent|on_hours_total',
            },
            msg='Incorrect constraints',
        )

        upper_bound_flow_rate = outputs[1].relative_maximum

        assert upper_bound_flow_rate.dims == tuple(model.get_coords())

        assert_var_equal(
            model['TestComponent(Out2)|flow_rate'],
            model.add_variables(lower=0, upper=300 * upper_bound_flow_rate, coords=model.get_coords()),
        )
        assert_var_equal(model['TestComponent|on'], model.add_variables(binary=True, coords=model.get_coords()))
        assert_var_equal(model['TestComponent(Out2)|on'], model.add_variables(binary=True, coords=model.get_coords()))

        assert_conequal(
            model.constraints['TestComponent(Out2)|flow_rate|lb'],
            model.variables['TestComponent(Out2)|flow_rate'] >= model.variables['TestComponent(Out2)|on'] * 0.3 * 300,
        )
        assert_conequal(
            model.constraints['TestComponent(Out2)|flow_rate|ub'],
            model.variables['TestComponent(Out2)|flow_rate']
            <= model.variables['TestComponent(Out2)|on'] * 300 * upper_bound_flow_rate,
        )

        assert_conequal(
            model.constraints['TestComponent|on|lb'],
            model.variables['TestComponent|on']
            >= (
                model.variables['TestComponent(In1)|on']
                + model.variables['TestComponent(Out1)|on']
                + model.variables['TestComponent(Out2)|on']
            )
            / (3 + 1e-5),
        )
        assert_conequal(
            model.constraints['TestComponent|on|ub'],
            model.variables['TestComponent|on']
            <= (
                model.variables['TestComponent(In1)|on']
                + model.variables['TestComponent(Out1)|on']
                + model.variables['TestComponent(Out2)|on']
            )
            + 1e-5,
        )

    @pytest.mark.parametrize(
        'in1_previous_flow_rate, out1_previous_flow_rate, out2_previous_flow_rate, previous_on_hours',
        [
            (None, None, None, 0),
            (np.array([0, 1e-6, 1e-4, 5]), None, None, 2),
            (np.array([0, 5, 0, 5]), None, None, 1),
            (np.array([0, 5, 0, 0]), 3, 0, 1),
            (np.array([0, 0, 2, 0, 4, 5]), [3, 4, 5], None, 4),
        ],
    )
    def test_previous_states_with_multiple_flows_parameterized(
        self,
        basic_flow_system_linopy_coords,
        coords_config,
        in1_previous_flow_rate,
        out1_previous_flow_rate,
        out2_previous_flow_rate,
        previous_on_hours,
    ):
        """Test that flow model constraints are correctly generated with different previous flow rates and constraint factors."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow(
                'In1',
                'Fernwärme',
                relative_minimum=np.ones(10) * 0.1,
                size=100,
                previous_flow_rate=in1_previous_flow_rate,
                on_off_parameters=fx.OnOffParameters(consecutive_on_hours_min=3),
            ),
        ]
        outputs = [
            fx.Flow(
                'Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200, previous_flow_rate=out1_previous_flow_rate
            ),
            fx.Flow(
                'Out2',
                'Gas',
                relative_minimum=np.ones(10) * 0.3,
                relative_maximum=ub_out2,
                size=300,
                previous_flow_rate=out2_previous_flow_rate,
            ),
        ]
        comp = flixopt.elements.Component(
            'TestComponent',
            inputs=inputs,
            outputs=outputs,
            on_off_parameters=fx.OnOffParameters(consecutive_on_hours_min=3),
        )
        flow_system.add_elements(comp)
        create_linopy_model(flow_system)

        assert_conequal(
            comp.submodel.constraints['TestComponent|consecutive_on_hours|initial'],
            comp.submodel.variables['TestComponent|consecutive_on_hours'].isel(time=0)
            == comp.submodel.variables['TestComponent|on'].isel(time=0) * (previous_on_hours + 1),
        )


class TestTransmissionModel:
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
            transmission.in1.submodel.on_off.on.solution.values,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'On does not work properly',
        )

        assert_almost_equal_numeric(
            transmission.in1.submodel.flow_rate.solution.values * 0.8 - 20,
            transmission.out1.submodel.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

    def test_transmission_balanced(self, basic_flow_system, highs_solver):
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
            inputs=[
                fx.Flow(
                    'Q_th_Last',
                    bus='Wärme lokal',
                    size=1,
                    fixed_relative_profile=flow_system.components['Wärmelast'].inputs[0].fixed_relative_profile
                    * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                )
            ],
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1a', bus='Wärme lokal', size=fx.InvestParameters(specific_effects=5, maximum_size=1000)),
            out1=fx.Flow('Rohr1b', 'Fernwärme', size=1000),
            in2=fx.Flow('Rohr2a', 'Fernwärme', size=fx.InvestParameters()),
            out2=fx.Flow('Rohr2b', bus='Wärme lokal', size=1000),
            balanced=True,
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        calculation = create_calculation_and_solve(flow_system, highs_solver, 'test_transmission_advanced')

        # Assertions
        assert_almost_equal_numeric(
            transmission.in1.submodel.on_off.on.solution.values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'On does not work properly',
        )

        assert_almost_equal_numeric(
            calculation.results.model.variables['Rohr(Rohr1b)|flow_rate'].solution.values,
            transmission.out1.submodel.flow_rate.solution.values,
            'Flow rate of Rohr__Rohr1b is not correct',
        )

        assert_almost_equal_numeric(
            transmission.in1.submodel.flow_rate.solution.values * 0.8
            - np.array([20 if val > 0.1 else 0 for val in transmission.in1.submodel.flow_rate.solution.values]),
            transmission.out1.submodel.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

        assert_almost_equal_numeric(
            transmission.in1.submodel._investment.size.solution.item(),
            transmission.in2.submodel._investment.size.solution.item(),
            'The Investments are not equated correctly',
        )

    def test_transmission_unbalanced(self, basic_flow_system, highs_solver):
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
            inputs=[
                fx.Flow(
                    'Q_th_Last',
                    bus='Wärme lokal',
                    size=1,
                    fixed_relative_profile=flow_system.components['Wärmelast'].inputs[0].fixed_relative_profile
                    * np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                )
            ],
        )

        transmission = fx.Transmission(
            'Rohr',
            relative_losses=0.2,
            absolute_losses=20,
            in1=fx.Flow('Rohr1a', bus='Wärme lokal', size=fx.InvestParameters(specific_effects=50, maximum_size=1000)),
            out1=fx.Flow('Rohr1b', 'Fernwärme', size=1000),
            in2=fx.Flow(
                'Rohr2a', 'Fernwärme', size=fx.InvestParameters(specific_effects=100, minimum_size=10, optional=False)
            ),
            out2=fx.Flow('Rohr2b', bus='Wärme lokal', size=1000),
            balanced=False,
        )

        flow_system.add_elements(transmission, boiler, boiler2, last2)

        calculation = create_calculation_and_solve(flow_system, highs_solver, 'test_transmission_advanced')

        # Assertions
        assert_almost_equal_numeric(
            transmission.in1.submodel.on_off.on.solution.values,
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            'On does not work properly',
        )

        assert_almost_equal_numeric(
            calculation.results.model.variables['Rohr(Rohr1b)|flow_rate'].solution.values,
            transmission.out1.submodel.flow_rate.solution.values,
            'Flow rate of Rohr__Rohr1b is not correct',
        )

        assert_almost_equal_numeric(
            transmission.in1.submodel.flow_rate.solution.values * 0.8
            - np.array([20 if val > 0.1 else 0 for val in transmission.in1.submodel.flow_rate.solution.values]),
            transmission.out1.submodel.flow_rate.solution.values,
            'Losses are not computed correctly',
        )

        assert transmission.in1.submodel._investment.size.solution.item() > 11

        assert_almost_equal_numeric(
            transmission.in2.submodel._investment.size.solution.item(),
            10,
            'Sizing does not work properly',
        )
