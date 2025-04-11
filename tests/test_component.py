import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
import flixopt.elements

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestComponentModel:

    def test_flow_label_check(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        _ = basic_flow_system_linopy
        inputs = [
            fx.Flow('Q_th_Last', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
            fx.Flow('Q_Gas', 'Fernwärme', relative_minimum=np.ones(10) * 0.1)
        ]
        outputs = [
            fx.Flow('Q_th_Last', 'Gas', relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Q_Gas', 'Gas', relative_minimum=np.ones(10) * 0.01)
        ]
        with pytest.raises(ValueError, match='Flow names must be unique!'):
            _ = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)

    def test_component(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
            fx.Flow('In2', 'Fernwärme', relative_minimum=np.ones(10) * 0.1)
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Out2', 'Gas', relative_minimum=np.ones(10) * 0.01)
        ]
        comp = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)
        flow_system.add_elements(comp)
        _ = create_linopy_model(flow_system)

        assert {'TestComponent(In1)|flow_rate',
 'TestComponent(In1)|total_flow_hours',
 'TestComponent(In2)|flow_rate',
 'TestComponent(In2)|total_flow_hours',
 'TestComponent(Out1)|flow_rate',
 'TestComponent(Out1)|total_flow_hours',
 'TestComponent(Out2)|flow_rate',
 'TestComponent(Out2)|total_flow_hours'} == set(comp.model.variables)

        assert {'TestComponent(In1)|total_flow_hours',
 'TestComponent(In2)|total_flow_hours',
 'TestComponent(Out1)|total_flow_hours',
 'TestComponent(Out2)|total_flow_hours'} == set(comp.model.constraints)

    def test_on_with_multiple_flows(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        ub_out2 = np.linspace(1, 1.5, 10).round(2)
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = [
            fx.Flow('Out1', 'Gas', relative_minimum=np.ones(10) * 0.2, size=200),
            fx.Flow('Out2', 'Gas', relative_minimum=np.ones(10) * 0.3,
                    relative_maximum = ub_out2, size=300),
        ]
        comp = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs,
                                          on_off_parameters=fx.OnOffParameters())
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        assert {
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
        } == set(comp.model.variables)

        assert {
            'TestComponent(In1)|total_flow_hours',
            'TestComponent(In1)|on_con1',
            'TestComponent(In1)|on_con2',
            'TestComponent(In1)|on_hours_total',
            'TestComponent(Out1)|total_flow_hours',
            'TestComponent(Out1)|on_con1',
            'TestComponent(Out1)|on_con2',
            'TestComponent(Out1)|on_hours_total',
            'TestComponent(Out2)|total_flow_hours',
            'TestComponent(Out2)|on_con1',
            'TestComponent(Out2)|on_con2',
            'TestComponent(Out2)|on_hours_total',
            'TestComponent|on_con1',
            'TestComponent|on_con2',
            'TestComponent|on_hours_total',
        } == set(comp.model.constraints)

        assert_var_equal(model['TestComponent(Out2)|flow_rate'],
                         model.add_variables(lower=0, upper=300 * ub_out2, coords=(timesteps,)))
        assert_var_equal(model['TestComponent|on'], model.add_variables(binary=True, coords = (timesteps,)))
        assert_var_equal(model['TestComponent(Out2)|on'], model.add_variables(binary=True, coords=(timesteps,)))

        assert_conequal(model.constraints['TestComponent(Out2)|on_con1'], model.variables['TestComponent(Out2)|on'] * 0.3 * 300 <= model.variables['TestComponent(Out2)|flow_rate'])
        assert_conequal(model.constraints['TestComponent(Out2)|on_con2'], model.variables['TestComponent(Out2)|on'] * 300 * ub_out2 >= model.variables['TestComponent(Out2)|flow_rate'])

        assert_conequal(model.constraints['TestComponent|on_con1'],
                        model.variables['TestComponent|on'] * 1e-5 <= model.variables['TestComponent(In1)|flow_rate'] + model.variables['TestComponent(Out1)|flow_rate'] + model.variables['TestComponent(Out2)|flow_rate'])
        # TODO: Might there be a better way to no use 1e-5?
        assert_conequal(model.constraints['TestComponent|on_con2'],
                        model.variables['TestComponent|on'] * (100 + 200 + 300 * ub_out2)/3
                        >= (model.variables['TestComponent(In1)|flow_rate']
                            + model.variables['TestComponent(Out1)|flow_rate']
                            + model.variables['TestComponent(Out2)|flow_rate']) / 3
                        )

    def test_on_with_single_flow(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        inputs = [
            fx.Flow('In1', 'Fernwärme', relative_minimum=np.ones(10) * 0.1, size=100),
        ]
        outputs = []
        comp = flixopt.elements.Component(
            'TestComponent', inputs=inputs, outputs=outputs, on_off_parameters=fx.OnOffParameters()
        )
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        assert {
            'TestComponent(In1)|flow_rate',
            'TestComponent(In1)|total_flow_hours',
            'TestComponent(In1)|on',
            'TestComponent(In1)|on_hours_total',
            'TestComponent|on',
            'TestComponent|on_hours_total',
        } == set(comp.model.variables)

        assert {
            'TestComponent(In1)|total_flow_hours',
            'TestComponent(In1)|on_con1',
            'TestComponent(In1)|on_con2',
            'TestComponent(In1)|on_hours_total',
            'TestComponent|on_con1',
            'TestComponent|on_con2',
            'TestComponent|on_hours_total',
        } == set(comp.model.constraints)

        assert_var_equal(
            model['TestComponent(In1)|flow_rate'], model.add_variables(lower=0, upper=100, coords=(timesteps,))
        )
        assert_var_equal(model['TestComponent|on'], model.add_variables(binary=True, coords=(timesteps,)))
        assert_var_equal(model['TestComponent(In1)|on'], model.add_variables(binary=True, coords=(timesteps,)))

        assert_conequal(
            model.constraints['TestComponent(In1)|on_con1'],
            model.variables['TestComponent(In1)|on'] * 0.1 * 100 <= model.variables['TestComponent(In1)|flow_rate'],
        )
        assert_conequal(
            model.constraints['TestComponent(In1)|on_con2'],
            model.variables['TestComponent(In1)|on'] * 100 >= model.variables['TestComponent(In1)|flow_rate'],
        )

        assert_conequal(
            model.constraints['TestComponent|on_con1'],
            model.variables['TestComponent|on'] * 0.1 * 100 <= model.variables['TestComponent(In1)|flow_rate'],
        )
        assert_conequal(
            model.constraints['TestComponent|on_con2'],
            model.variables['TestComponent|on'] * 100 >= model.variables['TestComponent(In1)|flow_rate'],
        )

