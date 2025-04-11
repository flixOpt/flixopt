import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx
import flixopt.elements

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestComponentModel:
    """Test the FlowModel class."""

    def test_component(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        inputs = [
            fx.Flow('Q_th_Last', 'Fernwärme', relative_minimum=np.ones(10) * 0.1),
            fx.Flow('Q_Gas', 'Fernwärme', relative_minimum=np.ones(10) * 0.1)
        ]
        outputs = [
            fx.Flow('Q_th_Last', 'Gas', relative_minimum=np.ones(10) * 0.01),
            fx.Flow('Q_Gas', 'Gas', relative_minimum=np.ones(10) * 0.01)
        ]
        comp = flixopt.elements.Component('TestComponent', inputs=inputs, outputs=outputs)
        flow_system.add_elements(comp)
        model = create_linopy_model(flow_system)

        assert {'WärmelastTest(Q_th_Last)|flow_rate', 'GastarifTest(Q_Gas)|flow_rate'}.issubset(set(comp.model.variables))
        assert set(comp.model.constraints) == {'TestBus|balance'}

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate'] == model.variables['WärmelastTest(Q_th_Last)|flow_rate']
        )

    def test_bus_penalty(self, basic_flow_system_linopy):
        """Test that flow model constraints are correctly generated."""
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        bus = fx.Bus('TestBus')
        flow_system.add_elements(bus,
                                 fx.Sink('WärmelastTest', sink=fx.Flow('Q_th_Last', 'TestBus')),
                                 fx.Source('GastarifTest', source=fx.Flow('Q_Gas', 'TestBus')))
        model = create_linopy_model(flow_system)

        assert set(bus.model.variables) == {'TestBus|excess_input',
                                            'TestBus|excess_output',
                                            'WärmelastTest(Q_th_Last)|flow_rate',
                                            'GastarifTest(Q_Gas)|flow_rate'}
        assert set(bus.model.constraints) == {'TestBus|balance'}

        assert_var_equal(model.variables['TestBus|excess_input'], model.add_variables(lower=0, coords = (timesteps,)))
        assert_var_equal(model.variables['TestBus|excess_output'], model.add_variables(lower=0, coords=(timesteps,)))

        assert_conequal(
            model.constraints['TestBus|balance'],
            model.variables['GastarifTest(Q_Gas)|flow_rate'] - model.variables['WärmelastTest(Q_th_Last)|flow_rate'] + model.variables['TestBus|excess_input'] -  model.variables['TestBus|excess_output'] ==  0
        )

        assert_conequal(
            model.constraints['TestBus->Penalty'],
            model.variables['TestBus->Penalty'] == (model.variables['TestBus|excess_input'] * 1e5 * model.hours_per_step).sum() + (model.variables['TestBus|excess_output'] * 1e5 * model.hours_per_step).sum(),
        )
