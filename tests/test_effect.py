import numpy as np
import pandas as pd
import pytest
import xarray as xr

import flixopt as fx

from .conftest import assert_conequal, assert_var_equal, create_linopy_model


class TestBusModel:
    """Test the FlowModel class."""

    def test_minimal(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        effect = fx.Effect('Effect1', '€', 'Testing Effect')

        flow_system.add_elements(effect)
        model = create_linopy_model(flow_system)

        assert set(effect.model.variables) == {'Effect1(invest)|total',
                                                 'Effect1(operation)|total',
                                                 'Effect1(operation)|total_per_timestep',
                                                 'Effect1|total',}
        assert set(effect.model.constraints) == {'Effect1(invest)|total',
                                                 'Effect1(operation)|total',
                                                 'Effect1(operation)|total_per_timestep',
                                                 'Effect1|total',}

        assert_var_equal(model.variables['Effect1|total'], model.add_variables())
        assert_var_equal(model.variables['Effect1(invest)|total'], model.add_variables())
        assert_var_equal(model.variables['Effect1(operation)|total'], model.add_variables())
        assert_var_equal(model.variables['Effect1(operation)|total_per_timestep'], model.add_variables(coords=(timesteps,)))

        assert_conequal(model.constraints['Effect1|total'],
                        model.variables['Effect1|total'] == model.variables['Effect1(operation)|total'] + model.variables['Effect1(invest)|total'])
        assert_conequal(model.constraints['Effect1(invest)|total'], model.variables['Effect1(invest)|total'] == 0)
        assert_conequal(model.constraints['Effect1(operation)|total'],
                        model.variables['Effect1(operation)|total'] == model.variables['Effect1(operation)|total_per_timestep'].sum())
        assert_conequal(model.constraints['Effect1(operation)|total_per_timestep'],
                        model.variables['Effect1(operation)|total_per_timestep'] ==0)

    def test_bounds(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        effect = fx.Effect('Effect1', '€', 'Testing Effect',
                           minimum_operation=1.0,
                           maximum_operation=1.1,
                           minimum_invest=2.0,
                           maximum_invest=2.1,
                           minimum_total=3.0,
                           maximum_total=3.1,
                           minimum_operation_per_hour=4.0,
                           maximum_operation_per_hour=4.1
                           )

        flow_system.add_elements(effect)
        model = create_linopy_model(flow_system)

        assert set(effect.model.variables) == {'Effect1(invest)|total',
                                                 'Effect1(operation)|total',
                                                 'Effect1(operation)|total_per_timestep',
                                                 'Effect1|total',}
        assert set(effect.model.constraints) == {'Effect1(invest)|total',
                                                 'Effect1(operation)|total',
                                                 'Effect1(operation)|total_per_timestep',
                                                 'Effect1|total',}

        assert_var_equal(model.variables['Effect1|total'], model.add_variables(lower=3.0, upper=3.1))
        assert_var_equal(model.variables['Effect1(invest)|total'], model.add_variables(lower=2.0, upper=2.1))
        assert_var_equal(model.variables['Effect1(operation)|total'], model.add_variables(lower=1.0, upper=1.1))
        assert_var_equal(
            model.variables['Effect1(operation)|total_per_timestep'], model.add_variables(
                lower=4.0 * model.hours_per_step, upper=4.1* model.hours_per_step, coords=(timesteps,))
        )

        assert_conequal(model.constraints['Effect1|total'],
                        model.variables['Effect1|total'] == model.variables['Effect1(operation)|total'] + model.variables['Effect1(invest)|total'])
        assert_conequal(model.constraints['Effect1(invest)|total'], model.variables['Effect1(invest)|total'] == 0)
        assert_conequal(model.constraints['Effect1(operation)|total'],
                        model.variables['Effect1(operation)|total'] == model.variables['Effect1(operation)|total_per_timestep'].sum())
        assert_conequal(model.constraints['Effect1(operation)|total_per_timestep'],
                        model.variables['Effect1(operation)|total_per_timestep'] ==0)

    def test_shares(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        effect1 = fx.Effect('Effect1', '€', 'Testing Effect',
                           specific_share_to_other_effects_operation={
                               'Effect2': 1.1,
                               'Effect3': 1.2
                           },
                           specific_share_to_other_effects_invest={
                               'Effect2': 2.1,
                               'Effect3': 2.2
                           }
                           )
        effect2 = fx.Effect('Effect2', '€', 'Testing Effect')
        effect3 = fx.Effect('Effect3', '€', 'Testing Effect')
        flow_system.add_elements(effect1, effect2, effect3)
        model = create_linopy_model(flow_system)

        assert set(effect2.model.variables) == {
            'Effect2(invest)|total',
            'Effect2(operation)|total',
            'Effect2(operation)|total_per_timestep',
            'Effect2|total',
            'Effect1(invest)->Effect2(invest)',
            'Effect1(operation)->Effect2(operation)',
        }
        assert set(effect2.model.constraints) == {
            'Effect2(invest)|total',
            'Effect2(operation)|total',
            'Effect2(operation)|total_per_timestep',
            'Effect2|total',
            'Effect1(invest)->Effect2(invest)',
            'Effect1(operation)->Effect2(operation)',
        }

        assert_conequal(
            model.constraints['Effect2(invest)|total'],
            model.variables['Effect2(invest)|total'] == model.variables['Effect1(invest)->Effect2(invest)'],
        )

        assert_conequal(
            model.constraints['Effect2(operation)|total_per_timestep'],
            model.variables['Effect2(operation)|total_per_timestep'] == model.variables['Effect1(operation)->Effect2(operation)'],
        )

        assert_conequal(
            model.constraints['Effect1(operation)->Effect2(operation)'],
            model.variables['Effect1(operation)->Effect2(operation)']
            == model.variables['Effect1(operation)|total_per_timestep'] * 1.1
        )

        assert_conequal(
            model.constraints['Effect1(invest)->Effect2(invest)'],
            model.variables['Effect1(invest)->Effect2(invest)']
            == model.variables['Effect1(invest)|total'] * 2.1,
        )


