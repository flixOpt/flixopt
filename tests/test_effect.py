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

        assert set(effect.model.variables) == {
            'Effect1(nontemporal)',
            'Effect1(temporal)',
            'Effect1(temporal)|per_timestep',
            'Effect1',
        }
        assert set(effect.model.constraints) == {
            'Effect1(nontemporal)',
            'Effect1(temporal)',
            'Effect1(temporal)|per_timestep',
            'Effect1',
        }

        assert_var_equal(model.variables['Effect1'], model.add_variables())
        assert_var_equal(model.variables['Effect1(nontemporal)'], model.add_variables())
        assert_var_equal(model.variables['Effect1(temporal)'], model.add_variables())
        assert_var_equal(model.variables['Effect1(temporal)|per_timestep'], model.add_variables(coords=(timesteps,)))

        assert_conequal(
            model.constraints['Effect1'],
            model.variables['Effect1']
            == model.variables['Effect1(temporal)'] + model.variables['Effect1(nontemporal)'],
        )
        assert_conequal(model.constraints['Effect1(nontemporal)'], model.variables['Effect1(nontemporal)'] == 0)
        assert_conequal(
            model.constraints['Effect1(temporal)'],
            model.variables['Effect1(temporal)'] == model.variables['Effect1(temporal)|per_timestep'].sum(),
        )
        assert_conequal(
            model.constraints['Effect1(temporal)|per_timestep'],
            model.variables['Effect1(temporal)|per_timestep'] == 0,
        )

    def test_bounds(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        timesteps = flow_system.time_series_collection.timesteps
        effect = fx.Effect(
            'Effect1',
            '€',
            'Testing Effect',
            minimum_temporal=1.0,
            maximum_temporal=1.1,
            minimum_nontemporal=2.0,
            maximum_nontemporal=2.1,
            minimum_total=3.0,
            maximum_total=3.1,
            minimum_per_hour=4.0,
            maximum_per_hour=4.1,
        )

        flow_system.add_elements(effect)
        model = create_linopy_model(flow_system)

        assert set(effect.model.variables) == {
            'Effect1(nontemporal)',
            'Effect1(temporal)',
            'Effect1(temporal)|per_timestep',
            'Effect1',
        }
        assert set(effect.model.constraints) == {
            'Effect1(nontemporal)',
            'Effect1(temporal)',
            'Effect1(temporal)|per_timestep',
            'Effect1',
        }

        assert_var_equal(model.variables['Effect1'], model.add_variables(lower=3.0, upper=3.1))
        assert_var_equal(model.variables['Effect1(nontemporal)'], model.add_variables(lower=2.0, upper=2.1))
        assert_var_equal(model.variables['Effect1(temporal)'], model.add_variables(lower=1.0, upper=1.1))
        assert_var_equal(
            model.variables['Effect1(temporal)|per_timestep'],
            model.add_variables(
                lower=4.0 * model.hours_per_step, upper=4.1 * model.hours_per_step, coords=(timesteps,)
            ),
        )

        assert_conequal(
            model.constraints['Effect1'],
            model.variables['Effect1']
            == model.variables['Effect1(temporal)'] + model.variables['Effect1(nontemporal)'],
        )
        assert_conequal(model.constraints['Effect1(nontemporal)'], model.variables['Effect1(nontemporal)'] == 0)
        assert_conequal(
            model.constraints['Effect1(temporal)'],
            model.variables['Effect1(temporal)'] == model.variables['Effect1(temporal)|per_timestep'].sum(),
        )
        assert_conequal(
            model.constraints['Effect1(temporal)|per_timestep'],
            model.variables['Effect1(temporal)|per_timestep'] == 0,
        )

    def test_shares(self, basic_flow_system_linopy):
        flow_system = basic_flow_system_linopy
        effect1 = fx.Effect(
            'Effect1',
            '€',
            'Testing Effect',
            specific_share_to_other_effects_operation={'Effect2': 1.1, 'Effect3': 1.2},
            specific_share_to_other_effects_invest={'Effect2': 2.1, 'Effect3': 2.2},
        )
        effect2 = fx.Effect('Effect2', '€', 'Testing Effect')
        effect3 = fx.Effect('Effect3', '€', 'Testing Effect')
        flow_system.add_elements(effect1, effect2, effect3)
        model = create_linopy_model(flow_system)

        assert set(effect2.model.variables) == {
            'Effect2(nontemporal)',
            'Effect2(temporal)',
            'Effect2(temporal)|per_timestep',
            'Effect2',
            'Effect1(nontemporal)->Effect2(nontemporal)',
            'Effect1(temporal)->Effect2(temporal)',
        }
        assert set(effect2.model.constraints) == {
            'Effect2(nontemporal)',
            'Effect2(temporal)',
            'Effect2(temporal)|per_timestep',
            'Effect2',
            'Effect1(nontemporal)->Effect2(nontemporal)',
            'Effect1(temporal)->Effect2(temporal)',
        }

        assert_conequal(
            model.constraints['Effect2(nontemporal)'],
            model.variables['Effect2(nontemporal)'] == model.variables['Effect1(nontemporal)->Effect2(nontemporal)'],
        )

        assert_conequal(
            model.constraints['Effect2(temporal)|per_timestep'],
            model.variables['Effect2(temporal)|per_timestep']
            == model.variables['Effect1(temporal)->Effect2(temporal)'],
        )

        assert_conequal(
            model.constraints['Effect1(temporal)->Effect2(temporal)'],
            model.variables['Effect1(temporal)->Effect2(temporal)']
            == model.variables['Effect1(temporal)|per_timestep'] * 1.1,
        )

        assert_conequal(
            model.constraints['Effect1(nontemporal)->Effect2(nontemporal)'],
            model.variables['Effect1(nontemporal)->Effect2(nontemporal)']
            == model.variables['Effect1(nontemporal)'] * 2.1,
        )
