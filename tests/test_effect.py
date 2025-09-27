import numpy as np
import xarray as xr

import flixopt as fx

from .conftest import (
    assert_conequal,
    assert_sets_equal,
    assert_var_equal,
    create_calculation_and_solve,
    create_linopy_model,
)


class TestEffectModel:
    """Test the FlowModel class."""

    def test_minimal(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        effect = fx.Effect('Effect1', '€', 'Testing Effect')

        flow_system.add_elements(effect)
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(effect.submodel.variables),
            {
                'Effect1(nontemporal)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(effect.submodel.constraints),
            {
                'Effect1(nontemporal)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(model.variables['Effect1'], model.add_variables(coords=model.get_coords(['year', 'scenario'])))
        assert_var_equal(
            model.variables['Effect1(nontemporal)'], model.add_variables(coords=model.get_coords(['year', 'scenario']))
        )
        assert_var_equal(
            model.variables['Effect1(temporal)'],
            model.add_variables(coords=model.get_coords(['year', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(temporal)|per_timestep'], model.add_variables(coords=model.get_coords())
        )

        assert_conequal(
            model.constraints['Effect1'],
            model.variables['Effect1']
            == model.variables['Effect1(temporal)'] + model.variables['Effect1(nontemporal)'],
        )
        assert_conequal(model.constraints['Effect1(nontemporal)'], model.variables['Effect1(nontemporal)'] == 0)
        assert_conequal(
            model.constraints['Effect1(temporal)'],
            model.variables['Effect1(temporal)'] == model.variables['Effect1(temporal)|per_timestep'].sum('time'),
        )
        assert_conequal(
            model.constraints['Effect1(temporal)|per_timestep'],
            model.variables['Effect1(temporal)|per_timestep'] == 0,
        )

    def test_bounds(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
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

        assert_sets_equal(
            set(effect.submodel.variables),
            {
                'Effect1(nontemporal)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(effect.submodel.constraints),
            {
                'Effect1(nontemporal)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model.variables['Effect1'],
            model.add_variables(lower=3.0, upper=3.1, coords=model.get_coords(['year', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(nontemporal)'],
            model.add_variables(lower=2.0, upper=2.1, coords=model.get_coords(['year', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(temporal)'],
            model.add_variables(lower=1.0, upper=1.1, coords=model.get_coords(['year', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(temporal)|per_timestep'],
            model.add_variables(
                lower=4.0 * model.hours_per_step,
                upper=4.1 * model.hours_per_step,
                coords=model.get_coords(['time', 'year', 'scenario']),
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
            model.variables['Effect1(temporal)'] == model.variables['Effect1(temporal)|per_timestep'].sum('time'),
        )
        assert_conequal(
            model.constraints['Effect1(temporal)|per_timestep'],
            model.variables['Effect1(temporal)|per_timestep'] == 0,
        )

    def test_shares(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
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

        assert_sets_equal(
            set(effect2.submodel.variables),
            {
                'Effect2(nontemporal)',
                'Effect2(temporal)',
                'Effect2(temporal)|per_timestep',
                'Effect2',
                'Effect1(nontemporal)->Effect2(nontemporal)',
                'Effect1(temporal)->Effect2(temporal)',
            },
            msg='Incorrect variables for effect2',
        )

        assert_sets_equal(
            set(effect2.submodel.constraints),
            {
                'Effect2(nontemporal)',
                'Effect2(temporal)',
                'Effect2(temporal)|per_timestep',
                'Effect2',
                'Effect1(nontemporal)->Effect2(nontemporal)',
                'Effect1(temporal)->Effect2(temporal)',
            },
            msg='Incorrect constraints for effect2',
        )

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


class TestEffectResults:
    def test_shares(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        flow_system.effects['costs'].specific_share_to_other_effects_operation['Effect1'] = 0.5
        flow_system.add_elements(
            fx.Effect(
                'Effect1',
                '€',
                'Testing Effect',
                specific_share_to_other_effects_operation={'Effect2': 1.1, 'Effect3': 1.2},
                specific_share_to_other_effects_invest={'Effect2': 2.1, 'Effect3': 2.2},
            ),
            fx.Effect('Effect2', '€', 'Testing Effect', specific_share_to_other_effects_operation={'Effect3': 5}),
            fx.Effect('Effect3', '€', 'Testing Effect'),
            fx.linear_converters.Boiler(
                'Boiler',
                eta=0.5,
                Q_th=fx.Flow(
                    'Q_th',
                    bus='Fernwärme',
                    size=fx.InvestParameters(specific_effects=10, minimum_size=20, optional=False),
                ),
                Q_fu=fx.Flow('Q_fu', bus='Gas'),
            ),
        )

        results = create_calculation_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 60), 'Sim1').results

        effect_share_factors = {
            'operation': {
                ('costs', 'Effect1'): 0.5,
                ('costs', 'Effect2'): 0.5 * 1.1,
                ('costs', 'Effect3'): 0.5 * 1.1 * 5 + 0.5 * 1.2,  # This is where the issue lies
                ('Effect1', 'Effect2'): 1.1,
                ('Effect1', 'Effect3'): 1.2 + 1.1 * 5,
                ('Effect2', 'Effect3'): 5,
            },
            'invest': {
                ('Effect1', 'Effect2'): 2.1,
                ('Effect1', 'Effect3'): 2.2,
            },
        }
        for key, value in effect_share_factors['operation'].items():
            np.testing.assert_allclose(results.effect_share_factors['operation'][key].values, value)

        for key, value in effect_share_factors['invest'].items():
            np.testing.assert_allclose(results.effect_share_factors['invest'][key].values, value)

        xr.testing.assert_allclose(
            results.effects_per_component['operation'].sum('component').sel(effect='costs', drop=True),
            results.solution['costs(temporal)|total_per_timestep'].fillna(0),
        )

        xr.testing.assert_allclose(
            results.effects_per_component['operation'].sum('component').sel(effect='Effect1', drop=True),
            results.solution['Effect1(temporal)|total_per_timestep'].fillna(0),
        )

        xr.testing.assert_allclose(
            results.effects_per_component['operation'].sum('component').sel(effect='Effect2', drop=True),
            results.solution['Effect2(temporal)|total_per_timestep'].fillna(0),
        )

        xr.testing.assert_allclose(
            results.effects_per_component['operation'].sum('component').sel(effect='Effect3', drop=True),
            results.solution['Effect3(temporal)|total_per_timestep'].fillna(0),
        )

        # Invest mode checks
        xr.testing.assert_allclose(
            results.effects_per_component['invest'].sum('component').sel(effect='costs', drop=True),
            results.solution['costs(nontemporal)|total'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['invest'].sum('component').sel(effect='Effect1', drop=True),
            results.solution['Effect1(nontemporal)|total'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['invest'].sum('component').sel(effect='Effect2', drop=True),
            results.solution['Effect2(nontemporal)|total'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['invest'].sum('component').sel(effect='Effect3', drop=True),
            results.solution['Effect3(nontemporal)|total'],
        )

        # Total mode checks
        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='costs', drop=True),
            results.solution['costs|total'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='Effect1', drop=True),
            results.solution['Effect1|total'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='Effect2', drop=True),
            results.solution['Effect2|total'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='Effect3', drop=True),
            results.solution['Effect3|total'],
        )
