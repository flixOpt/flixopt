import numpy as np
import xarray as xr

import flixopt as fx

from .conftest import (
    assert_conequal,
    assert_sets_equal,
    assert_var_equal,
    create_linopy_model,
    create_optimization_and_solve,
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
                'Effect1(periodic)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(effect.submodel.constraints),
            {
                'Effect1(periodic)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model.variables['Effect1'], model.add_variables(coords=model.get_coords(['period', 'scenario']))
        )
        assert_var_equal(
            model.variables['Effect1(periodic)'], model.add_variables(coords=model.get_coords(['period', 'scenario']))
        )
        assert_var_equal(
            model.variables['Effect1(temporal)'],
            model.add_variables(coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(temporal)|per_timestep'], model.add_variables(coords=model.get_coords())
        )

        assert_conequal(
            model.constraints['Effect1'],
            model.variables['Effect1'] == model.variables['Effect1(temporal)'] + model.variables['Effect1(periodic)'],
        )
        # In minimal/bounds tests with no contributing components, periodic totals should be zero
        assert_conequal(model.constraints['Effect1(periodic)'], model.variables['Effect1(periodic)'] == 0)
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
            minimum_periodic=2.0,
            maximum_periodic=2.1,
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
                'Effect1(periodic)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect variables',
        )

        assert_sets_equal(
            set(effect.submodel.constraints),
            {
                'Effect1(periodic)',
                'Effect1(temporal)',
                'Effect1(temporal)|per_timestep',
                'Effect1',
            },
            msg='Incorrect constraints',
        )

        assert_var_equal(
            model.variables['Effect1'],
            model.add_variables(lower=3.0, upper=3.1, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(periodic)'],
            model.add_variables(lower=2.0, upper=2.1, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(temporal)'],
            model.add_variables(lower=1.0, upper=1.1, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            model.variables['Effect1(temporal)|per_timestep'],
            model.add_variables(
                lower=4.0 * model.hours_per_step,
                upper=4.1 * model.hours_per_step,
                coords=model.get_coords(['time', 'period', 'scenario']),
            ),
        )

        assert_conequal(
            model.constraints['Effect1'],
            model.variables['Effect1'] == model.variables['Effect1(temporal)'] + model.variables['Effect1(periodic)'],
        )
        # In minimal/bounds tests with no contributing components, periodic totals should be zero
        assert_conequal(model.constraints['Effect1(periodic)'], model.variables['Effect1(periodic)'] == 0)
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
        )
        effect2 = fx.Effect(
            'Effect2',
            '€',
            'Testing Effect',
            share_from_temporal={'Effect1': 1.1},
            share_from_periodic={'Effect1': 2.1},
        )
        effect3 = fx.Effect(
            'Effect3',
            '€',
            'Testing Effect',
            share_from_temporal={'Effect1': 1.2},
            share_from_periodic={'Effect1': 2.2},
        )
        flow_system.add_elements(effect1, effect2, effect3)
        model = create_linopy_model(flow_system)

        assert_sets_equal(
            set(effect2.submodel.variables),
            {
                'Effect2(periodic)',
                'Effect2(temporal)',
                'Effect2(temporal)|per_timestep',
                'Effect2',
                'Effect1(periodic)->Effect2(periodic)',
                'Effect1(temporal)->Effect2(temporal)',
            },
            msg='Incorrect variables for effect2',
        )

        assert_sets_equal(
            set(effect2.submodel.constraints),
            {
                'Effect2(periodic)',
                'Effect2(temporal)',
                'Effect2(temporal)|per_timestep',
                'Effect2',
                'Effect1(periodic)->Effect2(periodic)',
                'Effect1(temporal)->Effect2(temporal)',
            },
            msg='Incorrect constraints for effect2',
        )

        assert_conequal(
            model.constraints['Effect2(periodic)'],
            model.variables['Effect2(periodic)'] == model.variables['Effect1(periodic)->Effect2(periodic)'],
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
            model.constraints['Effect1(periodic)->Effect2(periodic)'],
            model.variables['Effect1(periodic)->Effect2(periodic)'] == model.variables['Effect1(periodic)'] * 2.1,
        )


class TestEffectResults:
    def test_shares(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        effect1 = fx.Effect('Effect1', '€', 'Testing Effect', share_from_temporal={'costs': 0.5})
        effect2 = fx.Effect(
            'Effect2',
            '€',
            'Testing Effect',
            share_from_temporal={'Effect1': 1.1},
            share_from_periodic={'Effect1': 2.1},
        )
        effect3 = fx.Effect(
            'Effect3',
            '€',
            'Testing Effect',
            share_from_temporal={'Effect1': 1.2, 'Effect2': 5},
            share_from_periodic={'Effect1': 2.2},
        )
        flow_system.add_elements(
            effect1,
            effect2,
            effect3,
            fx.linear_converters.Boiler(
                'Boiler',
                thermal_efficiency=0.5,
                thermal_flow=fx.Flow(
                    'Q_th',
                    bus='Fernwärme',
                    size=fx.InvestParameters(effects_of_investment_per_size=10, minimum_size=20, mandatory=True),
                ),
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            ),
        )

        results = create_optimization_and_solve(flow_system, fx.solvers.HighsSolver(0.01, 60), 'Sim1').results

        effect_share_factors = {
            'temporal': {
                ('costs', 'Effect1'): 0.5,
                ('costs', 'Effect2'): 0.5 * 1.1,
                ('costs', 'Effect3'): 0.5 * 1.1 * 5 + 0.5 * 1.2,  # This is where the issue lies
                ('Effect1', 'Effect2'): 1.1,
                ('Effect1', 'Effect3'): 1.2 + 1.1 * 5,
                ('Effect2', 'Effect3'): 5,
            },
            'periodic': {
                ('Effect1', 'Effect2'): 2.1,
                ('Effect1', 'Effect3'): 2.2,
            },
        }
        for key, value in effect_share_factors['temporal'].items():
            np.testing.assert_allclose(results.effect_share_factors['temporal'][key].values, value)

        for key, value in effect_share_factors['periodic'].items():
            np.testing.assert_allclose(results.effect_share_factors['periodic'][key].values, value)

        xr.testing.assert_allclose(
            results.effects_per_component['temporal'].sum('component').sel(effect='costs', drop=True),
            results.solution['costs(temporal)|per_timestep'].fillna(0),
        )

        xr.testing.assert_allclose(
            results.effects_per_component['temporal'].sum('component').sel(effect='Effect1', drop=True),
            results.solution['Effect1(temporal)|per_timestep'].fillna(0),
        )

        xr.testing.assert_allclose(
            results.effects_per_component['temporal'].sum('component').sel(effect='Effect2', drop=True),
            results.solution['Effect2(temporal)|per_timestep'].fillna(0),
        )

        xr.testing.assert_allclose(
            results.effects_per_component['temporal'].sum('component').sel(effect='Effect3', drop=True),
            results.solution['Effect3(temporal)|per_timestep'].fillna(0),
        )

        # periodic mode checks
        xr.testing.assert_allclose(
            results.effects_per_component['periodic'].sum('component').sel(effect='costs', drop=True),
            results.solution['costs(periodic)'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['periodic'].sum('component').sel(effect='Effect1', drop=True),
            results.solution['Effect1(periodic)'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['periodic'].sum('component').sel(effect='Effect2', drop=True),
            results.solution['Effect2(periodic)'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['periodic'].sum('component').sel(effect='Effect3', drop=True),
            results.solution['Effect3(periodic)'],
        )

        # Total mode checks
        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='costs', drop=True),
            results.solution['costs'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='Effect1', drop=True),
            results.solution['Effect1'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='Effect2', drop=True),
            results.solution['Effect2'],
        )

        xr.testing.assert_allclose(
            results.effects_per_component['total'].sum('component').sel(effect='Effect3', drop=True),
            results.solution['Effect3'],
        )
