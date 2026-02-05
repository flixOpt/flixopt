import numpy as np
import pytest
import xarray as xr

import flixopt as fx

from ...conftest import (
    assert_var_equal,
    create_linopy_model,
)

pytestmark = pytest.mark.skip(reason='Superseded: model-building tests implicitly covered by tests/test_math/')


class TestEffectModel:
    """Test the FlowModel class."""

    def test_minimal(self, basic_flow_system_linopy_coords, coords_config):
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        effect = fx.Effect('Effect1', '€', 'Testing Effect')

        flow_system.add_elements(effect)
        model = create_linopy_model(flow_system)

        # Check that batched effect variables exist in the model
        # Effects are now batched: effect|periodic, effect|temporal, effect|per_timestep, effect|total
        expected_vars = {'effect|periodic', 'effect|temporal', 'effect|per_timestep', 'effect|total'}
        for var_name in expected_vars:
            assert var_name in model.variables, f'Variable {var_name} should exist'

        # Check that Effect1 is in the effect coordinate
        effect_coords = model.variables['effect|total'].coords['effect'].values
        # Note: The effect names include 'costs' (default) and 'Effect1'
        assert 'Effect1' in effect_coords, 'Effect1 should be in effect coordinates'

        # Check that batched effect constraints exist in the model
        expected_cons = {'effect|periodic', 'effect|temporal', 'effect|per_timestep', 'effect|total'}
        for con_name in expected_cons:
            assert con_name in model.constraints, f'Constraint {con_name} should exist'

        # Access individual effect variables using batched model + sel
        effect_label = 'Effect1'
        effect_total = model.variables['effect|total'].sel(effect=effect_label)
        effect_periodic = model.variables['effect|periodic'].sel(effect=effect_label)
        effect_temporal = model.variables['effect|temporal'].sel(effect=effect_label)
        effect_per_ts = model.variables['effect|per_timestep'].sel(effect=effect_label)

        # Check variable bounds - verify they have no bounds (minimal effect without bounds)
        assert_var_equal(effect_total, model.add_variables(coords=model.get_coords(['period', 'scenario'])))
        assert_var_equal(effect_periodic, model.add_variables(coords=model.get_coords(['period', 'scenario'])))
        assert_var_equal(effect_temporal, model.add_variables(coords=model.get_coords(['period', 'scenario'])))
        assert_var_equal(effect_per_ts, model.add_variables(coords=model.get_coords()))

        # Constraints exist and have the effect in coordinates (structure verified by integration tests)
        assert 'Effect1' in model.constraints['effect|total'].coords['effect'].values
        assert 'Effect1' in model.constraints['effect|periodic'].coords['effect'].values
        assert 'Effect1' in model.constraints['effect|temporal'].coords['effect'].values
        assert 'Effect1' in model.constraints['effect|per_timestep'].coords['effect'].values

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

        # Check that batched effect variables exist in the model
        expected_vars = {'effect|periodic', 'effect|temporal', 'effect|per_timestep', 'effect|total'}
        for var_name in expected_vars:
            assert var_name in model.variables, f'Variable {var_name} should exist'

        # Check that Effect1 is in the effect coordinate
        effect_coords = model.variables['effect|total'].coords['effect'].values
        assert 'Effect1' in effect_coords, 'Effect1 should be in effect coordinates'

        # Check that batched effect constraints exist in the model
        expected_cons = {'effect|periodic', 'effect|temporal', 'effect|per_timestep', 'effect|total'}
        for con_name in expected_cons:
            assert con_name in model.constraints, f'Constraint {con_name} should exist'

        # Access individual effect variables using batched model + sel
        effect_label = 'Effect1'
        effect_total = model.variables['effect|total'].sel(effect=effect_label)
        effect_periodic = model.variables['effect|periodic'].sel(effect=effect_label)
        effect_temporal = model.variables['effect|temporal'].sel(effect=effect_label)
        effect_per_ts = model.variables['effect|per_timestep'].sel(effect=effect_label)

        # Check variable bounds - verify they have the specified bounds
        assert_var_equal(
            effect_total,
            model.add_variables(lower=3.0, upper=3.1, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            effect_periodic,
            model.add_variables(lower=2.0, upper=2.1, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            effect_temporal,
            model.add_variables(lower=1.0, upper=1.1, coords=model.get_coords(['period', 'scenario'])),
        )
        assert_var_equal(
            effect_per_ts,
            model.add_variables(
                lower=4.0 * model.timestep_duration,
                upper=4.1 * model.timestep_duration,
                coords=model.get_coords(['time', 'period', 'scenario']),
            ),
        )

        # Constraints exist and have the effect in coordinates (structure verified by integration tests)
        assert 'Effect1' in model.constraints['effect|total'].coords['effect'].values
        assert 'Effect1' in model.constraints['effect|periodic'].coords['effect'].values
        assert 'Effect1' in model.constraints['effect|temporal'].coords['effect'].values
        assert 'Effect1' in model.constraints['effect|per_timestep'].coords['effect'].values

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

        # Check that batched effect variables exist in the model
        expected_vars = {'effect|periodic', 'effect|temporal', 'effect|per_timestep', 'effect|total'}
        for var_name in expected_vars:
            assert var_name in model.variables, f'Variable {var_name} should exist'

        # Check that all effects are in the effect coordinate
        effect_coords = model.variables['effect|total'].coords['effect'].values
        for effect_name in ['Effect1', 'Effect2', 'Effect3']:
            assert effect_name in effect_coords, f'{effect_name} should be in effect coordinates'

        # Check that batched effect constraints exist in the model
        expected_cons = {'effect|periodic', 'effect|temporal', 'effect|per_timestep', 'effect|total'}
        for con_name in expected_cons:
            assert con_name in model.constraints, f'Constraint {con_name} should exist'

        # Check share allocation variables exist (e.g., share|temporal_from_effect for effect-to-effect shares)
        # These are managed by the EffectsModel
        assert 'share|temporal' in model.variables, 'Temporal share variable should exist'

        # Access individual effect variables using batched model + sel
        _effect2_periodic = model.variables['effect|periodic'].sel(effect='Effect2')
        _effect2_temporal = model.variables['effect|temporal'].sel(effect='Effect2')
        _effect2_per_ts = model.variables['effect|per_timestep'].sel(effect='Effect2')

        # The effect constraints are verified through the TestEffectResults tests
        # which test that the actual optimization produces correct results


class TestEffectResults:
    def test_shares(self, basic_flow_system_linopy_coords, coords_config, highs_solver):
        flow_system = basic_flow_system_linopy_coords
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
                    size=fx.InvestParameters(
                        effects_of_investment_per_size=10, minimum_size=20, maximum_size=200, mandatory=True
                    ),
                ),
                fuel_flow=fx.Flow('Q_fu', bus='Gas'),
            ),
        )

        flow_system.optimize(highs_solver)

        # Use the new statistics accessor
        statistics = flow_system.statistics

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
            np.testing.assert_allclose(statistics.effect_share_factors['temporal'][key].values, value)

        for key, value in effect_share_factors['periodic'].items():
            np.testing.assert_allclose(statistics.effect_share_factors['periodic'][key].values, value)

        # Temporal effects checks using new API
        xr.testing.assert_allclose(
            statistics.temporal_effects.sel(effect='costs', drop=True).sum('contributor'),
            flow_system.solution['effect|per_timestep'].sel(effect='costs', drop=True).fillna(0),
        )

        xr.testing.assert_allclose(
            statistics.temporal_effects.sel(effect='Effect1', drop=True).sum('contributor'),
            flow_system.solution['effect|per_timestep'].sel(effect='Effect1', drop=True).fillna(0),
        )

        xr.testing.assert_allclose(
            statistics.temporal_effects.sel(effect='Effect2', drop=True).sum('contributor'),
            flow_system.solution['effect|per_timestep'].sel(effect='Effect2', drop=True).fillna(0),
        )

        xr.testing.assert_allclose(
            statistics.temporal_effects.sel(effect='Effect3', drop=True).sum('contributor'),
            flow_system.solution['effect|per_timestep'].sel(effect='Effect3', drop=True).fillna(0),
        )

        # Periodic effects checks using new API
        xr.testing.assert_allclose(
            statistics.periodic_effects.sel(effect='costs', drop=True).sum('contributor'),
            flow_system.solution['effect|periodic'].sel(effect='costs', drop=True),
        )

        xr.testing.assert_allclose(
            statistics.periodic_effects.sel(effect='Effect1', drop=True).sum('contributor'),
            flow_system.solution['effect|periodic'].sel(effect='Effect1', drop=True),
        )

        xr.testing.assert_allclose(
            statistics.periodic_effects.sel(effect='Effect2', drop=True).sum('contributor'),
            flow_system.solution['effect|periodic'].sel(effect='Effect2', drop=True),
        )

        xr.testing.assert_allclose(
            statistics.periodic_effects.sel(effect='Effect3', drop=True).sum('contributor'),
            flow_system.solution['effect|periodic'].sel(effect='Effect3', drop=True),
        )

        # Total effects checks using new API
        xr.testing.assert_allclose(
            statistics.total_effects.sel(effect='costs', drop=True).sum('contributor'),
            flow_system.solution['effect|total'].sel(effect='costs', drop=True),
        )

        xr.testing.assert_allclose(
            statistics.total_effects.sel(effect='Effect1', drop=True).sum('contributor'),
            flow_system.solution['effect|total'].sel(effect='Effect1', drop=True),
        )

        xr.testing.assert_allclose(
            statistics.total_effects.sel(effect='Effect2', drop=True).sum('contributor'),
            flow_system.solution['effect|total'].sel(effect='Effect2', drop=True),
        )

        xr.testing.assert_allclose(
            statistics.total_effects.sel(effect='Effect3', drop=True).sum('contributor'),
            flow_system.solution['effect|total'].sel(effect='Effect3', drop=True),
        )


class TestPenaltyAsObjective:
    """Test that Penalty cannot be set as the objective effect."""

    def test_penalty_cannot_be_created_as_objective(self):
        """Test that creating a Penalty effect with is_objective=True raises ValueError."""

        with pytest.raises(ValueError, match='Penalty.*cannot be set as the objective'):
            fx.Effect('Penalty', '€', 'Test Penalty', is_objective=True)

    def test_penalty_cannot_be_set_as_objective_via_setter(self):
        """Test that setting Penalty as objective via setter raises ValueError."""
        import pandas as pd

        # Create a fresh flow system without pre-existing objective
        flow_system = fx.FlowSystem(timesteps=pd.date_range('2020-01-01', periods=10, freq='h'))
        penalty_effect = fx.Effect('Penalty', '€', 'Test Penalty', is_objective=False)

        flow_system.add_elements(penalty_effect)

        with pytest.raises(ValueError, match='Penalty.*cannot be set as the objective'):
            flow_system.effects.objective_effect = penalty_effect
