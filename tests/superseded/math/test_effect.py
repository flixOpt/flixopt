import numpy as np
import pytest

import flixopt as fx

from ...conftest import (
    create_linopy_model,
)


class TestEffectModel:
    """Test the EffectModel class with new batched architecture."""

    def test_minimal(self, basic_flow_system_linopy_coords, coords_config):
        """Test that effect model variables and constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        effect = fx.Effect('Effect1', '€', 'Testing Effect')

        flow_system.add_elements(effect)
        model = create_linopy_model(flow_system)

        # Check effect variables exist with new naming
        assert 'effect|total' in model.variables
        assert 'effect|temporal' in model.variables
        assert 'effect|periodic' in model.variables
        assert 'effect|per_timestep' in model.variables

        # Check Effect1 is in the effect dimension
        assert 'Effect1' in model.variables['effect|total'].coords['effect'].values

        # Check constraints exist
        assert 'effect|total' in model.constraints
        assert 'effect|temporal' in model.constraints
        assert 'effect|periodic' in model.constraints
        assert 'effect|per_timestep' in model.constraints

    def test_bounds(self, basic_flow_system_linopy_coords, coords_config):
        """Test that effect bounds are correctly applied."""
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

        # Check bounds on effect|total
        total_var = model.variables['effect|total'].sel(effect='Effect1')
        assert (total_var.lower.values >= 3.0).all()
        assert (total_var.upper.values <= 3.1).all()

        # Check bounds on effect|temporal
        temporal_var = model.variables['effect|temporal'].sel(effect='Effect1')
        assert (temporal_var.lower.values >= 1.0).all()
        assert (temporal_var.upper.values <= 1.1).all()

        # Check bounds on effect|periodic
        periodic_var = model.variables['effect|periodic'].sel(effect='Effect1')
        assert (periodic_var.lower.values >= 2.0).all()
        assert (periodic_var.upper.values <= 2.1).all()

        # Check bounds on effect|per_timestep (per hour bounds scaled by timestep duration)
        per_timestep_var = model.variables['effect|per_timestep'].sel(effect='Effect1')
        # Just check the bounds are set (approximately 4.0 * 1h = 4.0)
        assert (per_timestep_var.lower.values >= 3.9).all()
        assert (per_timestep_var.upper.values <= 4.2).all()

    def test_shares(self, basic_flow_system_linopy_coords, coords_config):
        """Test that effect shares are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        effect1 = fx.Effect('Effect1', '€', 'Testing Effect')
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

        # Check all effects exist
        effects_in_model = list(model.variables['effect|total'].coords['effect'].values)
        assert 'Effect1' in effects_in_model
        assert 'Effect2' in effects_in_model
        assert 'Effect3' in effects_in_model

        # Check share variables exist
        assert 'share|temporal' in model.variables
        assert 'share|periodic' in model.variables

        # Check share constraints exist for effects with shares
        assert 'share|temporal(Effect2)' in model.constraints
        assert 'share|temporal(Effect3)' in model.constraints
        assert 'share|periodic(Effect2)' in model.constraints
        assert 'share|periodic(Effect3)' in model.constraints

        # Check that Effect1 is a contributor to the shares
        temporal_shares = model.variables['share|temporal']
        assert 'Effect1' in temporal_shares.coords['contributor'].values


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

        # Use the new stats accessor
        stats = flow_system.stats

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
            np.testing.assert_allclose(stats.effect_share_factors['temporal'][key].values, value)

        for key, value in effect_share_factors['periodic'].items():
            np.testing.assert_allclose(stats.effect_share_factors['periodic'][key].values, value)

        # Temporal effects checks - compare values directly
        np.testing.assert_allclose(
            stats.temporal_effects.sel(effect='costs').sum('contributor').values,
            flow_system.solution['costs(temporal)|per_timestep'].fillna(0).values,
        )

        np.testing.assert_allclose(
            stats.temporal_effects.sel(effect='Effect1').sum('contributor').values,
            flow_system.solution['Effect1(temporal)|per_timestep'].fillna(0).values,
        )

        np.testing.assert_allclose(
            stats.temporal_effects.sel(effect='Effect2').sum('contributor').values,
            flow_system.solution['Effect2(temporal)|per_timestep'].fillna(0).values,
        )

        np.testing.assert_allclose(
            stats.temporal_effects.sel(effect='Effect3').sum('contributor').values,
            flow_system.solution['Effect3(temporal)|per_timestep'].fillna(0).values,
        )

        # Periodic effects checks - compare values directly
        np.testing.assert_allclose(
            stats.periodic_effects.sel(effect='costs').sum('contributor').values,
            flow_system.solution['costs(periodic)'].values,
        )

        np.testing.assert_allclose(
            stats.periodic_effects.sel(effect='Effect1').sum('contributor').values,
            flow_system.solution['Effect1(periodic)'].values,
        )

        np.testing.assert_allclose(
            stats.periodic_effects.sel(effect='Effect2').sum('contributor').values,
            flow_system.solution['Effect2(periodic)'].values,
        )

        np.testing.assert_allclose(
            stats.periodic_effects.sel(effect='Effect3').sum('contributor').values,
            flow_system.solution['Effect3(periodic)'].values,
        )

        # Total effects checks - compare values directly
        np.testing.assert_allclose(
            stats.total_effects.sel(effect='costs').sum('contributor').values,
            flow_system.solution['costs'].values,
        )

        np.testing.assert_allclose(
            stats.total_effects.sel(effect='Effect1').sum('contributor').values,
            flow_system.solution['Effect1'].values,
        )

        np.testing.assert_allclose(
            stats.total_effects.sel(effect='Effect2').sum('contributor').values,
            flow_system.solution['Effect2'].values,
        )

        np.testing.assert_allclose(
            stats.total_effects.sel(effect='Effect3').sum('contributor').values,
            flow_system.solution['Effect3'].values,
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
