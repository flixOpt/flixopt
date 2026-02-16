"""Tests for legacy solution access patterns.

These tests verify that CONFIG.Legacy.solution_access enables backward-compatible
access to solution variables using the old naming convention.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import flixopt as fx

from .conftest import make_flow_system


class TestLegacySolutionAccess:
    """Tests for legacy solution access patterns."""

    def test_effect_access(self, optimize):
        """Test legacy effect access: solution['costs'] -> solution['effect|total'].sel(effect='costs')."""
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Source('Src', outputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, effects_per_flow_hour=1)]),
            fx.Sink(
                'Snk', inputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, fixed_relative_profile=np.array([1, 1]))]
            ),
        )
        fs = optimize(fs)

        # Legacy access should work
        legacy_result = fs.solution['costs'].item()
        # New access
        new_result = fs.solution['effect|total'].sel(effect='costs').item()

        assert_allclose(legacy_result, new_result, rtol=1e-10)
        assert_allclose(legacy_result, 20.0, rtol=1e-5)  # 2 timesteps * 10 flow * 1 cost

    def test_flow_rate_access(self, optimize):
        """Test legacy flow rate access: solution['Src(heat)|flow_rate'] -> solution['flow|rate'].sel(flow='Src(heat)')."""
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Source('Src', outputs=[fx.Flow(bus='Heat', flow_id='heat', size=10)]),
            fx.Sink(
                'Snk', inputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, fixed_relative_profile=np.array([1, 1]))]
            ),
        )
        fs = optimize(fs)

        # Legacy access should work
        legacy_result = fs.solution['Src(heat)|flow_rate'].values[:-1]  # Exclude trailing NaN
        # New access
        new_result = fs.solution['flow|rate'].sel(flow='Src(heat)').values[:-1]

        assert_allclose(legacy_result, new_result, rtol=1e-10)
        assert_allclose(legacy_result, [10, 10], rtol=1e-5)

    def test_flow_size_access(self, optimize):
        """Test legacy flow size access: solution['Src(heat)|size'] -> solution['flow|size'].sel(flow='Src(heat)')."""
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Source(
                'Src',
                outputs=[
                    fx.Flow(
                        bus='Heat', flow_id='heat', size=fx.InvestParameters(fixed_size=50), effects_per_flow_hour=1
                    )
                ],
            ),
            fx.Sink(
                'Snk', inputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, fixed_relative_profile=np.array([5, 5]))]
            ),
        )
        fs = optimize(fs)

        # Legacy access should work
        legacy_result = fs.solution['Src(heat)|size'].item()
        # New access
        new_result = fs.solution['flow|size'].sel(flow='Src(heat)').item()

        assert_allclose(legacy_result, new_result, rtol=1e-10)
        assert_allclose(legacy_result, 50.0, rtol=1e-5)

    def test_storage_charge_state_access(self, optimize):
        """Test legacy storage charge state access: solution['Battery|charge_state'] -> solution['storage|charge'].sel(storage='Battery')."""
        fs = make_flow_system(3)
        fs.add_elements(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Source('Grid', outputs=[fx.Flow(bus='Elec', flow_id='elec', size=100, effects_per_flow_hour=1)]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=10),
                discharging=fx.Flow(bus='Elec', size=10),
                capacity_in_flow_hours=50,
                initial_charge_state=25,
            ),
            fx.Sink(
                'Load',
                inputs=[fx.Flow(bus='Elec', flow_id='elec', size=10, fixed_relative_profile=np.array([1, 1, 1]))],
            ),
        )
        fs = optimize(fs)

        # Legacy access should work
        legacy_result = fs.solution['Battery|charge_state'].values
        # New access
        new_result = fs.solution['storage|charge'].sel(storage='Battery').values

        assert_allclose(legacy_result, new_result, rtol=1e-10)
        # Initial charge state is 25
        assert legacy_result[0] == 25.0

    def test_legacy_access_disabled_by_default(self):
        """Test that legacy access is disabled when CONFIG.Legacy.solution_access is False."""
        # Save current setting
        original_setting = fx.CONFIG.Legacy.solution_access

        try:
            # Disable legacy access
            fx.CONFIG.Legacy.solution_access = False

            fs = make_flow_system(2)
            fs.add_elements(
                fx.Bus('Heat'),
                fx.Effect('costs', '€', is_standard=True, is_objective=True),
                fx.Source('Src', outputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, effects_per_flow_hour=1)]),
                fx.Sink(
                    'Snk',
                    inputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, fixed_relative_profile=np.array([1, 1]))],
                ),
            )
            solver = fx.solvers.HighsSolver(log_to_console=False)
            fs.optimize(solver)

            # Legacy access should raise KeyError
            with pytest.raises(KeyError):
                _ = fs.solution['costs']

            # New access should work
            result = fs.solution['effect|total'].sel(effect='costs').item()
            assert_allclose(result, 20.0, rtol=1e-5)

        finally:
            # Restore original setting
            fx.CONFIG.Legacy.solution_access = original_setting

    def test_legacy_access_emits_deprecation_warning(self, optimize):
        """Test that legacy access emits DeprecationWarning."""
        fs = make_flow_system(2)
        fs.add_elements(
            fx.Bus('Heat'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Source('Src', outputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, effects_per_flow_hour=1)]),
            fx.Sink(
                'Snk', inputs=[fx.Flow(bus='Heat', flow_id='heat', size=10, fixed_relative_profile=np.array([1, 1]))]
            ),
        )
        fs = optimize(fs)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = fs.solution['costs']

            # Should have exactly one DeprecationWarning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert 'Legacy solution access' in str(deprecation_warnings[0].message)
            assert "solution['effect|total'].sel(effect='costs')" in str(deprecation_warnings[0].message)
