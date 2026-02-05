"""
Integration tests for complex energy systems.

.. deprecated::
    STALE — These regression baseline tests are partially superseded by tests/test_math/:
    - test_simple_flow_system → test_math/test_conversion.py + test_math/test_effects.py
    - test_model_components → test_math/test_conversion.py (boiler/CHP flow rates)
    - test_basic_flow_system → spread across test_math/ (effects, conversion, storage)
    - test_piecewise_conversion → test_math/test_piecewise.py
    The test_math tests provide isolated, analytically verified coverage per feature.
    These integration tests served as snapshot baselines for complex multi-component systems.
    Kept temporarily for reference. Safe to delete.
"""

import pytest

from ..conftest import (
    assert_almost_equal_numeric,
)

pytestmark = pytest.mark.skip(reason='Stale: superseded by tests/test_math/ — see module docstring')


class TestFlowSystem:
    def test_simple_flow_system(self, simple_flow_system, highs_solver):
        """
        Test the effects of the simple energy system model
        """
        simple_flow_system.optimize(highs_solver)

        # Cost assertions using new API (flow_system.solution)
        assert_almost_equal_numeric(
            simple_flow_system.solution['costs'].item(), 81.88394666666667, 'costs doesnt match expected value'
        )

        # CO2 assertions
        assert_almost_equal_numeric(
            simple_flow_system.solution['CO2'].item(), 255.09184, 'CO2 doesnt match expected value'
        )

    def test_model_components(self, simple_flow_system, highs_solver):
        """
        Test the component flows of the simple energy system model
        """
        simple_flow_system.optimize(highs_solver)

        # Boiler assertions using new API
        assert_almost_equal_numeric(
            simple_flow_system.solution['Boiler(Q_th)|flow_rate'].values,
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )

        # CHP unit assertions using new API
        assert_almost_equal_numeric(
            simple_flow_system.solution['CHP_unit(Q_th)|flow_rate'].values,
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )


class TestComplex:
    def test_basic_flow_system(self, flow_system_base, highs_solver):
        flow_system_base.optimize(highs_solver)

        # Assertions using flow_system.solution (the new API)
        assert_almost_equal_numeric(
            flow_system_base.solution['costs'].item(),
            -11597.873624489237,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['costs(temporal)|per_timestep'].values,
            [
                -2.38500000e03,
                -2.21681333e03,
                -2.38500000e03,
                -2.17599000e03,
                -2.35107029e03,
                -2.38500000e03,
                0.00000000e00,
                -1.68897826e-10,
                -2.16914486e-12,
            ],
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['CO2(temporal)->costs(temporal)'].sum().item(),
            258.63729669618675,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['Kessel(Q_th)->costs(temporal)'].sum().item(),
            0.01,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['Kessel->costs(temporal)'].sum().item(),
            -0.0,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['Gastarif(Q_Gas)->costs(temporal)'].sum().item(),
            39.09153113079115,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['Einspeisung(P_el)->costs(temporal)'].sum().item(),
            -14196.61245231646,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['KWK->costs(temporal)'].sum().item(),
            0.0,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['Kessel(Q_th)->costs(periodic)'].values,
            1000 + 500,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['Speicher->costs(periodic)'].values,
            800 + 1,
            'costs doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['CO2(temporal)'].values,
            1293.1864834809337,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['CO2(periodic)'].values,
            0.9999999999999994,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['Kessel(Q_th)|flow_rate'].values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['KWK(Q_th)|flow_rate'].values,
            [
                7.50000000e01,
                6.97111111e01,
                7.50000000e01,
                7.50000000e01,
                7.39330280e01,
                7.50000000e01,
                0.00000000e00,
                3.12638804e-14,
                3.83693077e-14,
            ],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_base.solution['KWK(P_el)|flow_rate'].values,
            [
                6.00000000e01,
                5.57688889e01,
                6.00000000e01,
                6.00000000e01,
                5.91464224e01,
                6.00000000e01,
                0.00000000e00,
                2.50111043e-14,
                3.06954462e-14,
            ],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['Speicher|netto_discharge'].values,
            [-45.0, -69.71111111, 15.0, -10.0, 36.06697198, -55.0, 20.0, 20.0, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )
        # charge_state includes extra timestep for final charge state (len = timesteps + 1)
        assert_almost_equal_numeric(
            flow_system_base.solution['Speicher|charge_state'].values,
            [0.0, 40.5, 100.0, 77.0, 79.84, 37.38582802, 83.89496178, 57.18336484, 32.60869565, 10.0],
            'Speicher charge_state doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_base.solution['Speicher|PiecewiseEffects|costs'].values,
            800,
            'Speicher|PiecewiseEffects|costs doesnt match expected value',
        )

    def test_piecewise_conversion(self, flow_system_piecewise_conversion, highs_solver):
        flow_system_piecewise_conversion.optimize(highs_solver)

        # Compare expected values with actual values using new API
        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['costs'].item(),
            -10710.997365760755,
            'costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['CO2'].item(),
            1278.7939026086956,
            'CO2 doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['Kessel(Q_th)|flow_rate'].values,
            [0, 0, 0, 45, 0, 0, 0, 0, 0],
            'Kessel doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['KWK(Q_th)|flow_rate'].values,
            [45.0, 45.0, 64.5962087, 100.0, 61.3136, 45.0, 45.0, 12.86469565, 0.0],
            'KWK Q_th doesnt match expected value',
        )
        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['KWK(P_el)|flow_rate'].values,
            [40.0, 40.0, 47.12589407, 60.0, 45.93221818, 40.0, 40.0, 10.91784108, -0.0],
            'KWK P_el doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['Speicher|netto_discharge'].values,
            [-15.0, -45.0, 25.4037913, -35.0, 48.6864, -25.0, -25.0, 7.13530435, 20.0],
            'Speicher nettoFlow doesnt match expected value',
        )

        assert_almost_equal_numeric(
            flow_system_piecewise_conversion.solution['Speicher|PiecewiseEffects|costs'].values,
            454.74666666666667,
            'Speicher investcosts_segmented_costs doesnt match expected value',
        )


if __name__ == '__main__':
    pytest.main(['-v'])
