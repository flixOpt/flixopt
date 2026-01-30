import pytest

from .conftest import (
    assert_almost_equal_numeric,
)


class TestFlowSystem:
    def test_simple_flow_system(self, simple_flow_system, highs_solver):
        """
        Test the effects of the simple energy system model
        """
        simple_flow_system.optimize(highs_solver)

        # Cost assertions using new API (flow_system.solution)
        assert_almost_equal_numeric(
            simple_flow_system.solution['effect|total'].sel(effect='costs').item(),
            81.88394666666667,
            'costs doesnt match expected value',
        )

        # CO2 assertions
        assert_almost_equal_numeric(
            simple_flow_system.solution['effect|total'].sel(effect='CO2').item(),
            255.09184,
            'CO2 doesnt match expected value',
        )

    def test_model_components(self, simple_flow_system, highs_solver):
        """
        Test the component flows of the simple energy system model
        """
        simple_flow_system.optimize(highs_solver)

        # Boiler assertions using new API
        assert_almost_equal_numeric(
            simple_flow_system.solution['flow|rate'].sel(flow='Boiler(Q_th)').values,
            [0, 0, 0, 28.4864, 35, 0, 0, 0, 0],
            'Q_th doesnt match expected value',
        )

        # CHP unit assertions using new API
        assert_almost_equal_numeric(
            simple_flow_system.solution['flow|rate'].sel(flow='CHP_unit(Q_th)').values,
            [30.0, 26.66666667, 75.0, 75.0, 75.0, 20.0, 20.0, 20.0, 20.0],
            'Q_th doesnt match expected value',
        )


class TestComplex:
    def test_basic_flow_system(self, flow_system_base, highs_solver):
        flow_system_base.optimize(highs_solver)
        sol = flow_system_base.solution

        # Check objective value (the most important invariant)
        # Objective = costs effect total + penalty effect total
        objective_value = flow_system_base.model.objective.value
        assert_almost_equal_numeric(
            objective_value,
            -11831.803,  # Updated for batched model implementation
            'Objective value doesnt match expected value',
        )

        # 'costs' now represents just the costs effect's total (not including penalty)
        # This is semantically correct - penalty is a separate effect
        costs_total = sol['effect|total'].sel(effect='costs').item()
        penalty_total = sol['effect|total'].sel(effect='Penalty').item()
        assert_almost_equal_numeric(
            costs_total + penalty_total,
            objective_value,
            'costs + penalty should equal objective',
        )

        # Check periodic investment costs (should be stable regardless of solution path)
        assert_almost_equal_numeric(
            sol['share|periodic'].sel(contributor='Kessel(Q_th)', effect='costs').values,
            500.0,  # effects_per_size contribution
            'Kessel periodic costs doesnt match expected value',
        )
        assert_almost_equal_numeric(
            sol['share|periodic'].sel(contributor='Speicher', effect='costs').values,
            1.0,  # effects_per_capacity contribution
            'Speicher periodic costs doesnt match expected value',
        )

        # Check CO2 effect values
        assert_almost_equal_numeric(
            sol['effect|periodic'].sel(effect='CO2').values,
            1.0,
            'CO2 periodic doesnt match expected value',
        )

        # Check piecewise effects
        assert_almost_equal_numeric(
            sol['storage|piecewise_effects|share'].sel(storage='Speicher', effect='costs').values,
            800,
            'Speicher piecewise_effects costs doesnt match expected value',
        )

        # Check that solution has all expected variable types
        assert 'costs' in sol['effect|total'].coords['effect'].values, 'costs effect should be in solution'
        assert 'Penalty' in sol['effect|total'].coords['effect'].values, 'Penalty effect should be in solution'
        assert 'CO2' in sol['effect|total'].coords['effect'].values, 'CO2 effect should be in solution'
        assert 'PE' in sol['effect|total'].coords['effect'].values, 'PE effect should be in solution'
        assert 'Kessel(Q_th)' in sol['flow|rate'].coords['flow'].values, 'Kessel flow_rate should be in solution'
        assert 'KWK(Q_th)' in sol['flow|rate'].coords['flow'].values, 'KWK flow_rate should be in solution'
        assert 'storage|charge' in sol.data_vars, 'Storage charge should be in solution'

    def test_piecewise_conversion(self, flow_system_piecewise_conversion, highs_solver):
        flow_system_piecewise_conversion.optimize(highs_solver)
        sol = flow_system_piecewise_conversion.solution

        # Check objective value
        objective_value = flow_system_piecewise_conversion.model.objective.value
        assert_almost_equal_numeric(
            objective_value,
            -10910.997,  # Updated for batched model implementation
            'Objective value doesnt match expected value',
        )

        # costs + penalty should equal objective
        costs_total = sol['effect|total'].sel(effect='costs').item()
        penalty_total = sol['effect|total'].sel(effect='Penalty').item()
        assert_almost_equal_numeric(
            costs_total + penalty_total,
            objective_value,
            'costs + penalty should equal objective',
        )

        # Check structural aspects - variables exist
        assert 'costs' in sol['effect|total'].coords['effect'].values, 'costs effect should be in solution'
        assert 'CO2' in sol['effect|total'].coords['effect'].values, 'CO2 effect should be in solution'
        assert 'Kessel(Q_th)' in sol['flow|rate'].coords['flow'].values, 'Kessel flow_rate should be in solution'
        assert 'KWK(Q_th)' in sol['flow|rate'].coords['flow'].values, 'KWK flow_rate should be in solution'

        # Check piecewise effects cost
        assert_almost_equal_numeric(
            sol['storage|piecewise_effects|share'].sel(storage='Speicher', effect='costs').values,
            454.75,
            'Speicher piecewise_effects costs doesnt match expected value',
        )


if __name__ == '__main__':
    pytest.main(['-v'])
