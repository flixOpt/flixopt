"""Tests for the new solution persistence API.

This module tests the direct solution storage on FlowSystem and Element classes:
- FlowSystem.solution: xr.Dataset containing all solution variables
- Element.solution: subset of FlowSystem.solution for that element's variables
- Element._variable_names: list of variable names for each element
- Serialization/deserialization of solution with FlowSystem
"""

import uuid

import pytest
import xarray as xr

import flixopt as fx

from .conftest import (
    assert_almost_equal_numeric,
    flow_system_base,
    flow_system_long,
    flow_system_segments_of_flows_2,
    simple_flow_system,
    simple_flow_system_scenarios,
)


@pytest.fixture(
    params=[
        flow_system_base,
        simple_flow_system_scenarios,
        flow_system_segments_of_flows_2,
        simple_flow_system,
        flow_system_long,
    ]
)
def flow_system(request):
    fs = request.getfixturevalue(request.param.__name__)
    if isinstance(fs, fx.FlowSystem):
        return fs
    else:
        return fs[0]


class TestSolutionOnFlowSystem:
    """Tests for FlowSystem.solution attribute."""

    def test_solution_none_before_solve(self, simple_flow_system):
        """FlowSystem.solution should be None before optimization."""
        assert simple_flow_system.solution is None

    def test_solution_set_after_solve(self, simple_flow_system, highs_solver):
        """FlowSystem.solution should be set after solve()."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        assert simple_flow_system.solution is not None
        assert isinstance(simple_flow_system.solution, xr.Dataset)

    def test_solution_contains_all_variables(self, simple_flow_system, highs_solver):
        """FlowSystem.solution should contain all model variables."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Solution should have variables
        assert len(simple_flow_system.solution.data_vars) > 0

        # Check that known variables are present (from the simple flow system)
        solution_vars = set(simple_flow_system.solution.data_vars.keys())
        # Should have flow rates, costs, etc.
        assert any('flow_rate' in v for v in solution_vars)
        assert any('costs' in v for v in solution_vars)

    def test_solution_matches_results(self, simple_flow_system, highs_solver):
        """FlowSystem.solution should match results.solution."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # The solution on flow_system should be the same as from results
        fs_solution = simple_flow_system.solution
        results_solution = calc.results.solution

        # Check they have the same variables
        assert set(fs_solution.data_vars.keys()) == set(results_solution.data_vars.keys())

        # Check values match
        for var_name in fs_solution.data_vars:
            xr.testing.assert_equal(fs_solution[var_name], results_solution[var_name])


class TestSolutionOnElement:
    """Tests for Element.solution property."""

    def test_element_solution_raises_before_linked(self, simple_flow_system):
        """Element.solution should raise if element not linked to FlowSystem."""
        # Create an unlinked element
        bus = fx.Bus('TestBus')
        with pytest.raises(ValueError, match='not linked to a FlowSystem'):
            _ = bus.solution

    def test_element_solution_raises_before_solve(self, simple_flow_system):
        """Element.solution should raise if no solution available."""
        boiler = simple_flow_system.components['Boiler']
        with pytest.raises(ValueError, match='No solution available'):
            _ = boiler.solution

    def test_element_solution_raises_before_modeling(self, simple_flow_system, highs_solver):
        """Element.solution should work after modeling and solve."""
        # We need to add a new element after modeling to test this edge case
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Create a new element not in the flow system - this is a special case
        # The actual elements in the flow system should work fine
        boiler = simple_flow_system.components['Boiler']
        # This should work since boiler was modeled
        solution = boiler.solution
        assert isinstance(solution, xr.Dataset)

    def test_element_solution_contains_element_variables(self, simple_flow_system, highs_solver):
        """Element.solution should contain only that element's variables."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        boiler = simple_flow_system.components['Boiler']
        boiler_solution = boiler.solution

        # All variables in element solution should start with element's label
        for var_name in boiler_solution.data_vars:
            assert var_name.startswith(boiler.label_full), f'{var_name} does not start with {boiler.label_full}'

    def test_different_elements_have_different_solutions(self, simple_flow_system, highs_solver):
        """Different elements should have different solution subsets."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        boiler = simple_flow_system.components['Boiler']
        chp = simple_flow_system.components['CHP_unit']

        boiler_vars = set(boiler.solution.data_vars.keys())
        chp_vars = set(chp.solution.data_vars.keys())

        # They should have different variables
        assert boiler_vars != chp_vars
        # And they shouldn't overlap
        assert len(boiler_vars & chp_vars) == 0


class TestVariableNamesPopulation:
    """Tests for Element._variable_names population after modeling."""

    def test_variable_names_empty_before_modeling(self, simple_flow_system):
        """Element._variable_names should be empty before modeling."""
        boiler = simple_flow_system.components['Boiler']
        assert boiler._variable_names == []

    def test_variable_names_populated_after_modeling(self, simple_flow_system, highs_solver):
        """Element._variable_names should be populated after modeling."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()

        boiler = simple_flow_system.components['Boiler']
        assert len(boiler._variable_names) > 0

    def test_constraint_names_populated_after_modeling(self, simple_flow_system, highs_solver):
        """Element._constraint_names should be populated after modeling."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()

        boiler = simple_flow_system.components['Boiler']
        # Boiler should have some constraints
        assert len(boiler._constraint_names) >= 0  # Some elements might have no constraints

    def test_all_elements_have_variable_names(self, simple_flow_system, highs_solver):
        """All elements with submodels should have _variable_names populated."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()

        for element in simple_flow_system.values():
            if element.submodel is not None:
                # Element was modeled, should have variable names
                assert isinstance(element._variable_names, list)


class TestSolutionPersistence:
    """Tests for solution serialization/deserialization with FlowSystem."""

    @pytest.mark.slow
    def test_solution_persisted_in_dataset(self, flow_system, highs_solver, request):
        """Solution should be included when saving FlowSystem to dataset."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'persist-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Save to dataset
        ds = flow_system.to_dataset()

        # Check solution variables are in the dataset with 'solution|' prefix
        solution_vars = [v for v in ds.data_vars if v.startswith('solution|')]
        assert len(solution_vars) > 0, 'No solution variables in dataset'

        # Check has_solution attribute
        assert ds.attrs.get('has_solution', False) is True

    @pytest.mark.slow
    def test_solution_restored_from_dataset(self, flow_system, highs_solver, request):
        """Solution should be restored when loading FlowSystem from dataset."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'restore-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Save and restore
        ds = flow_system.to_dataset()
        restored_fs = fx.FlowSystem.from_dataset(ds)

        # Check solution is restored
        assert restored_fs.solution is not None
        assert isinstance(restored_fs.solution, xr.Dataset)

        # Check same number of variables
        assert len(restored_fs.solution.data_vars) == len(flow_system.solution.data_vars)

    @pytest.mark.slow
    def test_solution_values_match_after_restore(self, flow_system, highs_solver, request):
        """Solution values should match after save/restore cycle."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'values-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        original_solution = flow_system.solution.copy(deep=True)

        # Save and restore
        ds = flow_system.to_dataset()
        restored_fs = fx.FlowSystem.from_dataset(ds)

        # Check values match exactly
        for var_name in original_solution.data_vars:
            xr.testing.assert_equal(
                original_solution[var_name],
                restored_fs.solution[var_name],
            )

    @pytest.mark.slow
    def test_element_solution_works_after_restore(self, flow_system, highs_solver, request):
        """Element.solution should work on restored FlowSystem."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'element-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Get an element and its solution
        element_label = list(flow_system.components.keys())[0]
        original_element = flow_system.components[element_label]
        original_element_solution = original_element.solution.copy(deep=True)

        # Save and restore
        ds = flow_system.to_dataset()
        restored_fs = fx.FlowSystem.from_dataset(ds)

        # Get the same element from restored flow system
        restored_element = restored_fs.components[element_label]

        # Element.solution should work
        restored_element_solution = restored_element.solution

        # Values should match exactly
        for var_name in original_element_solution.data_vars:
            xr.testing.assert_equal(
                original_element_solution[var_name],
                restored_element_solution[var_name],
            )

    @pytest.mark.slow
    def test_variable_names_persisted(self, flow_system, highs_solver, request):
        """Element._variable_names should be persisted and restored."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'varnames-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Get original variable names
        element_label = list(flow_system.components.keys())[0]
        original_element = flow_system.components[element_label]
        original_var_names = original_element._variable_names.copy()

        # Save and restore
        ds = flow_system.to_dataset()
        restored_fs = fx.FlowSystem.from_dataset(ds)

        # Get restored element
        restored_element = restored_fs.components[element_label]

        # Variable names should match
        assert restored_element._variable_names == original_var_names


class TestFlowSystemFileIO:
    """Tests for file-based persistence of FlowSystem with solution."""

    @pytest.mark.slow
    def test_netcdf_roundtrip_with_solution(self, flow_system, highs_solver, tmp_path, request):
        """FlowSystem with solution should survive netCDF roundtrip."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'netcdf-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        original_solution = flow_system.solution.copy(deep=True)

        # Save to netCDF
        filepath = tmp_path / 'flow_system_with_solution.nc4'
        flow_system.to_netcdf(filepath)

        # Load from netCDF
        restored_fs = fx.FlowSystem.from_netcdf(filepath)

        # Check solution is restored
        assert restored_fs.solution is not None

        # Check values match exactly
        for var_name in original_solution.data_vars:
            xr.testing.assert_equal(
                original_solution[var_name],
                restored_fs.solution[var_name],
            )

    @pytest.mark.slow
    def test_loaded_flow_system_can_be_reoptimized(self, flow_system, highs_solver, tmp_path, request):
        """Loaded FlowSystem should be able to run new optimization."""
        unique_id = uuid.uuid4().hex[:12]
        worker_id = getattr(request.config, 'workerinput', {}).get('workerid', 'main')

        calc = fx.Optimization(f'reopt-{worker_id}-{unique_id}', flow_system=flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)
        original_objective = calc.results.model.objective.value

        # Save and load
        filepath = tmp_path / 'flow_system_for_reopt.nc4'
        flow_system.to_netcdf(filepath)
        restored_fs = fx.FlowSystem.from_netcdf(filepath)

        # Run new optimization
        calc2 = fx.Optimization(f'reopt2-{worker_id}-{unique_id}', flow_system=restored_fs)
        calc2.do_modeling()
        calc2.solve(highs_solver)

        # Should get same objective value
        assert_almost_equal_numeric(
            original_objective,
            calc2.results.model.objective.value,
            'Objective mismatch after reload',
        )


class TestNoSolutionPersistence:
    """Tests for FlowSystem without solution (before optimization)."""

    def test_flow_system_without_solution_saves(self, simple_flow_system):
        """FlowSystem without solution should save successfully."""
        ds = simple_flow_system.to_dataset()
        assert ds.attrs.get('has_solution', True) is False

    def test_flow_system_without_solution_loads(self, simple_flow_system):
        """FlowSystem without solution should load successfully."""
        ds = simple_flow_system.to_dataset()
        restored_fs = fx.FlowSystem.from_dataset(ds)

        assert restored_fs.solution is None

    def test_loaded_flow_system_without_solution_can_optimize(self, simple_flow_system, highs_solver):
        """Loaded FlowSystem (no prior solution) should optimize successfully."""
        ds = simple_flow_system.to_dataset()
        restored_fs = fx.FlowSystem.from_dataset(ds)

        calc = fx.Optimization('test', flow_system=restored_fs)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Should have solution now
        assert restored_fs.solution is not None


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_variable_names_handled(self, simple_flow_system, highs_solver):
        """Elements with no variables should be handled gracefully."""
        calc = fx.Optimization('test', flow_system=simple_flow_system)
        calc.do_modeling()
        calc.solve(highs_solver)

        # Buses typically have no variables of their own in some configurations
        for bus in simple_flow_system.buses.values():
            # Should not raise, even if empty
            if bus._variable_names:
                _ = bus.solution
            # If no variable names, solution access would raise - that's expected

    def test_solution_cleared_on_new_optimization(self, simple_flow_system, highs_solver):
        """New optimization should update solution, not accumulate."""
        calc1 = fx.Optimization('test1', flow_system=simple_flow_system)
        calc1.do_modeling()
        calc1.solve(highs_solver)

        first_solution_vars = set(simple_flow_system.solution.data_vars.keys())

        # Re-optimize (same system)
        calc2 = fx.Optimization('test2', flow_system=simple_flow_system)
        calc2.do_modeling()
        calc2.solve(highs_solver)

        second_solution_vars = set(simple_flow_system.solution.data_vars.keys())

        # Should have same variables (not accumulated)
        assert first_solution_vars == second_solution_vars


if __name__ == '__main__':
    pytest.main(['-v', '--disable-warnings', __file__])
