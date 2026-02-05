"""
Tests for FlowSystem locking behavior (read-only after optimization).

A FlowSystem becomes locked (read-only) when it has a solution.
This prevents accidental modifications to a system that has already been optimized.
"""

import copy
import warnings

import pytest

import flixopt as fx

from ..conftest import build_simple_flow_system


class TestIsLocked:
    """Test the is_locked property."""

    def test_not_locked_initially(self, simple_flow_system):
        """A new FlowSystem should not be locked."""
        assert simple_flow_system.is_locked is False

    def test_not_locked_after_build_model(self, simple_flow_system):
        """FlowSystem should not be locked after build_model (no solution yet)."""
        simple_flow_system.build_model()
        assert simple_flow_system.is_locked is False

    def test_locked_after_optimization(self, simple_flow_system, highs_solver):
        """FlowSystem should be locked after optimization."""
        simple_flow_system.optimize(highs_solver)
        assert simple_flow_system.is_locked is True

    def test_not_locked_after_reset(self, simple_flow_system, highs_solver):
        """FlowSystem should not be locked after reset."""
        simple_flow_system.optimize(highs_solver)
        assert simple_flow_system.is_locked is True

        simple_flow_system.reset()
        assert simple_flow_system.is_locked is False


class TestAddElementsLocking:
    """Test that add_elements respects locking."""

    def test_add_elements_before_optimization(self, simple_flow_system):
        """Should be able to add elements before optimization."""
        new_bus = fx.Bus('NewBus')
        simple_flow_system.add_elements(new_bus)
        assert 'NewBus' in simple_flow_system.buses

    def test_add_elements_raises_when_locked(self, simple_flow_system, highs_solver):
        """Should raise RuntimeError when adding elements to a locked FlowSystem."""
        simple_flow_system.optimize(highs_solver)

        new_bus = fx.Bus('NewBus')
        with pytest.raises(RuntimeError, match='Cannot add elements.*reset\\(\\)'):
            simple_flow_system.add_elements(new_bus)

    def test_add_elements_after_reset(self, simple_flow_system, highs_solver):
        """Should be able to add elements after reset."""
        simple_flow_system.optimize(highs_solver)
        simple_flow_system.reset()

        new_bus = fx.Bus('NewBus')
        simple_flow_system.add_elements(new_bus)
        assert 'NewBus' in simple_flow_system.buses

    def test_add_elements_invalidates_model(self, simple_flow_system):
        """Adding elements to a FlowSystem with a model should invalidate the model."""
        simple_flow_system.build_model()
        assert simple_flow_system.model is not None

        new_bus = fx.Bus('NewBus')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            simple_flow_system.add_elements(new_bus)
            assert len(w) == 1
            assert 'model will be invalidated' in str(w[0].message)

        assert simple_flow_system.model is None


class TestAddCarriersLocking:
    """Test that add_carriers respects locking."""

    def test_add_carriers_before_optimization(self, simple_flow_system):
        """Should be able to add carriers before optimization."""
        carrier = fx.Carrier('biogas', '#00FF00', 'kW')
        simple_flow_system.add_carriers(carrier)
        assert 'biogas' in simple_flow_system.carriers

    def test_add_carriers_raises_when_locked(self, simple_flow_system, highs_solver):
        """Should raise RuntimeError when adding carriers to a locked FlowSystem."""
        simple_flow_system.optimize(highs_solver)

        carrier = fx.Carrier('biogas', '#00FF00', 'kW')
        with pytest.raises(RuntimeError, match='Cannot add carriers.*reset\\(\\)'):
            simple_flow_system.add_carriers(carrier)

    def test_add_carriers_after_reset(self, simple_flow_system, highs_solver):
        """Should be able to add carriers after reset."""
        simple_flow_system.optimize(highs_solver)
        simple_flow_system.reset()

        carrier = fx.Carrier('biogas', '#00FF00', 'kW')
        simple_flow_system.add_carriers(carrier)
        assert 'biogas' in simple_flow_system.carriers

    def test_add_carriers_invalidates_model(self, simple_flow_system):
        """Adding carriers to a FlowSystem with a model should invalidate the model."""
        simple_flow_system.build_model()
        assert simple_flow_system.model is not None

        carrier = fx.Carrier('biogas', '#00FF00', 'kW')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            simple_flow_system.add_carriers(carrier)
            assert len(w) == 1
            assert 'model will be invalidated' in str(w[0].message)

        assert simple_flow_system.model is None


class TestReset:
    """Test the reset method."""

    def test_reset_clears_solution(self, simple_flow_system, highs_solver):
        """Reset should clear the solution."""
        simple_flow_system.optimize(highs_solver)
        assert simple_flow_system.solution is not None

        simple_flow_system.reset()
        assert simple_flow_system.solution is None

    def test_reset_clears_model(self, simple_flow_system, highs_solver):
        """Reset should clear the model."""
        simple_flow_system.optimize(highs_solver)
        assert simple_flow_system.model is not None

        simple_flow_system.reset()
        assert simple_flow_system.model is None

    def test_reset_clears_element_submodels(self, simple_flow_system, highs_solver):
        """Reset should clear element submodels."""
        simple_flow_system.optimize(highs_solver)

        # Check that elements have submodels after optimization
        boiler = simple_flow_system.components['Boiler']
        assert boiler.submodel is not None
        assert len(boiler._variable_names) > 0

        simple_flow_system.reset()

        # Check that submodels are cleared
        assert boiler.submodel is None
        assert len(boiler._variable_names) == 0

    def test_reset_returns_self(self, simple_flow_system, highs_solver):
        """Reset should return self for method chaining."""
        simple_flow_system.optimize(highs_solver)
        result = simple_flow_system.reset()
        assert result is simple_flow_system

    def test_reset_allows_reoptimization(self, simple_flow_system, highs_solver):
        """After reset, FlowSystem can be optimized again."""
        simple_flow_system.optimize(highs_solver)
        original_cost = simple_flow_system.solution['costs'].item()

        simple_flow_system.reset()
        simple_flow_system.optimize(highs_solver)

        assert simple_flow_system.solution is not None
        # Cost should be the same since system structure didn't change
        assert simple_flow_system.solution['costs'].item() == pytest.approx(original_cost)


class TestCopy:
    """Test the copy method."""

    @pytest.fixture(scope='class')
    def optimized_flow_system(self):
        """Pre-optimized flow system shared across TestCopy (tests only work with copies)."""
        fs = build_simple_flow_system()
        solver = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=300)
        fs.optimize(solver)
        return fs

    def test_copy_creates_new_instance(self, simple_flow_system):
        """Copy should create a new FlowSystem instance."""
        copy_fs = simple_flow_system.copy()
        assert copy_fs is not simple_flow_system

    def test_copy_preserves_elements(self, simple_flow_system):
        """Copy should preserve all elements."""
        copy_fs = simple_flow_system.copy()

        assert set(copy_fs.components.keys()) == set(simple_flow_system.components.keys())
        assert set(copy_fs.buses.keys()) == set(simple_flow_system.buses.keys())

    def test_copy_does_not_copy_solution(self, optimized_flow_system):
        """Copy should not include the solution."""
        assert optimized_flow_system.solution is not None

        copy_fs = optimized_flow_system.copy()
        assert copy_fs.solution is None

    def test_copy_does_not_copy_model(self, optimized_flow_system):
        """Copy should not include the model."""
        assert optimized_flow_system.model is not None

        copy_fs = optimized_flow_system.copy()
        assert copy_fs.model is None

    def test_copy_is_not_locked(self, optimized_flow_system):
        """Copy should not be locked even if original is."""
        assert optimized_flow_system.is_locked is True

        copy_fs = optimized_flow_system.copy()
        assert copy_fs.is_locked is False

    def test_copy_can_be_modified(self, optimized_flow_system):
        """Copy should be modifiable even if original is locked."""
        copy_fs = optimized_flow_system.copy()
        new_bus = fx.Bus('NewBus')
        copy_fs.add_elements(new_bus)  # Should not raise
        assert 'NewBus' in copy_fs.buses

    def test_copy_can_be_optimized_independently(self, optimized_flow_system):
        """Copy can be optimized independently of original."""
        original_cost = optimized_flow_system.solution['costs'].item()

        copy_fs = optimized_flow_system.copy()
        solver = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=300)
        copy_fs.optimize(solver)

        # Both should have solutions
        assert optimized_flow_system.solution is not None
        assert copy_fs.solution is not None

        # Costs should be equal (same system)
        assert copy_fs.solution['costs'].item() == pytest.approx(original_cost)

    def test_python_copy_uses_copy_method(self, optimized_flow_system):
        """copy.copy() should use the custom copy method."""
        copy_fs = copy.copy(optimized_flow_system)
        assert copy_fs.solution is None
        assert copy_fs.is_locked is False

    def test_python_deepcopy_uses_copy_method(self, optimized_flow_system):
        """copy.deepcopy() should use the custom copy method."""
        copy_fs = copy.deepcopy(optimized_flow_system)
        assert copy_fs.solution is None
        assert copy_fs.is_locked is False


class TestLoadedFlowSystem:
    """Test that loaded FlowSystems respect locking."""

    def test_loaded_fs_with_solution_is_locked(self, simple_flow_system, highs_solver, tmp_path):
        """A FlowSystem loaded from file with solution should be locked."""
        simple_flow_system.optimize(highs_solver)
        filepath = tmp_path / 'test_fs.nc'
        simple_flow_system.to_netcdf(filepath)

        loaded_fs = fx.FlowSystem.from_netcdf(filepath)
        assert loaded_fs.is_locked is True

    def test_loaded_fs_can_be_reset(self, simple_flow_system, highs_solver, tmp_path):
        """A loaded FlowSystem can be reset to allow modifications."""
        simple_flow_system.optimize(highs_solver)
        filepath = tmp_path / 'test_fs.nc'
        simple_flow_system.to_netcdf(filepath)

        loaded_fs = fx.FlowSystem.from_netcdf(filepath)
        loaded_fs.reset()

        assert loaded_fs.is_locked is False
        new_bus = fx.Bus('NewBus')
        loaded_fs.add_elements(new_bus)  # Should not raise


class TestInvalidate:
    """Test the invalidate method for manual model invalidation."""

    def test_invalidate_resets_connected_and_transformed(self, simple_flow_system):
        """Invalidate should reset the connected_and_transformed flag."""
        simple_flow_system.connect_and_transform()
        assert simple_flow_system.connected_and_transformed is True

        simple_flow_system.invalidate()
        assert simple_flow_system.connected_and_transformed is False

    def test_invalidate_clears_model(self, simple_flow_system):
        """Invalidate should clear the model."""
        simple_flow_system.build_model()
        assert simple_flow_system.model is not None

        simple_flow_system.invalidate()
        assert simple_flow_system.model is None

    def test_invalidate_raises_when_locked(self, simple_flow_system, highs_solver):
        """Invalidate should raise RuntimeError when FlowSystem has a solution."""
        simple_flow_system.optimize(highs_solver)

        with pytest.raises(RuntimeError, match='Cannot invalidate.*reset\\(\\)'):
            simple_flow_system.invalidate()

    def test_invalidate_returns_self(self, simple_flow_system):
        """Invalidate should return self for method chaining."""
        simple_flow_system.connect_and_transform()
        result = simple_flow_system.invalidate()
        assert result is simple_flow_system

    def test_invalidate_allows_retransformation(self, simple_flow_system, highs_solver):
        """After invalidate, connect_and_transform should run again."""
        simple_flow_system.connect_and_transform()
        assert simple_flow_system.connected_and_transformed is True

        simple_flow_system.invalidate()
        assert simple_flow_system.connected_and_transformed is False

        # Should be able to connect_and_transform again
        simple_flow_system.connect_and_transform()
        assert simple_flow_system.connected_and_transformed is True

    def test_modify_element_and_invalidate(self, simple_flow_system, highs_solver):
        """Test the workflow: optimize -> reset -> modify -> invalidate -> re-optimize."""
        # First optimization
        simple_flow_system.optimize(highs_solver)
        original_cost = simple_flow_system.solution['costs'].item()

        # Reset to unlock
        simple_flow_system.reset()

        # Modify an element attribute (increase gas price, which should increase costs)
        gas_tariff = simple_flow_system.components['Gastarif']
        original_effects = gas_tariff.outputs[0].effects_per_flow_hour
        # Double the cost effect
        gas_tariff.outputs[0].effects_per_flow_hour = {effect: value * 2 for effect, value in original_effects.items()}

        # Invalidate to trigger re-transformation
        simple_flow_system.invalidate()

        # Re-optimize
        simple_flow_system.optimize(highs_solver)
        new_cost = simple_flow_system.solution['costs'].item()

        # Cost should have increased due to higher gas price
        assert new_cost > original_cost

    def test_invalidate_needed_after_transform_before_optimize(self, simple_flow_system, highs_solver):
        """Invalidate is needed to apply changes made after connect_and_transform but before optimize."""
        # Connect and transform (but don't optimize yet)
        simple_flow_system.connect_and_transform()

        # Modify an attribute - double the gas costs
        gas_tariff = simple_flow_system.components['Gastarif']
        original_effects = gas_tariff.outputs[0].effects_per_flow_hour
        gas_tariff.outputs[0].effects_per_flow_hour = {effect: value * 2 for effect, value in original_effects.items()}

        # Call invalidate to ensure re-transformation
        simple_flow_system.invalidate()
        assert simple_flow_system.connected_and_transformed is False

        # Now optimize - the doubled values should take effect
        simple_flow_system.optimize(highs_solver)
        cost_with_doubled = simple_flow_system.solution['costs'].item()

        # Reset and use original values
        simple_flow_system.reset()
        gas_tariff.outputs[0].effects_per_flow_hour = {
            effect: value / 2 for effect, value in gas_tariff.outputs[0].effects_per_flow_hour.items()
        }
        simple_flow_system.optimize(highs_solver)
        cost_with_original = simple_flow_system.solution['costs'].item()

        # The doubled costs should result in higher total cost
        assert cost_with_doubled > cost_with_original

    def test_reset_already_invalidates(self, simple_flow_system, highs_solver):
        """Reset already invalidates, so modifications after reset take effect."""
        # First optimization
        simple_flow_system.optimize(highs_solver)
        original_cost = simple_flow_system.solution['costs'].item()

        # Reset - this already calls _invalidate_model()
        simple_flow_system.reset()
        assert simple_flow_system.connected_and_transformed is False

        # Modify an element attribute
        gas_tariff = simple_flow_system.components['Gastarif']
        original_effects = gas_tariff.outputs[0].effects_per_flow_hour
        gas_tariff.outputs[0].effects_per_flow_hour = {effect: value * 2 for effect, value in original_effects.items()}

        # Re-optimize - changes take effect because reset already invalidated
        simple_flow_system.optimize(highs_solver)
        new_cost = simple_flow_system.solution['costs'].item()

        # Cost should have increased
        assert new_cost > original_cost
