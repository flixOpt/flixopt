"""Tests for property guards and invalidation mechanism.

These tests verify that:
1. Label is immutable (raises AttributeError)
2. Parameter modification before linking works normally
3. Parameter modification after linking invalidates FlowSystem
4. _connected_and_transformed becomes False after modification
5. Nested object modification triggers invalidation
6. Full workflow: modify -> re-optimize works correctly
"""

import numpy as np
import pandas as pd
import pytest

import flixopt as fx
from flixopt.interface import InvestParameters, StatusParameters
from flixopt.linear_converters import HeatPump


class TestLabelImmutability:
    """Test that labels cannot be modified after creation."""

    def test_bus_label_immutable(self):
        """Bus label should raise AttributeError when modified."""
        bus = fx.Bus('TestBus', carrier='heat')

        with pytest.raises(AttributeError, match="Cannot modify 'label'"):
            bus.label = 'NewLabel'

    def test_effect_label_immutable(self):
        """Effect label should raise AttributeError when modified."""
        effect = fx.Effect('TestEffect', unit='EUR', is_standard=True, is_objective=True)

        with pytest.raises(AttributeError, match="Cannot modify 'label'"):
            effect.label = 'NewLabel'

    def test_flow_label_immutable(self):
        """Flow label should raise AttributeError when modified."""
        flow = fx.Flow('heat', bus='thermal', size=100)

        with pytest.raises(AttributeError, match="Cannot modify 'label'"):
            flow.label = 'NewLabel'

    def test_component_label_immutable(self):
        """Component label should raise AttributeError when modified."""
        source = fx.Source(
            'Source',
            outputs=[fx.Flow('heat', bus='thermal', size=100)],
        )

        with pytest.raises(AttributeError, match="Cannot modify 'label'"):
            source.label = 'NewLabel'


class TestModificationBeforeLinking:
    """Test that parameters can be modified freely before linking to FlowSystem."""

    def test_bus_carrier_modification_before_linking(self):
        """Bus carrier can be modified before linking."""
        bus = fx.Bus('TestBus', carrier='heat')

        # Should work without error
        bus.carrier = 'electricity'
        assert bus.carrier == 'electricity'

    def test_effect_unit_modification_before_linking(self):
        """Effect unit can be modified before linking."""
        effect = fx.Effect('TestEffect', unit='EUR', is_standard=True, is_objective=True)

        # Should work without error
        effect.unit = 'USD'
        assert effect.unit == 'USD'

    def test_flow_size_modification_before_linking(self):
        """Flow size can be modified before linking."""
        flow = fx.Flow('heat', bus='thermal', size=100)

        # Should work without error
        flow.size = 200
        assert flow.size == 200


class TestInvalidationAfterLinking:
    """Test that modifications after linking trigger FlowSystem invalidation."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple FlowSystem for testing."""
        timesteps = pd.date_range('2020-01-01', periods=2, freq='h', name='time')

        bus_th = fx.Bus('thermal', carrier='heat')
        effect = fx.Effect('costs', unit='EUR', is_standard=True, is_objective=True)

        source = fx.Source(
            'Source',
            outputs=[fx.Flow('heat', bus='thermal', size=100, effects_per_flow_hour=10)],
        )

        sink = fx.Sink(
            'Sink',
            inputs=[fx.Flow('heat', bus='thermal', size=100)],
        )

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(bus_th, effect, source, sink)

        return fs, source, sink

    def test_flow_modification_invalidates_system(self, simple_system):
        """Modifying a flow parameter after linking should invalidate the system."""
        fs, source, sink = simple_system

        # First, connect and transform
        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        # Modify a flow parameter
        source.outputs[0].size = 200

        # System should be invalidated
        assert fs._connected_and_transformed is False

    def test_component_modification_invalidates_system(self, simple_system):
        """Modifying a component parameter after linking should invalidate the system."""
        fs, source, sink = simple_system

        # First, connect and transform
        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        # Modify source's output flow size (component-level modification through flows)
        source.outputs[0].relative_maximum = 0.9

        # System should be invalidated
        assert fs._connected_and_transformed is False

    def test_effect_modification_invalidates_system(self, simple_system):
        """Modifying an effect parameter after linking should invalidate the system."""
        fs, source, sink = simple_system

        # First, connect and transform
        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        # Modify the effect
        fs.effects['costs'].maximum_total = 1000

        # System should be invalidated
        assert fs._connected_and_transformed is False


class TestNestedObjectInvalidation:
    """Test that modifications to nested objects trigger invalidation."""

    @pytest.fixture
    def system_with_status_parameters(self):
        """Create a system with status parameters for testing."""
        timesteps = pd.date_range('2020-01-01', periods=2, freq='h', name='time')

        bus_th = fx.Bus('thermal', carrier='heat')
        effect = fx.Effect('costs', unit='EUR', is_standard=True, is_objective=True)

        source = fx.Source(
            'Source',
            outputs=[
                fx.Flow(
                    'heat',
                    bus='thermal',
                    size=100,
                    status_parameters=StatusParameters(effects_per_startup=10),
                )
            ],
        )

        sink = fx.Sink(
            'Sink',
            inputs=[fx.Flow('heat', bus='thermal', size=100)],
        )

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(bus_th, effect, source, sink)

        return fs, source

    def test_status_parameters_modification_invalidates(self, system_with_status_parameters):
        """Modifying StatusParameters should invalidate the FlowSystem."""
        fs, source = system_with_status_parameters

        # First, connect and transform
        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        # Modify status parameters on the flow
        source.outputs[0].status_parameters.min_uptime = 5

        # System should be invalidated
        assert fs._connected_and_transformed is False

    @pytest.fixture
    def system_with_invest_parameters(self):
        """Create a system with invest parameters for testing."""
        timesteps = pd.date_range('2020-01-01', periods=2, freq='h', name='time')

        bus_th = fx.Bus('thermal', carrier='heat')
        effect = fx.Effect('costs', unit='EUR', is_standard=True, is_objective=True)

        source = fx.Source(
            'Source',
            outputs=[
                fx.Flow(
                    'heat',
                    bus='thermal',
                    size=InvestParameters(minimum_size=10, maximum_size=100, effects_of_investment_per_size=5),
                )
            ],
        )

        sink = fx.Sink(
            'Sink',
            inputs=[fx.Flow('heat', bus='thermal', size=100)],
        )

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(bus_th, effect, source, sink)

        return fs, source

    def test_invest_parameters_modification_invalidates(self, system_with_invest_parameters):
        """Modifying InvestParameters should invalidate the FlowSystem."""
        fs, source = system_with_invest_parameters

        # First, connect and transform
        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        # Modify invest parameters on the flow
        source.outputs[0].size.maximum_size = 200

        # System should be invalidated
        assert fs._connected_and_transformed is False


class TestListAndDictProperties:
    """Test that list/dict properties return copies."""

    def test_component_inputs_returns_copy(self):
        """Component.inputs should return a copy, not the internal list."""
        sink = fx.Sink(
            'Sink',
            inputs=[fx.Flow('in1', bus='thermal', size=10)],
        )

        # Get the inputs list
        inputs = sink.inputs

        # Modifying the returned list should not affect the component
        inputs.append(fx.Flow('in2', bus='thermal', size=20))

        # Original should still have only one input
        assert len(sink.inputs) == 1

    def test_flow_effects_per_flow_hour_returns_copy(self):
        """Flow.effects_per_flow_hour should return a copy."""
        flow = fx.Flow('TestFlow', bus='thermal', size=100, effects_per_flow_hour={'costs': 10})

        # Get the effects dict
        effects = flow.effects_per_flow_hour

        # Modifying the returned dict should not affect the flow
        effects['new_effect'] = 20

        # Original should still have only one effect
        assert 'new_effect' not in flow.effects_per_flow_hour


class TestFullWorkflow:
    """Test that modifying then re-optimizing works correctly."""

    @pytest.fixture
    def optimizable_system(self):
        """Create a simple optimizable system."""
        timesteps = pd.date_range('2020-01-01', periods=2, freq='h', name='time')

        bus_th = fx.Bus('thermal', carrier='heat')
        effect = fx.Effect('costs', unit='EUR', is_standard=True, is_objective=True)

        source = fx.Source(
            'Source',
            outputs=[fx.Flow('heat', bus='thermal', size=100, effects_per_flow_hour=10)],
        )

        sink = fx.Sink(
            'Sink',
            inputs=[fx.Flow('heat', bus='thermal', size=100, fixed_relative_profile=np.array([0.5, 0.8]))],
        )

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(bus_th, effect, source, sink)

        return fs, source, sink

    def test_modify_without_reset_raises_error(self, optimizable_system):
        """Test that modifying a solved system without reset() raises an error."""
        fs, source, sink = optimizable_system
        solver = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60)

        # First optimization
        fs.optimize(solver)

        # Attempting to modify without reset should raise an error
        with pytest.raises(RuntimeError, match='Call flow_system.reset\\(\\) first'):
            source.outputs[0].effects_per_flow_hour = 20

    def test_modify_and_reoptimize(self, optimizable_system):
        """Test that we can modify parameters and re-optimize after reset()."""
        fs, source, sink = optimizable_system
        solver = fx.solvers.HighsSolver(mip_gap=0, time_limit_seconds=60)

        # First optimization
        fs.optimize(solver)
        first_result = fs.solution['costs'].item()

        # Reset before modification (mandatory)
        fs.reset()

        # Modify source cost
        source.outputs[0].effects_per_flow_hour = 20

        # System should be invalidated
        assert fs._connected_and_transformed is False

        # Re-optimize (should reconnect automatically)
        fs.optimize(solver)
        second_result = fs.solution['costs'].item()

        # Results should be different (doubled cost per flow hour)
        assert second_result > first_result
        # Cost should approximately double
        assert abs(second_result / first_result - 2.0) < 0.1


class TestStorageProperties:
    """Test Storage-specific property guards."""

    @pytest.fixture
    def storage_system(self):
        """Create a system with a storage component."""
        timesteps = pd.date_range('2020-01-01', periods=2, freq='h', name='time')

        bus_th = fx.Bus('thermal', carrier='heat')
        effect = fx.Effect('costs', unit='EUR', is_standard=True, is_objective=True)

        source = fx.Source(
            'Source',
            outputs=[fx.Flow('heat', bus='thermal', size=100, effects_per_flow_hour=10)],
        )

        storage = fx.Storage(
            'Battery',
            charging=fx.Flow('charge', bus='thermal', size=50),
            discharging=fx.Flow('discharge', bus='thermal', size=50),
            capacity_in_flow_hours=100,
            eta_charge=0.95,
            eta_discharge=0.95,
            initial_charge_state=0.5,
        )

        sink = fx.Sink(
            'Sink',
            inputs=[fx.Flow('heat', bus='thermal', size=100)],
        )

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(bus_th, effect, source, storage, sink)

        return fs, storage

    def test_storage_capacity_invalidates(self, storage_system):
        """Modifying storage capacity should invalidate the system."""
        fs, storage = storage_system

        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        storage.capacity_in_flow_hours = 200
        assert fs._connected_and_transformed is False

    def test_storage_eta_invalidates(self, storage_system):
        """Modifying storage efficiency should invalidate the system."""
        fs, storage = storage_system

        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        storage.eta_charge = 0.90
        assert fs._connected_and_transformed is False

    def test_storage_initial_charge_invalidates(self, storage_system):
        """Modifying initial charge state should invalidate the system."""
        fs, storage = storage_system

        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        storage.initial_charge_state = 0.8
        assert fs._connected_and_transformed is False


class TestHeatPumpProperties:
    """Test HeatPump-specific property guards."""

    @pytest.fixture
    def heatpump_system(self):
        """Create a system with a heat pump."""
        timesteps = pd.date_range('2020-01-01', periods=2, freq='h', name='time')

        bus_th = fx.Bus('thermal', carrier='heat')
        bus_el = fx.Bus('electric', carrier='electricity')
        effect = fx.Effect('costs', unit='EUR', is_standard=True, is_objective=True)

        source_el = fx.Source(
            'GridPower',
            outputs=[fx.Flow('el', bus='electric', size=100, effects_per_flow_hour=0.15)],
        )

        heatpump = HeatPump(
            'HP',
            cop=3.5,
            electrical_flow=fx.Flow('electricity', bus='electric', size=50),
            thermal_flow=fx.Flow('heat', bus='thermal', size=None),
        )

        sink = fx.Sink(
            'HeatDemand',
            inputs=[fx.Flow('heat', bus='thermal', size=100)],
        )

        fs = fx.FlowSystem(timesteps)
        fs.add_elements(bus_th, bus_el, effect, source_el, heatpump, sink)

        return fs, heatpump

    def test_heatpump_cop_invalidates(self, heatpump_system):
        """Modifying heat pump COP should invalidate the system."""
        fs, heatpump = heatpump_system

        fs.connect_and_transform()
        assert fs._connected_and_transformed is True

        heatpump.cop = 4.0
        assert fs._connected_and_transformed is False
