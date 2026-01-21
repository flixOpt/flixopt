import flixopt as fx

from .conftest import assert_conequal, create_linopy_model


class TestBusModel:
    """Test the FlowModel class."""

    def test_bus(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        # Check batched variables exist
        assert 'flow|rate' in model.variables
        # Check flows are in the coordinate
        flow_rate_coords = list(model.variables['flow|rate'].coords['flow'].values)
        assert 'WärmelastTest(Q_th_Last)' in flow_rate_coords
        assert 'GastarifTest(Q_Gas)' in flow_rate_coords
        # Check balance constraint exists
        assert 'TestBus|balance' in model.constraints

        # Access batched flow rate variable and select individual flows
        flow_rate = model.variables['flow|rate']
        gas_flow = flow_rate.sel(flow='GastarifTest(Q_Gas)', drop=True)
        heat_flow = flow_rate.sel(flow='WärmelastTest(Q_th_Last)', drop=True)

        assert_conequal(
            model.constraints['TestBus|balance'],
            gas_flow == heat_flow,
        )

    def test_bus_penalty(self, basic_flow_system_linopy_coords, coords_config):
        """Test that flow model constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=1e5)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        # Check batched variables exist
        assert 'flow|rate' in model.variables
        flow_rate_coords = list(model.variables['flow|rate'].coords['flow'].values)
        assert 'WärmelastTest(Q_th_Last)' in flow_rate_coords
        assert 'GastarifTest(Q_Gas)' in flow_rate_coords
        # Check balance constraint exists
        assert 'TestBus|balance' in model.constraints

        # Verify batched variables exist and are accessible
        assert 'flow|rate' in model.variables
        assert 'bus|virtual_supply' in model.variables
        assert 'bus|virtual_demand' in model.variables

        # Access batched variables and select individual elements
        virtual_supply = model.variables['bus|virtual_supply'].sel(bus='TestBus', drop=True)
        virtual_demand = model.variables['bus|virtual_demand'].sel(bus='TestBus', drop=True)

        # Verify virtual supply/demand have correct lower bound (>= 0)
        assert float(virtual_supply.lower.min()) == 0.0
        assert float(virtual_demand.lower.min()) == 0.0

        # Verify the balance constraint exists
        assert 'TestBus|balance' in model.constraints

        # Penalty is now added as shares to the Penalty effect's temporal model
        # Check that the penalty shares exist in the model
        assert 'TestBus->Penalty(temporal)' in model.constraints
        assert 'TestBus->Penalty(temporal)' in model.variables

        # Verify penalty effect exists in the effects collection
        penalty_effect = flow_system.effects.penalty_effect
        assert penalty_effect is not None

    def test_bus_with_coords(self, basic_flow_system_linopy_coords, coords_config):
        """Test bus behavior across different coordinate configurations."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow('Q_th_Last', 'TestBus')]),
            fx.Source('GastarifTest', outputs=[fx.Flow('Q_Gas', 'TestBus')]),
        )
        model = create_linopy_model(flow_system)

        # Check batched variables exist
        assert 'flow|rate' in model.variables
        flow_rate_coords = list(model.variables['flow|rate'].coords['flow'].values)
        assert 'WärmelastTest(Q_th_Last)' in flow_rate_coords
        assert 'GastarifTest(Q_Gas)' in flow_rate_coords
        # Check balance constraint exists
        assert 'TestBus|balance' in model.constraints

        # Access batched flow rate variable and select individual flows
        flow_rate = model.variables['flow|rate']
        gas_flow = flow_rate.sel(flow='GastarifTest(Q_Gas)', drop=True)
        heat_flow = flow_rate.sel(flow='WärmelastTest(Q_th_Last)', drop=True)

        assert_conequal(
            model.constraints['TestBus|balance'],
            gas_flow == heat_flow,
        )

        # Just verify coordinate dimensions are correct
        if flow_system.scenarios is not None:
            assert 'scenario' in gas_flow.dims
        assert 'time' in gas_flow.dims
