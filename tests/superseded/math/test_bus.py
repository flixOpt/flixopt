import numpy as np

import flixopt as fx

from ...conftest import create_linopy_model


class TestBusModel:
    """Test the BusModel class with new batched architecture."""

    def test_bus(self, basic_flow_system_linopy_coords, coords_config):
        """Test that bus balance constraint is correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow(bus='TestBus', flow_id='Q_th_Last')]),
            fx.Source('GastarifTest', outputs=[fx.Flow(bus='TestBus', flow_id='Q_Gas')]),
        )
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist with new naming
        flow_rate = model.variables['flow|rate']
        assert 'WärmelastTest(Q_th_Last)' in flow_rate.coords['flow'].values
        assert 'GastarifTest(Q_Gas)' in flow_rate.coords['flow'].values

        # Check bus balance constraint exists
        assert 'bus|balance' in model.constraints
        assert 'TestBus' in model.constraints['bus|balance'].coords['bus'].values

        # Check balance constraint structure: supply - demand == 0
        balance = model.constraints['bus|balance'].sel(bus='TestBus')
        np.testing.assert_array_equal(balance.rhs.values, 0)
        assert (balance.sign.values == '=').all()

    def test_bus_penalty(self, basic_flow_system_linopy_coords, coords_config):
        """Test that bus penalty creates virtual supply/demand variables."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=1e5)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow(bus='TestBus', flow_id='Q_th_Last')]),
            fx.Source('GastarifTest', outputs=[fx.Flow(bus='TestBus', flow_id='Q_Gas')]),
        )
        model = create_linopy_model(flow_system)

        # Check virtual supply/demand variables exist
        assert 'bus|virtual_supply' in model.variables
        assert 'bus|virtual_demand' in model.variables

        virtual_supply = model.variables['bus|virtual_supply'].sel(bus='TestBus')
        virtual_demand = model.variables['bus|virtual_demand'].sel(bus='TestBus')

        # Check bounds are correct (lower=0, no upper bound)
        assert (virtual_supply.lower.values >= 0).all()
        assert (virtual_demand.lower.values >= 0).all()

        # Check balance constraint exists and RHS is 0
        balance = model.constraints['bus|balance'].sel(bus='TestBus')
        np.testing.assert_array_equal(balance.rhs.values, 0)
        assert (balance.sign.values == '=').all()

        # Check penalty share variable and constraint exist
        assert 'TestBus->Penalty(temporal)' in model.variables
        assert 'TestBus->Penalty(temporal)' in model.constraints

    def test_bus_with_coords(self, basic_flow_system_linopy_coords, coords_config):
        """Test bus behavior across different coordinate configurations."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config
        bus = fx.Bus('TestBus', imbalance_penalty_per_flow_hour=None)
        flow_system.add_elements(
            bus,
            fx.Sink('WärmelastTest', inputs=[fx.Flow(bus='TestBus', flow_id='Q_th_Last')]),
            fx.Source('GastarifTest', outputs=[fx.Flow(bus='TestBus', flow_id='Q_Gas')]),
        )
        model = create_linopy_model(flow_system)

        # Check flow variables exist
        flow_rate = model.variables['flow|rate']
        assert 'WärmelastTest(Q_th_Last)' in flow_rate.coords['flow'].values
        assert 'GastarifTest(Q_Gas)' in flow_rate.coords['flow'].values

        # Check bus balance constraint exists
        balance = model.constraints['bus|balance'].sel(bus='TestBus')
        np.testing.assert_array_equal(balance.rhs.values, 0)

        # Verify coordinate dimensions are correct
        gas_var = flow_rate.sel(flow='GastarifTest(Q_Gas)')
        if flow_system.scenarios is not None:
            assert 'scenario' in gas_var.dims
        assert 'time' in gas_var.dims
