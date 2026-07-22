import numpy as np
import pytest

import flixopt as fx

from ...conftest import create_linopy_model


class TestStorageModel:
    """Test that storage model variables and constraints are correctly generated."""

    def test_basic_storage(self, basic_flow_system_linopy_coords, coords_config):
        """Test that basic storage model variables and constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a simple storage
        storage = fx.Storage(
            'TestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,  # 30 kWh storage capacity
            initial_charge_state=0,  # Start empty
            prevent_simultaneous_charge_and_discharge=True,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist with new naming
        flow_rate = model.variables['flow|rate']
        assert 'TestStorage(Q_th_in)' in flow_rate.coords['flow'].values
        assert 'TestStorage(Q_th_out)' in flow_rate.coords['flow'].values

        # Check storage variables exist
        assert 'storage|charge' in model.variables
        assert 'storage|netto' in model.variables
        charge = model.variables['storage|charge']
        netto = model.variables['storage|netto']
        assert 'TestStorage' in charge.coords['storage'].values
        assert 'TestStorage' in netto.coords['storage'].values

        # Check constraints exist
        assert 'storage|netto_eq' in model.constraints
        assert 'storage|balance' in model.constraints
        assert 'storage|initial_charge_state' in model.constraints

        # Check variable bounds
        in_rate = flow_rate.sel(flow='TestStorage(Q_th_in)')
        out_rate = flow_rate.sel(flow='TestStorage(Q_th_out)')
        assert (in_rate.lower.values >= 0).all()
        assert (in_rate.upper.values <= 20).all()
        assert (out_rate.lower.values >= 0).all()
        assert (out_rate.upper.values <= 20).all()

        # Check charge bounds
        cs = charge.sel(storage='TestStorage')
        assert (cs.lower.values >= 0).all()
        assert (cs.upper.values <= 30).all()

    def test_lossy_storage(self, basic_flow_system_linopy_coords, coords_config):
        """Test that lossy storage model variables and constraints are correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a simple storage
        storage = fx.Storage(
            'TestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,  # 30 kWh storage capacity
            initial_charge_state=0,  # Start empty
            eta_charge=0.9,  # Charging efficiency
            eta_discharge=0.8,  # Discharging efficiency
            relative_loss_per_hour=0.05,  # 5% loss per hour
            prevent_simultaneous_charge_and_discharge=True,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'TestStorage(Q_th_in)' in flow_rate.coords['flow'].values
        assert 'TestStorage(Q_th_out)' in flow_rate.coords['flow'].values

        # Check storage variables exist
        assert 'storage|charge' in model.variables
        assert 'storage|netto' in model.variables

        # Check constraints exist
        assert 'storage|netto_eq' in model.constraints
        assert 'storage|balance' in model.constraints
        assert 'storage|initial_charge_state' in model.constraints

        # Check variable bounds
        in_rate = flow_rate.sel(flow='TestStorage(Q_th_in)')
        out_rate = flow_rate.sel(flow='TestStorage(Q_th_out)')
        assert (in_rate.lower.values >= 0).all()
        assert (in_rate.upper.values <= 20).all()
        assert (out_rate.lower.values >= 0).all()
        assert (out_rate.upper.values <= 20).all()

        # Check charge bounds
        charge = model.variables['storage|charge'].sel(storage='TestStorage')
        assert (charge.lower.values >= 0).all()
        assert (charge.upper.values <= 30).all()

    def test_charge_state_bounds(self, basic_flow_system_linopy_coords, coords_config):
        """Test that storage with time-varying charge state bounds is correctly generated."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create a simple storage with time-varying bounds
        storage = fx.Storage(
            'TestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,  # 30 kWh storage capacity
            initial_charge_state=3,
            prevent_simultaneous_charge_and_discharge=True,
            relative_maximum_charge_state=np.array([0.14, 0.22, 0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.78, 0.86]),
            relative_minimum_charge_state=np.array([0.07, 0.11, 0.15, 0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43]),
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check that flow rate variables exist
        flow_rate = model.variables['flow|rate']
        assert 'TestStorage(Q_th_in)' in flow_rate.coords['flow'].values
        assert 'TestStorage(Q_th_out)' in flow_rate.coords['flow'].values

        # Check storage variables exist
        assert 'storage|charge' in model.variables
        assert 'storage|netto' in model.variables

        # Check constraints exist
        assert 'storage|netto_eq' in model.constraints
        assert 'storage|balance' in model.constraints
        assert 'storage|initial_charge_state' in model.constraints

        # Check variable bounds - time-varying
        in_rate = flow_rate.sel(flow='TestStorage(Q_th_in)')
        out_rate = flow_rate.sel(flow='TestStorage(Q_th_out)')
        assert (in_rate.lower.values >= 0).all()
        assert (in_rate.upper.values <= 20).all()
        assert (out_rate.lower.values >= 0).all()
        assert (out_rate.upper.values <= 20).all()

        # Check charge has time-varying bounds
        charge = model.variables['storage|charge'].sel(storage='TestStorage')
        # Lower bounds should be at least min_relative * capacity
        assert (charge.lower.values >= 0.07 * 30 - 0.1).all()  # Small tolerance
        # Upper bounds should be at most max_relative * capacity
        assert (charge.upper.values <= 0.86 * 30 + 0.1).all()

    def test_storage_with_investment(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with investment parameters."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with investment parameters
        storage = fx.Storage(
            'InvestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=fx.InvestParameters(
                effects_of_investment={'costs': 100},
                effects_of_investment_per_size={'costs': 10},
                minimum_size=20,
                maximum_size=100,
                mandatory=False,
            ),
            initial_charge_state=0,
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
            prevent_simultaneous_charge_and_discharge=True,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check storage variables exist
        assert 'storage|charge' in model.variables
        charge = model.variables['storage|charge']
        assert 'InvestStorage' in charge.coords['storage'].values

        # Check investment variables exist
        assert 'storage|size' in model.variables
        assert 'storage|invested' in model.variables
        size_var = model.variables['storage|size']
        invested_var = model.variables['storage|invested']
        assert 'InvestStorage' in size_var.coords['storage'].values
        assert 'InvestStorage' in invested_var.coords['storage'].values

        # Check investment constraints exist
        assert 'storage|size|ub' in model.constraints
        assert 'storage|size|lb' in model.constraints

        # Check variable bounds
        size = size_var.sel(storage='InvestStorage')
        assert (size.lower.values >= 0).all()  # Optional investment
        assert (size.upper.values <= 100).all()

    def test_storage_with_final_state_constraints(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with final state constraints."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with final state constraints
        storage = fx.Storage(
            'FinalStateStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,
            initial_charge_state=10,  # Start with 10 kWh
            minimal_final_charge_state=15,  # End with at least 15 kWh
            maximal_final_charge_state=25,  # End with at most 25 kWh
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check final state constraints exist
        assert 'storage|final_charge_min' in model.constraints
        assert 'storage|final_charge_max' in model.constraints

        # Check initial charge state constraint exists
        assert 'storage|initial_charge_state' in model.constraints

    def test_storage_cyclic_initialization(self, basic_flow_system_linopy_coords, coords_config):
        """Test storage with cyclic initialization."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with cyclic initialization
        storage = fx.Storage(
            'CyclicStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,
            initial_charge_state='equals_final',  # Cyclic initialization
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check cyclic constraint exists
        assert 'storage|initial_equals_final' in model.constraints

    @pytest.mark.parametrize(
        'prevent_simultaneous',
        [True, False],
    )
    def test_simultaneous_charge_discharge(self, basic_flow_system_linopy_coords, coords_config, prevent_simultaneous):
        """Test prevent_simultaneous_charge_and_discharge parameter."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create storage with or without simultaneous charge/discharge prevention
        storage = fx.Storage(
            'SimultaneousStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=30,
            initial_charge_state=0,
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
            prevent_simultaneous_charge_and_discharge=prevent_simultaneous,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Binary status variables should exist when preventing simultaneous operation
        if prevent_simultaneous:
            assert 'flow|status' in model.variables
            status_var = model.variables['flow|status']
            assert 'SimultaneousStorage(Q_th_in)' in status_var.coords['flow'].values
            assert 'SimultaneousStorage(Q_th_out)' in status_var.coords['flow'].values

            # Check for constraint that enforces either charging or discharging
            assert 'storage|prevent_simultaneous' in model.constraints

    @pytest.mark.parametrize(
        'mandatory,minimum_size,expected_invested',
        [
            (False, None, True),  # Optional with no min_size -> invested variable
            (False, 20, True),  # Optional with min_size -> invested variable
            (True, None, False),  # Mandatory with no min_size -> no invested variable needed
            (True, 20, False),  # Mandatory with min_size -> no invested variable needed
        ],
    )
    def test_investment_parameters(
        self,
        basic_flow_system_linopy_coords,
        coords_config,
        mandatory,
        minimum_size,
        expected_invested,
    ):
        """Test different investment parameter combinations."""
        flow_system, coords_config = basic_flow_system_linopy_coords, coords_config

        # Create investment parameters
        invest_params = {
            'effects_of_investment': {'costs': 100},
            'effects_of_investment_per_size': {'costs': 10},
            'mandatory': mandatory,
            'maximum_size': 100,
        }
        if minimum_size is not None:
            invest_params['minimum_size'] = minimum_size

        # Create storage with specified investment parameters
        storage = fx.Storage(
            'InvestStorage',
            charging=fx.Flow('Q_th_in', bus='Fernwärme', size=20),
            discharging=fx.Flow('Q_th_out', bus='Fernwärme', size=20),
            capacity_in_flow_hours=fx.InvestParameters(**invest_params),
            initial_charge_state=0,
            eta_charge=0.9,
            eta_discharge=0.9,
            relative_loss_per_hour=0.05,
        )

        flow_system.add_elements(storage)
        model = create_linopy_model(flow_system)

        # Check size variable exists
        assert 'storage|size' in model.variables
        size_var = model.variables['storage|size']
        assert 'InvestStorage' in size_var.coords['storage'].values

        # Check invested variable based on mandatory flag
        if expected_invested:
            assert 'storage|invested' in model.variables
            invested_var = model.variables['storage|invested']
            assert 'InvestStorage' in invested_var.coords['storage'].values

            # Check constraints for optional investment
            assert 'storage|size|lb' in model.constraints
            assert 'storage|size|ub' in model.constraints
