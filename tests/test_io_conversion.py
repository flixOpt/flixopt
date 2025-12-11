"""Tests for the IO conversion utilities for backwards compatibility."""

import xarray as xr

from flixopt.io import (
    PARAMETER_RENAMES,
    VALUE_RENAMES,
    _rename_keys_recursive,
    convert_old_dataset,
    convert_old_netcdf,
    load_dataset_from_netcdf,
    save_dataset_to_netcdf,
)


class TestRenameKeysRecursive:
    """Tests for the _rename_keys_recursive function."""

    def test_simple_key_rename(self):
        """Test basic key renaming."""
        old = {'minimum_operation': 100}
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert 'minimum_temporal' in result
        assert 'minimum_operation' not in result
        assert result['minimum_temporal'] == 100

    def test_nested_key_rename(self):
        """Test key renaming in nested structures."""
        old = {
            'components': {
                'Boiler': {
                    'on_off_parameters': {
                        'on_hours_total_max': 50,
                    }
                }
            }
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert 'status_parameters' in result['components']['Boiler']
        assert 'on_off_parameters' not in result['components']['Boiler']
        assert result['components']['Boiler']['status_parameters']['on_hours_max'] == 50

    def test_class_name_rename(self):
        """Test that __class__ values are also renamed."""
        old = {
            '__class__': 'OnOffParameters',
            'on_hours_total_max': 100,
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['__class__'] == 'StatusParameters'
        assert result['on_hours_max'] == 100

    def test_value_rename(self):
        """Test value renaming for specific keys."""
        old = {'initial_charge_state': 'lastValueOfSim'}
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['initial_charge_state'] == 'equals_final'

    def test_list_handling(self):
        """Test that lists are processed correctly."""
        old = {
            'flows': [
                {'flow_hours_total_max': 100},
                {'flow_hours_total_min': 50},
            ]
        }
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['flows'][0]['flow_hours_max'] == 100
        assert result['flows'][1]['flow_hours_min'] == 50

    def test_unchanged_keys_preserved(self):
        """Test that keys not in rename map are preserved."""
        old = {'label': 'MyComponent', 'size': 100}
        result = _rename_keys_recursive(old, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result['label'] == 'MyComponent'
        assert result['size'] == 100

    def test_empty_dict(self):
        """Test handling of empty dict."""
        result = _rename_keys_recursive({}, PARAMETER_RENAMES, VALUE_RENAMES)
        assert result == {}

    def test_empty_list(self):
        """Test handling of empty list."""
        result = _rename_keys_recursive([], PARAMETER_RENAMES, VALUE_RENAMES)
        assert result == []

    def test_scalar_values(self):
        """Test that scalar values are returned unchanged."""
        assert _rename_keys_recursive(42, PARAMETER_RENAMES, VALUE_RENAMES) == 42
        assert _rename_keys_recursive('string', PARAMETER_RENAMES, VALUE_RENAMES) == 'string'
        assert _rename_keys_recursive(None, PARAMETER_RENAMES, VALUE_RENAMES) is None


class TestParameterRenames:
    """Tests to verify all expected parameter renames are in the mapping."""

    def test_effect_parameters(self):
        """Test Effect parameter renames are defined."""
        assert PARAMETER_RENAMES['minimum_operation'] == 'minimum_temporal'
        assert PARAMETER_RENAMES['maximum_operation'] == 'maximum_temporal'
        assert PARAMETER_RENAMES['minimum_invest'] == 'minimum_periodic'
        assert PARAMETER_RENAMES['maximum_invest'] == 'maximum_periodic'
        assert PARAMETER_RENAMES['minimum_investment'] == 'minimum_periodic'
        assert PARAMETER_RENAMES['maximum_investment'] == 'maximum_periodic'
        assert PARAMETER_RENAMES['minimum_operation_per_hour'] == 'minimum_per_hour'
        assert PARAMETER_RENAMES['maximum_operation_per_hour'] == 'maximum_per_hour'

    def test_invest_parameters(self):
        """Test InvestParameters renames are defined."""
        assert PARAMETER_RENAMES['fix_effects'] == 'effects_of_investment'
        assert PARAMETER_RENAMES['specific_effects'] == 'effects_of_investment_per_size'
        assert PARAMETER_RENAMES['divest_effects'] == 'effects_of_retirement'
        assert PARAMETER_RENAMES['piecewise_effects'] == 'piecewise_effects_of_investment'

    def test_flow_parameters(self):
        """Test Flow/OnOffParameters renames are defined."""
        assert PARAMETER_RENAMES['flow_hours_total_max'] == 'flow_hours_max'
        assert PARAMETER_RENAMES['flow_hours_total_min'] == 'flow_hours_min'
        assert PARAMETER_RENAMES['on_hours_total_max'] == 'on_hours_max'
        assert PARAMETER_RENAMES['on_hours_total_min'] == 'on_hours_min'
        assert PARAMETER_RENAMES['switch_on_total_max'] == 'switch_on_max'

    def test_bus_parameters(self):
        """Test Bus parameter renames are defined."""
        assert PARAMETER_RENAMES['excess_penalty_per_flow_hour'] == 'imbalance_penalty_per_flow_hour'

    def test_component_parameters(self):
        """Test component parameter renames are defined."""
        assert PARAMETER_RENAMES['source'] == 'outputs'
        assert PARAMETER_RENAMES['sink'] == 'inputs'
        assert PARAMETER_RENAMES['prevent_simultaneous_sink_and_source'] == 'prevent_simultaneous_flow_rates'

    def test_linear_converter_parameters(self):
        """Test linear converter parameter renames are defined."""
        assert PARAMETER_RENAMES['Q_fu'] == 'fuel_flow'
        assert PARAMETER_RENAMES['P_el'] == 'electrical_flow'
        assert PARAMETER_RENAMES['Q_th'] == 'thermal_flow'
        assert PARAMETER_RENAMES['Q_ab'] == 'heat_source_flow'
        assert PARAMETER_RENAMES['eta'] == 'thermal_efficiency'
        assert PARAMETER_RENAMES['eta_th'] == 'thermal_efficiency'
        assert PARAMETER_RENAMES['eta_el'] == 'electrical_efficiency'
        assert PARAMETER_RENAMES['COP'] == 'cop'

    def test_class_renames(self):
        """Test class name renames are defined."""
        assert PARAMETER_RENAMES['OnOffParameters'] == 'StatusParameters'
        assert PARAMETER_RENAMES['on_off_parameters'] == 'status_parameters'
        assert PARAMETER_RENAMES['FullCalculation'] == 'Optimization'
        assert PARAMETER_RENAMES['AggregatedCalculation'] == 'ClusteredOptimization'
        assert PARAMETER_RENAMES['SegmentedCalculation'] == 'SegmentedOptimization'
        assert PARAMETER_RENAMES['CalculationResults'] == 'Results'
        assert PARAMETER_RENAMES['AggregationParameters'] == 'ClusteringParameters'

    def test_time_series_data_parameters(self):
        """Test TimeSeriesData parameter renames are defined."""
        assert PARAMETER_RENAMES['agg_group'] == 'aggregation_group'
        assert PARAMETER_RENAMES['agg_weight'] == 'aggregation_weight'


class TestValueRenames:
    """Tests for value renaming."""

    def test_initial_charge_state_value(self):
        """Test initial_charge_state value rename is defined."""
        assert VALUE_RENAMES['initial_charge_state']['lastValueOfSim'] == 'equals_final'


class TestConvertOldDataset:
    """Tests for convert_old_dataset function."""

    def test_converts_attrs(self):
        """Test that dataset attrs are converted."""
        ds = xr.Dataset(attrs={'minimum_operation': 100, 'maximum_invest': 500})
        result = convert_old_dataset(ds)
        assert 'minimum_temporal' in result.attrs
        assert 'maximum_periodic' in result.attrs
        assert 'minimum_operation' not in result.attrs
        assert 'maximum_invest' not in result.attrs

    def test_nested_attrs_conversion(self):
        """Test conversion of nested attrs structures."""
        ds = xr.Dataset(
            attrs={
                'components': {
                    'Boiler': {
                        '__class__': 'OnOffParameters',
                        'on_hours_total_max': 100,
                    }
                }
            }
        )
        result = convert_old_dataset(ds)
        assert result.attrs['components']['Boiler']['__class__'] == 'StatusParameters'
        assert result.attrs['components']['Boiler']['on_hours_max'] == 100

    def test_custom_renames(self):
        """Test that custom renames can be provided."""
        ds = xr.Dataset(attrs={'custom_old': 'value'})
        result = convert_old_dataset(ds, key_renames={'custom_old': 'custom_new'}, value_renames={})
        assert 'custom_new' in result.attrs
        assert 'custom_old' not in result.attrs

    def test_returns_same_object(self):
        """Test that the function modifies and returns the same dataset object."""
        ds = xr.Dataset(attrs={'minimum_operation': 100})
        result = convert_old_dataset(ds)
        # Note: attrs are modified in place, so the object should be the same
        assert result is ds


class TestConvertOldNetcdf:
    """Tests for convert_old_netcdf function."""

    def test_load_and_convert(self, tmp_path):
        """Test loading and converting a netCDF file."""
        # Create an old-style dataset and save it
        old_ds = xr.Dataset(
            {'var1': (['time'], [1, 2, 3])},
            coords={'time': [0, 1, 2]},
            attrs={
                'components': {
                    'Boiler': {
                        '__class__': 'OnOffParameters',
                        'on_hours_total_max': 100,
                    }
                }
            },
        )
        input_path = tmp_path / 'old_system.nc'
        save_dataset_to_netcdf(old_ds, input_path)

        # Convert
        result = convert_old_netcdf(input_path)

        # Verify conversion
        assert result.attrs['components']['Boiler']['__class__'] == 'StatusParameters'
        assert result.attrs['components']['Boiler']['on_hours_max'] == 100

    def test_load_convert_and_save(self, tmp_path):
        """Test loading, converting, and saving to new file."""
        # Create an old-style dataset and save it
        old_ds = xr.Dataset(
            {'var1': (['time'], [1, 2, 3])},
            coords={'time': [0, 1, 2]},
            attrs={'minimum_operation': 100},
        )
        input_path = tmp_path / 'old_system.nc'
        output_path = tmp_path / 'new_system.nc'
        save_dataset_to_netcdf(old_ds, input_path)

        # Convert and save
        convert_old_netcdf(input_path, output_path)

        # Load the new file and verify
        loaded = load_dataset_from_netcdf(output_path)
        assert 'minimum_temporal' in loaded.attrs
        assert loaded.attrs['minimum_temporal'] == 100


class TestFullConversionScenario:
    """Integration tests for full conversion scenarios."""

    def test_complex_flowsystem_structure(self):
        """Test conversion of a complex FlowSystem-like structure."""
        old_structure = {
            '__class__': 'FlowSystem',
            'components': {
                'Boiler': {
                    '__class__': 'LinearConverter',
                    'Q_fu': ':::Boiler|fuel',
                    'eta': 0.9,
                    'on_off_parameters': {
                        '__class__': 'OnOffParameters',
                        'on_hours_total_max': 100,
                        'switch_on_total_max': 10,
                    },
                },
                'HeatPump': {
                    '__class__': 'HeatPumpWithSource',
                    'COP': 3.5,
                    'Q_ab': ':::HeatPump|ambient',
                },
                'Battery': {
                    '__class__': 'Storage',
                    'initial_charge_state': 'lastValueOfSim',
                },
                'Grid': {
                    '__class__': 'Source',
                    'source': [{'__class__': 'Flow', 'flow_hours_total_max': 1000}],
                },
                'Demand': {
                    '__class__': 'Sink',
                    'sink': [{'__class__': 'Flow', 'flow_hours_total_min': 500}],
                },
            },
            'effects': {
                'costs': {
                    '__class__': 'Effect',
                    'minimum_operation': 0,
                    'maximum_invest': 1000,
                    'minimum_operation_per_hour': 0,
                },
            },
            'buses': {
                'heat_bus': {
                    '__class__': 'Bus',
                    'excess_penalty_per_flow_hour': 1000,
                },
            },
        }

        result = _rename_keys_recursive(old_structure, PARAMETER_RENAMES, VALUE_RENAMES)

        # Verify component conversions
        boiler = result['components']['Boiler']
        assert boiler['fuel_flow'] == ':::Boiler|fuel'
        assert boiler['thermal_efficiency'] == 0.9
        assert boiler['status_parameters']['__class__'] == 'StatusParameters'
        assert boiler['status_parameters']['on_hours_max'] == 100
        assert boiler['status_parameters']['switch_on_max'] == 10

        heat_pump = result['components']['HeatPump']
        assert heat_pump['cop'] == 3.5
        assert heat_pump['heat_source_flow'] == ':::HeatPump|ambient'

        battery = result['components']['Battery']
        assert battery['initial_charge_state'] == 'equals_final'

        grid = result['components']['Grid']
        assert 'outputs' in grid
        assert grid['outputs'][0]['flow_hours_max'] == 1000

        demand = result['components']['Demand']
        assert 'inputs' in demand
        assert demand['inputs'][0]['flow_hours_min'] == 500

        # Verify effect conversions
        costs = result['effects']['costs']
        assert costs['minimum_temporal'] == 0
        assert costs['maximum_periodic'] == 1000
        assert costs['minimum_per_hour'] == 0

        # Verify bus conversions
        heat_bus = result['buses']['heat_bus']
        assert heat_bus['imbalance_penalty_per_flow_hour'] == 1000

    def test_invest_parameters_conversion(self):
        """Test conversion of InvestParameters."""
        old_structure = {
            '__class__': 'InvestParameters',
            'fix_effects': {'costs': 1000},
            'specific_effects': {'costs': 100},
            'divest_effects': {'costs': 500},
            'piecewise_effects': {'__class__': 'PiecewiseEffects'},
        }

        result = _rename_keys_recursive(old_structure, PARAMETER_RENAMES, VALUE_RENAMES)

        assert result['effects_of_investment'] == {'costs': 1000}
        assert result['effects_of_investment_per_size'] == {'costs': 100}
        assert result['effects_of_retirement'] == {'costs': 500}
        assert result['piecewise_effects_of_investment']['__class__'] == 'PiecewiseEffects'
