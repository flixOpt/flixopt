"""Unit tests for BackwardsCompatibleDataset functionality."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flixopt.results import BackwardsCompatibleDataset


class TestBackwardsCompatibleDataset:
    """Test suite for BackwardsCompatibleDataset class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with new variable names for testing."""
        data = {
            # Effect variables (new names)
            'costs': xr.DataArray([100.0], dims=['scalar']),
            'emissions': xr.DataArray([50.0], dims=['scalar']),
            # Cross-effect variables with new naming
            'Boiler_01(Natural_Gas)->costs(temporal)': xr.DataArray([20.0, 25.0, 30.0], dims=['time']),
            'Boiler_01(Natural_Gas)->costs(nontemporal)': xr.DataArray([500.0], dims=['scalar']),
            'HeatPump_02(Electricity)->emissions(temporal)': xr.DataArray([5.0, 6.0, 7.0], dims=['time']),
            'Storage_01(Electricity)->emissions(nontemporal)': xr.DataArray([10.0], dims=['scalar']),
            # Parameter variables with new naming
            'minimum_temporal': xr.DataArray([10.0], dims=['scalar']),
            'maximum_temporal': xr.DataArray([200.0], dims=['scalar']),
            'minimum_nontemporal': xr.DataArray([5.0], dims=['scalar']),
            'maximum_nontemporal': xr.DataArray([500.0], dims=['scalar']),
            'minimum_temporal_per_hour': xr.DataArray([1.0], dims=['scalar']),
            'maximum_temporal_per_hour': xr.DataArray([10.0], dims=['scalar']),
            # Regular variables (no renaming needed)
            'flow_rate': xr.DataArray([75.0, 85.0, 95.0], dims=['time']),
            'charge_state': xr.DataArray([40.0, 60.0, 80.0], dims=['time']),
        }

        coords = {'time': pd.date_range('2023-01-01', periods=3, freq='h'), 'scalar': ['value']}

        return xr.Dataset(data, coords=coords)

    @pytest.fixture
    def bc_dataset(self, sample_dataset):
        """Create a BackwardsCompatibleDataset instance."""
        return BackwardsCompatibleDataset(sample_dataset)

    def test_init(self, sample_dataset):
        """Test BackwardsCompatibleDataset initialization."""
        bc_dataset = BackwardsCompatibleDataset(sample_dataset)
        assert bc_dataset._dataset is sample_dataset
        assert bc_dataset._mapping_cache == {}

    @pytest.mark.parametrize(
        'deprecated_name,expected_name',
        [
            ('costs|total', 'costs'),
            # Add more direct mappings here as they are added to DEPRECATED_VARIABLE_MAPPING
        ],
    )
    def test_direct_mapping_access(self, bc_dataset, sample_dataset, deprecated_name, expected_name):
        """Test accessing variables via direct mapping."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = bc_dataset[deprecated_name]

            # Should issue exactly one deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert f"'{deprecated_name}' is deprecated" in str(w[0].message)
            assert f"Use '{expected_name}' instead" in str(w[0].message)

            # Should return the correct data
            expected = sample_dataset[expected_name]
            assert np.array_equal(result.values, expected.values)
            assert result.dims == expected.dims

    @pytest.mark.parametrize(
        'deprecated_key,expected_key',
        [
            # (operation) -> (temporal) pattern substitutions
            ('Boiler_01(Natural_Gas)->costs(operation)', 'Boiler_01(Natural_Gas)->costs(temporal)'),
            ('HeatPump_02(Electricity)->emissions(operation)', 'HeatPump_02(Electricity)->emissions(temporal)'),
            # (invest) -> (nontemporal) pattern substitutions
            ('Boiler_01(Natural_Gas)->costs(invest)', 'Boiler_01(Natural_Gas)->costs(nontemporal)'),
            ('Storage_01(Electricity)->emissions(invest)', 'Storage_01(Electricity)->emissions(nontemporal)'),
        ],
    )
    def test_pattern_substitution_cross_effects(self, bc_dataset, sample_dataset, deprecated_key, expected_key):
        """Test pattern-based substitution for cross-effect variables."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = bc_dataset[deprecated_key]

            # Check warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert f"'{deprecated_key}' is deprecated" in str(w[0].message)
            assert f"Use '{expected_key}' instead" in str(w[0].message)

            # Check data correctness
            expected = sample_dataset[expected_key]
            assert np.array_equal(result.values, expected.values)

    @pytest.mark.parametrize(
        'deprecated_name,expected_name',
        [
            ('minimum_operation', 'minimum_temporal'),
            ('maximum_operation', 'maximum_temporal'),
            ('minimum_invest', 'minimum_nontemporal'),
            ('maximum_invest', 'maximum_nontemporal'),
            ('minimum_operation_per_hour', 'minimum_temporal_per_hour'),
            ('maximum_operation_per_hour', 'maximum_temporal_per_hour'),
        ],
    )
    def test_regex_patterns(self, bc_dataset, sample_dataset, deprecated_name, expected_name):
        """Test regex-based pattern matching for parameter names."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = bc_dataset[deprecated_name]

            # Check warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert f"'{deprecated_name}' is deprecated" in str(w[0].message)
            assert f"Use '{expected_name}' instead" in str(w[0].message)

            # Check data correctness
            expected = sample_dataset[expected_name]
            assert np.array_equal(result.values, expected.values)

    def test_no_renaming_for_existing_variables(self, bc_dataset, sample_dataset):
        """Test that existing variables are accessed without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = bc_dataset['flow_rate']

            # Should not issue any warnings
            assert len(w) == 0

            # Should return correct data
            expected = sample_dataset['flow_rate']
            assert np.array_equal(result.values, expected.values)

    @pytest.mark.parametrize(
        'nonexistent_key,description',
        [
            ('nonexistent_variable', 'completely nonexistent variable'),
            ('missing_operation', 'regex pattern that would transform but target missing'),
            ('missing_invest', 'regex pattern that would transform but target missing'),
            ('Component->missing(operation)', 'cross-effect pattern that would transform but target missing'),
            ('Component->missing(invest)', 'cross-effect pattern that would transform but target missing'),
            ('fake|mapping', 'direct mapping pattern but not in mapping dict'),
            ('costs|wrong', 'partial direct mapping but wrong suffix'),
        ],
    )
    def test_nonexistent_variable_error(self, bc_dataset, nonexistent_key, description):
        """Test that accessing nonexistent variables raises appropriate errors."""
        # Just test that KeyError is raised, don't try to match the message for complex keys
        if '->' in nonexistent_key or '|' in nonexistent_key:
            with pytest.raises(KeyError):
                bc_dataset[nonexistent_key]
        else:
            with pytest.raises(KeyError, match=nonexistent_key):
                bc_dataset[nonexistent_key]

    @pytest.mark.parametrize(
        'variable_name,should_exist',
        [
            # Existing variables (current names)
            ('costs', True),
            ('emissions', True),
            ('flow_rate', True),
            ('charge_state', True),
            ('minimum_temporal', True),
            ('maximum_nontemporal', True),
            # Direct mapping deprecated names
            ('costs|total', True),
            # Pattern substitution deprecated names
            ('Boiler_01(Natural_Gas)->costs(operation)', True),
            ('Storage_01(Electricity)->emissions(invest)', True),
            ('HeatPump_02(Electricity)->emissions(operation)', True),
            # Regex pattern deprecated names
            ('minimum_operation', True),
            ('maximum_invest', True),
            ('minimum_operation_per_hour', True),
            ('maximum_operation_per_hour', True),
            # Non-existent variables
            ('nonexistent_var', False),
            ('another_missing(operation)', False),
            ('fake_operation', False),
            ('not_real_invest', False),
            ('', False),  # Empty string
        ],
    )
    def test_contains_method(self, bc_dataset, variable_name, should_exist):
        """Test __contains__ method with backwards compatibility."""
        if should_exist:
            assert variable_name in bc_dataset
        else:
            assert variable_name not in bc_dataset

    @pytest.mark.parametrize(
        'deprecated_key,expected_key,mapping_type',
        [
            ('costs|total', 'costs', 'direct_mapping'),
            (
                'Boiler_01(Natural_Gas)->costs(operation)',
                'Boiler_01(Natural_Gas)->costs(temporal)',
                'pattern_substitution',
            ),
            ('minimum_operation', 'minimum_temporal', 'regex_pattern'),
            ('maximum_invest', 'maximum_nontemporal', 'regex_pattern'),
        ],
    )
    def test_caching_mechanism(self, bc_dataset, deprecated_key, expected_key, mapping_type):
        """Test that the mapping cache works correctly for all mapping types."""
        # Initial cache should be empty for this key
        assert deprecated_key not in bc_dataset._mapping_cache

        # First access should populate cache
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result1 = bc_dataset[deprecated_key]

        # Check that mapping was cached
        assert deprecated_key in bc_dataset._mapping_cache
        assert bc_dataset._mapping_cache[deprecated_key] == expected_key

        # Second access should use cache (same result, no additional processing)
        cache_before = bc_dataset._mapping_cache.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result2 = bc_dataset[deprecated_key]

        # Cache should be unchanged and results should be identical
        assert bc_dataset._mapping_cache == cache_before
        assert np.array_equal(result1.values, result2.values)

    def test_get_deprecated_mappings(self, bc_dataset):
        """Test the get_deprecated_mappings debugging method."""
        mappings = bc_dataset.get_deprecated_mappings()

        # Should include direct mappings
        assert 'costs|total' in mappings
        assert mappings['costs|total'] == 'costs'

        # Check some specific pattern-based mappings that should exist
        # These correspond to the actual variables in our sample dataset
        expected_mappings = [
            ('Boiler_01(Natural_Gas)->costs(operation)', 'Boiler_01(Natural_Gas)->costs(temporal)'),
            ('Boiler_01(Natural_Gas)->costs(invest)', 'Boiler_01(Natural_Gas)->costs(nontemporal)'),
            ('HeatPump_02(Electricity)->emissions(operation)', 'HeatPump_02(Electricity)->emissions(temporal)'),
            ('Storage_01(Electricity)->emissions(invest)', 'Storage_01(Electricity)->emissions(nontemporal)'),
        ]

        for old_name, new_name in expected_mappings:
            if new_name in bc_dataset._dataset:  # Only check if target exists
                assert old_name in mappings, f"Expected mapping '{old_name}' -> '{new_name}' not found"
                assert mappings[old_name] == new_name

        # Should be a reasonable number of mappings
        assert len(mappings) >= 3  # At least direct + some pattern mappings

        # Debug print to help understand what mappings were found
        print(f'\nFound {len(mappings)} deprecated mappings:')
        for old, new in mappings.items():
            print(f'  {old} -> {new}')

    def test_dataset_method_delegation(self, bc_dataset, sample_dataset):
        """Test that dataset methods are properly delegated."""
        # Test len
        assert len(bc_dataset) == len(sample_dataset)

        # Test keys
        assert list(bc_dataset.keys()) == list(sample_dataset.keys())

        # Test iteration
        bc_vars = list(bc_dataset)
        dataset_vars = list(sample_dataset)
        assert bc_vars == dataset_vars

        # Test values method exists
        assert hasattr(bc_dataset, 'values')

        # Test items method exists
        assert hasattr(bc_dataset, 'items')

    def test_raw_dataset_property(self, bc_dataset, sample_dataset):
        """Test _raw_dataset property returns the original dataset."""
        assert bc_dataset._raw_dataset is sample_dataset

    def test_getattr_delegation(self, bc_dataset, sample_dataset):
        """Test that unknown attributes are delegated to the wrapped dataset."""
        # Test accessing dataset attributes
        assert bc_dataset.sizes == sample_dataset.sizes
        assert bc_dataset.coords.keys() == sample_dataset.coords.keys()
        assert list(bc_dataset.data_vars.keys()) == list(sample_dataset.data_vars.keys())

    @pytest.mark.parametrize(
        'test_key,should_trigger_pattern,description',
        [
            # Cases that should NOT trigger cross-effect pattern substitution (no ->)
            ('operation_only', False, 'standalone operation word without arrow'),
            ('invest_only', False, 'standalone invest word without arrow'),
            ('some_operation_name', False, 'operation in middle of name without arrow'),
            ('invest_cost', False, 'invest at start of name without arrow'),
            # Cases that SHOULD trigger cross-effect pattern substitution (has ->)
            ('NonExistent->something(operation)', True, 'has arrow and operation pattern'),
            ('Component->effect(invest)', True, 'has arrow and invest pattern'),
            ('A->B(operation)', True, 'minimal arrow with operation'),
            ('X->Y(invest)', True, 'minimal arrow with invest'),
            # Cases that should NOT trigger because missing parentheses
            ('Component->operation', False, 'has arrow but no parentheses around operation'),
            ('Component->invest', False, 'has arrow but no parentheses around invest'),
        ],
    )
    def test_pattern_condition_functions(self, bc_dataset, test_key, should_trigger_pattern, description):
        """Test that pattern conditions work correctly."""
        # All these test keys should fail with KeyError since they don't exist
        # But the important thing is whether they trigger pattern matching attempt
        with pytest.raises(KeyError):
            bc_dataset[test_key]

        # We can indirectly test if pattern was attempted by checking the cache
        # If pattern was attempted but failed, no cache entry should be made
        assert test_key not in bc_dataset._mapping_cache

    def test_multiple_pattern_matching(self, bc_dataset, sample_dataset):
        """Test behavior when multiple patterns could potentially match."""
        # Create a variable that could match multiple patterns
        # This tests the order of pattern application

        # Add a variable that has both cross-effect pattern and could match regex
        sample_dataset['test_operation'] = xr.DataArray([123.0], dims=['scalar'])
        bc_dataset_new = BackwardsCompatibleDataset(sample_dataset)

        # This should match the regex pattern (*_operation -> *_temporal)
        # not the cross-effect pattern (since no -> in key)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _ = bc_dataset_new['test_operation']  # Should access as-is, no renaming

            # Should not issue warning since 'test_operation' exists as-is
            assert len(w) == 0

    @pytest.mark.parametrize(
        'test_input,expected_exception,description',
        [
            # Empty and None inputs
            ('', KeyError, 'empty string'),
            (None, TypeError, 'None input'),
            # Partial pattern matches that shouldn't transform
            ('not_a_real_operation_var', KeyError, 'operation in name but no real match'),
            ('operation_but_no_suffix', KeyError, 'operation word but not as suffix'),
            ('prefix_invest_middle', KeyError, 'invest in middle but not as suffix'),
            ('something_operation_something', KeyError, 'operation in middle with suffixes'),
            # Special characters
            ('costs|', KeyError, 'partial pipe character'),
            ('|total', KeyError, 'pipe at start'),
            ('costs||total', KeyError, 'double pipe'),
            ('costs|total|extra', KeyError, 'extra parts after valid pattern'),
            # Arrow patterns that don't match
            ('->', KeyError, 'just arrow'),
            ('->operation', KeyError, 'arrow at start'),
            ('operation->', KeyError, 'arrow at end'),
            ('->->', KeyError, 'double arrow'),
            # Mixed patterns
            ('costs|total(operation)', KeyError, 'mixing direct and pattern syntax'),
            ('minimum_operation|total', KeyError, 'mixing regex and direct syntax'),
        ],
    )
    def test_edge_cases(self, bc_dataset, test_input, expected_exception, description):
        """Test various edge cases."""
        with pytest.raises(expected_exception):
            bc_dataset[test_input]

    def test_performance_with_many_accesses(self, bc_dataset):
        """Test that caching provides performance benefits."""
        deprecated_key = 'costs|total'

        # Access the same key many times
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(100):
                result = bc_dataset[deprecated_key]
                assert result is not None

        # Should have cached the result
        assert deprecated_key in bc_dataset._mapping_cache

    def test_regex_pattern_coverage(self):
        """Test that regex patterns cover expected cases."""
        patterns = BackwardsCompatibleDataset._get_regex_patterns()

        # Should have patterns for operation, invest, and operation_per_hour
        descriptions = [desc for _, _, desc in patterns]
        assert any('operation suffix to temporal' in desc for desc in descriptions)
        assert any('invest suffix to nontemporal' in desc for desc in descriptions)
        assert any('operation_per_hour to temporal_per_hour' in desc for desc in descriptions)

    def test_warning_message_format(self, bc_dataset):
        """Test that warning messages have the correct format."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            bc_dataset['costs|total']

            warning_msg = str(w[0].message)
            assert "'costs|total' is deprecated" in warning_msg
            assert "Use 'costs' instead" in warning_msg

    @pytest.mark.parametrize(
        'variable_name,value,description',
        [
            # Similar names that should NOT trigger renaming
            ('operational_cost', 999.0, 'operational (not operation) suffix'),
            ('investment_return', 888.0, 'investment (not invest) prefix'),
            ('operation_mode', 777.0, 'operation as prefix not suffix'),
            ('invest_strategy', 666.0, 'invest as prefix not suffix'),
            ('cooperative_effort', 555.0, 'contains operation but different context'),
            ('reinvestment_plan', 444.0, 'contains invest but different context'),
            ('temporal_sequence', 333.0, 'temporal as prefix (target name)'),
            ('nontemporal_data', 222.0, 'nontemporal as prefix (target name)'),
        ],
    )
    def test_no_false_positives_for_similar_names(self, bc_dataset, sample_dataset, variable_name, value, description):
        """Test that similar but different variable names don't trigger false matches."""
        # Add variable with similar name that shouldn't trigger renaming
        sample_dataset[variable_name] = xr.DataArray([value], dims=['scalar'])
        bc_dataset_new = BackwardsCompatibleDataset(sample_dataset)

        # Should be accessed without any renaming warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = bc_dataset_new[variable_name]

            # Should not trigger deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0, f'Unexpected deprecation warning for {variable_name}: {description}'

            # Should return correct value
            assert result.values == value
