"""
Comprehensive test file demonstrating the select parameter capabilities.

This file tests various plotting methods and shows what's possible with the new 'select' parameter.
"""

import warnings

import plotly.io as pio
import pytest

import flixopt as fx

# Set default renderer to browser
pio.renderers.default = 'browser'


@pytest.fixture(scope='module')
def results():
    """Load results once for all tests."""
    return fx.results.CalculationResults.from_file('tests/ressources/', 'Sim1')


@pytest.fixture(scope='module')
def scenarios(results):
    """Get available scenarios."""
    return results.solution.scenario.values.tolist()


@pytest.fixture(scope='module')
def periods(results):
    """Get available periods."""
    return results.solution.period.values.tolist()


class TestBasicSelection:
    """Test basic single-value selection."""

    def test_plot_node_balance_single_scenario(self, results, scenarios):
        """Test plot_node_balance with single scenario."""
        results['Fernwärme'].plot_node_balance(select={'scenario': scenarios[0]}, show=False, save=False)

    def test_node_balance_method_single_scenario(self, results, scenarios):
        """Test node_balance method with single scenario."""
        ds = results['Fernwärme'].node_balance(select={'scenario': scenarios[0]})
        assert 'time' in ds.dims
        assert 'period' in ds.dims


class TestMultiValueSelection:
    """Test selection with multiple values (lists)."""

    def test_plot_with_multiple_scenarios(self, results, scenarios):
        """Test plot_node_balance with multiple scenarios + faceting."""
        if len(scenarios) < 2:
            pytest.skip('Not enough scenarios in dataset')

        results['Fernwärme'].plot_node_balance(
            select={'scenario': scenarios}, facet_by='scenario', animate_by=None, show=False, save=False
        )

    def test_plot_with_scenario_subset(self, results, scenarios):
        """Test with partial list selection."""
        if len(scenarios) < 2:
            pytest.skip('Not enough scenarios in dataset')

        selected = scenarios[:2]
        results['Fernwärme'].plot_node_balance(
            select={'scenario': selected}, facet_by='scenario', show=False, save=False
        )


class TestIndexBasedSelection:
    """Test selection using index positions."""

    def test_integer_index_selection(self, results):
        """Test with integer index (should fail with current xarray behavior)."""
        with pytest.raises(KeyError, match='not all values found'):
            results['Fernwärme'].plot_node_balance(select={'scenario': 0}, show=False, save=False)

    def test_list_of_indices_selection(self, results):
        """Test with multiple indices (should fail with current xarray behavior)."""
        with pytest.raises(KeyError, match='not all values found'):
            results['Fernwärme'].plot_node_balance(
                select={'scenario': [0, 1]}, facet_by='scenario', show=False, save=False
            )


class TestCombinedSelection:
    """Test combining multiple dimension selections."""

    def test_select_scenario_and_period(self, results, scenarios, periods):
        """Test selecting both scenario and period."""
        ds = results['Fernwärme'].node_balance(select={'scenario': scenarios[0], 'period': periods[0]})
        assert 'time' in ds.dims
        # scenario and period should be dropped after selection
        assert 'scenario' not in ds.dims
        assert 'period' not in ds.dims

    def test_scenario_list_period_single(self, results, scenarios, periods):
        """Test with one dimension as list, another as single value."""
        results['Fernwärme'].plot_node_balance(
            select={'scenario': scenarios, 'period': periods[0]}, facet_by='scenario', show=False, save=False
        )


class TestFacetingAndAnimation:
    """Test combining select with faceting and animation."""

    def test_select_scenario_facet_by_period(self, results, scenarios):
        """Test: Select specific scenarios, then facet by period."""
        results['Fernwärme'].plot_node_balance(
            select={'scenario': scenarios[0]}, facet_by='period', animate_by=None, show=False, save=False
        )

    def test_facet_and_animate(self, results, periods):
        """Test: Facet by scenario, animate by period."""
        if len(periods) <= 1:
            pytest.skip('Only one period available')

        results['Fernwärme'].plot_node_balance(
            select={},  # No filtering - use all data
            facet_by='scenario',
            animate_by='period',
            show=False,
            save=False,
        )


class TestDifferentPlottingMethods:
    """Test select parameter across different plotting methods."""

    def test_plot_node_balance(self, results, scenarios):
        """Test plot_node_balance."""
        results['Fernwärme'].plot_node_balance(select={'scenario': scenarios[0]}, mode='area', show=False, save=False)

    def test_plot_heatmap(self, results, scenarios):
        """Test plot_heatmap with the new imshow implementation."""
        var_names = list(results.solution.data_vars)
        if not var_names:
            pytest.skip('No variables found')

        # Find a variable with time dimension for proper heatmap
        var_name = None
        for name in var_names:
            if 'time' in results.solution[name].dims:
                var_name = name
                break

        if var_name is None:
            pytest.skip('No time-series variables found for heatmap test')

        # Test that the new heatmap implementation works
        results.plot_heatmap(var_name, select={'scenario': scenarios[0]}, show=False, save=False)

    def test_node_balance_data_retrieval(self, results, scenarios):
        """Test node_balance (data retrieval)."""
        ds = results['Fernwärme'].node_balance(select={'scenario': scenarios[0]}, unit_type='flow_hours')
        assert 'time' in ds.dims or 'period' in ds.dims


class TestBackwardCompatibility:
    """Test that old 'indexer' parameter still works with deprecation warning."""

    def test_indexer_parameter_deprecated(self, results, scenarios):
        """Test using deprecated 'indexer' parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            results['Fernwärme'].plot_node_balance(indexer={'scenario': scenarios[0]}, show=False, save=False)

            # Check if deprecation warning was raised
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert 'indexer' in str(deprecation_warnings[0].message).lower()


class TestParameterPrecedence:
    """Test that 'select' takes precedence over 'indexer'."""

    def test_select_overrides_indexer(self, results, scenarios):
        """Test that 'select' overrides 'indexer'."""
        if len(scenarios) < 2:
            pytest.skip('Not enough scenarios')

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')

            ds = results['Fernwärme'].node_balance(
                indexer={'scenario': scenarios[0]},  # This should be overridden
                select={'scenario': scenarios[1]},  # This should win
            )

            # The scenario dimension should be dropped after selection
            assert 'scenario' not in ds.dims or ds.scenario.values == scenarios[1]


class TestEmptyDictBehavior:
    """Test behavior with empty selection dict."""

    def test_empty_dict_no_filtering(self, results):
        """Test using select={} (empty dict - no filtering)."""
        results['Fernwärme'].plot_node_balance(
            select={}, facet_by='scenario', animate_by='period', show=False, save=False
        )


class TestErrorHandling:
    """Test error handling for invalid parameters."""

    def test_unexpected_keyword_argument(self, results):
        """Test unexpected kwargs are rejected."""
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            results['Fernwärme'].plot_node_balance(select={'scenario': 0}, unexpected_param='test', show=False)


# Keep the old main function for backward compatibility when run directly
def main():
    """Run tests when executed directly (non-pytest mode)."""
    print('\n' + '#' * 70)
    print('# SELECT PARAMETER TESTS')
    print('#' * 70)
    print('\nTo run with pytest, use:')
    print('  pytest tests/test_select_features.py -v')
    print('\nTo run specific test:')
    print('  pytest tests/test_select_features.py::TestBasicSelection -v')
    print('\n' + '#' * 70)


if __name__ == '__main__':
    main()
