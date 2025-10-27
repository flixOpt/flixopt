"""Example demonstrating the PyPSA-style statistics accessor pattern in flixopt.

This script shows how to use the new statistics accessor for calculating
and visualizing optimization results with a clean, chainable API.
"""

import flixopt as fx
from flixopt.results import CalculationResults

# This example assumes you have run an optimization and saved the results
# For a complete example, first run one of the examples in examples/ directory
# and ensure results are saved.


def demonstrate_statistics_accessor():
    """Demonstrate the PyPSA-style statistics accessor pattern."""

    print('=' * 80)
    print('PyPSA-Style Statistics Accessor Pattern Demo')
    print('=' * 80)
    print()

    # Load existing results (adjust path as needed)
    # You need to have run an optimization first and saved the results
    try:
        results = CalculationResults.from_file('examples/01_Simple/results', 'simple_example')
        print(f'✓ Loaded results: {results.name}')
        print()
    except Exception as e:
        print(f'❌ Could not load results: {e}')
        print()
        print('Please run an optimization example first:')
        print('  python examples/01_Simple/simple_example.py')
        print()
        return

    # Check that statistics accessor is available
    print('Statistics accessor available:', hasattr(results, 'statistics'))
    print(f'Accessor object: {results.statistics}')
    print()

    # Example 1: Get raw statistics data
    print('-' * 80)
    print('Example 1: Get raw statistics data')
    print('-' * 80)
    try:
        # Call the method to get StatisticPlotter
        plotter = results.statistics.flow_summary()
        print(f'Plotter object: {plotter}')

        # Use .data property to get raw data (recommended)
        data = plotter.data
        print(f'Data type: {type(data)}')
        print(f'Data variables: {list(data.data_vars)}')
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 2: Create a bar chart with enhanced API
    print('-' * 80)
    print('Example 2: Create interactive bar chart')
    print('-' * 80)
    try:
        fig = results.statistics.flow_summary().plot.bar(title='Flow Summary', ylabel='Flow Rate [MW]')
        print(f'Figure created: {type(fig)}')
        print('Opening in browser...')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 3: Energy balance visualization
    print('-' * 80)
    print('Example 3: Energy balance analysis')
    print('-' * 80)
    try:
        # Get energy balance and plot
        fig = results.statistics.energy_balance().plot.bar()
        print('Energy balance chart created')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 4: Storage states time series
    print('-' * 80)
    print('Example 4: Storage charge states over time')
    print('-' * 80)
    try:
        # Check if there are storages
        if results.storages:
            fig = results.statistics.storage_states(aggregate_scenarios=True).plot.line()
            print('Storage states chart created')
            fig.show()
        else:
            print('No storage components found in this example')
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 5: Component effects
    print('-' * 80)
    print('Example 5: Effects per component')
    print('-' * 80)
    try:
        fig = results.statistics.component_effects(effect_mode='total').plot.bar()
        print('Component effects chart created')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 6: Component sizes
    print('-' * 80)
    print('Example 6: Component sizes/capacities')
    print('-' * 80)
    try:
        fig = results.statistics.component_sizes().plot.bar()
        print('Component sizes chart created')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 7: Using different plot types
    print('-' * 80)
    print('Example 7: Time series with line plot')
    print('-' * 80)
    try:
        # Get flow summary without time aggregation
        fig = results.statistics.flow_summary(aggregate_time=False).plot.line()
        print('Time series line chart created')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 8: Chained operations
    print('-' * 80)
    print('Example 8: Chained operations')
    print('-' * 80)
    try:
        # Get energy balance for specific components (if we know component names)
        component_names = list(results.components.keys())[:3]  # First 3 components
        print(f'Analyzing components: {component_names}')

        fig = results.statistics.energy_balance(components=component_names, aggregate_time=True).plot.bar()
        print('Filtered energy balance chart created')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 9: NEW - Faceting (subplots)
    print('-' * 80)
    print('Example 9: Faceting - Multiple subplots by dimension')
    print('-' * 80)
    try:
        # Create faceted bar chart if scenarios exist
        data = results.statistics.flow_summary(aggregate_scenarios=False).data
        if 'scenario' in data.dims:
            fig = results.statistics.flow_summary(aggregate_scenarios=False).plot.bar(
                facet_by='scenario',
                facet_cols=2,  # 2 columns of subplots
                title='Flow Summary by Scenario',
                ylabel='Flow Rate [MW]',
            )
            print('Faceted bar chart created (subplots by scenario)')
            fig.show()
        else:
            print('No scenario dimension found - skipping faceting example')
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 10: NEW - Animation
    print('-' * 80)
    print('Example 10: Animation - Animated over dimension')
    print('-' * 80)
    try:
        # Create animated chart over time
        data = results.statistics.flow_summary(aggregate_time=False).data
        if 'time' in data.dims and len(data.time) > 1:
            fig = results.statistics.flow_summary(aggregate_time=False).plot.bar(
                animate_by='time', title='Flow Summary Animation Over Time', ylabel='Flow Rate [MW]'
            )
            print('Animated bar chart created (animated over time)')
            print('Use the play button in the browser to see animation')
            fig.show()
        else:
            print('Not enough time steps for animation - skipping')
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    # Example 11: NEW - Combined faceting and custom colors
    print('-' * 80)
    print('Example 11: Advanced - Faceting with custom colors')
    print('-' * 80)
    try:
        fig = results.statistics.energy_balance(aggregate_time=True).plot.bar(
            mode='grouped',  # Grouped instead of stacked
            ylabel='Energy [MWh]',
            title='Energy Balance Comparison',
        )
        print('Grouped bar chart with custom settings created')
        fig.show()
        print()
    except Exception as e:
        print(f'Error: {e}')
        print()

    print('=' * 80)
    print('Demo completed!')
    print('=' * 80)
    print()
    print('Key features demonstrated:')
    print('  ✓ Clean, chainable API: results.statistics.method().plot.type()')
    print('  ✓ Lazy evaluation: data computed only when needed')
    print('  ✓ Multiple plot types: bar, line, scatter, area')
    print('  ✓ Filtering and aggregation options')
    print('  ✓ Access to raw data: plotter.data')
    print('  ✓ Interactive Plotly visualizations')
    print('  ✓ NEW: Multi-dimensional faceting (subplots)')
    print('  ✓ NEW: Animation over dimensions')
    print('  ✓ NEW: Integrated color processing')
    print('  ✓ NEW: Consistent API across all plot types')
    print()


if __name__ == '__main__':
    demonstrate_statistics_accessor()
