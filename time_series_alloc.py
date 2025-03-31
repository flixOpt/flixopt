
import numpy as np
import xarray as xr
import pandas as pd

from flixopt.core import DataConverter, TimeSeriesAllocator

class Element:
    def __init__(self, name: str, data: xr.DataArray):
        self.name = name
        self.data = data


# Example script to demonstrate both classes
def main():
    print("Demonstrating DataConverter and TimeSeriesAllocator Classes")
    print("=" * 70)

    # Create timesteps for our examples
    start_date = pd.Timestamp('2025-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(10)]
    timesteps = pd.DatetimeIndex(dates, name='time')

    # Create scenarios for our examples
    scenario_names = ['low', 'medium', 'high']
    scenarios = pd.Index(scenario_names, name='scenario')

    print(f"Created {len(timesteps)} timesteps from {timesteps[0]} to {timesteps[-1]}")
    print(f"Created {len(scenarios)} scenarios: {', '.join(scenarios)}")
    print("\n")

    # Part 1: Demonstrate DataConverter with different types
    print("Part 1: DataConverter Examples")
    print("-" * 30)

    # Example 1: Converting a scalar value
    print("Example 1: Converting a scalar value (42)")
    scalar_value = 42
    scalar_da = DataConverter.as_dataarray(scalar_value, timesteps)
    print(f"  Shape: {scalar_da.shape}, Dimensions: {scalar_da.dims}")
    print(f"  First few values: {scalar_da.values[:3]}")
    print(f"  All values are the same: {np.all(scalar_da.values == scalar_value)}")
    print()

    # Example 2: Converting a 1D numpy array
    print("Example 2: Converting a 1D numpy array")
    array_1d = np.arange(len(timesteps)) * 10
    array_da = DataConverter.as_dataarray(array_1d, timesteps)
    print(f"  Shape: {array_da.shape}, Dimensions: {array_da.dims}")
    print(f"  First few values: {array_da.values[:3]}")
    print(f"  Values match input: {np.all(array_da.values == array_1d)}")
    print()

    # Example 3: Converting a pandas Series with time index
    print("Example 3: Converting a pandas Series with time index")
    series = pd.Series(np.random.rand(len(timesteps)) * 100, index=timesteps)
    series_da = DataConverter.as_dataarray(series, timesteps)
    print(f"  Shape: {series_da.shape}, Dimensions: {series_da.dims}")
    print(f"  First few values: {series_da.values[:3]}")
    print(f"  Values match input: {np.all(series_da.values == series.values)}")
    print()

    # Example 4: Converting with scenarios
    print("Example 4: Converting data with scenarios")
    # Create 2D array with shape (scenarios, timesteps)
    array_2d = np.random.rand(len(scenarios), len(timesteps)) * 100
    array_2d_da = DataConverter.as_dataarray(array_2d, timesteps, scenarios)
    print(f"  Shape: {array_2d_da.shape}, Dimensions: {array_2d_da.dims}")
    print(f"  Values for first scenario: {array_2d_da.sel(scenario='low').values[:3]}")
    print(f"  Values match input: {np.all(array_2d_da.values == array_2d)}")
    print()

    # Example 5: Broadcasting a 1D array to scenarios
    print("Example 5: Broadcasting a 1D array to scenarios")
    broadcast_da = DataConverter.as_dataarray(array_1d, timesteps, scenarios)
    print(f"  Shape: {broadcast_da.shape}, Dimensions: {broadcast_da.dims}")
    print(f"  Original shape: {array_1d.shape}")
    print(f"  All scenarios have identical values: {np.all(broadcast_da.sel(scenario='low').values == broadcast_da.sel(scenario='medium').values)}")
    print("\n")

    # Part 2: Demonstrate TimeSeriesAllocator
    print("Part 2: TimeSeriesAllocator Examples")
    print("-" * 35)

    # Create a TimeSeriesAllocator instance
    print("Creating TimeSeriesAllocator with timesteps and scenarios")
    allocator = TimeSeriesAllocator(timesteps, scenarios)
    print(f"  Regular timesteps: {len(allocator.timesteps)}")
    print(f"  Extended timesteps: {len(allocator.timesteps_extra)}")
    print(f"  Added extra timestep: {allocator.timesteps_extra[-1]}")
    print(f"  Hours per timestep: {allocator.hours_per_timestep.values[0]:.1f} hours")
    print()

    # Add data arrays to the allocator
    print("Adding data arrays to the allocator")

    # Example 1: Add a scalar value (broadcast to all timesteps and scenarios)
    constant_val = 42
    constant_da = allocator.add_data_array("constant", constant_val)
    print("  Added 'constant' (scalar value 42)")
    print(f"    Shape: {constant_da.shape}")
    print(f"    Values: All {constant_val}")
    print()

    # Example 2: Add a 1D array (mapped to timesteps, broadcast to scenarios)
    ramp_values = np.linspace(10, 100, len(timesteps))
    ramp_da = allocator.add_data_array("ramp", ramp_values)
    print("  Added 'ramp' (linear values from 10 to 100)")
    print(f"    Shape: {ramp_da.shape}")
    print(f"    First few values: {ramp_da.sel(scenario='low').values[:3]}")
    print()

    # Example 3: Add a 2D array (scenarios × timesteps)
    demand_values = np.zeros((len(scenarios), len(timesteps)))
    # Low scenario: constant demand
    demand_values[0, :] = 50
    # Medium scenario: linearly increasing
    demand_values[1, :] = np.linspace(50, 100, len(timesteps))
    # High scenario: exponentially increasing
    demand_values[2, :] = 50 * np.exp(np.linspace(0, 1, len(timesteps)))

    demand_da = allocator.add_data_array("demand", demand_values)
    print("  Added 'demand' (different profile per scenario)")
    print(f"    Shape: {demand_da.shape}")
    for i, scenario in enumerate(scenarios):
        print(f"    {scenario} scenario first value: {demand_da.sel(scenario=scenario).values[0]:.1f}")
    print()

    # Example 4: Add data with extra timestep
    forecast_values = np.random.normal(size=(len(scenarios), len(timesteps) + 1)) * 10 + 100
    forecast_da = allocator.add_data_array("forecast", forecast_values, has_extra_timestep=True)
    print("  Added 'forecast' (with extra timestep)")
    print(f"    Shape: {forecast_da.shape}")
    print(f"    Last regular timestep: {timesteps[-1]}")
    print(f"    Extra timestep: {allocator.timesteps_extra[-1]}")
    print()

    # Demonstrate selection functionality
    print("Demonstrating selection functionality")
    # Select a subset of timesteps
    subset_timesteps = timesteps[3:7]
    print(f"  Selecting timesteps from {subset_timesteps[0]} to {subset_timesteps[-1]}")
    allocator.set_selection(timesteps=subset_timesteps)

    # Access data with the selection applied
    demand_subset = allocator["demand"]
    print(f"    Original demand shape: {demand_da.shape}")
    print(f"    Selected demand shape: {demand_subset.shape}")
    print()

    # Select a single scenario
    print("  Selecting only the 'high' scenario")
    allocator.set_selection(scenarios=pd.Index(['high'], name='scenario'))
    demand_high = allocator["demand"]
    print(f"    Shape after scenario selection: {demand_high.shape}")
    print()

    # Clear the selection
    print("  Clearing all selections")
    allocator.clear_selection()
    demand_full = allocator["demand"]
    print(f"    Shape after clearing selection: {demand_full.shape}")
    print()

    print("Examples completed successfully!")

if __name__ == "__main__":
    main()
