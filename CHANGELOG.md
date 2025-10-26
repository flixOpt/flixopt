# Changelog

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Formatting is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) & [Gitmoji](https://gitmoji.dev).
For more details regarding the individual PRs and contributors, please refer to our [GitHub releases](https://github.com/flixOpt/flixopt/releases).

!!! tip

    If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

---

<!-- This text won't be rendered
Note: The CI will automatically append a "What's Changed" section to the changelog for github releases.
This contains all commits, PRs, and contributors.
Therefore, the Changelog should focus on the user-facing changes.

Please remove all irrelevant sections before releasing.
Please keep the format of the changelog consistent with the other releases, so the extraction for mkdocs works.
---

## [Template] - ????-??-??

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).


### ✨ Added

### 💥 Breaking Changes

### ♻️ Changed

### 🗑️ Deprecated

### 🔥 Removed

### 🐛 Fixed

### 🔒 Security

### 📦 Dependencies

### 📝 Docs

### 👷 Development

### 🚧 Known Issues

---

## [Unreleased] - ????-??-??

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### ✨ Added
- Support for plotting kwargs in `results.py`, passed to plotly express and matplotlib.
- **Color management system**: New `color_processing.py` module with `process_colors()` function for unified color handling across plotting backends
  - Supports flexible color inputs: colorscale names (e.g., 'turbo', 'plasma'), color lists, and label-to-color dictionaries
  - Automatic fallback handling when requested colorscales are unavailable
  - Seamless integration with both Plotly and Matplotlib colorscales
  - Automatic rgba→hex color conversion for Matplotlib compatibility
- **Component color grouping**: Added `setup_colors()` method to `CalculationResults` and `SegmentedCalculationResults` to create color mappings with similar colors for all variables of a component
  - Allows grouping components by custom colorscales: `{'CHP': 'red', 'Greys': ['Gastarif', 'Einspeisung'], 'Storage': 'blue'}`
  - Colors are automatically assigned using default colorscale if not specified
  - For segmented calculations, colors are propagated to all segments for consistent visualization
  - Explicit `colors` arguments in plot methods override configured colors (when provided)
- **Plotting configuration**: New `CONFIG.Plotting` section with extensive customization options:
  - `default_show`: Control default visibility of plots
  - `default_engine`: Choose between 'plotly' or 'matplotlib'
  - `default_dpi`: Configure resolution for saved plots (with matplotlib)
  - `default_facet_cols`: Set default columns for faceted plots
  - `default_sequential_colorscale`: Default for heatmaps and continuous data (default: 'turbo')
  - `default_qualitative_colorscale`: Default for categorical plots (default: 'plotly')

### 💥 Breaking Changes

### ♻️ Changed
- **Template integration**: Plotly templates now fully control plot styling without hardcoded overrides
- **Dataset first plotting**: Underlying plotting methods in `plotting.py` now use `xr.Dataset` as the main datatype. DataFrames are automatically converted via `_ensure_dataset()`. Both DataFrames and Datasets can be passed to plotting functions without code changes.
- **Color terminology**: Standardized terminology from "colormap" to "colorscale" throughout the codebase for consistency with Plotly conventions
- **Default colorscales**: Changed default sequential colorscale from 'viridis' to 'turbo' for better perceptual uniformity; qualitative colorscale now defaults to 'plotly'
- **Aggregation plotting**: `Aggregation.plot()` now respects `CONFIG.Plotting.default_qualitative_colorscale` and uses `process_colors()` for consistent color handling
- **netcdf engine**: FOllowing the xarray revert in `xarray==2025.09.02` and after running some benchmarks, we go back to using the netcdf4 engine

### 🗑️ Deprecated

### 🔥 Removed
- Removed `plotting.pie_with_plotly()` method as it was not used
- Removed `ColorProcessor` class - replaced by simpler `process_colors()` function
- Removed `resolve_colors()` helper function - color resolution now handled directly by `process_colors()`

### 🐛 Fixed
- Improved error messages for `engine='matplotlib'` with multidimensional data
- Better dimension validation in `results.plot_heatmap()`

### 🔒 Security

### 📦 Dependencies

### 📝 Docs
- Moved `linked_periods` into correct section of the docstring (was in deprecated params)
- Updated terminology in docstrings from "colormap" to "colorscale" for consistency
- Enhanced examples to demonstrate `setup_colors()` usage:
  - `simple_example.py`: Shows automatic color assignment and optional custom configuration
  - `scenario_example.py`: Demonstrates component grouping with custom colorscales

### 👷 Development
- Fixed concurrency issue in CI
- **Code architecture**: Extracted color processing logic into dedicated `color_processing.py` module for better separation of concerns
- Refactored from class-based (`ColorProcessor`) to function-based color handling for simpler API and reduced complexity

### 🚧 Known Issues

---

Until here -->

## [3.1.1] - 2025-10-20
**Summary**: Fixed a bug when acessing the `effects_per_component` dataset in results without periodic effects.

If upgrading from v2.x, see the [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0) and [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/).

### 🐛 Fixed
- Fixed ValueError in effects_per_component when all periodic effects are scalars/NaN by explicitly creating mode-specific templates (via _create_template_for_mode) with correct dimensions

### 👷 Development
- Converted all remaining numpy style docstrings to google style

---

## [3.1.0] - 2025-10-19

**Summary**: This release adds faceting and animation support for multidimensional plots and redesigns the documentation website. Plotting results across scenarios or periods is now significantly simpler (Plotly only).

If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### ✨ Added
- **Faceting and animation for multidimensional plots**: All plotting methods now support `facet_by` and `animate_by` parameters to create subplot grids and animations from multidimensional data (scenarios, periods, etc.). *Plotly only.*
- **Flexible data selection with `select` parameter**: Select data using single values, lists, slices, or index arrays for precise control over what gets plotted
- **Heatmap fill control**: New `fill` parameter in heatmap methods controls how missing values are filled after reshaping (`'ffill'` or `'bfill'`)
- **Smart line styling for mixed variables**: Area plots now automatically style variables containing both positive and negative values with dashed lines, while stacking purely positive or negative variables

### ♻️ Changed
- **Breaking: Selection behavior**: Plotting methods no longer automatically select the first value for non-time dimensions. Use the `select` parameter for explicit selection of scenarios, periods, or other dimensions
- **Better error messages**: Enhanced error messages when using Matplotlib with multidimensional data, with clearer guidance on dimension requirements and suggestions to use Plotly
- **Improved examples**: Enhanced `scenario_example.py` with better demonstration of new features
- **Robust validation**: Improved dimension validation in `plot_heatmap()` with clearer error messages

### 🗑️ Deprecated
- **`indexer` parameter**: Use the new `select` parameter instead. The `indexer` parameter will be removed in v4.0.0
- **`heatmap_timeframes` and `heatmap_timesteps_per_frame` parameters**: Use the new `reshape_time=(timeframes, timesteps_per_frame)` parameter instead in heatmap plotting methods
- **`color_map` parameter**: Use the new `colors` parameter instead in heatmap plotting methods

### 🐛 Fixed
- Fixed cryptic errors when working with empty buses by adding proper validation
- Added early validation for non-existent periods when using linked periods with tuples

### 📝 Documentation
- **Redesigned documentation website** with custom css

### 👷 Development
- Renamed internal `_apply_indexer_to_data()` to `_apply_selection_to_data()` for consistency with new API naming

---

## [3.0.3] - 2025-10-16
**Summary**: Hotfixing new plotting parameter `style`. Continue to use `mode`.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### 🐛 Fixed
- Reverted breaking change from v3.0.0: continue to use `mode parameter in plotting instead of new `style`
- Renamed new `mode` parameter in plotting methods to `unit_type`

### 📝 Docs
- Updated Migration Guide and added missing entries.
- Improved Changelog of v3.0.0

---

## [3.0.2] - 2025-10-15
**Summary**: This is a follow-up release to **[v3.0.0](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0)**, improving the documentation.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### 📝 Docs
- Update the Readme
- Add a project roadmap to the docs
- Change Development status to "Production/Stable"
- Regroup parts in docs

---

## [3.0.1] - 2025-10-14
**Summary**: This is a follow-up release to **[v3.0.0](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0)**, adding a Migration Guide and bugfixing the docs.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### 📝 Docs
- Fixed deployed docs
- Added Migration Guide for flixopt 3

### 👷 Development
- Added missing type hints

---

## [3.0.0] - 2025-10-13
**Summary**: This release introduces new model dimensions (periods and scenarios) for multi-period investments and stochastic modeling, along with a redesigned effect sharing system and enhanced I/O capabilities.

**Note**: If upgrading from v2.x, see the [Migration Guide](https://flixopt.github.io/flixopt/latest/user-guide/migration-guide-v3/) and [v3.0.0 release notes](https://github.com/flixOpt/flixOpt/releases/tag/v3.0.0).

### ✨ Added

**New model dimensions:**

- **Period dimension**: Enables multi-period investment modeling with distinct decisions in each period for transformation pathway optimization
- **Scenario dimension**: Supports stochastic modeling with weighted scenarios for robust decision-making under uncertainty (demand, prices, weather)
    - Control variable independence across scenarios via `scenario_independent_sizes` and `scenario_independent_flow_rates` parameters
    - By default, investment sizes are shared across scenarios while flow rates vary per scenario

**Redesigned effect sharing system:**

Effects now use intuitive `share_from_*` syntax that clearly shows contribution sources:

```python
costs = fx.Effect('costs', '€', 'Total costs',
    share_from_temporal={'CO2': 0.2},      # From temporal effects
    share_from_periodic={'land': 100})     # From periodic effects
```

This replaces `specific_share_to_other_effects_*` parameters and inverts the direction for clearer relationships.

**Enhanced I/O and data handling:**

- NetCDF/JSON serialization for all Interface objects and FlowSystem with round-trip support
- FlowSystem manipulation: `sel()`, `isel()`, `resample()`, `copy()`, `__eq__()` methods
- Direct access to FlowSystem from results without manual restoring (lazily loaded)
- New `FlowResults` class and precomputed DataArrays for sizes/flow_rates/flow_hours
- `effects_per_component` dataset for component impact evaluation, including all indirect effects through effect shares

**Other additions:**

- Balanced storage - charging and discharging sizes can be forced equal via `balanced` parameter
- New Storage parameters: `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` for final state control
- Improved filter methods in results
- Example for 2-stage investment decisions leveraging FlowSystem resampling

### 💥 Breaking Changes

**API and Behavior Changes:**

- **Effect system redesigned** (no deprecation):
  - **Terminology changes**: Effect domains renamed for clarity: `operation` → `temporal`, `invest`/`investment` → `periodic`
  - **Sharing system**: The old `specific_share_to_other_effects_*` parameters were completely replaced with the new `share_from_temporal` and `share_from_periodic` syntax (see 🔥 Removed section)
- **FlowSystem independence**: FlowSystems cannot be shared across multiple Calculations anymore. A copy of the FlowSystem is created instead, making every Calculation independent. Each Subcalculation in `SegmentedCalculation` now has its own distinct `FlowSystem` object
- **Bus and Effect object assignment**: Direct assignment of Bus/Effect objects is no longer supported. Use labels (strings) instead:
  - `Flow.bus` must receive a string label, not a Bus object
  - Effect shares must use effect labels (strings) in dictionaries, not Effect objects
- **Logging defaults** (from v2.2.0): Console and file logging are now disabled by default. Enable explicitly with `CONFIG.Logging.console = True` and `CONFIG.apply()`

**Class and Method Renaming:**

- Renamed class `SystemModel` to `FlowSystemModel`
- Renamed class `Model` to `Submodel`
- Renamed `mode` parameter in plotting methods to `style`
- `Calculation.do_modeling()` now returns the `Calculation` object instead of its `linopy.Model`. Callers that previously accessed the linopy model directly should now use `calculation.do_modeling().model` instead of `calculation.do_modeling()`

**Variable Renaming in Results:**

- Investment binary variable: `is_invested` → `invested` in `InvestmentModel`
- Switch tracking variables in `OnOffModel`:
  - `switch_on` → `switch|on`
  - `switch_off` → `switch|off`
  - `switch_on_nr` → `switch|count`
- Effect submodel variables (following terminology changes):
  - `Effect(invest)|total` → `Effect(periodic)`
  - `Effect(operation)|total` → `Effect(temporal)`
  - `Effect(operation)|total_per_timestep` → `Effect(temporal)|per_timestep`
  - `Effect|total` → `Effect`

**Data Structure Changes:**

- `relative_minimum_charge_state` and `relative_maximum_charge_state` don't have an extra timestep anymore. Use the new `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` parameters for final state control

### ♻️ Changed

- Type system overhaul - added clear separation between temporal and non-temporal data throughout codebase for better clarity
- Enhanced FlowSystem interface with improved `__repr__()` and `__str__()` methods
- Improved Model Structure - Views and organisation is now divided into:
    - Model: The main Model (linopy.Model) that is used to create and store the variables and constraints for the FlowSystem.
    - Submodel: The base class for all submodels. Each is a subset of the Model, for simpler access and clearer code.
- Made docstrings in `config.py` more compact and easier to read
- Improved format handling in configuration module
- Enhanced console output to support both `stdout` and `stderr` stream selection
- Added `show_logger_name` parameter to `CONFIG.Logging` for displaying logger names in messages

### 🗑️ Deprecated

- The `agg_group` and `agg_weight` parameters of `TimeSeriesData` are deprecated and will be removed in a future version. Use `aggregation_group` and `aggregation_weight` instead.
- The `active_timesteps` parameter of `Calculation` is deprecated and will be removed in a future version. Use the new `sel(time=...)` method on the FlowSystem instead.
- **InvestParameters** parameters renamed for improved clarity around investment and retirement effects:
    - `fix_effects` → `effects_of_investment`
    - `specific_effects` → `effects_of_investment_per_size`
    - `divest_effects` → `effects_of_retirement`
    - `piecewise_effects` → `piecewise_effects_of_investment`
- **Effect** parameters renamed:
    - `minimum_investment` → `minimum_periodic`
    - `maximum_investment` → `maximum_periodic`
    - `minimum_operation` → `minimum_temporal`
    - `maximum_operation` → `maximum_temporal`
    - `minimum_operation_per_hour` → `minimum_per_hour`
    - `maximum_operation_per_hour` → `maximum_per_hour`
- **Component** parameters renamed:
    - `Source.source` → `Source.outputs`
    - `Sink.sink` → `Sink.inputs`
    - `SourceAndSink.source` → `SourceAndSink.outputs`
    - `SourceAndSink.sink` → `SourceAndSink.inputs`
    - `SourceAndSink.prevent_simultaneous_sink_and_source` → `SourceAndSink.prevent_simultaneous_flow_rates`

### 🔥 Removed

- **Effect share parameters**: The old `specific_share_to_other_effects_*` parameters were replaced WITHOUT DEPRECATION
    - `specific_share_to_other_effects_operation` → `share_from_temporal` (with inverted direction)
    - `specific_share_to_other_effects_invest` → `share_from_periodic` (with inverted direction)

### 🐛 Fixed

- Enhanced NetCDF I/O with proper attribute preservation for DataArrays
- Improved error handling and validation in serialization processes
- Better type consistency across all framework components
- Added extra validation in `config.py` to improve error handling

### 📝 Docs

- Reorganized mathematical notation docs: moved to lowercase `mathematical-notation/` with subdirectories (`elements/`, `features/`, `modeling-patterns/`)
- Added comprehensive documentation pages: `dimensions.md` (time/period/scenario), `effects-penalty-objective.md`, modeling patterns
- Enhanced all element pages with implementation details, cross-references, and "See Also" sections
- Rewrote README and landing page with clearer vision, roadmap, and universal applicability emphasis
- Removed deprecated `docs/SUMMARY.md`, updated `mkdocs.yml` for new structure
- Tightened docstrings in core modules with better cross-referencing
- Added recipes section to docs

### 🚧 Known Issues

- IO for single Interfaces/Elements to Datasets might not work properly if the Interface/Element is not part of a fully transformed and connected FlowSystem. This arises from Numeric Data not being stored as xr.DataArray by the user. To avoid this, always use the `to_dataset()` on Elements inside a FlowSystem that's connected and transformed.

### 👷 Development

- **Centralized deprecation pattern**: Added `_handle_deprecated_kwarg()` helper method to `Interface` base class that provides reusable deprecation handling with consistent warnings, conflict detection, and optional value transformation. Applied across 5 classes (InvestParameters, Source, Sink, SourceAndSink, Effect) reducing deprecation boilerplate by 72%.
- FlowSystem data management simplified - removed `time_series_collection` pattern in favor of direct timestep properties
- Change modeling hierarchy to allow for more flexibility in future development. This leads to minimal changes in the access and creation of Submodels and their variables.
- Added new module `.modeling` that contains modeling primitives and utilities
- Clearer separation between the main Model and "Submodels"
- Improved access to the Submodels and their variables, constraints and submodels
- Added `__repr__()` for Submodels to easily inspect its content
- Enhanced data handling methods
    - `fit_to_model_coords()` method for data alignment
    - `fit_effects_to_model_coords()` method for effect data processing
    - `connect_and_transform()` method replacing several operations
- **Testing improvements**: Eliminated warnings during test execution
    - Updated deprecated code patterns in tests and examples (e.g., `sink`/`source` → `inputs`/`outputs`, `'H'` → `'h'` frequency)
    - Refactored plotting logic to handle test environments explicitly with non-interactive backends
    - Added comprehensive warning filters in `__init__.py` and `pyproject.toml` to suppress third-party library warnings
    - Improved test fixtures with proper figure cleanup to prevent memory leaks
    - Enhanced backend detection and handling in `plotting.py` for both Matplotlib and Plotly
    - Always run dependent tests in order

---

## [2.2.0] - 2025-10-11
**Summary:** This release is a Configuration and Logging management release.

### ✨ Added
- Added `CONFIG.reset()` method to restore configuration to default values
- Added configurable log file rotation settings: `CONFIG.Logging.max_file_size` and `CONFIG.Logging.backup_count`
- Added configurable log format settings: `CONFIG.Logging.date_format` and `CONFIG.Logging.format`
- Added configurable console settings: `CONFIG.Logging.console_width` and `CONFIG.Logging.show_path`
- Added `CONFIG.Logging.Colors` nested class for customizable log level colors using ANSI escape codes (works with both standard and Rich handlers)
- All examples now enable console logging to demonstrate proper logging usage
- Console logging now outputs to `sys.stdout` instead of `sys.stderr` for better compatibility with output redirection

### 💥 Breaking Changes
- Console logging is now disabled by default (`CONFIG.Logging.console = False`). Enable it explicitly in your scripts with `CONFIG.Logging.console = True` and `CONFIG.apply()`
- File logging is now disabled by default (`CONFIG.Logging.file = None`). Set a file path to enable file logging

### ♻️ Changed
- Logging and Configuration management changed
- Improved default logging colors: DEBUG is now gray (`\033[90m`) for de-emphasized messages, INFO uses terminal default color (`\033[0m`) for clean output

### 🗑️ Deprecated
- `change_logging_level()` function is now deprecated in favor of `CONFIG.Logging.level` and `CONFIG.apply()`. Will be removed in version 3.0.0.

### 🔥 Removed
- Removed unused `config.merge_configs` function from configuration module

### 👷 Development
- Greatly expanded test coverage for `config.py` module
- Added `@pytest.mark.xdist_group` to `TestConfigModule` tests to prevent global config interference

---

## [2.1.11] - 2025-10-05
**Summary:** Important bugfix in `Storage` leading to wrong results due to incorrect discharge losses.

### ♻️ Changed
- Using `h5netcdf` instead of `netCDF4` for dataset I/O operations. This follows the update in `xarray==2025.09.01`

### 🐛 Fixed
- Fix `charge_state` Constraint in `Storage` leading to incorrect losses in discharge and therefore incorrect charge states and discharge values.

### 📦 Dependencies
- Updated `renovate.config` to treat CalVer packages (xarray and dask) with more care
- Updated packaging configuration

---

## [2.1.10] - 2025-09-29
**Summary:** This release is a Documentation and Development release.

### 📝 Docs
- Improved CHANGELOG.md formatting by adding better categories and formating by Gitmoji.
- Added a script to extract the release notes from the CHANGELOG.md file for better organized documentation.

### 👷 Development
- Improved `renovate.config`
- Sped up CI by not running examples in every run and using `pytest-xdist`

---

## [2.1.9] - 2025-09-23

**Summary:** Small bugfix release addressing network visualization error handling.

### 🐛 Fixed
- Fix error handling in network visualization if `networkx` is not installed

---

## [2.1.8] - 2025-09-22

**Summary:** Code quality improvements, enhanced documentation, and bug fixes for heat pump components and visualization features.

### ✨ Added
- Extra Check for HeatPumpWithSource.COP to be strictly > 1 to avoid division by zero
- Apply deterministic color assignment by using sorted() in `plotting.py`
- Add missing args in docstrings in `plotting.py`, `solvers.py`, and `core.py`.

### ♻️ Changed
- Greatly improved docstrings and documentation of all public classes
- Make path handling to be gentle about missing .html suffix in `plotting.py`
- Default for `relative_losses` in `Transmission` is now 0 instead of None
- Setter of COP in `HeatPumpWithSource` now completely overwrites the conversion factors, which is safer.
- Fix some docstrings in plotting.py
- Change assertions to raise Exceptions in `plotting.py`

### 🐛 Fixed

**Core Components:**
- Fix COP getter and setter of `HeatPumpWithSource` returning and setting wrong conversion factors
- Fix custom compression levels in `io.save_dataset_to_netcdf`
- Fix `total_max` did not work when total min was not used

**Visualization:**
- Fix color scheme selection in network_app; color pickers now update when a scheme is selected

### 📝 Docs
- Fix broken links in docs
- Fix some docstrings in plotting.py

### 👷 Development
- Pin dev dependencies to specific versions
- Improve CI workflows to run faster and smarter

---

## [2.1.7] - 2025-09-13

**Summary:** Maintenance release to improve Code Quality, CI and update the dependencies. There are no changes or new features.

### ✨ Added
- Added `__version__` to flixopt

### 👷 Development
- ruff format the whole Codebase
- Added renovate config
- Added pre-commit
- lint and format in CI
- improved CI
- Updated Dependencies
- Updated Issue Templates

---

## [2.1.6] - 2025-09-02

**Summary:** Enhanced Sink/Source components with multi-flow support and new interactive network visualization.

### ✨ Added
- **Network Visualization**: Added `FlowSystem.start_network_app()` and `FlowSystem.stop_network_app()` to easily visualize the network structure of a flow system in an interactive Dash web app
  - *Note: This is still experimental and might change in the future*

### ♻️ Changed
- **Multi-Flow Support**: `Sink`, `Source`, and `SourceAndSink` now accept multiple `flows` as `inputs` and `outputs` instead of just one. This enables modeling more use cases with these classes
- **Flow Control**: Both `Sink` and `Source` now have a `prevent_simultaneous_flow_rates` argument to prevent simultaneous flow rates of more than one of their flows

### 🗑️ Deprecated
- For the classes `Sink`, `Source` and `SourceAndSink`: `.sink`, `.source` and `.prevent_simultaneous_sink_and_source` are deprecated in favor of the new arguments `inputs`, `outputs` and `prevent_simultaneous_flow_rates`

### 🐛 Fixed
- Fixed testing issue with new `linopy` version 0.5.6

### 👷 Development
- Added dependency "nbformat>=4.2.0" to dev dependencies to resolve issue with plotly CI

---

## [2.1.5] - 2025-07-08

### 🐛 Fixed
- Fixed Docs deployment

---

## [2.1.4] - 2025-07-08

### 🐛 Fixed
- Fixing release notes of 2.1.3, as well as documentation build.

---

## [2.1.3] - 2025-07-08

### 🐛 Fixed
- Using `Effect.maximum_operation_per_hour` raised an error, needing an extra timestep. This has been fixed thanks to @PRse4.

---

## [2.1.2] - 2025-06-14

### 🐛 Fixed
- Storage losses per hour were not calculated correctly, as mentioned by @brokenwings01. This might have led to issues when modeling large losses and long timesteps.
  - Old implementation:     $c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i)) \cdot \Delta \text{t}_{i}$
  - Correct implementation: $c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i)) ^{\Delta \text{t}_{i}}$

### 🚧 Known Issues
- Just to mention: Plotly >= 6 may raise errors if "nbformat" is not installed. We pinned plotly to <6, but this may be fixed in the future.

---

## [2.1.1] - 2025-05-08

### ♻️ Changed
- Improved docstring and tests

### 🐛 Fixed
- Fixed bug in the `_ElementResults.constraints` not returning the constraints but rather the variables

---
## [2.1.0] - 2025-04-11

### ✨ Added
- Python 3.13 support added
- Logger warning if relative_minimum is used without on_off_parameters in Flow
- Greatly improved internal testing infrastructure by leveraging linopy's testing framework

### 💥 Breaking Changes
- Restructured the modeling of the On/Off state of Flows or Components
  - Variable renaming: `...|consecutive_on_hours` → `...|ConsecutiveOn|hours`
  - Variable renaming: `...|consecutive_off_hours` → `...|ConsecutiveOff|hours`
  - Constraint renaming: `...|consecutive_on_hours_con1` → `...|ConsecutiveOn|con1`
  - Similar pattern for all consecutive on/off constraints

### 🐛 Fixed
- Fixed the lower bound of `flow_rate` when using optional investments without OnOffParameters
- Fixed bug that prevented divest effects from working
- Added lower bounds of 0 to two unbounded vars (numerical improvement)

---

## [2.0.1] - 2025-04-10

### ✨ Added
- Logger warning if relative_minimum is used without on_off_parameters in Flow

### 🐛 Fixed
- Replace "|" with "__" in filenames when saving figures (Windows compatibility)
- Fixed bug that prevented the load factor from working without InvestmentParameters

## [2.0.0] - 2025-03-29

**Summary:** 💥 **MAJOR RELEASE** - Complete framework migration from Pyomo to Linopy with redesigned architecture.

### ✨ Added

**Model Capabilities:**
- Full model serialization support - save and restore unsolved Models
- Enhanced model documentation with YAML export containing human-readable mathematical formulations
- Extend flixopt models with native linopy language support
- Full Model Export/Import capabilities via linopy.Model

**Results & Data:**
- Unified solution exploration through `Calculation.results` attribute
- Compression support for result files
- `to_netcdf/from_netcdf` methods for FlowSystem and core components
- xarray integration for TimeSeries with improved datatypes support

### 💥 Breaking Changes

**Framework Migration:**
- **Optimization Engine**: Complete migration from Pyomo to Linopy optimization framework
- **Package Import**: Framework renamed from flixOpt to flixopt (`import flixopt as fx`)
- **Data Architecture**: Redesigned data handling to rely on xarray.Dataset throughout the package
- **Results System**: Results handling completely redesigned with new `CalculationResults` class

**Variable Structure:**
- Restructured the modeling of the On/Off state of Flows or Components
  - Variable renaming: `...|consecutive_on_hours` → `...|ConsecutiveOn|hours`
  - Variable renaming: `...|consecutive_off_hours` → `...|ConsecutiveOff|hours`
  - Constraint renaming: `...|consecutive_on_hours_con1` → `...|ConsecutiveOn|con1`
  - Similar pattern for all consecutive on/off constraints

### 🔥 Removed
- **Pyomo dependency** (replaced by linopy)
- **Period concepts** in time management (simplified to timesteps)

### 🐛 Fixed
- Improved infeasible model detection and reporting
- Enhanced time series management and serialization
- Reduced file size through improved compression

### 📝 Docs
- Google Style Docstrings throughout the codebase
