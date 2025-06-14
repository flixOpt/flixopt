# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [gitmoji](https://gitmoji.dev)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 💥 Breaking Changes
* **💥 BREAKING**: Removed `kind` in favor of `style` in plotting functions
* **💥 BREAKING**: Renamed `TimeSeries.active_data` to `TimeSeries.selected_data`
* **💥 BREAKING**: `CalculationResults.flow_system` now returns the restored FlowSystem instead of the `xr.Dataset`. The data can be found under `flow_system_data`

### ✨ Added
#### Major Features
* **Scenarios**: Model uncertainties or **Multi-Period Transformations**
  * Scenarios are passed to a `FlowSystem` with `scenario_weight` multipliers
  * Total objective effect of each scenario forms the optimization objective
  * Sizes might be optimized for each scenario separately, globally or only for a subset of all scenarios (See `InvestmentParameters`).
* **Balanced Storage**: Storage charging and discharging sizes can now be forced to be equal when optimizing by choosing `balanced=True`

#### Results & Analysis
* **New dedicated `FlowResults` class**
  * Dedicated xr.DataArrays combining all **flow_rates**, **flow_hours**, or **sizes** of flows
  * Use `effects_per_component()` to retrieve all effects results for every Component, including indirect effects (ElementA → CO2 → Costs)

#### API Improvements
* Support for pandas.Series and pandas.DataFrame when setting Element parameters (internally converted to xarray.DataArrays)
* Improved internal datatypes for clearer data format requirements:
  * `Scalar` for scalar values only
  * `TimestepData` for time-indexed data (with optional scenario dimension)
  * `ScenarioData` for scenario-dimensional data

#### Plotting & Visualization
* All plotting styles available for both plotly and matplotlib plots: `stacked_bar`, `line`, `area`
* Added `grouped_bar` plotting style
* Changed default legend location in plots (now on the right side)

### 🗑️ Deprecated
* `Calculation.active_timesteps` → Use `Calculation.selected_timesteps` instead
* ⚠️ Loading Results from prior versions will raise warnings due to FlowResults incompatibility. Some new features cannot be used.

### 🐛 Fixed
* Fixed formatting issues in YAML model documentation (line breaks)

### Known Issues
* Scenarios are not yet supported in `AggregatedCalculation` and `SegmentedCalculation`

## [2.1.2] - 2025-06-14

### 🐛 Fixed
* **Critical Fix**: Storage losses per hour calculation corrected (thanks @brokenwings01)
  * **Impact**: Affects modeling of large losses and long timesteps
  * **Old**: `c(t_i) · (1-ċ_rel,loss(t_i)) · Δt_i`
  * **Correct**: `c(t_i) · (1-ċ_rel,loss(t_i))^Δt_i`

### Known Issues
* Plotly >= 6 may raise errors if "nbformat" is not installed (pinned to <6 for now)

## [2.1.1] - 2025-05-08

### 🐛 Fixed
* Fixed `_ElementResults.constraints` returning variables instead of constraints

### Changed
* Improved docstrings and tests

## [2.1.0] - 2025-04-11

### 💥 Breaking Changes
* **💥 BREAKING**: Restructured On/Off state modeling for Flows and Components
  * **♻️ Variable renaming**: `...|consecutive_on_hours` → `...|ConsecutiveOn|hours`
  * **♻️ Variable renaming**: `...|consecutive_off_hours` → `...|ConsecutiveOff|hours`
  * **♻️ Constraint renaming**: `...|consecutive_on_hours_con1` → `...|ConsecutiveOn|con1`
  * Similar pattern applied to all consecutive on/off constraints

### ✨ Added
* **Python 3.13 support**
* Enhanced testing infrastructure leveraging linopy's testing framework
* Logger warnings for `relative_minimum` usage without `on_off_parameters` in Flow

### 🐛 Fixed
* Fixed `flow_rate` lower bound issues with optional investments without OnOffParameters
* Fixed divest effects functionality
* Added missing lower bounds of 0 to unbounded variables (numerical stability improvement)

## [2.0.1] - 2025-04-10

### 🐛 Fixed
* **Windows Compatibility**: Replace "|" with "__" in figure filenames
* Fixed load factor functionality without InvestmentParameters

### ✨ Added
* Logger warning for `relative_minimum` usage without `on_off_parameters` in Flow

## [2.0.0] - 2025-03-29

### 💥 Breaking Changes
* **💥 BREAKING**: Complete migration from Pyomo to Linopy optimization framework
* **💥 BREAKING**: Redesigned data handling using xarray.Dataset throughout
* **💥 BREAKING**: Framework renamed from flixOpt to flixopt (`import flixopt as fx`)
* **💥 BREAKING**: Complete redesign of Results handling with new `CalculationResults` class
* **🔥 BREAKING**: Removed Pyomo dependency
* **🔥 BREAKING**: Removed Period concepts (simplified to timesteps)

### ✨ Added
#### Major Features
* **Full model serialization**: Save and restore unsolved Models
* **Enhanced model documentation**: YAML export with human-readable mathematical formulations
* **Native linopy integration**: Extend flixopt models with linopy language support
* **Model Export/Import**: Full capabilities via linopy.Model

#### Results & Analysis
* **Unified solution exploration** through `Calculation.results` attribute
* **Compression support** for result files
* **xarray integration** for TimeSeries with improved datatypes support

#### API Improvements
* `to_netcdf/from_netcdf` methods for FlowSystem and core components
* Google Style Docstrings throughout codebase

### 🐛 Fixed
* **Improved infeasible model detection and reporting**
* Enhanced time series management and serialization
* Reduced file sizes through better compression

### 🔥 Removed
* **BREAKING**: Pyomo dependency (replaced by linopy)
* **BREAKING**: Period concepts in time management (simplified to timesteps)