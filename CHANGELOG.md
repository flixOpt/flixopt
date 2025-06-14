# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
* Fixed formating issues in the yaml model documentation (line breaks)

### Breaking Changes:
* Removed `kind` in favor of `style` in plotting functions.
* Renamed `TimeSeries.active_data` to `TimeSeries.selected_data`
* `CalculationResults.flow_system` now returns the restorded FlowSystem instead of the `xr.Dataset`. The data can be found under `flow_system_data`.

### Added
* **Scenarios:**
  * Scenarios can now be used to model uncertainties in the flow system, such as:
  * Scenarios are passed to a `FlowSystem`. The total objective effect of each scenario is multiplied by a `scenario_weight`. This forms the objective of the optimization.
* **`CalculationResults`:** 
  * New dedicated `FlowResults`. 
  * Dedicated xr.DataArrays for all **flow_rates**, **flow_hours**, and **sizes** of flows are availlable.
  * Use `effects_per_component()` to retrieve all effects results for every Component. This includes indirect effects that hove their origin in an element, but are inflicted by another effect (ElementA --> CO2 --> Costs))
* Balanced storage - Storage charging and discharging sizes can now be forced to be equal when optimizing their size by choosing `balanced=True`.
* Plotting styles can now be changed for all plots. (stacked_bar, line, area)
* Added plotting style `grouped_bar`
* Support for pandas.Series and pandas.DataFrame when setting parameters of Elements (internally converted to xarray.DataArrays)

### Changed
* Improved internal Datatypes to make needed data format more obvious: `Scalar` for only scalar values, `TimestepData` for time-indexed data (which might have a scenario dimension), `ScenarioData` for data with a scenario dimension.
* `InvestmentParameters` now have a `investment_scenarios` parameter to define which scenarios to define how to optimize the size across scenarios
* Changed legend location in plots

### Deprecations
* Renamed `Calculation.active_timesteps` to `Calculation.selected_timesteps`
* A warning is raised if Results prior ti this version are loaded, as this prevents the FLowResults from being created.

### Known Issues
* Scenarios are not yet supported in `AggregatedCalculation` and `SegmentedCalculation`

### Development
* Greatly improved testing by directly asserting for the correctness of the created equations and variables (and their bounds).

## [2.1.2] - 2025-06-14

### Fixed
- Storage losses per hour where not calculated correctly, as mentioned by @brokenwings01. This might have lead to issues with modeling large losses and long timesteps. 
  - Old implementation:     $c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i)) \cdot \Delta \text{t}_{i}$
  - Correct implementation: $c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i)) ^{\Delta \text{t}_{i}}$

### Known issues
- Just to mention: Plotly >= 6 may raise errors if "nbformat" is not installed. We pinned plotly to <6, but this may be fixed in the future.

## [2.1.1] - 2025-05-08

### Fixed
- Fixed bug in the `_ElementResults.constraints` not returning the constraints but rather the variables

### Changed
- Improved docstring and tests

## [2.1.0] - 2025-04-11

### Added
- Python 3.13 support added
- Logger warning if relative_minimum is used without on_off_parameters in Flow
- Greatly improved internal testing infrastructure by leveraging linopy's testing framework

### Fixed
- Fixed the lower bound of `flow_rate` when using optional investments without OnOffParameters
- Fixed bug that prevented divest effects from working
- Added lower bounds of 0 to two unbounded vars (numerical improvement)

### Changed
- **BREAKING**: Restructured the modeling of the On/Off state of Flows or Components
  - Variable renaming: `...|consecutive_on_hours` → `...|ConsecutiveOn|hours`
  - Variable renaming: `...|consecutive_off_hours` → `...|ConsecutiveOff|hours`
  - Constraint renaming: `...|consecutive_on_hours_con1` → `...|ConsecutiveOn|con1`
  - Similar pattern for all consecutive on/off constraints

## [2.0.1] - 2025-04-10

### Added
- Logger warning if relative_minimum is used without on_off_parameters in Flow

### Fixed
- Replace "|" with "__" in filenames when saving figures (Windows compatibility)
- Fixed bug that prevented the load factor from working without InvestmentParameters

## [2.0.0] - 2025-03-29

### Changed
- **BREAKING**: Complete migration from Pyomo to Linopy optimization framework
- **BREAKING**: Redesigned data handling to rely on xarray.Dataset throughout the package
- **BREAKING**: Framework renamed from flixOpt to flixopt (`import flixopt as fx`)
- **BREAKING**: Results handling completely redesigned with new `CalculationResults` class

### Added
- Full model serialization support - save and restore unsolved Models
- Enhanced model documentation with YAML export containing human-readable mathematical formulations
- Extend flixopt models with native linopy language support
- Full Model Export/Import capabilities via linopy.Model
- Unified solution exploration through `Calculation.results` attribute
- Compression support for result files
- `to_netcdf/from_netcdf` methods for FlowSystem and core components
- xarray integration for TimeSeries with improved datatypes support
- Google Style Docstrings throughout the codebase

### Fixed
- Improved infeasible model detection and reporting
- Enhanced time series management and serialization
- Reduced file size through improved compression

### Removed
- **BREAKING**: Pyomo dependency (replaced by linopy)
- Period concepts in time management (simplified to timesteps)