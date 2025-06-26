# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
* **BREAKING**: FlowSystems can not be shared across multiple Calculations anymore. A copy of the FlowSystem is created instead, making every Calculation independent
* **BREAKING**: Type system overhaul - replaced `NumericDataTS` with `NumericDataUser` throughout codebase for better clarity
* **BREAKING**: `relative_minimum_charge_state` and `relative_maximum_charge_state` don't have an extra timestep anymore. The final charge state can now be constrained by parameters `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` instead
* FlowSystem data management simplified - removed `time_series_collection` pattern in favor of direct timestep properties
* Enhanced FlowSystem interface with improved `__repr__()` and `__str__()` methods
* *Internal*: Removed intermediate `TimeSeries` and `TimeSeriesCollection` classes, replaced directly with `xr.DataArray` or `TimeSeriesData` (inheriting from `xr.DataArray`)

### Added
* **NEW**: Complete serialization infrastructure through `Interface` base class
   * IO for all Interfaces and the FlowSystem with round-trip serialization support
   * Automatic DataArray extraction and restoration
   * NetCDF export/import capabilities for all Interface objects and FlowSystem
   * JSON export for documentation purposes
   * Recursive handling of nested Interface objects
* **NEW**: FlowSystem data manipulation methods
   * `sel()` and `isel()` methods for temporal data selection
   * `resample()` method for temporal resampling
   * `copy()` method with deep copying support
   * `__eq__()` method for FlowSystem comparison
* **NEW**: Storage component enhancements
   * `relative_minimum_final_charge_state` parameter for final state control
   * `relative_maximum_final_charge_state` parameter for final state control
* *Internal*: Enhanced data handling methods
   * `fit_to_model_coords()` method for data alignment
   * `fit_effects_to_model_coords()` method for effect data processing
   * `connect_and_transform()` method replacing separate operations
* **NEW**: Core data handling improvements
   * `get_dataarray_stats()` function for statistical summaries
   * Enhanced `DataConverter` class with better TimeSeriesData support

### Fixed
* Enhanced NetCDF I/O with proper attribute preservation for DataArrays
* Improved error handling and validation in serialization processes
* Better type consistency across all framework components

### Know Issues
* Plotly >= 6 may raise errors if "nbformat" is not installed. We pinned plotly to <6, but this may be fixed in the future.
* IO for single Interfaces/Elemenets to Datasets might not work properly if the Interface/Element is not part of a fully transformed and connected FlowSystem. This arrises from Numeric Data not being stored as xr.DataArray by the user. TO avoid this, always use the `to_dataset()` on Elements inside a FlowSystem thats connected and transformed.

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