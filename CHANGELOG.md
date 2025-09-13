# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- This text won't be rendered
Take Care: The CI will automatically append a "Whats CHanged" section to the changelog.
This contains all COmmits, PR's and Contributers.
Therefore, the Changelog should focus on the user-facing changes.

## [Unreleased] - ????-??-??

### Changed

### Added

### Deprecated

### Fixed

### Known issues

### Development

Until here -->

## [2.1.7] - 2025-09-13

This update is a maintenance release to improve Code Quality, CI and update the dependencies.
There are no changes or new features.

### Added
- Added __version__ to flixopt

### Development
- ruff format the whole Codebase
- Added renovate config
- Added pre-commit
- lint and format in CI
- improved CI
- Updated Dependencies
- Updated Issue Templates


## [2.1.6] - 2025-09-02

### Changed
- `Sink`, `Source` and `SourceAndSink` now accept multiple `flows` as `inputs` and `outputs` instead of just one. This enables to model more use cases using these classes. [[#291](https://github.com/flixOpt/flixopt/pull/291) by [@FBumann](https://github.com/FBumann)]
- Further, both `Sink` and `Source` now have a `prevent_simultaneous_flow_rates` argument to prevent simultaneous flow rates of more than one of their Flows. [[#291](https://github.com/flixOpt/flixopt/pull/291) by [@FBumann](https://github.com/FBumann)]

### Added
- Added `FlowSystem.start_network_app()` and `FlowSystem.stop_network_app()` to easily visualize the network structure of a flow system in an interactive dash web app. This is still experimental and might change in the future. [[#293](https://github.com/flixOpt/flixopt/pull/293) by [@FBumann](https://github.com/FBumann)]

### Deprecated
- For the classes `Sink`, `Source` and `SourceAndSink`: `.sink`, `.source` and `.prevent_simultaneous_sink_and_source` are deprecated in favor of the new arguments `inputs`, `outputs` and `prevent_simultaneous_flow_rates`. [[#291](https://github.com/flixOpt/flixopt/pull/291) by [@FBumann](https://github.com/FBumann)]

### Fixed
- Fixed testing issue with new `linopy` version 0.5.6 [[#296](https://github.com/flixOpt/flixopt/pull/296) by [@FBumann](https://github.com/FBumann)]

## [2.1.5] - 2025-07-08

### Fixed
- Fixed Docs deployment

## [2.1.4] - 2025-07-08

### Fixed
- Fixing release notes of 2.1.3, as well as documentation build.


## [2.1.3] - 2025-07-08

### Fixed
- Using `Effect.maximum_operation_per_hour` raised an error, needing an extra timestep. This has been fixed thanks to @PRse4.

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
