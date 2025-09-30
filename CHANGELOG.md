# Changelog

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Formatting is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) & [Gitmoji](https://gitmoji.dev).
For more details regarding the individual PRs and contributors, please refer to our [GitHub releases](https://github.com/flixOpt/flixopt/releases).

<!-- This text won't be rendered
Note: The CI will automatically append a "What's Changed" section to the changelog for github releases.
This contains all commits, PRs, and contributors.
Therefore, the Changelog should focus on the user-facing changes.

Please remove all irrelevant sections before releasing.
Please keep the format of the changelog consistent with the other releases, so the extraction for mkdocs works.
---

## [Template] - ????-??-??

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
This release introduces new model dimensions (periods and scenarios) for multi-period investments and stochastic modeling, along with a redesigned effect sharing system and enhanced I/O capabilities.

### ✨ Added

**New model dimensions:**
* **Period dimension**: Enables multi-period investment modeling with distinct decisions in each period for transformation pathway optimization
* **Scenario dimension**: Supports stochastic modeling with weighted scenarios for robust decision-making under uncertainty (demand, prices, weather)

**Redesigned effect sharing system:**
Effects now use intuitive `share_from_*` syntax that clearly shows contribution sources:
```python
costs = fx.Effect('costs', '€', 'Total costs',
    share_from_temporal={'CO2': 0.2},      # From temporal effects
    share_from_periodic={'land': 100})     # From periodic effects
```
This replaces `specific_share_to_other_effects_*` parameters and inverts the direction for clearer relationships.

**Enhanced I/O and data handling:**
* NetCDF/JSON serialization for all Interface objects and FlowSystem with round-trip support
* FlowSystem manipulation: `sel()`, `isel()`, `resample()`, `copy()`, `__eq__()` methods
* Direct access to FlowSystem from results without manual restoring (lazily loaded)
* New `FlowResults` class and precomputed DataArrays for sizes/flow_rates/flow_hours
* `effects_per_component()` dataset for component impact evaluation, including all indirect effects through effect shares

**Other additions:**
* Balanced storage - charging and discharging sizes can be forced equal via `balanced` parameter
* New Storage parameters: `relative_minimum_final_charge_state` and `relative_maximum_final_charge_state` for final state control
* Improved filter methods in results
* Example for 2-stage investment decisions leveraging FlowSystem resampling

### 💥 Breaking Changes
* `relative_minimum_charge_state` and `relative_maximum_charge_state` don't have an extra timestep anymore.
* Renamed class `SystemModel` to `FlowSystemModel`
* Renamed class `Model` to `Submodel`
* Renamed `mode` parameter in plotting methods to `style`
* `Calculation.do_modeling()` now returns the `Calculation` object instead of its `linopy.Model`. Callers that previously accessed the linopy model directly should now use `calculation.do_modeling().model` instead of `calculation.do_modeling()`.

### ♻️ Changed
* FlowSystems cannot be shared across multiple Calculations anymore. A copy of the FlowSystem is created instead, making every Calculation independent
* Each Subcalculation in `SegmentedCalculation` now has its own distinct `FlowSystem` object
* Type system overhaul - added clear separation between temporal and non-temporal data throughout codebase for better clarity
* Enhanced FlowSystem interface with improved `__repr__()` and `__str__()` methods
* Improved Model Structure - Views and organisation is now divided into:
  * Model: The main Model (linopy.Model) that is used to create and store the variables and constraints for the flow_system.
  * Submodel: The base class for all submodels. Each is a subset of the Model, for simpler access and clearer code.

### 🗑️ Deprecated
* The `agg_group` and `agg_weight` parameters of `TimeSeriesData` are deprecated and will be removed in a future version. Use `aggregation_group` and `aggregation_weight` instead.
* The `active_timesteps` parameter of `Calculation` is deprecated and will be removed in a future version. Use the new `sel(time=...)` method on the FlowSystem instead.
* The assignment of Bus Objects to Flow.bus is deprecated and will be removed in a future version. Use the label of the Bus instead.
* The usage of Effects objects in Dicts to assign shares to Effects is deprecated and will be removed in a future version. Use the label of the Effect instead.
* Effect parameters renamed:
  - `minimum_investment` → `minimum_periodic`
  - `maximum_investment` → `maximum_periodic`
  - `minimum_operation` → `minimum_temporal`
  - `maximum_operation` → `maximum_temporal`
  - `minimum_operation_per_hour` → `minimum_per_hour`
  - `maximum_operation_per_hour` → `maximum_per_hour`

### 🔥 Removed
* **Effect share parameters**: The old `specific_share_to_other_effects_*` parameters were replaced WITHOUT DEPRECATION
  - `specific_share_to_other_effects_operation` → `share_from_temporal` (with inverted direction)
  - `specific_share_to_other_effects_invest` → `share_from_periodic` (with inverted direction)

### 🐛 Fixed
* Enhanced NetCDF I/O with proper attribute preservation for DataArrays
* Improved error handling and validation in serialization processes
* Better type consistency across all framework components

### 🚧 Known Issues
* IO for single Interfaces/Elements to Datasets might not work properly if the Interface/Element is not part of a fully transformed and connected FlowSystem. This arises from Numeric Data not being stored as xr.DataArray by the user. To avoid this, always use the `to_dataset()` on Elements inside a FlowSystem that's connected and transformed.

### 👷 Development
* FlowSystem data management simplified - removed `time_series_collection` pattern in favor of direct timestep properties
* Change modeling hierarchy to allow for more flexibility in future development. This leads to minimal changes in the access and creation of Submodels and their variables.
* Added new module `.modeling` that contains Modelling primitives and utilities
* Clearer separation between the main Model and "Submodels"
* Improved access to the Submodels and their variables, constraints and submodels
* Added `__repr__()` for Submodels to easily inspect its content
* Enhanced data handling methods
   * `fit_to_model_coords()` method for data alignment
   * `fit_effects_to_model_coords()` method for effect data processing
   * `connect_and_transform()` method replacing several operations


Until here -->
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
