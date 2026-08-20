# Glossary

Key concepts and terminology used throughout flixOpt.

## System Elements

| Concept | Description |
|---------|-------------|
| **Bus** | A connection point where energy or material flows meet. Acts as a junction that enforces flow balance (inputs = outputs). Examples: heat network, electricity grid, gas bus. |
| **Flow** | Movement of energy or material between a component and a bus. Has a direction (into or out of component), a **size** (capacity), and a **flow_rate** (actual power at each timestep). |
| **Component** | Physical or logical element that transforms, stores, or transfers flows. Connects to buses via flows. |
| **Carrier** | Type of energy or material (electricity, heat, gas, water). Assigned to buses for semantic organization and automatic plot coloring. |
| **Effect** | Any measurable metric to track or optimize (costs, CO2 emissions, energy use). One effect is the **objective** (minimized/maximized), others can be constrained or tracked. |
| **FlowSystem** | The complete model container. Holds all buses, components, flows, effects, and the time definition. Entry point for optimization and result access. |

## Component Types

| Concept | Description |
|---------|-------------|
| **LinearConverter** | Transforms input flows to output flows via linear conversion factors. Examples: boiler (gas → heat), heat pump (electricity → heat), turbine. |
| **Storage** | Accumulates and releases energy over time. Tracks charge state evolution. Examples: battery, thermal tank, reservoir. |
| **Source** | System boundary providing supply from outside. Examples: grid connection, fuel supplier, well. |
| **Sink** | System boundary consuming demand. Examples: building load, export, waste disposal. |
| **SourceAndSink** | Combined source and sink at the same bus. Used when both import and export are possible. |
| **Transmission** | Transports flows between locations with optional efficiency losses. Example: district heating pipe, power line. |

## Time and Dimensions

| Concept | Description |
|---------|-------------|
| **timesteps** | The basic time resolution of the model. A sequence of time points (e.g., hourly for one year = 8760 timesteps). All variables are indexed over timesteps. |
| **timestep_duration** | Length of each timestep in hours. Used to convert between power (kW) and energy (kWh). Inferred from the datetime index if not specified. |
| **period** | Long-term planning horizon dimension. Multiple periods enable multi-year investment planning (e.g., 2025, 2030, 2035). Each period has its own investment decisions. |
| **scenario** | Uncertainty dimension representing different futures (e.g., weather scenarios, price scenarios). Operations vary per scenario; investments are typically shared across scenarios. |
| **cluster** | Aggregation dimension used when time-series clustering is applied. Represents typical periods that stand in for many similar original periods. |

## Time-Series Clustering

| Concept | Description |
|---------|-------------|
| **typical period** | A representative time segment (e.g., typical day) selected or computed to represent a cluster of similar original periods. |
| `n_clusters` | Number of clusters (typical periods) to create. Each cluster represents multiple similar original periods. Example: 12 typical days for a year. |
| `cluster_duration` | Length of each cluster period. Accepts int/float (hours) or pandas Timedelta strings (e.g., `24`, `'24h'`, `'1D'`). |
| `cluster_weight` | How many original periods each cluster represents. Used to scale results back to full resolution. |
| `cluster_mode` | Storage behavior during clustering: `'intercluster_cyclic'` (seasonal storage), `'cyclic'` (daily cycling), `'independent'` (no linking). |
| **expand()** | Transform method to restore full time resolution after clustered optimization. Maps cluster solutions back to all original timesteps. |

## Variables and Parameters

| Concept | Description |
|---------|-------------|
| **flow_rate** | Decision variable: actual power/flow at each timestep [kW, m3/h]. Bounded by the flow's size. |
| **size** | Capacity or nominal rating of a flow [kW, m3/h]. Can be fixed (scalar), unbounded (None), or an investment decision (`InvestParameters`). In the solution, all investment variables use the `|size` suffix. |
| **capacity_in_flow_hours** | Storage capacity parameter [kWh, m3]. Distinct from flow `size` (which is power-based). In the solution, storage capacity is also accessed via `|size` for consistency with other investment variables. |
| **charge_state** | Storage variable: current amount stored [kWh, m3]. Evolves based on charging/discharging flows. Also called SOC (State of Charge) in energy system contexts. |
| **status** | Binary variable indicating whether equipment is operating (1) or off (0) at each timestep. Enabled via `StatusParameters`. |
| **conversion_factor** | Linear multiplier between input and output flows in a LinearConverter. Can be time-varying. Related to but not identical to efficiency. |
| **efficiency** | Ratio of useful output to input. For LinearConverter: output = efficiency * input. For Storage: `eta_charge` and `eta_discharge`. |

## Feature Parameters

| Concept | Description |
|---------|-------------|
| **InvestParameters** | Configuration for investment sizing decisions. Defines sizing bounds (`minimum_size`, `maximum_size`, `fixed_size`) and investment-related effects (capex). |
| **StatusParameters** | Configuration for binary on/off modeling. Enables startup effects, minimum uptime/downtime constraints, and operational mode tracking. |
| **Piece** | Single segment of a piecewise linear function, defined by start and end points. |
| **Piecewise** | Collection of pieces forming a piecewise linear approximation. Used for non-linear relationships like efficiency curves. |
| **PiecewiseConversion** | Multi-flow piecewise relationships where all flows change together based on operating point. |
| **PiecewiseEffects** | Piecewise relationship mapping a variable (origin) to effect contributions at varying rates. |

## Effects System

| Concept | Description |
|---------|-------------|
| **temporal effect** | Effect accumulated over timesteps from operations (e.g., fuel costs, emissions per MWh). Formula: `effect(t) = flow_rate(t) * cost_per_unit * dt`. |
| **periodic effect** | Time-independent effect per period (e.g., investment costs, fixed fees). Independent of operational decisions. |
| **total effect** | Sum of temporal and periodic effects: `E_total = E_periodic + sum(E_temporal(t))`. |
| **effects_per_flow_hour** | Cost/impact per unit of flow-hours. Parameter on Flow for operational costs (e.g., `{'costs': 50}` for 50 EUR/MWh). |
| **effects_of_investment_per_size** | Cost/impact per unit of installed capacity. Parameter on InvestParameters (e.g., `{'costs': 800}` for 800 EUR/kW). |
| **share_from_temporal** | Cross-effect linking where one effect contributes to another (e.g., CO2 → costs via carbon pricing). |
| **Penalty** | Built-in effect for soft constraints. Excess/shortage penalties on buses contribute to Penalty, which is added to the objective. |

## Operational Constraints

| Concept | Description |
|---------|-------------|
| **startup** | Transition from off (status=0) to on (status=1). Can incur costs via `effects_per_startup`. |
| **uptime** | Continuous duration equipment operates. Can be constrained with `min_uptime`, `max_uptime` in StatusParameters. |
| **downtime** | Continuous duration equipment is off. Can be constrained with `min_downtime`, `max_downtime` in StatusParameters. |
| **flow_hours** | Total energy delivered by a flow: sum of flow_rate * timestep_duration. Can be constrained with `flow_hours_min`, `flow_hours_max`. |
| **excess_penalty** | Penalty applied when bus has more supply than demand. Soft constraint alternative to strict balance. |
| **shortage_penalty** | Penalty applied when bus has more demand than supply (unmet demand). |

## Weights and Aggregation

| Concept | Description |
|---------|-------------|
| **scenario_weight** | Probability or importance of each scenario. Temporal effects are weighted by scenario weight in the objective. Default: equal weights, normalized to sum to 1. |
| **period_weight** | Importance/duration of each period. Computed automatically from period index intervals. Used for multi-year cost aggregation. |
| **cluster_weight** | Number of original periods each cluster represents. Used to scale clustered results to full resolution. |

## Solution and Results

| Concept | Description |
|---------|-------------|
| **solution** | xarray Dataset containing all optimization results. Access via `flow_system.solution` or element-specific `.solution` attributes. |
| **statistics** | Accessor providing aggregated result analysis. Methods: `flow_rates`, `flow_hours`, `sizes`, `charge_states`, `total_effects`. |
| **statistics.plot** | Visualization accessor. Methods: `balance()`, `heatmap()`, `sankey()`, `effects()`, `storage()`. |

## Optimization

| Concept | Description |
|---------|-------------|
| **optimize()** | Main entry point to build and solve the optimization model. Returns solution status. |
| **build_model()** | Build the linopy optimization model without solving. Allows adding custom constraints before solving. |
| **solve()** | Solve a previously built model. |
| **segmented optimization** | Rolling horizon approach that solves the problem in overlapping time windows. Useful for large problems or online optimization. |

## Transform Methods

| Concept | Description |
|---------|-------------|
| **transform.sel()** | Select a subset of the FlowSystem along dimensions (time, period, scenario). Returns a new FlowSystem. |
| **transform.cluster()** | Apply time-series clustering to reduce problem size. Returns a clustered FlowSystem. |
| **transform.expand()** | Restore full time resolution from a clustered solution. Reconstructs original timesteps from typical periods. |
| **transform.resample()** | Change time resolution (e.g., hourly to 4-hourly). Returns a resampled FlowSystem. |

## Mathematical Notation

| Symbol | Type | Description |
|--------|------|-------------|
| $p(t)$ | Variable | Flow rate at timestep $t$ |
| $P$ | Variable/Parameter | Size (capacity) of flow or storage |
| $E(t)$ | Variable | Charge state of storage at timestep $t$ |
| $s(t)$ | Variable | Binary status (on/off) at timestep $t$ |
| $s^{start}(t)$ | Variable | Binary startup indicator at timestep $t$ |
| $\eta$ | Parameter | Efficiency factor |
| $\Delta t$ | Parameter | Timestep duration (hours) |
| $w_s$ | Parameter | Scenario weight |
| $w_y$ | Parameter | Period weight |
