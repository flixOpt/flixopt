
# Mathematical Notation

This section provides the **mathematical formulations** underlying FlixOpt's optimization models. It is intended as **reference documentation** for users who want to understand the mathematical details behind the high-level FlixOpt API described in the [FlixOpt Concepts](../core-concepts.md) guide.

**For typical usage**, refer to the [FlixOpt Concepts](../core-concepts.md) guide, [Examples](../../examples/index.md), and [API Reference](../../api-reference/index.md) - you don't need to understand these mathematical formulations to use FlixOpt effectively.

---

## Naming Conventions

FlixOpt uses the following naming conventions:

- All optimization variables are denoted by italic letters (e.g., $x$, $y$, $z$)
- All parameters and constants are denoted by non italic small letters (e.g., $\text{a}$, $\text{b}$, $\text{c}$)
- All Sets are denoted by greek capital letters (e.g., $\mathcal{F}$, $\mathcal{E}$)
- All units of a set are denoted by greek small letters (e.g., $\mathcal{f}$, $\mathcal{e}$)
- The letter $i$ is used to denote an index (e.g., $i=1,\dots,\text n$)
- All time steps are denoted by the letter $\text{t}$ (e.g., $\text{t}_0$, $\text{t}_1$, $\text{t}_i$)

## Dimensions and Time Steps

FlixOpt supports multi-dimensional optimization with up to three dimensions: **time** (mandatory), **period** (optional), and **scenario** (optional).

**All mathematical formulations in this documentation are independent of whether periods or scenarios are present.** The equations shown are written with time index $\text{t}_i$ only, but automatically expand to additional dimensions when periods/scenarios are added.

For complete details on dimensions, their relationships, and influence on formulations, see **[Dimensions](dimensions.md)**.

### Time Steps

Time steps are defined as a sequence of discrete time steps $\text{t}_i \in \mathcal{T} \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}\}$ (left-aligned in its timespan).
From this sequence, the corresponding time intervals $\Delta \text{t}_i \in \Delta \mathcal{T}$ are derived as

$$\Delta \text{t}_i = \text{t}_{i+1} - \text{t}_i \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}-1\}$$

The final time interval $\Delta \text{t}_\text n$ defaults to $\Delta \text{t}_\text n = \Delta \text{t}_{\text n-1}$, but is of course customizable.
Non-equidistant time steps are also supported.

---

## Documentation Structure

This reference is organized to match the FlixOpt API structure:

### Elements
Mathematical formulations for core FlixOpt elements (corresponding to [`flixopt.elements`][flixopt.elements]):

- [Flow](elements/Flow.md) - Flow rate constraints and bounds
- [Bus](elements/Bus.md) - Nodal balance equations
- [Storage](elements/Storage.md) - Storage balance and charge state evolution
- [LinearConverter](elements/LinearConverter.md) - Linear conversion relationships

**User API:** When you create a `Flow`, `Bus`, `Storage`, or `LinearConverter` in your FlixOpt model, these mathematical formulations are automatically applied.

### Features
Mathematical formulations for optional features (corresponding to parameters in FlixOpt classes):

- [InvestParameters](features/InvestParameters.md) - Investment decision modeling
- [StatusParameters](features/StatusParameters.md) - Binary active/inactive operation
- [Piecewise](features/Piecewise.md) - Piecewise linear approximations

**User API:** When you pass `invest_parameters` or `status_parameters` to a `Flow` or component, these formulations are applied.

### System-Level
- [Effects, Penalty & Objective](effects-penalty-objective.md) - Cost allocation and objective function

**User API:** When you create [`Effect`][flixopt.effects.Effect] objects and set `effects_per_flow_hour`, these formulations govern how costs are calculated.

### Modeling Patterns (Advanced)
**Internal implementation details** - These low-level patterns are used internally by Elements and Features. They are documented here for:

- Developers extending FlixOpt
- Advanced users debugging models or understanding solver behavior
- Researchers comparing mathematical formulations

**Normal users do not need to read this section** - the patterns are automatically applied when you use Elements and Features:

- [Bounds and States](modeling-patterns/bounds-and-states.md) - Variable bounding patterns
- [Duration Tracking](modeling-patterns/duration-tracking.md) - Consecutive time period tracking
- [State Transitions](modeling-patterns/state-transitions.md) - State change modeling

---

## Quick Reference

### Components Cross-Reference

| Concept | Documentation | Python Class |
|---------|---------------|--------------|
| **Flow rate bounds** | [Flow](elements/Flow.md) | [`Flow`][flixopt.elements.Flow] |
| **Bus balance** | [Bus](elements/Bus.md) | [`Bus`][flixopt.elements.Bus] |
| **Storage balance** | [Storage](elements/Storage.md) | [`Storage`][flixopt.components.Storage] |
| **Linear conversion** | [LinearConverter](elements/LinearConverter.md) | [`LinearConverter`][flixopt.components.LinearConverter] |

### Features Cross-Reference

| Concept | Documentation | Python Class |
|---------|---------------|--------------|
| **Binary investment** | [InvestParameters](features/InvestParameters.md) | [`InvestParameters`][flixopt.interface.InvestParameters] |
| **On/off operation** | [StatusParameters](features/StatusParameters.md) | [`StatusParameters`][flixopt.interface.StatusParameters] |
| **Piecewise segments** | [Piecewise](features/Piecewise.md) | [`Piecewise`][flixopt.interface.Piecewise] |

### Modeling Patterns Cross-Reference

| Pattern | Documentation | Implementation |
|---------|---------------|----------------|
| **Basic bounds** | [bounds-and-states](modeling-patterns/bounds-and-states.md#basic-bounds) | [`BoundingPatterns.basic_bounds()`][flixopt.modeling.BoundingPatterns.basic_bounds] |
| **Bounds with state** | [bounds-and-states](modeling-patterns/bounds-and-states.md#bounds-with-state) | [`BoundingPatterns.bounds_with_state()`][flixopt.modeling.BoundingPatterns.bounds_with_state] |
| **Scaled bounds** | [bounds-and-states](modeling-patterns/bounds-and-states.md#scaled-bounds) | [`BoundingPatterns.scaled_bounds()`][flixopt.modeling.BoundingPatterns.scaled_bounds] |
| **Duration tracking** | [duration-tracking](modeling-patterns/duration-tracking.md) | [`ModelingPrimitives.consecutive_duration_tracking()`][flixopt.modeling.ModelingPrimitives.consecutive_duration_tracking] |
| **State transitions** | [state-transitions](modeling-patterns/state-transitions.md) | [`BoundingPatterns.state_transition_bounds()`][flixopt.modeling.BoundingPatterns.state_transition_bounds] |

### Python Class Lookup

| Class | Documentation | API Reference |
|-------|---------------|---------------|
| `Flow` | [Flow](elements/Flow.md) | [`Flow`][flixopt.elements.Flow] |
| `Bus` | [Bus](elements/Bus.md) | [`Bus`][flixopt.elements.Bus] |
| `Storage` | [Storage](elements/Storage.md) | [`Storage`][flixopt.components.Storage] |
| `LinearConverter` | [LinearConverter](elements/LinearConverter.md) | [`LinearConverter`][flixopt.components.LinearConverter] |
| `InvestParameters` | [InvestParameters](features/InvestParameters.md) | [`InvestParameters`][flixopt.interface.InvestParameters] |
| `StatusParameters` | [StatusParameters](features/StatusParameters.md) | [`StatusParameters`][flixopt.interface.StatusParameters] |
| `Piecewise` | [Piecewise](features/Piecewise.md) | [`Piecewise`][flixopt.interface.Piecewise] |
