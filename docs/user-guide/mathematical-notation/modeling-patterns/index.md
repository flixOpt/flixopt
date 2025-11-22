# Modeling Patterns

This section documents the fundamental mathematical patterns used throughout FlixOpt for constructing optimization models. These patterns are implemented in `flixopt.modeling` and provide reusable building blocks for creating constraints.

## Overview

The modeling patterns are organized into three categories:

1. **[Bounds and States](bounds-and-states.md)** - Variable bounding with optional state control
2. **[Duration Tracking](duration-tracking.md)** - Tracking consecutive durations of states
3. **[State Transitions](state-transitions.md)** - Modeling state changes and transitions

## Pattern Categories

### Bounding Patterns

These patterns define how optimization variables are constrained within bounds:

- **Basic Bounds** - Simple upper and lower bounds on variables
- **Bounds with State** - Binary-controlled bounds (active/inactive states)
- **Scaled Bounds** - Bounds dependent on another variable (e.g., size)
- **Scaled Bounds with State** - Combination of scaling and binary control

### Tracking Patterns

These patterns track properties over time:

- **Expression Tracking** - Creating auxiliary variables that track expressions
- **Consecutive Duration Tracking** - Tracking how long a state has been active
- **Mutual Exclusivity** - Ensuring only one of multiple options is active

### Transition Patterns

These patterns model changes between states:

- **State Transitions** - Tracking switches between binary states (on→off, off→on)
- **Continuous Transitions** - Linking continuous variable changes to switches
- **Level Changes with Binaries** - Controlled increases/decreases in levels

## Usage in Components

These patterns are used throughout FlixOpt components:

- [`Flow`][flixopt.elements.Flow] uses **scaled bounds with state** for flow rate constraints
- [`Storage`][flixopt.components.Storage] uses **basic bounds** for charge state
- [`StatusParameters`](../features/StatusParameters.md) uses **state transitions** for startup/shutdown
- [`InvestParameters`](../features/InvestParameters.md) uses **bounds with state** for investment decisions

## Implementation

All patterns are implemented in [`flixopt.modeling`][flixopt.modeling] module:

- [`ModelingPrimitives`][flixopt.modeling.ModelingPrimitives] - Core constraint patterns
- [`BoundingPatterns`][flixopt.modeling.BoundingPatterns] - Specialized bounding patterns
