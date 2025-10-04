# Mathematical Documentation Cross-Reference

This document provides a comprehensive mapping between mathematical concepts, their documentation, implementation, and tests. Use this to quickly locate where specific mathematical formulations are defined and implemented.

## Components

| Concept | Documentation | Equations | Implementation | Key Location |
|---------|---------------|-----------|----------------|--------------|
| **Flow rate bounds** | [Flow.md](../elements/Flow.md) | eq. (1) | `FlowModel._do_modeling()` | `flixopt/elements.py:350+` |
| **Bus balance** | [Bus.md](../elements/Bus.md) | eq. (1), (2) | `BusModel._do_modeling()` | `flixopt/elements.py:751` |
| **Bus excess/deficit** | [Bus.md](../elements/Bus.md) | eq. (3), (4) | `BusModel._do_modeling()` | `flixopt/elements.py:763` |
| **Storage charge bounds** | [Storage.md](../elements/Storage.md) | eq. (1) | `StorageModel._do_modeling()` | `flixopt/components.py:~820` |
| **Storage balance** | [Storage.md](../elements/Storage.md) | eq. (3) | `StorageModel._do_modeling()` | `flixopt/components.py:838-842` |
| **Linear conversion** | [LinearConverter.md](../elements/LinearConverter.md) | eq. (1), (2) | Component-specific | `flixopt/components.py:37+` |

## Features

| Concept | Documentation | Equations | Implementation | Key Location |
|---------|---------------|-----------|----------------|--------------|
| **Binary investment** | [InvestParameters.md](../features/InvestParameters.md) | eq. (1) | `InvestParametersFeature` | `flixopt/interface.py:663+` |
| **Continuous investment sizing** | [InvestParameters.md](../features/InvestParameters.md) | eq. (2) | `InvestParametersFeature` | `flixopt/interface.py:663+` |
| **Investment fixed effects** | [InvestParameters.md](../features/InvestParameters.md) | eq. (4) | Effect calculation | `flixopt/features.py` |
| **Investment specific effects** | [InvestParameters.md](../features/InvestParameters.md) | eq. (5) | Effect calculation | `flixopt/features.py` |
| **Investment divestment effects** | [InvestParameters.md](../features/InvestParameters.md) | eq. (8) | Effect calculation | `flixopt/features.py` |
| **On/off state variable** | [OnOffParameters.md](../features/OnOffParameters.md) | eq. (1) | `OnOffFeature` | `flixopt/interface.py:918+` |
| **State transitions** | [OnOffParameters.md](../features/OnOffParameters.md) | eq. (2), (3) | `OnOffFeature` | Uses `state_transition_bounds()` |
| **Switching effects** | [OnOffParameters.md](../features/OnOffParameters.md) | eq. (4) | Effect calculation | `flixopt/features.py` |
| **Running effects** | [OnOffParameters.md](../features/OnOffParameters.md) | eq. (5) | Effect calculation | `flixopt/features.py` |
| **Minimum on-time** | [OnOffParameters.md](../features/OnOffParameters.md) | eq. (8) | Duration tracking | Uses `consecutive_duration_tracking()` |
| **Piecewise segments** | [Piecewise.md](../features/Piecewise.md) | eq. (1)-(4) | `PiecewiseModel` | `flixopt/features.py:382+` |

## Modeling Patterns

| Pattern | Documentation | Equations | Implementation | Key Location |
|---------|---------------|-----------|----------------|--------------|
| **Basic bounds** | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#basic-bounds) | eq. (1) | `BoundingPatterns.basic_bounds()` | `flixopt/modeling.py:393` |
| **Bounds with state** | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#bounds-with-state) | eq. (2) | `BoundingPatterns.bounds_with_state()` | `flixopt/modeling.py:427` |
| **Scaled bounds** | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#scaled-bounds) | eq. (3) | `BoundingPatterns.scaled_bounds()` | `flixopt/modeling.py:473` |
| **Scaled bounds with state** | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#scaled-bounds-with-state) | eq. (4), (5) | `BoundingPatterns.scaled_bounds_with_state()` | `flixopt/modeling.py:516` |
| **Expression tracking** | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#expression-tracking) | eq. (6), (7) | `ModelingPrimitives.expression_tracking_variable()` | `flixopt/modeling.py:201` |
| **Mutual exclusivity** | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#mutual-exclusivity) | eq. (8) | `ModelingPrimitives.mutual_exclusivity_constraint()` | `flixopt/modeling.py:345` |
| **Duration tracking** | [duration-tracking.md](../modeling-patterns/duration-tracking.md) | eq. (1)-(4) | `ModelingPrimitives.consecutive_duration_tracking()` | `flixopt/modeling.py:240` |
| **Minimum duration** | [duration-tracking.md](../modeling-patterns/duration-tracking.md#minimum-duration-constraints) | eq. (5) | Part of `consecutive_duration_tracking()` | `flixopt/modeling.py:240` |
| **State transitions** | [state-transitions.md](../modeling-patterns/state-transitions.md#binary-state-transitions) | eq. (1)-(4) | `BoundingPatterns.state_transition_bounds()` | `flixopt/modeling.py:573` |
| **Continuous transitions** | [state-transitions.md](../modeling-patterns/state-transitions.md#continuous-transitions) | eq. (5), (6) | `BoundingPatterns.continuous_transition_bounds()` | `flixopt/modeling.py:618` |
| **Level changes with binaries** | [state-transitions.md](../modeling-patterns/state-transitions.md#level-changes-with-binaries) | eq. (7)-(12) | `BoundingPatterns.link_changes_to_level_with_binaries()` | `flixopt/modeling.py:684` |

## Effects System

| Concept | Documentation | Equations | Implementation | Key Location |
|---------|---------------|-----------|----------------|--------------|
| **Share to effect (invest)** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (1) | Effect aggregation | `flixopt/effects.py` |
| **Share to effect (operation)** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (2) | Effect aggregation | `flixopt/effects.py` |
| **Cross-effect contribution** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (3), (4) | Effect cross-links | `flixopt/effects.py` |
| **Total effect** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (6) | Effect total calculation | `flixopt/effects.py` |
| **Effect bounds** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (7), (8) | Effect constraints | `flixopt/effects.py` |
| **Penalty** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (9) | Penalty aggregation | `flixopt/effects.py` |
| **Objective function** | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | eq. (10) | Objective creation | `flixopt/flow_system.py` |

## Quick Lookup by Python Class

| Python Class | Mathematical Documentation | Location |
|--------------|---------------------------|----------|
| `Flow` | [Flow.md](../elements/Flow.md) | `flixopt/elements.py:175` |
| `Bus` | [Bus.md](../elements/Bus.md) | `flixopt/elements.py:120` |
| `Storage` | [Storage.md](../elements/Storage.md) | `flixopt/components.py:237` |
| `LinearConverter` | [LinearConverter.md](../elements/LinearConverter.md) | `flixopt/components.py:37` |
| `InvestParameters` | [InvestParameters.md](../features/InvestParameters.md) | `flixopt/interface.py:663` |
| `OnOffParameters` | [OnOffParameters.md](../features/OnOffParameters.md) | `flixopt/interface.py:918` |
| `Piecewise` | [Piecewise.md](../features/Piecewise.md) | `flixopt/interface.py:83` |
| `Effect` | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | `flixopt/effects.py:32` |
| `ModelingPrimitives` | [modeling-patterns/](../modeling-patterns/index.md) | `flixopt/modeling.py:178` |
| `BoundingPatterns` | [modeling-patterns/](../modeling-patterns/index.md) | `flixopt/modeling.py:390` |

## Quick Lookup by Mathematical Concept

| Mathematical Concept | Documentation | Implementation Function |
|---------------------|---------------|------------------------|
| Variable bounds | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md) | `basic_bounds()`, `bounds_with_state()`, `scaled_bounds()`, `scaled_bounds_with_state()` |
| Binary state control | [bounds-and-states.md](../modeling-patterns/bounds-and-states.md#bounds-with-state) | `bounds_with_state()`, `scaled_bounds_with_state()` |
| Consecutive time periods | [duration-tracking.md](../modeling-patterns/duration-tracking.md) | `consecutive_duration_tracking()` |
| Switching between states | [state-transitions.md](../modeling-patterns/state-transitions.md) | `state_transition_bounds()`, `continuous_transition_bounds()` |
| Linear equality constraints | [LinearConverter.md](../elements/LinearConverter.md) | Component-specific implementations |
| Energy balance | [Storage.md](../elements/Storage.md), [Bus.md](../elements/Bus.md) | Component-specific implementations |
| Piecewise linear functions | [Piecewise.md](../features/Piecewise.md) | `PiecewiseModel._do_modeling()` |
| Investment decisions | [InvestParameters.md](../features/InvestParameters.md) | `InvestParametersFeature` |
| On/off operation | [OnOffParameters.md](../features/OnOffParameters.md) | `OnOffFeature` |
| Cost/effect allocation | [Effects, Penalty & Objective.md](../effects-penalty-objective.md) | Effect system in `flixopt/effects.py` |

## Navigation Tips

### Finding Documentation from Code
1. Find the class name in Python (e.g., `Storage`)
2. Look up the class in "Quick Lookup by Python Class" table
3. Follow the link to the mathematical documentation

### Finding Implementation from Docs
1. Find the equation number in documentation (e.g., Storage eq. 3)
2. Look up in the corresponding section in the cross-reference tables
3. Navigate to the implementation location

### Understanding a Mathematical Pattern
1. Identify the pattern type (bounds, transitions, duration, etc.)
2. Read the pattern documentation in [modeling-patterns/](../modeling-patterns/index.md)
3. See usage examples in component documentation
4. Find implementation in `flixopt/modeling.py`

## See Also

- [Mathematical Notation Overview](index.md) - Naming conventions and fundamentals
- [Modeling Patterns](../modeling-patterns/index.md) - Reusable mathematical building blocks
- [Effects, Penalty & Objective](../effects-penalty-objective.md) - System-level formulation
