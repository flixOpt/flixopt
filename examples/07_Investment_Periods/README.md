# Investment Periods Examples

This directory contains examples demonstrating the use of the **period dimension** and **linked_periods parameter** in flixopt's `InvestParameters`.

## Overview

The period dimension enables multi-period optimization, allowing you to model investment decisions across multiple time horizons (e.g., years 2020, 2025, 2030). The `linked_periods` parameter controls how investment decisions are connected across these periods.

## Examples

### 1. `investment_periods_example.py`

**Basic period-based investments**

Demonstrates:
- Using InvestParameters with the period dimension
- Period-specific investment costs (e.g., technology learning curves)
- Period-varying constraints (maximum sizes)
- Basic linked_periods usage with `(0, 1)` - linking all periods together

Key concepts:
- Investment costs that decrease over time
- Different maximum capacities per period
- Single investment decision shared across periods (linked_periods)

### 2. `linked_periods_advanced_example.py`

**Advanced linked_periods configurations**

Demonstrates:
- `linked_periods=None`: Independent investment per period
- `linked_periods=(0, 1)`: All periods fully linked (single decision)
- Custom linking patterns with arrays like `[1, 1, 2, 2, 3]`
- Practical use cases: phased rollouts, technology generations, upgrade cycles

Key concepts:
- Group-based linking (same group ID = linked periods)
- Sequential vs. persistent investments
- Technology replacement cycles
- Phased deployment strategies

## Understanding linked_periods

The `linked_periods` parameter accepts:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `None` | Independent decision per period | Equipment that can be installed/removed between periods |
| `(0, 1)` | All periods linked (1D array) | Long-lived infrastructure (buildings, major equipment) |
| `[1,1,2,2,3]` | Custom groups | Phased deployments, technology generations, upgrade cycles |

### Custom Linking Examples

```python
# Example: Two technology generations
linked_periods=np.array([1, 1, 1, 2, 2])  # Periods 0-2 linked, periods 3-4 linked separately

# Example: Completely independent periods
linked_periods=np.array([1, 2, 3, 4, 5])  # Each period is its own group

# Example: Early commitment, later flexibility
linked_periods=np.array([1, 1, 2, 3, 4])  # First two linked, rest independent
```

## Running the Examples

```bash
# Basic period example
PYTHONPATH=/path/to/flixopt python examples/07_Investment_Periods/investment_periods_example.py

# Advanced linked_periods example
PYTHONPATH=/path/to/flixopt python examples/07_Investment_Periods/linked_periods_advanced_example.py
```

## Key Parameters in InvestParameters

```python
InvestParameters(
    fixed_size=None,              # Scalar or per-period numpy array
    minimum_size=0,               # Scalar or per-period numpy array
    maximum_size=1000,            # Scalar or per-period numpy array
    optional=True,
    fix_effects={'costs': 5000},  # Scalar or per-period numpy array/dict
    specific_effects={            # Scalar or per-period numpy array/dict
        'costs': np.array([1000, 900, 800])  # Decreasing costs per period
    },
    linked_periods=(0, 1),        # None, (0,1), or numpy array
)
```

## Common Patterns

### Technology Learning Curve
```python
# Costs decrease over time due to technological improvements
specific_effects={'costs': np.array([1200, 1000, 800, 650, 500])}
linked_periods=None  # Can invest at different times
```

### Long-lived Infrastructure
```python
# Single investment that persists across all periods
linked_periods=(0, 1)  # All periods linked
```

### Phased Rollout
```python
# Two deployment phases
linked_periods=np.array([1, 1, 1, 2, 2])  # Phase 1: periods 0-2, Phase 2: periods 3-4
```

### Replacement Cycles
```python
# Equipment with 2-period lifetime, can be replaced
linked_periods=np.array([1, 1, 2, 2, 3])  # Gen 1, Gen 2, Gen 3
```

## Tips

1. **Start simple**: Use `linked_periods=(0, 1)` for most long-lived assets
2. **Model reality**: Match linking to actual equipment lifecycles
3. **Cost annualization**: Ensure investment costs are properly annualized to the period duration
4. **Check results**: Verify the `invested` binary variable to understand investment timing
5. **Solver settings**: Multi-period MIP problems may need longer solve times or relaxed MIP gaps
