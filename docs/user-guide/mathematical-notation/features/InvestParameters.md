# InvestParameters

[`InvestParameters`][flixopt.interface.InvestParameters] model investment decisions in optimization problems, enabling both binary (invest/don't invest) and continuous sizing choices with comprehensive cost modeling.

## Investment Decision Types

FlixOpt supports two main types of investment decisions:

### Binary Investment

Fixed-size investment creating a yes/no decision (e.g., install a 100 kW generator):

$$\label{eq:invest_binary}
v_\text{invest} = s_\text{invest} \cdot \text{size}_\text{fixed}
$$

With:

- $v_\text{invest}$ being the resulting investment size
- $s_\text{invest} \in \{0, 1\}$ being the binary investment decision
- $\text{size}_\text{fixed}$ being the predefined component size

**Behavior:**
- $s_\text{invest} = 0$: no investment ($v_\text{invest} = 0$)
- $s_\text{invest} = 1$: invest at fixed size ($v_\text{invest} = \text{size}_\text{fixed}$)

---

### Continuous Sizing

Variable-size investment with bounds (e.g., battery capacity from 10-1000 kWh):

$$\label{eq:invest_continuous}
s_\text{invest} \cdot \text{size}_\text{min} \leq v_\text{invest} \leq s_\text{invest} \cdot \text{size}_\text{max}
$$

With:

- $v_\text{invest}$ being the investment size variable (continuous)
- $s_\text{invest} \in \{0, 1\}$ being the binary investment decision
- $\text{size}_\text{min}$ being the minimum investment size (if investing)
- $\text{size}_\text{max}$ being the maximum investment size

**Behavior:**
- $s_\text{invest} = 0$: no investment ($v_\text{invest} = 0$)
- $s_\text{invest} = 1$: invest with size in $[\text{size}_\text{min}, \text{size}_\text{max}]$

This uses the **bounds with state** pattern described in [Bounds and States](../modeling-patterns/bounds-and-states.md#bounds-with-state).

---

### Optional vs. Mandatory Investment

The `mandatory` parameter controls whether investment is required:

**Optional Investment** (`mandatory=False`, default):
$$\label{eq:invest_optional}
s_\text{invest} \in \{0, 1\}
$$

The optimization can freely choose to invest or not.

**Mandatory Investment** (`mandatory=True`):
$$\label{eq:invest_mandatory}
s_\text{invest} = 1
$$

The investment must occur (useful for mandatory upgrades or replacements).

---

## Effect Modeling

Investment effects (costs, emissions, etc.) are modeled using three components:

### Fixed Effects

One-time effects incurred if investment is made, independent of size:

$$\label{eq:invest_fixed_effects}
E_{e,\text{fix}} = s_\text{invest} \cdot \text{fix}_e
$$

With:

- $E_{e,\text{fix}}$ being the fixed contribution to effect $e$
- $\text{fix}_e$ being the fixed effect value (e.g., fixed installation cost)

**Examples:**
- Fixed installation costs (permits, grid connection)
- One-time environmental impacts (land preparation)
- Fixed labor or administrative costs

---

### Specific Effects

Effects proportional to investment size (per-unit costs):

$$\label{eq:invest_specific_effects}
E_{e,\text{spec}} = v_\text{invest} \cdot \text{spec}_e
$$

With:

- $E_{e,\text{spec}}$ being the size-dependent contribution to effect $e$
- $\text{spec}_e$ being the specific effect value per unit size (e.g., €/kW)

**Examples:**
- Equipment costs (€/kW)
- Material requirements (kg steel/kW)
- Recurring costs (€/kW/year maintenance)

---

### Piecewise Effects

Non-linear effect relationships using piecewise linear approximations:

$$\label{eq:invest_piecewise_effects}
E_{e,\text{pw}} = \sum_{k=1}^{K} \lambda_k \cdot r_{e,k}
$$

Subject to:
$$
v_\text{invest} = \sum_{k=1}^{K} \lambda_k \cdot v_k
$$

With:

- $E_{e,\text{pw}}$ being the piecewise contribution to effect $e$
- $\lambda_k$ being the piecewise lambda variables (see [Piecewise](../features/Piecewise.md))
- $r_{e,k}$ being the effect rate at piece $k$
- $v_k$ being the size points defining the pieces

**Use cases:**
- Economies of scale (bulk discounts)
- Technology learning curves
- Threshold effects (capacity tiers with different costs)

See [Piecewise](../features/Piecewise.md) for detailed mathematical formulation.

---

### Retirement Effects

Effects incurred if investment is NOT made (when retiring/not replacing existing equipment):

$$\label{eq:invest_retirement_effects}
E_{e,\text{retirement}} = (1 - s_\text{invest}) \cdot \text{retirement}_e
$$

With:

- $E_{e,\text{retirement}}$ being the retirement contribution to effect $e$
- $\text{retirement}_e$ being the retirement effect value

**Behavior:**
- $s_\text{invest} = 0$: retirement effects are incurred
- $s_\text{invest} = 1$: no retirement effects

**Examples:**
- Demolition or disposal costs
- Decommissioning expenses
- Contractual penalties for not investing
- Opportunity costs or lost revenues

---

### Total Investment Effects

The total contribution to effect $e$ from an investment is:

$$\label{eq:invest_total_effects}
E_{e,\text{invest}} = E_{e,\text{fix}} + E_{e,\text{spec}} + E_{e,\text{pw}} + E_{e,\text{retirement}}
$$

Effects integrate into the overall system effects as described in [Effects, Penalty & Objective](../effects-penalty-objective.md).

---

## Integration with Components

Investment parameters modify component sizing:

### Without Investment
Component size is a fixed parameter:
$$
\text{size} = \text{size}_\text{nominal}
$$

### With Investment
Component size becomes a variable:
$$
\text{size} = v_\text{invest}
$$

This size variable then appears in component constraints. For example, flow rate bounds become:

$$
v_\text{invest} \cdot \text{rel}_\text{lower} \leq p(t) \leq v_\text{invest} \cdot \text{rel}_\text{upper}
$$

Using the **scaled bounds** pattern from [Bounds and States](../modeling-patterns/bounds-and-states.md#scaled-bounds).

---

## Cost Annualization

**Important:** All investment cost values must be properly weighted to match the optimization model's time horizon.

For long-term investments, costs should be annualized:

$$\label{eq:annualization}
\text{cost}_\text{annual} = \frac{\text{cost}_\text{capital} \cdot r}{1 - (1 + r)^{-n}}
$$

With:

- $\text{cost}_\text{capital}$ being the upfront investment cost
- $r$ being the discount rate
- $n$ being the equipment lifetime in years

**Example:** €1,000,000 equipment with 20-year life and 5% discount rate
$$
\text{cost}_\text{annual} = \frac{1{,}000{,}000 \cdot 0.05}{1 - (1.05)^{-20}} \approx €80{,}243/\text{year}
$$

---

## Implementation

**Python Class:** [`InvestParameters`][flixopt.interface.InvestParameters]

**Key Parameters:**

- `fixed_size`: For binary investments (mutually exclusive with continuous sizing)
- `minimum_size`, `maximum_size`: For continuous sizing
- `mandatory`: Whether investment is required (default: `False`)
- `effects_of_investment`: Fixed effects incurred when investing (replaces deprecated `fix_effects`)
- `effects_of_investment_per_size`: Per-unit effects proportional to size (replaces deprecated `specific_effects`)
- `piecewise_effects_of_investment`: Non-linear effect modeling (replaces deprecated `piecewise_effects`)
- `effects_of_retirement`: Effects for not investing (replaces deprecated `divest_effects`)

See the [`InvestParameters`][flixopt.interface.InvestParameters] API documentation for complete parameter list and usage examples.

**Used in:**
- [`Flow`][flixopt.elements.Flow] - Flexible capacity decisions
- [`Storage`][flixopt.components.Storage] - Storage sizing optimization
- [`LinearConverter`][flixopt.components.LinearConverter] - Converter capacity planning
- All components supporting investment decisions

---

## Examples

### Binary Investment (Solar Panels)
```python
solar_investment = InvestParameters(
    fixed_size=100,  # 100 kW system
    mandatory=False,  # Optional investment (default)
    effects_of_investment={'cost': 25000},  # Installation costs
    effects_of_investment_per_size={'cost': 1200},  # €1200/kW
)
```

### Continuous Sizing (Battery)
```python
battery_investment = InvestParameters(
    minimum_size=10,  # kWh
    maximum_size=1000,
    mandatory=False,  # Optional investment (default)
    effects_of_investment={'cost': 5000},  # Grid connection
    effects_of_investment_per_size={'cost': 600},  # €600/kWh
)
```

### With Retirement Costs (Replacement)
```python
boiler_replacement = InvestParameters(
    minimum_size=50,  # kW
    maximum_size=200,
    mandatory=False,  # Optional investment (default)
    effects_of_investment={'cost': 15000},
    effects_of_investment_per_size={'cost': 400},
    effects_of_retirement={'cost': 8000},  # Demolition if not replaced
)
```

### Economies of Scale (Piecewise)
```python
battery_investment = InvestParameters(
    minimum_size=10,
    maximum_size=1000,
    piecewise_effects_of_investment=PiecewiseEffects(
        piecewise_origin=Piecewise([
            Piece(0, 100),    # Small
            Piece(100, 500),  # Medium
            Piece(500, 1000), # Large
        ]),
        piecewise_shares={
            'cost': Piecewise([
                Piece(800, 750),  # €800-750/kWh
                Piece(750, 600),  # €750-600/kWh
                Piece(600, 500),  # €600-500/kWh (bulk discount)
            ])
        },
    ),
)
```
