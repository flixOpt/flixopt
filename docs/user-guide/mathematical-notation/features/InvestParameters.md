# InvestParameters

**InvestParameters** model investment decisions in optimization problems, enabling both binary (invest/don't invest) and continuous sizing choices with comprehensive cost modeling.

!!! info "Quick Reference"

    | Aspect | Key Equation | Description |
    |--------|--------------|-------------|
    | **Binary Investment** | $v_\text{invest} = s_\text{invest} \cdot \text{size}_\text{fixed}$ | Fixed-size yes/no decision |
    | **Continuous Sizing** | $s_\text{invest} \cdot \text{size}_\text{min} \leq v_\text{invest} \leq s_\text{invest} \cdot \text{size}_\text{max}$ | Variable size with bounds |
    | **Total Effects** | $E = E_\text{fix} + E_\text{spec} + E_\text{pw} + E_\text{retirement}$ | Fixed + specific + piecewise + retirement |

---

## Mathematical Formulation

### Investment Decision Types

??? abstract "Binary Investment"

    Fixed-size investment creating a yes/no decision (e.g., install a 100 kW generator):

    $$\label{eq:invest_binary}
    v_\text{invest} = s_\text{invest} \cdot \text{size}_\text{fixed}
    $$

    - $v_\text{invest}$ - Resulting investment size
    - $s_\text{invest} \in \{0, 1\}$ - Binary investment decision
    - $\text{size}_\text{fixed}$ - Predefined component size

    **Behavior:**

    - $s_\text{invest} = 0$: No investment ($v_\text{invest} = 0$)
    - $s_\text{invest} = 1$: Invest at fixed size ($v_\text{invest} = \text{size}_\text{fixed}$)

??? abstract "Continuous Sizing"

    Variable-size investment with bounds (e.g., battery capacity from 10-1000 kWh):

    $$\label{eq:invest_continuous}
    s_\text{invest} \cdot \text{size}_\text{min} \leq v_\text{invest} \leq s_\text{invest} \cdot \text{size}_\text{max}
    $$

    - $v_\text{invest}$ - Investment size variable (continuous)
    - $s_\text{invest} \in \{0, 1\}$ - Binary investment decision
    - $\text{size}_\text{min}$ - Minimum investment size (if investing)
    - $\text{size}_\text{max}$ - Maximum investment size

    **Behavior:**

    - $s_\text{invest} = 0$: No investment ($v_\text{invest} = 0$)
    - $s_\text{invest} = 1$: Invest with size in $[\text{size}_\text{min}, \text{size}_\text{max}]$

    Uses the [bounds with state](../modeling-patterns/bounds-and-states.md#bounds-with-state) pattern.

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

### Effect Modeling

Investment effects (costs, emissions, etc.) are modeled using four components:

??? note "Fixed Effects"

    One-time effects incurred if investment is made, independent of size:

    $$\label{eq:invest_fixed_effects}
    E_{e,\text{fix}} = s_\text{invest} \cdot \text{fix}_e
    $$

    - $E_{e,\text{fix}}$ - Fixed contribution to effect $e$
    - $\text{fix}_e$ - Fixed effect value (e.g., fixed installation cost)

    **Examples:** Installation costs (permits, grid connection), one-time environmental impacts, fixed labor/administrative costs

??? note "Specific Effects"

    Effects proportional to investment size (per-unit costs):

    $$\label{eq:invest_specific_effects}
    E_{e,\text{spec}} = v_\text{invest} \cdot \text{spec}_e
    $$

    - $E_{e,\text{spec}}$ - Size-dependent contribution to effect $e$
    - $\text{spec}_e$ - Specific effect value per unit size (e.g., €/kW)

    **Examples:** Equipment costs (€/kW), material requirements (kg steel/kW), recurring costs (€/kW/year maintenance)

??? note "Piecewise Effects"

    Non-linear effect relationships using piecewise linear approximations:

    $$\label{eq:invest_piecewise_effects}
    E_{e,\text{pw}} = \sum_{k=1}^{K} \lambda_k \cdot r_{e,k}
    $$

    Subject to:
    $$
    v_\text{invest} = \sum_{k=1}^{K} \lambda_k \cdot v_k
    $$

    - $E_{e,\text{pw}}$ - Piecewise contribution to effect $e$
    - $\lambda_k$ - Piecewise lambda variables (see [Piecewise](Piecewise.md))
    - $r_{e,k}$ - Effect rate at piece $k$
    - $v_k$ - Size points defining the pieces

    **Use cases:** Economies of scale (bulk discounts), technology learning curves, threshold effects (capacity tiers with different costs)

    See [Piecewise](Piecewise.md) for detailed mathematical formulation.

??? note "Retirement Effects"

    Effects incurred if investment is NOT made (when retiring/not replacing existing equipment):

    $$\label{eq:invest_retirement_effects}
    E_{e,\text{retirement}} = (1 - s_\text{invest}) \cdot \text{retirement}_e
    $$

    - $E_{e,\text{retirement}}$ - Retirement contribution to effect $e$
    - $\text{retirement}_e$ - Retirement effect value

    **Behavior:**

    - $s_\text{invest} = 0$: Retirement effects are incurred
    - $s_\text{invest} = 1$: No retirement effects

    **Examples:** Demolition or disposal costs, decommissioning expenses, contractual penalties for not investing, opportunity costs or lost revenues

### Total Investment Effects

The total contribution to effect $e$ from an investment is:

$$\label{eq:invest_total_effects}
E_{e,\text{invest}} = E_{e,\text{fix}} + E_{e,\text{spec}} + E_{e,\text{pw}} + E_{e,\text{retirement}}
$$

Effects integrate into the overall system effects as described in [Effects, Penalty & Objective](../effects-penalty-objective.md).

---

### Integration with Components

Investment parameters modify component sizing:

**Without Investment:**
Component size is a fixed parameter:
$$
\text{size} = \text{size}_\text{nominal}
$$

**With Investment:**
Component size becomes a variable:
$$
\text{size} = v_\text{invest}
$$

This size variable then appears in component constraints. For example, flow rate bounds become:

$$
v_\text{invest} \cdot \text{rel}_\text{lower} \leq p(t) \leq v_\text{invest} \cdot \text{rel}_\text{upper}
$$

Using the [scaled bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds) pattern.

---

### Cost Annualization

!!! warning "Important: Time Horizon Matching"

    All investment cost values must be properly weighted to match the optimization model's time horizon.

For long-term investments, costs should be annualized:

$$\label{eq:annualization}
\text{cost}_\text{annual} = \frac{\text{cost}_\text{capital} \cdot r}{1 - (1 + r)^{-n}}
$$

- $\text{cost}_\text{capital}$ - Upfront investment cost
- $r$ - Discount rate
- $n$ - Equipment lifetime in years

!!! example "Annualization Example"

    €1,000,000 equipment with 20-year life and 5% discount rate:

    $$
    \text{cost}_\text{annual} = \frac{1{,}000{,}000 \cdot 0.05}{1 - (1.05)^{-20}} \approx €80{,}243/\text{year}
    $$

---

## Implementation

[:octicons-code-24: `InvestParameters`][flixopt.interface.InvestParameters]{ .md-button .md-button--primary }

### Key Parameters

| Parameter | Mathematical Symbol | Description |
|-----------|---------------------|-------------|
| `fixed_size` | $\text{size}_\text{fixed}$ | For binary investments (mutually exclusive with continuous) |
| `minimum_size`, `maximum_size` | $\text{size}_\text{min}$, $\text{size}_\text{max}$ | For continuous sizing |
| `mandatory` | - | Whether investment is required (default: `False`) |
| `effects_of_investment` | $\text{fix}_e$ | Fixed effects when investing |
| `effects_of_investment_per_size` | $\text{spec}_e$ | Per-unit effects proportional to size |
| `piecewise_effects_of_investment` | $r_{e,k}$ | Non-linear effect modeling |
| `effects_of_retirement` | $\text{retirement}_e$ | Effects for not investing |

!!! tip "Parameter Naming Update"
    Recent versions use more descriptive parameter names:

    - `effects_of_investment` replaces `fix_effects`
    - `effects_of_investment_per_size` replaces `specific_effects`
    - `piecewise_effects_of_investment` replaces `piecewise_effects`
    - `effects_of_retirement` replaces `divest_effects`

### Used In

- [:octicons-code-16: `Flow`][flixopt.elements.Flow] - Flexible capacity decisions
- [:octicons-code-16: `Storage`][flixopt.components.Storage] - Storage sizing optimization
- [:octicons-code-16: `LinearConverter`][flixopt.components.LinearConverter] - Converter capacity planning
- All components supporting investment decisions

---

## Examples

=== "Binary Investment"

    **Solar Panels** - Fixed 100 kW system

    ```python
    solar_investment = InvestParameters(
        fixed_size=100,  # 100 kW system
        mandatory=False,  # Optional investment (default)
        effects_of_investment={'cost': 25000},  # Installation costs
        effects_of_investment_per_size={'cost': 1200},  # €1200/kW
    )
    ```

    **Result:** Binary decision: invest in 100 kW system or don't invest.

=== "Continuous Sizing"

    **Battery Storage** - Variable capacity from 10-1000 kWh

    ```python
    battery_investment = InvestParameters(
        minimum_size=10,  # kWh
        maximum_size=1000,
        mandatory=False,  # Optional investment (default)
        effects_of_investment={'cost': 5000},  # Grid connection
        effects_of_investment_per_size={'cost': 600},  # €600/kWh
    )
    ```

    **Result:** Optimization chooses optimal battery size (or $0$ for no investment).

=== "With Retirement"

    **Boiler Replacement** - Replace or decommission

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

    **Result:** Choose between:

    - Invest: Pay investment costs (€15k + €400/kW)
    - Don't invest: Pay retirement costs (€8k demolition)

=== "Economies of Scale"

    **Battery with Piecewise Costs** - Bulk discounts

    ```python
    battery_investment = InvestParameters(
        minimum_size=10,
        maximum_size=1000,
        piecewise_effects_of_investment=PiecewiseEffects(
            piecewise_origin=Piecewise([
                Piece(0, 100),    # Small: 0-100 kWh
                Piece(100, 500),  # Medium: 100-500 kWh
                Piece(500, 1000), # Large: 500-1000 kWh
            ]),
            piecewise_shares={
                'cost': Piecewise([
                    Piece(800, 750),  # €800-750/kWh (small)
                    Piece(750, 600),  # €750-600/kWh (medium)
                    Piece(600, 500),  # €600-500/kWh (large, bulk discount)
                ])
            },
        ),
    )
    ```

    **Result:** Unit cost decreases with larger investment sizes (economies of scale).

=== "Mandatory Investment"

    **Equipment Upgrade** - Must replace aging infrastructure

    ```python
    transformer_upgrade = InvestParameters(
        minimum_size=500,  # kVA
        maximum_size=2000,
        mandatory=True,  # Investment required
        effects_of_investment={'cost': 50000},
        effects_of_investment_per_size={'cost': 300},
    )
    ```

    **Result:** Optimization must invest, only decides on optimal size (500-2000 kVA).

---

## See Also

- [Flow](../elements/Flow.md) - Using InvestParameters with flows
- [Storage](../elements/Storage.md) - Storage investment modeling
- [LinearConverter](../elements/LinearConverter.md) - Converter capacity planning
- [StatusParameters](StatusParameters.md) - Combining with operational constraints
- [Piecewise](Piecewise.md) - Non-linear cost modeling
- [Bounds with State](../modeling-patterns/bounds-and-states.md#bounds-with-state) - Mathematical pattern
- [Scaled Bounds](../modeling-patterns/bounds-and-states.md#scaled-bounds) - Mathematical pattern
- [Effects, Penalty & Objective](../effects-penalty-objective.md) - How effects integrate into objective
