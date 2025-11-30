# InvestmentParameters

InvestmentParameters extend [SizingParameters](SizingParameters.md) to model WHEN to invest, not just how much.

!!! info "Relationship to SizingParameters"
    - **SizingParameters**: Determines capacity size (how much to build)
    - **InvestmentParameters**: Adds timing decisions (when to invest) with fixed lifetime

## Key Concept: Investment Timing

InvestmentParameters tracks:

1. **When** the investment occurs (at most once)
2. **How long** the investment is active (fixed lifetime)
3. **When** decommissioning occurs (lifetime periods after investment)

$$
\sum_{p} x^{invest}_p \leq 1 \quad \text{(invest at most once)}
$$

$$
x^{active}_p = 1 \iff \exists p' \leq p : x^{invest}_{p'} = 1 \land p - p' < L
$$

where $L$ is the lifetime in periods.

---

## Basic Usage

```python
solar_timing = fx.InvestmentParameters(
    lifetime=10,  # Investment lasts 10 periods
    minimum_size=50,
    maximum_size=200,
    effects_of_size={'costs': 15000},
    effects_per_size={'costs': 1200},
)
```

---

## Timing Controls

=== "Allow Investment"

    Restrict when investment can occur:

    ```python
    import xarray as xr

    fx.InvestmentParameters(
        lifetime=10,
        allow_investment=xr.DataArray(
            [1, 1, 1, 0, 0],  # Only allow in first 3 periods
            coords=[('period', [2020, 2025, 2030, 2035, 2040])],
        ),
    )
    ```

=== "Force Investment"

    Force investment in a specific period:

    ```python
    fx.InvestmentParameters(
        lifetime=10,
        force_investment=xr.DataArray(
            [0, 0, 1, 0, 0],  # Force in 2030
            coords=[('period', [2020, 2025, 2030, 2035, 2040])],
        ),
    )
    ```

=== "Previous Lifetime"

    Model existing capacity from before the optimization horizon:

    ```python
    fx.InvestmentParameters(
        lifetime=10,
        previous_lifetime=3,  # 3 periods of lifetime remaining
    )
    ```

---

## Investment-Period Effects

Effects that depend on WHEN the investment is made (technology learning curves, time-varying subsidies):

=== "Fixed by Investment Period"

    Effects that vary by investment timing:

    ```python
    # Cost decreases over time (learning curve)
    fx.InvestmentParameters(
        lifetime=10,
        effects_of_investment={
            'costs': xr.DataArray(
                [50000, 45000, 40000, 35000, 30000],
                coords=[('period', [2020, 2025, 2030, 2035, 2040])],
            )
        },
    )
    ```

=== "Per-Size by Investment Period"

    Size-dependent effects that vary by investment timing:

    ```python
    # Cost per kW decreases (technology improvement)
    fx.InvestmentParameters(
        lifetime=10,
        effects_of_investment_per_size={
            'costs': xr.DataArray(
                [1500, 1200, 1000, 900, 800],  # €/kW over time
                coords=[('period', [2020, 2025, 2030, 2035, 2040])],
            )
        },
    )
    ```

---

## Variables Created

| Variable | Description |
|----------|-------------|
| `size` | Capacity size |
| `invested` | Binary: is investment currently active? |
| `investment_occurs` | Binary: does investment happen this period? |
| `decommissioning_occurs` | Binary: does decommissioning happen this period? |
| `size_increase` | Size increase when investing |
| `size_decrease` | Size decrease when decommissioning |

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $P$ | $\mathbb{R}_{\geq 0}$ | Investment size (capacity) |
| $x^{invest}_p$ | $\{0, 1\}$ | Investment occurs in period $p$ |
| $x^{decom}_p$ | $\{0, 1\}$ | Decommissioning occurs in period $p$ |
| $x^{active}_p$ | $\{0, 1\}$ | Investment is active in period $p$ |
| $L$ | $\mathbb{Z}_{>0}$ | Lifetime in periods |

**Classes:** [`InvestmentParameters`][flixopt.interface.InvestmentParameters], [`InvestmentModel`][flixopt.features.InvestmentModel]
