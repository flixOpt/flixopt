# StatusParameters

StatusParameters add on/off behavior to flows — startup costs, minimum run times, cycling limits.

## Basic: Binary Status

A status variable $s(t) \in \{0, 1\}$ controls whether equipment is active:

```python
generator = fx.Flow(
    label='power', bus=elec_bus, size=100,
    relative_minimum=0.4,  # 40% min when ON
    status_parameters=fx.StatusParameters(
        effects_per_startup={'costs': 25000},  # €25k per startup
    ),
)
```

When $s(t) = 0$: flow is zero. When $s(t) = 1$: flow bounds apply.

---

## Startup Tracking

Detect transitions: $s^{start}(t) - s^{stop}(t) = s(t) - s(t-1)$

=== "Startup Costs"

    ```python
    fx.StatusParameters(
        effects_per_startup={'costs': 25000},
    )
    ```

=== "Running Costs"

    ```python
    fx.StatusParameters(
        effects_per_active_hour={'costs': 100},  # €/h while on
    )
    ```

=== "Startup Limit"

    ```python
    fx.StatusParameters(
        startup_limit=20,  # Max 20 starts per period
    )
    ```

---

## Duration Constraints

=== "Min Uptime"

    Once on, must stay on for minimum duration:

    $s^{start}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} s(j) \geq T_{up}^{min}$

    ```python
    fx.StatusParameters(min_uptime=8)  # 8 hours minimum
    ```

=== "Min Downtime"

    Once off, must stay off for minimum duration:

    $s^{stop}(t) = 1 \Rightarrow \sum_{j=t}^{t+k} (1 - s(j)) \geq T_{down}^{min}$

    ```python
    fx.StatusParameters(min_downtime=4)  # 4 hours cooling
    ```

=== "Max Uptime"

    Force shutdown after limit:

    $\sum_{j=t-k}^{t} s(j) \leq T_{up}^{max}$

    ```python
    fx.StatusParameters(max_uptime=18)  # Max 18h continuous
    ```

=== "Total Hours"

    Limit total operating hours per period:

    $H^{min} \leq \sum_t s(t) \cdot \Delta t \leq H^{max}$

    ```python
    fx.StatusParameters(
        active_hours_min=2000,
        active_hours_max=5000,
    )
    ```

---

## Reference

| Symbol | Type | Description |
|--------|------|-------------|
| $s(t)$ | $\{0, 1\}$ | Binary status (0=off, 1=on) |
| $s^{start}(t)$ | $\{0, 1\}$ | Startup indicator |
| $s^{stop}(t)$ | $\{0, 1\}$ | Shutdown indicator |
| $T_{up}^{min}$ | $\mathbb{Z}_{\geq 0}$ | Min uptime (`min_uptime`) |
| $T_{up}^{max}$ | $\mathbb{Z}_{\geq 0}$ | Max uptime (`max_uptime`) |
| $T_{down}^{min}$ | $\mathbb{Z}_{\geq 0}$ | Min downtime (`min_downtime`) |
| $H^{min}$ | $\mathbb{R}_{\geq 0}$ | Min total active hours (`active_hours_min`) |
| $H^{max}$ | $\mathbb{R}_{\geq 0}$ | Max total active hours (`active_hours_max`) |
| $\Delta t$ | $\mathbb{R}_{> 0}$ | Timestep duration (hours) |

**Classes:** [`StatusParameters`][flixopt.interface.StatusParameters], [`StatusModel`][flixopt.features.StatusModel]
