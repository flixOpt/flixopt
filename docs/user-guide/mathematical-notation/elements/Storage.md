# Storages
**Storages** have one incoming and one outgoing **[Flow](Flow.md)** with a charging and discharging efficiency.
A storage has a state of charge $c(\text{t}_i)$ which is limited by its `size` $\text C$ and relative bounds $\eqref{eq:Storage_Bounds}$.

$$ \label{eq:Storage_Bounds}
    \text C \cdot \text c^{\text{L}}_{\text{rel}}(\text t_{i})
    \leq c(\text{t}_i) \leq
    \text C \cdot \text c^{\text{U}}_{\text{rel}}(\text t_{i})
$$

Where:

- $\text C$ is the size of the storage
- $c(\text{t}_i)$ is the state of charge at time $\text{t}_i$
- $\text c^{\text{L}}_{\text{rel}}(\text t_{i})$ is the relative lower bound (typically 0)
- $\text c^{\text{U}}_{\text{rel}}(\text t_{i})$ is the relative upper bound (typically 1)

With $\text c^{\text{L}}_{\text{rel}}(\text t_{i}) = 0$ and $\text c^{\text{U}}_{\text{rel}}(\text t_{i}) = 1$,
Equation $\eqref{eq:Storage_Bounds}$ simplifies to

$$ 0 \leq c(\text t_{i}) \leq \text C $$

The state of charge $c(\text{t}_i)$ decreases by a fraction of the prior state of charge. The belonging parameter
$ \dot{ \text c}_\text{rel, loss}(\text{t}_i)$ expresses the "loss fraction per hour". The storage balance from  $\text{t}_i$ to $\text t_{i+1}$ is

$$
\begin{align*}
    c(\text{t}_{i+1}) &= c(\text{t}_{i}) \cdot (1-\dot{\text{c}}_\text{rel,loss}(\text{t}_i) \cdot \Delta \text{t}_{i}) \\
    &\quad + p_{f_\text{in}}(\text{t}_i) \cdot \Delta \text{t}_i \cdot \eta_\text{in}(\text{t}_i) \\
    &\quad - \frac{p_{f_\text{out}}(\text{t}_i) \cdot \Delta \text{t}_i}{\eta_\text{out}(\text{t}_i)}
    \tag{3}
\end{align*}
$$

Where:

- $c(\text{t}_{i+1})$ is the state of charge at time $\text{t}_{i+1}$
- $c(\text{t}_{i})$ is the state of charge at time $\text{t}_{i}$
- $\dot{\text{c}}_\text{rel,loss}(\text{t}_i)$ is the relative loss rate (self-discharge) per hour
- $\Delta \text{t}_{i}$ is the time step duration in hours
- $p_{f_\text{in}}(\text{t}_i)$ is the input flow rate at time $\text{t}_i$
- $\eta_\text{in}(\text{t}_i)$ is the charging efficiency at time $\text{t}_i$
- $p_{f_\text{out}}(\text{t}_i)$ is the output flow rate at time $\text{t}_i$
- $\eta_\text{out}(\text{t}_i)$ is the discharging efficiency at time $\text{t}_i$

---

## Mathematical Patterns Used

Storage formulation uses the following modeling patterns:

- **[Basic Bounds](modeling-patterns/bounds-and-states.md#basic-bounds)** - For charge state bounds (equation $\eqref{eq:Storage_Bounds}$)
- **[Scaled Bounds](modeling-patterns/bounds-and-states.md#scaled-bounds)** - For flow rate bounds relative to storage size

When combined with investment parameters, storage can use:
- **[Bounds with State](modeling-patterns/bounds-and-states.md#bounds-with-state)** - Investment decisions (see [InvestParameters](InvestParameters.md))

---

## Implementation

**Class:** [`Storage`][flixopt.components.Storage]

**Location:** `flixopt/components.py:237`

**Model Class:** [`StorageModel`][flixopt.components.StorageModel]

**Location:** `flixopt/components.py:800`

**Key Constraints:**
- Charge state bounds: `flixopt/components.py:~820`
- Storage balance equation (eq. 3): `flixopt/components.py:838-842`

**Variables Created:**
- `charge_state`: State of charge $c(\text{t}_i)$
- `charge_flow`: Input flow rate $p_{f_\text{in}}(\text{t}_i)$
- `discharge_flow`: Output flow rate $p_{f_\text{out}}(\text{t}_i)$

**Parameters:**
- `size`: Storage capacity $\text{C}$
- `relative_loss_per_hour`: Self-discharge rate $\dot{\text{c}}_\text{rel,loss}$
- `charge_state_start`: Initial charge $c(\text{t}_0)$
- `charge_state_end`: Final charge target $c(\text{t}_\text{end})$ (optional)
- `eta_charge`, `eta_discharge`: Charging/discharging efficiencies $\eta_\text{in}, \eta_\text{out}$

---

## See Also

- [Flow](Flow.md) - Input and output flow definitions
- [InvestParameters](InvestParameters.md) - Variable storage sizing
- [Modeling Patterns](modeling-patterns/index.md) - Mathematical building blocks
