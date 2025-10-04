A Bus is a simple nodal balance between its incoming and outgoing flow rates.

$$ \label{eq:bus_balance}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i)
$$

Optionally, a Bus can have a `excess_penalty_per_flow_hour` parameter, which allows to penaltize the balance for missing or excess flow-rates.
This is usefull as it handles a possible ifeasiblity gently.

This changes the balance to

$$ \label{eq:bus_balance-excess}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_ \text{in}}(\text{t}_i) + \phi_\text{in}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i) + \phi_\text{out}(\text{t}_i)
$$

The penalty term is defined as

$$ \label{eq:bus_penalty}
  s_{b \rightarrow \Phi}(\text{t}_i) =
      \text a_{b \rightarrow \Phi}(\text{t}_i) \cdot \Delta \text{t}_i
      \cdot [ \phi_\text{in}(\text{t}_i) + \phi_\text{out}(\text{t}_i) ]
$$

With:

- $\mathcal{F}_\text{in}$ and $\mathcal{F}_\text{out}$ being the set of all incoming and outgoing flows
- $p_{f_\text{in}}(\text{t}_i)$ and $p_{f_\text{out}}(\text{t}_i)$ being the flow-rate at time $\text{t}_i$ for flow $f_\text{in}$ and $f_\text{out}$, respectively
- $\phi_\text{in}(\text{t}_i)$ and $\phi_\text{out}(\text{t}_i)$ being the missing or excess flow-rate at time $\text{t}_i$, respectively
- $\text{t}_i$ being the time step
- $s_{b \rightarrow \Phi}(\text{t}_i)$ being the penalty term
- $\text a_{b \rightarrow \Phi}(\text{t}_i)$ being the penalty coefficient (`excess_penalty_per_flow_hour`)

---

## Implementation

**Class:** [`Bus`][flixopt.elements.Bus]

**Location:** `flixopt/elements.py:120`

**Model Class:** [`BusModel`][flixopt.elements.BusModel]

**Location:** `flixopt/elements.py:736`

**Key Constraints:**
- Bus balance equation (eq. $\eqref{eq:bus_balance}$ or $\eqref{eq:bus_balance-excess}$): `flixopt/elements.py:751`
- Excess/deficit bounds (when applicable): `flixopt/elements.py:763`

**Variables Created:**
- No additional variables for strict balance
- `excess_input`, `excess_output`: Excess/deficit variables $\phi_\text{in}(\text{t}_i), \phi_\text{out}(\text{t}_i)$ (when penalty is specified)

**Parameters:**
- `excess_penalty_per_flow_hour`: Penalty coefficient $\text{a}_{b \rightarrow \Phi}$ for balance violations

**Penalty Contribution:**
When excess is allowed, the penalty term $s_{b \rightarrow \Phi}(\text{t}_i)$ contributes to the overall system penalty $\Phi$ as described in [Effects, Penalty & Objective](Effects,%20Penalty%20&%20Objective.md#penalty).

---

## See Also

- [Flow](../elements/Flow.md) - Definition of flow rates in the balance
- [Effects, Penalty & Objective](Effects,%20Penalty%20&%20Objective.md) - How penalties are included in the objective function
- [Modeling Patterns](../modeling-patterns/index.md) - Mathematical building blocks
