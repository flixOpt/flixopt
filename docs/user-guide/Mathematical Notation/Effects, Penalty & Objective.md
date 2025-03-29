## Effects
[`Effects`][flixopt.effects.Effect] are used to allocate things like costs, emissions, or other "effects" occuring in the system.
These arise from so called **Shares**, which originate from **Elements** like [Flows](Flow.md).

**Example:**

[`Flows`][flixopt.elements.Flow] have an attribute called `effects_per_flow_hour`, defining the effect amount of per flow hour.
Assiziated effects could be:
- costs - given in [€/kWh]...
- ...or emissions - given in [kg/kWh].
- 
Effects are allocated seperatly for investments and operation.

### Shares to Effects

$$ \label{eq:Share_invest}
s_{l \rightarrow e, \text{inv}} = \sum_{v \in \mathcal{V}_{l, \text{inv}}} v \cdot \text a_{v \rightarrow e} 
$$

$$ \label{eq:Share_operation}
s_{l \rightarrow e, \text{op}}(\text{t}_i) = \sum_{v \in \mathcal{V}_{l,\text{op}}} v(\text{t}_i) \cdot \text a_{v \rightarrow e}(\text{t}_i)           
$$

With:

- $\text{t}_i$ being the time step
- $\mathcal{V_l}$ being the set of all optimization variables of element $e$
- $\mathcal{V}_{l, \text{inv}}$ being the set of all optimization variables of element $e$ related to investment
- $\mathcal{V}_{l, \text{op}}$ being the set of all optimization variables of element $e$ related to operation
- $v$ being an optimization variable of the element $l$
- $v(\text{t}_i)$ being an optimization variable of the element $l$ at timestep $\text{t}_i$
- $\text a_{v \rightarrow e}$ being the factor between the optimization variable $v$ to effect $e$
- $\text a_{v \rightarrow e}(\text{t}_i)$ being the factor between the optimization variable $v$ to effect $e$ for timestep $\text{t}_i$
- $s_{l \rightarrow e, \text{inv}}$ being the share of element $l$ to the investment part of effect $e$
- $s_{l \rightarrow e, \text{op}}(\text{t}_i)$ being the share of element $l$ to the operation part of effect $e$

### Shares between different Effects

Furthermore, the Effect $x$ can contribute a share to another Effect ${e} \in \mathcal{E}\backslash x$. 
This share is defined by the factor $\text r_{x \rightarrow e}$. 

For example, the Effect "CO$_2$ emissions" (unit: kg) 
can cause an additional share to Effect "monetary costs" (unit: €). 
In this case, the factor $\text a_{x \rightarrow e}$ is the specific CO$_2$ price in €/kg. However, circular references have to be avoided.

The overall sum of investment shares of an Effect $e$ is given by $\eqref{Effect_invest}$

$$ \label{eq:Effect_invest}
E_{e, \text{inv}} = 
\sum_{l \in \mathcal{L}} s_{l \rightarrow e,\text{inv}} + 
\sum_{x \in \mathcal{E}\backslash e} E_{x, \text{inv}}  \cdot \text{r}_{x \rightarrow  e,\text{inv}}
$$

The overall sum of operation shares is given by $\eqref{eq:Effect_Operation}$

$$ \label{eq:Effect_Operation}
E_{e, \text{op}}(\text{t}_{i}) = 
\sum_{l \in \mathcal{L}} s_{l \rightarrow e, \text{op}}(\text{t}_i) + 
\sum_{x \in \mathcal{E}\backslash e} E_{x, \text{op}}(\text{t}_i) \cdot \text{r}_{x \rightarrow {e},\text{op}}(\text{t}_i)
$$

and totals to $\eqref{eq:Effect_Operation_total}$
$$\label{eq:Effect_Operation_total}
E_{e,\text{op},\text{tot}} = \sum_{i=1}^n  E_{e,\text{op}}(\text{t}_{i})
$$

With:

- $\mathcal{L}$ being the set of all elements in the FlowSystem
- $\mathcal{E}$ being the set of all effects in the FlowSystem
- $\text r_{x \rightarrow e, \text{inv}}$ being the factor between the operation part of Effect $x$ and Effect $e$
- $\text r_{x \rightarrow e, \text{op}}(\text{t}_i)$ being the factor between the invest part of Effect $x$ and Effect $e$

- $\text{t}_i$ being the time step
- $s_{l \rightarrow e, \text{inv}}$ being the share of element $l$ to the investment part of effect $e$
- $s_{l \rightarrow e, \text{op}}(\text{t}_i)$ being the share of element $l$ to the operation part of effect $e$


The total of an effect $E_{e}$ is given as $\eqref{eq:Effect_Total}$

$$ \label{eq:Effect_Total}
E_{e} = E_{\text{inv},e} +E_{\text{op},\text{tot},e}
$$

### Constraining Effects

For each variable $v \in \{ E_{e,\text{inv}}, E_{e,\text{op},\text{tot}}, E_e\}$, a lower bound $v^\text{L}$ and upper bound $v^\text{U}$ can be defined as

$$ \label{eq:Bounds_Single}
\text v^\text{L} \leq v \leq \text v^\text{U}
$$

Furthermore, bounds for the operational shares can be set for each time step

$$ \label{eq:Bounds_Time_Steps}
\text E_{e,\text{op}}^\text{L}(\text{t}_i) \leq E_{e,\text{op}}(\text{t}_i) \leq \text E_{e,\text{op}}^\text{U}(\text{t}_i)
$$

## Penalty

Additionally to the user defined [Effects](#effects), a Penalty $\Phi$ is part of every FlixOpt Model.
Its used to prevent unsolvable problems and simplify troubleshooting.
Shares to the penalty can originate from every Element and are constructed similarly to 
$\eqref{Share_invest}$ and  $\eqref{Share_operation}$.

$$ \label{eq:Penalty}
\Phi = \sum_{l \in \mathcal{L}} \left( s_{l \rightarrow \Phi}  +\sum_{\text{t}_i \in \mathcal{T}} s_{l \rightarrow \Phi}(\text{t}_{i}) \right)
$$

With:

- $\mathcal{L}$ being the set of all elements in the FlowSystem
- $\mathcal{T}$ being the set of all timesteps
- $s_{l \rightarrow \Phi}$ being the share of element $l$ to the penalty

At the moment, penalties only occur in [Buses](#buses)

## Objective

The optimization objective of a FlixOpt Model is defined as $\eqref{eq:Objective}$
$$ \label{eq:Objective}
\min(E_{\Omega} + \Phi)
$$

With:

- $\Omega$ being the chosen **Objective [Effect](#effects)** (see $\eqref{eq:Effect_Total}$)
- $\Phi$ being the [Penalty](#penalty)

This approach allows for a multi-criteria optimization using both...
 - ... the **Weigted Sum**Method, as the chosen **Objective Effect** can incorporate other Effects.
 - ... the ($\epsilon$-constraint method) by constraining effects.