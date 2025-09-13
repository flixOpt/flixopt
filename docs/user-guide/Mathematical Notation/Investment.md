# Investments

## Current state
$$
\beta_{\text{invest}} \cdot \text{max}(\epsilon, \text V^{\text L}) \leq V \leq  \beta_{\text{invest}} \cdot \text V^{\text U}
$$
With:
- $V$ = size
- $V^{\text L}$ = minimum size
- $V^{\text U}$ = maximum size
- $\epsilon$ = epsilon, a small number (such as $1e^{-5}$)
- $\beta_{invest} \in {0,1}$ = wether the size is invested or not

_Please edit the use cases as needed_
## Quickfix 1: Optimize the single best size overall
### Single variable
This is already possible and should be, as this is a needed use case
An additional factor to when the size is actually available might me practical (Which indicates the (fixed) time of investment)
## Math
$$
V(p) = V * a(p)
$$
with:
- $V$ = size
- $a(p)$ = factor for availlability per period

Factor $a(p)$ is simply multiplied with relative minimum or maximum(t). This is already possible by doing this yourself.
Effectively, the relative minimum or maximum are altered before using the same constraiints as before.
THis might lead to some issues regariding minimum_load factor, or others, as the size is not 0 in a scenario where the component cant produce.
**Therefore this might not be the best choice. See (#Variable per Scenario)

## Variable per Scenario
- **size** and **invest** as a variable per period $V(s)$ and $\beta_{invest}(s)$
- with scenario $s \in S$

### Usecase 1: Optimize the size for each Scenario independently
Restrictions are seperatly for each scenario
No changes needed. This could be the default behaviour.

### Usecase 2: Optimize ONE size for ALL scenarios
The size is the same globally, but not a scalar, but a variable per scenario $V(s)$
#### 2a: The same size in all scenarios
$$
V(s) = V(s') \quad \forall s,s' \in S
$$

With:
- $V(s)$ and $V(s')$ = size
- $S$ = set of scenarios

#### 2b: The same size, but can be 0 prior to the first increment
- Find the Optimal time of investment.
- Force an investment in a certain scenario (parameter optional as a list/array ob booleans)
- Combine optional and minimum/maximum size to force an investment inside a range if scenarios

$$
\beta_{\text{invest}}(s) \leq \beta_{\text{invest}}(s+1) \quad \forall s \in \{1,2,\ldots,S-1\}
$$

$$
V(s') - V(s) \leq M \cdot (2 - \beta_{\text{invest}}(s) - \beta_{\text{invest}}(s')) \quad \forall s, s' \in S
$$
$$
V(s') - V(s) \geq M \cdot (2 - \beta_{\text{invest}}(s) - \beta_{\text{invest}}(s')) \quad \forall s, s' \in S
$$

This could be the default behaviour. (which would be consistent with other variables)


### Switch

$$
\begin{aligned}
& \text{SWITCH}_s \in \{0,1\} \quad \forall s \in \{1,2,\ldots,S\} \\
& \sum_{s=1}^{S} \text{SWITCH}_s = 1 \\
& \beta_{\text{invest}}(s) = \sum_{s'=1}^{s} \text{SWITCH}_{s'} \quad \forall s \in \{1,2,\ldots,S\} \\
\end{aligned}
$$

$$
\begin{aligned}
& V(s) \leq V_{\text{actual}} \quad \forall s \in \{1,2,\ldots,S\} \\
& V(s) \geq V_{\text{actual}} - M \cdot (1 - \beta_{\text{invest}}(s)) \quad \forall s \in \{1,2,\ldots,S\}
\end{aligned}
$$




### Usecase 3: Find the best scenario to increment the size (Timing of the investment)
The size can only increment once (based on a starting point). This allows to optimize the timing of an investment.
#### Math
Treat $\beta_{invest}$ like an ON/OFF variable, and introduce a SwitchOn, that can only be active once.

*Thoughts:*
- Treating $\beta_{invest}$ like an ON/OFF variable suggest using the already presentconstraints linked to On/OffModel
- The timing could be constraint to be first in scenario x, or last in scenario y
- Restrict the number of consecutive scenarios
THis might needs the OnOffModel to be more generic (HOURS). Further, the span between scenarios needs to be weighted (like dt_in_hours), or the scenarios need to be measureable (integers)


### Others

#### Usecase 4: Only increase/decrease the size
Start from a certain size. For each scenario, the size can increase, but never decrease. (Or the other way around).
This would mean that a size expansion is possible,

#### Usecase 5: Restrict the increment in size per scenario
Restrict how much the size can increase/decrease for in scenario, based on the prior scenario.





Many more are possible
