
# Mathematical Notation

## Naming Conventions

flixOpt uses the following naming conventions:

- All optimization variables are denoted by italic letters (e.g., $x$, $y$, $z$)
- All parameters and constants are denoted by non italic small letters (e.g., $\text{a}$, $\text{b}$, $\text{c}$)
- All Sets are denoted by greek capital letters (e.g., $\mathcal{F}$, $\mathcal{E}$)
- All units of a set are denoted by greek small letters (e.g., $\mathcal{f}$, $\mathcal{e}$)
- The letter $i$ is used to denote an index (e.g., $i=1,\dots,\text n$)
- All time steps are denoted by the letter $\text{t}$ (e.g., $\text{t}_0$, $\text{t}_1$, $\text{t}_i$)

## Timesteps
Time steps are defined as a sequence of discrete time steps $\text{t}_i \in \mathcal{T} \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}\}$ (left-aligned in its timespan).
From this sequence, the corresponding time intervals $\Delta \text{t}_i \in \Delta \mathcal{T}$ are derived as 

$$\Delta \text{t}_i = \text{t}_{i+1} - \text{t}_i \quad \text{for} \quad i \in \{1, 2, \dots, \text{n}-1\}$$

The final time interval $\Delta \text{t}_\text n$ defaults to $\Delta \text{t}_\text n = \Delta \text{t}_{\text n-1}$, but is of course customizable.
Non-equidistant time steps are also supported.

## Buses

The balance equation for a bus is:

$$ \label{eq:bus_balance}
  \sum_{f_\text{in} \in \mathcal{F}_\text{in}} p_{f_\text{in}}(\text{t}_i) =
  \sum_{f_\text{out} \in \mathcal{F}_\text{out}} p_{f_\text{out}}(\text{t}_i)
$$

Optionally, a Bus can have a `excess_penalty_per_flow_hour` parameter, which allows to penalize the balance for missing or excess flow-rates.
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

## Flows

The flow-rate is the main optimization variable of the Flow. It's limited by the size of the Flow and relative bounds \eqref{eq:flow_rate}.

$$ \label{eq:flow_rate}
    \text P \cdot \text p^{\text{L}}_{\text{rel}}(\text{t}_{i})
    \leq p(\text{t}_{i}) \leq
    \text P \cdot \text p^{\text{U}}_{\text{rel}}(\text{t}_{i})
$$

With:

- $\text P$ being the size of the Flow
- $p(\text{t}_{i})$ being the flow-rate at time $\text{t}_{i}$
- $\text p^{\text{L}}_{\text{rel}}(\text{t}_{i})$ being the relative lower bound (typically 0)
- $\text p^{\text{U}}_{\text{rel}}(\text{t}_{i})$ being the relative upper bound (typically 1)

With $\text p^{\text{L}}_{\text{rel}}(\text{t}_{i}) = 0$ and $\text p^{\text{U}}_{\text{rel}}(\text{t}_{i}) = 1$,
equation \eqref{eq:flow_rate} simplifies to

$$
    0 \leq p(\text{t}_{i}) \leq \text P
$$


This mathematical Formulation can be extended or changed when using [OnOffParameters](#omoffparameters)
to define the On/Off state of the Flow, or [InvestParameters](#investments),
which changes the size of the Flow from a constant to an optimization variable.

## LinearConverters
[`LinearConverters`][flixOpt.components.LinearConverter] define a ratio between incoming and outgoing [Flows](#flows).

$$ \label{eq:Linear-Transformer-Ratio}
    \sum_{f_{\text{in}} \in \mathcal F_{in}} \text a_{f_{\text{in}}}(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = \sum_{f_{\text{out}} \in \mathcal F_{out}}  \text b_{f_\text{out}}(\text{t}_i) \cdot p_{f_\text{out}}(\text{t}_i)
$$

With:

- $\mathcal F_{in}$ and $\mathcal F_{out}$ being the set of all incoming and outgoing flows
- $p_{f_\text{in}}(\text{t}_i)$ and $p_{f_\text{out}}(\text{t}_i)$ being the flow-rate at time $\text{t}_i$ for flow $f_\text{in}$ and $f_\text{out}$, respectively
- $\text a_{f_\text{in}}(\text{t}_i)$ and $\text b_{f_\text{out}}(\text{t}_i)$ being the ratio of the flow-rate at time $\text{t}_i$ for flow $f_\text{in}$ and $f_\text{out}$, respectively

With one incoming **Flow** and one outgoing **Flow**, this can be simplified to: 

$$ \label{eq:Linear-Transformer-Ratio-simple}
    \text a(\text{t}_i) \cdot p_{f_\text{in}}(\text{t}_i) = p_{f_\text{out}}(\text{t}_i)
$$

where $\text a$ can be interpreted as the conversion efficiency of the **LinearTransformer**.
#### Piecewise Concersion factors
The conversion efficiency can be defined as a piecewise function.

## Effects
## Features
### InvestParameters
### OnOffParameters

## Calculation Modes