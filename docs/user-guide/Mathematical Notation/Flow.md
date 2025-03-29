The flow_rate is the main optimization variable of the Flow. It's limited by the size of the Flow and relative bounds \eqref{eq:flow_rate}.

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


This mathematical Formulation can be extended or changed when using [OnOffParameters](#onoffparameters)
to define the On/Off state of the Flow, or [InvestParameters](#investments),
which changes the size of the Flow from a constant to an optimization variable.