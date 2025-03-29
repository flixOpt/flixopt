[`LinearConverters`][flixopt.components.LinearConverter] define a ratio between incoming and outgoing [Flows](#flows).

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
The conversion efficiency can be defined as a piecewise linear approximation. See [Piecewise](Piecewise.md) for more details.