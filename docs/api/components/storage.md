# Mathematical Formulation

**Storages** have one incoming and one outgoing **Flow** - $f_\text{in}$ and $f_\text{out}$ -
each with an efficiency $\eta_\text{in}$ and $\eta_\text{out}$.
Further, storages have a `size` $\text C$ and a state of charge $c(\text{t}_i)$.
Similarly to the flow-rate $p(\text{t}_i)$ of a [`Flow`][flixOpt.elements.Flow],
the `size` $\text C$ combined with a relative upper bound
$\text c^{\text{U}}_\text{rel}(\text t_{i})$ and lower bound $\text c^{\text{L}}_\text{rel}(\text t_{i})$
limits the state of charge $c(\text{t}_i)$ by $\eqref{eq:Storage_Bounds}$.

$$ \label{eq:Storage_Bounds}
    \text C \cdot \text c^{\text{L}}_{\text{rel}}(\text t_{i})
    \leq c(\text{t}_i) \leq
    \text C \cdot \text c^{\text{U}}_{\text{rel}}(\text t_{i})
$$

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

::: flixOpt.components.Storage


## Creating a Storage

```python
import flixOpt as fx

thermal_storage = fx.Storage(
    label="ThermalStorage",
    charging=fx.Flow("charging", "Wärme", size=100),
    discharging=fx.Flow("discharging", "Wärme", size=100),
    capacity_in_flow_hours=1000,  # 1000 kWh capacity
    relative_loss_per_hour=0.01,  # 1% loss per hour
    eta_charge=0.95,              # 95% charging efficiency
    eta_discharge=0.95            # 95% discharging efficiency
)
```
