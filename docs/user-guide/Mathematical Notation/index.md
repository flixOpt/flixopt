
# Mathematical Notation

## Naming Conventions

FlixOpt uses the following naming conventions:

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
