# Piecewise

Piecewise enables modeling of non-linear relationships through piecewise linear approximations while maintaining problem linearity, consisting of a collection of Pieces that define valid ranges for variables.

=== "Variables"

    | Symbol | Python Name | Description | Domain | Created When |
    |--------|-------------|-------------|--------|--------------|
    | $\beta_k$ | `beta` (per Piece) | Binary variable indicating if piece $k$ is active | $\{0,1\}$ | Always (for each Piece) |
    | $\beta_\text{zero}$ | `zero_point` | Binary variable allowing all variables to be zero | $\{0,1\}$ | `zero_point=True` specified |
    | $\lambda_{0,k}$ | `lambda0` (per Piece) | Fraction of start point $\text{v}_{\text{start},k}$ that is active | $[0,1]$ | Always (for each Piece) |
    | $\lambda_{1,k}$ | `lambda1` (per Piece) | Fraction of end point $\text{v}_{\text{end},k}$ that is active | $[0,1]$ | Always (for each Piece) |

=== "Constraints"

    **Active piece definition** (always active for each piece):

    $$\label{eq:active_piece}
    \beta_k = \lambda_{0,k} + \lambda_{1,k}
    $$

    Binary variable $\beta_k$ is 1 if piece $k$ is active (either $\lambda_{0,k}$ or $\lambda_{1,k}$ is non-zero), 0 otherwise.

    ---

    **Variable definition through piece** (always active for each variable):

    $$\label{eq:piece}
    v_k = \lambda_{0,k} \cdot \text{v}_{\text{start},k} + \lambda_{1,k} \cdot \text{v}_{\text{end},k}
    $$

    The variable value is the weighted sum of the piece's start and end points.

    ---

    **Single active piece** (when `zero_point=False` or not specified):

    $$\label{eq:piecewise_in_pieces}
    \sum_{k=1}^K \beta_k = 1
    $$

    Exactly one piece must be active. This ensures $v \in [\text{v}_{\text{start},k}, \text{v}_{\text{end},k}]$ for some $k$.

    ---

    **Optional zero with single active piece** (when `zero_point=True`):

    $$\label{eq:piecewise_in_pieces_zero}
    \sum_{k=1}^K \beta_k = \beta_\text{zero}
    $$

    Either one piece is active ($\beta_\text{zero} = 1$) or all are inactive ($\beta_\text{zero} = 0$, forcing all $\lambda$ to zero). This allows $v \in \{0\} \cup [\text{v}_{\text{start},k}, \text{v}_{\text{end},k}]$.

    ---

    **Combined piecewise relationship** (when multiple variables share pieces):

    $$\label{eq:piecewise_combined}
    \begin{align}
    v_1 &= \sum_{k=1}^K (\lambda_{0,k} \cdot \text{v}_{1,\text{start},k} + \lambda_{1,k} \cdot \text{v}_{1,\text{end},k}) \\
    v_2 &= \sum_{k=1}^K (\lambda_{0,k} \cdot \text{v}_{2,\text{start},k} + \lambda_{1,k} \cdot \text{v}_{2,\text{end},k})
    \end{align}
    $$

    Multiple variables share the same $\lambda$ and $\beta$ variables, creating coupled non-linear relationships.

    **Mathematical Patterns:** [SOS Type 2 Constraints](../modeling-patterns/bounds-and-states.md), [Piecewise Linear Approximation](../modeling-patterns/bounds-and-states.md)

=== "Parameters"

    | Symbol | Python Parameter | Description | Default |
    |--------|------------------|-------------|---------|
    | $K$ | - | Number of pieces | From `pieces` list length |
    | $\text{v}_{\text{end},k}$ | - | End point of piece $k$ | From `Piece.end` |
    | $\text{v}_{\text{start},k}$ | - | Start point of piece $k$ | From `Piece.start` |
    | - | `pieces` | List of Piece objects defining the linear segments | Required |
    | - | `zero_point` | Allow all variables to be zero | False |

=== "Use Cases"

    ## Continuous Efficiency Curve (Touching Pieces)

    ```python
    from flixopt import Piecewise, Piece

    efficiency_curve = Piecewise([
        Piece((0, 0), (25, 25)),    # Low load: 0-25 MW
        Piece((25, 25), (75, 75)),  # Medium load: 25-75 MW (touches at 25)
        Piece((75, 75), (100, 100)), # High load: 75-100 MW (touches at 75)
    ])
    ```

    **Variables:** $\beta_1, \beta_2, \beta_3$ (piece indicators), $\lambda_{0,1}, \lambda_{1,1}, ..., \lambda_{0,3}, \lambda_{1,3}$ (lambda variables)

    **Constraints:** $\eqref{eq:active_piece}$ for each piece, $\eqref{eq:piece}$ defining $v$, $\eqref{eq:piecewise_in_pieces}$ ensuring exactly one piece active

    **Behavior:** Creates smooth continuous function without gaps, allowing operation anywhere from 0-100 MW.

    ---

    ## Forbidden Operating Range (Gap Between Pieces)

    ```python
    from flixopt import Piecewise, Piece

    turbine_operation = Piecewise([
        Piece((0, 0), (0, 0)),      # Off state (point)
        Piece((40, 40), (100, 100)), # Operating range (gap: 0-40 forbidden)
    ])
    ```

    **Variables:** $\beta_1, \beta_2$, $\lambda_{0,1}, \lambda_{1,1}, \lambda_{0,2}, \lambda_{1,2}$

    **Constraints:** $\eqref{eq:active_piece}$, $\eqref{eq:piece}$, $\eqref{eq:piecewise_in_pieces}$

    **Behavior:** Equipment must be either completely off (0) or operating between 40-100 MW. The range 0-40 MW is forbidden due to the gap.

    ---

    ## Variable COP Heat Pump (Two Coupled Variables)

    ```python
    from flixopt import LinearConverter, Flow, Piecewise, Piece, PiecewiseConversion

    # COP varies: 2.5 at low load to 4.0 at high load
    electricity_to_heat = Piecewise([
        Piece((0, 0), (50, 125)),      # 0-50 kW elec → 0-125 kW heat (COP 2.5)
        Piece((50, 125), (100, 350)),  # 50-100 kW elec → 125-350 kW heat (COP 3.5-4.5)
    ])

    heat_pump = LinearConverter(
        label='heat_pump',
        inputs=[Flow(label='electricity_in', bus='electricity', size=100)],
        outputs=[Flow(label='heat_out', bus='heating', size=350)],
        piecewise_conversion=PiecewiseConversion(
            origin_flow='electricity_in',
            piecewise_shares={'heat_out': electricity_to_heat},
        ),
    )
    ```

    **Variables:** Shared $\beta_1, \beta_2$, $\lambda_{0,1}, \lambda_{1,1}, \lambda_{0,2}, \lambda_{1,2}$ for both electricity and heat

    **Constraints:** $\eqref{eq:piecewise_combined}$ coupling electricity input to heat output with variable COP

    **Behavior:** Electricity input and heat output are coupled through shared lambda variables, modeling load-dependent COP.

    ---

    ## Optional Operation with Zero Point

    ```python
    from flixopt import Piecewise, Piece

    optional_operation = Piecewise(
        pieces=[
            Piece((10, 10), (50, 50)),   # Low operating range
            Piece((50, 50), (100, 100)), # High operating range
        ],
        zero_point=True,  # Allow complete shutdown
    )
    ```

    **Variables:** $\beta_1, \beta_2$, $\beta_\text{zero}$, $\lambda_{0,1}, \lambda_{1,1}, \lambda_{0,2}, \lambda_{1,2}$

    **Constraints:** $\eqref{eq:active_piece}$, $\eqref{eq:piece}$, $\eqref{eq:piecewise_in_pieces_zero}$ with zero point

    **Behavior:** Equipment can be completely off ($v=0$, $\beta_\text{zero}=0$) or operating in 10-100 range ($\beta_\text{zero}=1$, one piece active).

    ---

    ## Economies of Scale (Investment Costs)

    ```python
    from flixopt import InvestParameters, Piecewise, Piece, PiecewiseEffects

    # Cost per kWh decreases with scale
    battery_cost = InvestParameters(
        minimum_size=10,
        maximum_size=1000,
        piecewise_effects_of_investment=PiecewiseEffects(
            piecewise_origin=Piecewise([
                Piece((0, 0), (100, 100)),    # Small
                Piece((100, 100), (500, 500)),  # Medium
                Piece((500, 500), (1000, 1000)), # Large
            ]),
            piecewise_shares={
                'cost': Piecewise([
                    Piece((0, 0), (100, 80000)),    # €800/kWh
                    Piece((100, 80000), (500, 350000)), # €750-600/kWh
                    Piece((500, 350000), (1000, 850000)), # €600-500/kWh (bulk discount)
                ])
            },
        ),
    )
    ```

    **Variables:** Shared $\beta_k$, $\lambda_{0,k}$, $\lambda_{1,k}$ for size and cost

    **Constraints:** $\eqref{eq:piecewise_combined}$ coupling size to cost with decreasing unit cost

    **Behavior:** Investment size and total cost are coupled through piecewise relationship modeling economies of scale.

---

## Piece Relationship Patterns

### Touching Pieces (Continuous Function)
Pieces that share boundary points create smooth, continuous functions without gaps or overlaps.
**Use case:** Efficiency curves, performance maps

### Gaps Between Pieces (Forbidden Regions)
Non-contiguous pieces with gaps represent forbidden operating regions.
**Use case:** Minimum load requirements, safety zones, equipment limitations

### Overlapping Pieces (Flexible Operation)
Pieces with overlapping domains provide optimization flexibility, allowing the solver to choose which segment to operate in.
**Use case:** Multiple operating modes, flexible efficiency options

---

## Implementation

- **Feature Class:** [`Piecewise`][flixopt.interface.Piecewise]
- **Model Class:** [`PiecewiseModel`][flixopt.features.PiecewiseModel]
- **Helper Class:** [`Piece`][flixopt.interface.Piece]
- **Used by:** [`LinearConverter`](../elements/LinearConverter.md) (via `PiecewiseConversion`) · [`InvestParameters`](InvestParameters.md) (via `PiecewiseEffects`)

## See Also

- **Elements:** [LinearConverter](../elements/LinearConverter.md)
- **Features:** [InvestParameters](InvestParameters.md)
- **Patterns:** [Modeling Patterns](../modeling-patterns/index.md)
- **System-Level:** [Effects, Penalty & Objective](../effects-penalty-objective.md)
