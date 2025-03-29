# Piecewise

A Piecewise is a collection of [`Pieces`][flixopt.interface.Piece], which each define a valid range for a variable $v$

$$ \label{eq:active_piece}
    \beta_\text{k} = \lambda_\text{0, k} + \lambda_\text{1, k}
$$

$$ \label{eq:piece}
    v_\text{k} = \lambda_\text{0, k} * \text{v}_{\text{start,k}} + \lambda_\text{1,k} * \text{v}_{\text{end,k}}
$$

$$ \label{eq:piecewise_in_pieces}
\sum_{k=1}^k \beta_{k} = 1
$$

With:

- $v$: The variable to be defined by the Piecewise
- $\text{v}_{\text{start,k}}$: the start point of the piece for variable $v$
- $\text{v}_{\text{end,k}}$: the end point of the piece for variable $v$
- $\beta_\text{k} \in \{0, 1\}$: defining wether the Piece $k$ is active
- $\lambda_\text{0,k} \in [0, 1]$: A variable defining the fraction of $\text{v}_{\text{start,k}}$ that is active
- $\lambda_\text{1,k} \in [0, 1]$: A variable defining the fraction of $\text{v}_{\text{end,k}}$ that is active

Which can also be described as $v \in 0 \cup [\text{v}_\text{start}, \text{v}_\text{end}]$.

Instead of \eqref{eq:piecewise_in_pieces}, the following constraint is used to also allow all variables to be zero:

$$ \label{eq:piecewise_in_pieces_zero}
\sum_{k=1}^k \beta_{k} = \beta_\text{zero}
$$

With:

- $\beta_\text{zero} \in \{0, 1\}$.

Which can also be described as $v \in \{0\} \cup [\text{v}_{\text{start_k}}, \text{v}_{\text{end_k}}]$


## Combining multiple Piecewises

Piecewise allows representing non-linear relationships. 
This is a powerful technique in linear optimization to model non-linear behaviors while maintaining the problem's linearity.

Therefore, each Piecewise must have the same number of Pieces $k$.

The variables described in [Piecewise](#piecewise) are created for each Piece, but nor for each Piecewise.
Rather, \eqref{eq:piece} is the only constraint that is created for each Piecewise, using the start and endpoints $\text{v}_{\text{start,k}}$ and $\text{v}_{\text{end,k}}$ of each Piece for the corresponding variable $v$
