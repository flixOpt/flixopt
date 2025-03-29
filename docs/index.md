# FlixOpt

**FlixOpt** is a Python-based optimization framework designed to tackle energy and material flow problems using mixed-integer linear programming (MILP).

It borrows concepts from both [FINE](https://github.com/FZJ-IEK3-VSA/FINE) and [oemof.solph](https://github.com/oemof/oemof-solph).

## Why FlixOpt?

FlixOpt is designed as a general-purpose optimization framework to get your model running quickly, without sacrificing flexibility down the road:

- **Easy to Use API**: FlixOpt provides a Pythonic, object-oriented interface that makes mathematical optimization more accessible to Python developers.

- **Approachable Learning Curve**: Designed to be accessible from the start, with options for more detailed models down the road.

- **Domain Independence**: While frameworks like oemof and FINE excel at energy system modeling with domain-specific components, FlixOpt offers a more general mathematical approach that can be applied across different fields.

- **Extensibility**: Easily add custom constraints or variables to any FlixOpt Model using [linopy](https://github.com/PyPSA/linopy). Tailor any FlixOpt model to your specific needs without loosing the convenience of the framework.

- **Solver Agnostic**: Work with different solvers through a consistent interface.

- **Results File I/O**: Built to analyze results independent of running the optimization.

<figure markdown>
  ![FlixOpt Conceptual Usage](./images/architecture_flixOpt.png)
  <figcaption>Conceptual Usage and IO operations of FlixOpt</figcaption>
</figure>

## Installation

```bash
pip install flixopt
```

For more detailed installation options, see the [Getting Started](getting-started.md) guide.

## License

FlixOpt is released under the MIT License. See [LICENSE](https://github.com/flixopt/flixopt/blob/main/LICENSE) for details.

## Citation

If you use FlixOpt in your research or project, please cite:

- **Main Citation:** [DOI:10.18086/eurosun.2022.04.07](https://doi.org/10.18086/eurosun.2022.04.07)
- **Short Overview:** [DOI:10.13140/RG.2.2.14948.24969](https://doi.org/10.13140/RG.2.2.14948.24969)

*A more sophisticated paper is in progress*
