# Frequently Asked Questions

Common questions about using flixOpt.

## General

### What is flixOpt?

flixOpt is a Python framework for modeling and optimizing energy and material flow systems. It handles both operational optimization (dispatch) and investment optimization (capacity expansion planning).

### What types of problems can flixOpt solve?

flixOpt can solve:

- Operational dispatch with fixed capacities
- Capacity expansion planning with investment decisions
- Multi-period planning with sequential investments
- Combined operational and investment optimization
- Multi-commodity systems with energy carriers
- Stochastic optimization with scenarios

### Which solvers does flixOpt support?

flixOpt supports:

- **HiGHS** (default, included)
- **Gurobi** (commercial, academic licenses available)
- we did support more, but saw no usage. Adding them back would be little effort

### Is flixOpt free to use?

Yes! flixOpt is released under the MIT License, which allows free use for commercial and academic purposes. However, some solvers like Gurobi and CPLEX require commercial licenses (though academic licenses are available).

## Installation & Setup

### How do I install flixOpt?

```bash
pip install flixopt
```

For full features:
```bash
pip install "flixopt[full]"
```

See [Installation](../home/installation.md) for details.

### Do I need to install a solver separately?

No! The HiGHS solver is included with flixOpt and works out of the box. Other solvers are optional.

## Advanced Topics

### Can I add custom constraints?

Yes! You can add custom constraints directly to the optimization model using linopy. See the Advanced Usage section.

### Does flixOpt support stochastic optimization?

Currently, you can optimize the expected value across scenarios. Further stochastic optimization is on the roadmap.

## Troubleshooting

### Where can I get help?

- Check [Troubleshooting](troubleshooting.md)
- Search [GitHub Discussions](https://github.com/flixOpt/flixopt/discussions)
- Open an [issue](https://github.com/flixOpt/flixopt/issues) if you found a bug
- See [Support](support.md) for more resources

## Not Finding Your Answer?

- Browse the [User Guide](index.md)
- Check the [Examples](../examples/index.md)
- Review the [API Reference](../api-reference/)
- Ask in [GitHub Discussions](https://github.com/flixOpt/flixopt/discussions)
