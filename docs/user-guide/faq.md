# Frequently Asked Questions

## What is flixOpt?

flixOpt is a Python framework for modeling and optimizing energy and material flow systems. It handles both operational optimization (dispatch) and investment optimization (capacity expansion).

## Which solvers does flixOpt support?

- **HiGHS** (default, included)
- **Gurobi** (commercial, academic licenses available)

## How do I install flixOpt?

```bash
pip install flixopt
```

For full features:
```bash
pip install "flixopt[full]"
```

## Do I need to install a solver separately?

No. HiGHS is included and works out of the box.

## Can I add custom constraints?

Yes. You can add custom constraints directly to the optimization model using linopy.

## Where can I get help?

- Check [Troubleshooting](troubleshooting.md)
- Open an [issue on GitHub](https://github.com/flixOpt/flixopt/issues)
