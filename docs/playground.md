---
hide:
  - toc
---

# Playground

Try flixopt directly in your browser — no installation required!

The playground runs flixopt entirely in your browser using WebAssembly
(via [Pyodide](https://pyodide.org/) and [marimo](https://marimo.io/)).
You can **edit and re-run any cell** to experiment with the model.

<a href="quickstart/" class="md-button md-button--primary" target="_blank">
Launch Quickstart Playground
</a>

## What's included

The quickstart tutorial builds a simple gas-boiler-workshop energy system:

- Define a time horizon and heat demand profile
- Build an energy system with buses, sources, and converters
- Optimize using the HiGHS solver
- Visualize results with interactive Plotly charts

## Browser requirements

- Modern browser (Chrome, Firefox, Safari, Edge)
- First load takes ~30–60 seconds to download the Python runtime and packages
- Subsequent visits use cached data and load faster

!!! note "Experimental Feature"
    The WASM playground is experimental. If you encounter issues,
    please [open an issue](https://github.com/flixOpt/flixopt/issues).
