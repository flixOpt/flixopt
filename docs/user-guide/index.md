# User Guide

Welcome to the flixOpt User Guide! This guide will help you master energy and material flow optimization with flixOpt.

## What is flixOpt?

flixOpt is a comprehensive framework for modeling and optimizing energy and material flow systems. It supports:

- **Operational Optimization** - Dispatch optimization with fixed capacities
- **Investment Optimization** - Capacity expansion planning with binary or continuous sizing
- **Multi-Period Planning** - Sequential investment decisions across multiple periods
- **Scenario Analysis** - Stochastic modeling with weighted scenarios

## Key Features

<div class="grid cards" markdown>

- :material-puzzle: **Flexible Components**

    ---

    Flow, Bus, Storage, LinearConverter - build any system topology

- :material-cog: **Advanced Modeling**

    ---

    Investment decisions, On/Off states, Piecewise linearization

- :material-calculator: **Multiple Solvers**

    ---

    HiGHS (default), Gurobi, CPLEX - choose what fits your needs

- :material-chart-line: **Built-in Analysis**

    ---

    Plotting, export, and result exploration tools

</div>

## Learning Path

This guide follows a sequential learning path:

| Step | Section | What You'll Learn |
|------|---------|-------------------|
| 1 | [Core Concepts](core-concepts.md) | Fundamental building blocks: FlowSystem, Bus, Flow, Components, Effects |
| 2 | [Building Models](building-models/index.md) | How to construct models step by step |
| 3 | [Running Optimizations](optimization/index.md) | Solver configuration and execution |
| 4 | [Analyzing Results](results/index.md) | Extracting and visualizing outcomes |
| 5 | [FlowSystem Accessors](flow-system-accessors.md) | Optimize, transform, statistics, and topology APIs |
| 6 | [Mathematical Notation](mathematical-notation/index.md) | Deep dive into the math behind each element |
| 7 | [Recipes](recipes/index.md) | Common patterns and solutions |

## Quick Links

### Getting Started

- [Quick Start](../home/quick-start.md) - Build your first model in 5 minutes
- [Minimal Example](../examples/00-Minimal Example.md) - Simplest possible model
- [Core Concepts](core-concepts.md) - Understand the fundamentals

### Reference

- [FlowSystem Accessors](flow-system-accessors.md) - Optimize, transform, statistics, topology APIs
- [Mathematical Notation](mathematical-notation/index.md) - Detailed specifications
- [API Reference](../api-reference/index.md) - Complete class documentation
- [Examples](../examples/index.md) - Working code to learn from

### Help

- [FAQ](faq.md) - Frequently asked questions
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Community](support.md) - Get help from the community

## Use Cases

flixOpt handles any flow-based optimization problem:

**Energy Systems**: Power dispatch, CHP optimization, renewable integration, battery storage, district heating

**Industrial Applications**: Process optimization, multi-commodity networks, supply chains, resource allocation
