---
title: 'Flixopt: A framework for optimization of flow-systems'
tags:
  - flow-system
  - mixed-integer linear programming
  - MILP
  - optimization-based framework
  - optimization of operation and design
  - investment decision
  - energy system optimization
authors:
  - name: Felix Panitz
    orcid: 0009-0007-7030-6987
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Felix Bumann
    orcid: 0009-0006-0765-4789
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Peter Stange
    orcid: 0009-0001-6407-1495
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Technische Universität Dresden, Germany
   index: 1
   ror: 042aqky30
 - name: Fraunhofer Research Institution for Energy Infrastructures and Geotechnologies IEG, Germany
   index: 2
   ror: 00y718461
date: 28 November 2025
bibliography: paper.bib

---

# Flixopt: A Flexible Optimization Framework for Energy and Material Flow Systems

## Summary

Flixopt is an open-source Python framework for the mixed-integer linear optimization of systems with energy and material flows. It addresses the growing complexity of modern supply systems—characterized by fluctuating renewables, sector coupling, storage integration, and diverse technology portfolios—by providing a general, extensible, and vectorized modeling architecture. Built on the linear optimization package Linopy [@Hofmann2023], Flixopt allows users to flexibly represent system topologies, configure operational and investment decisions, and evaluate multiple metrics through its novel *Effects* concept. While developed primarily for energy-system studies such as district heating, the framework is domain-agnostic and suitable for any flow-based system where modular modeling and transparency are essential.

## Statement of Need

Existing energy system modeling frameworks often focus on specific technologies or sectors, restrict the representation of time structures, or embed implicit assumptions into predefined component libraries. Reviews such as Hoffmann et al. [@HOFFMANN2024100190] highlight the need for more flexible, framework-based modeling tools capable of adapting to diverse research questions. Other established tools—such as oemof.solph [@oemof_2020] or PyPSA [@PyPSA]—excel in particular domains but provide less granular control over multi-metric evaluations or lack transparent vector-based formulations.

Flixopt fills this gap by enabling:

* full flexibility in timestep definitions, including non-equidistant structures;
* modular construction of flows, components, and buses;
* detailed representation of operational and investment behavior via optional *Features*;
* arbitrary evaluation metrics through *Effects*, supporting multiobjective and constrained formulations.

This combination allows researchers to represent system structure and evaluation logic without adopting predefined assumptions or technology templates.

## Software Description

Flixopt organizes system models into *Flows*, *Components*, *Buses*, and *Effects*. Flows represent the transport of energy or material; components transform, store, generate, or consume flows; buses enforce nodal balance; and effects gather contributions to metrics such as cost, emissions, primary energy demand, or user-defined indicators. Effects may reference each other, enabling constructs such as CO₂ taxes, weighted-sum objectives, or ε-constraints.

Each user-defined element is associated with an internal model that generates constraints and variables in a vectorized form using Linopy. Optional features add investment decisions, on/off states, minimum up- or downtimes, mutual exclusivity of flows, or piecewise-linear relations. These capabilities remain inactive unless explicitly required, keeping models compact and computationally efficient.

Flixopt supports several calculation modes. The default mode performs full time-resolved optimization. Aggregated modes use clustered typical periods, following principles discussed in the literature on time-series aggregation (e.g., @TSAM2020), reducing computational burden for long time horizons. A segmented mode decomposes large operational problems into smaller, sequentially optimized blocks.

![Overview of Flixopt´s system architecture and workflow \label{fig:architecture}](figures/architecture.svg)

## Use Cases

Flixopt has been used in a variety of studies, particularly in district heating and building supply optimization. Examples include:

* **District heating transformation**: Optimization of investment and operation for portfolios including heat pumps, biomass boilers, and thermal storage, as in SmartBioGrid [@SBG_2023] and related follow-up analyses.
* **Building-level energy systems**: Integrated optimization of heat pumps, PV, thermal storage, and optional batteries under physical constraints such as roof area.
* **Strategic long-term planning**: Multi-decade scenarios with typical-period aggregation and iterative investment decisions, similar to approaches seen in Welder et al. [@WELDER20181130].

Across these applications, Flixopt reduces the need for custom equation writing while maintaining full transparency and flexibility.

## Conclusion

Flixopt provides a general, extensible, and transparent optimization framework for systems with energy and material flows. By combining a modular element structure, vector-based modeling, flexible time handling, and a powerful multi-metric evaluation system, it enables researchers and practitioners to address diverse optimization tasks without committing to predefined templates or assumptions. Its architecture, grounded in established modeling principles [@williams_model_2013; @kallrath_2002], and its open-source implementation make it a robust foundation for future developments in energy system optimization.

## References

References are managed via the accompanying `paper.bib` file, which will be rendered automatically by JOSS.

# Variante näher am paper (und zu lang)!
## Introduction

With the ongoing transformation of the energy supply, the complexity of systems is increasing. Fuel and technology switches, sector coupling, fluctuating energy sources, and the integration of storage solutions are becoming state of the art. Due to new limitations as well as newly acquired degrees of freedom, simple demand-driven operation is increasingly replaced by cost‑optimized operation of many individual units within supply systems. Mathematical optimization — particularly mixed‑integer linear programming (MILP) — has therefore become one of the standard methods for determining optimal operation and design of energy supply systems.

A wide range of frameworks for modeling and optimization of energy systems exists. An overview of 63 existing frameworks is given in Hoffmann et al. (2024). However, existing frameworks are often tailored to particular use cases, do not allow flexible extension, restrict timestep handling, or lack support for evaluation metrics beyond monetary cost. In addition to the need for greater flexibility and adaptability, efficient handling of equations in a vector‑based form has become an important requirement, as it enables improved mathematical modeling, enhances computational performance, and supports transparency and maintainability of the model’s equation set.

Flixopt is a Python framework for mixed‑integer linear programming and optimization of complex systems with energy and material flows. It builds on a MATLAB® framework that has been migrated to Python and incorporates principles from earlier work as well as ideas from the Python package oemof/solph. Flixopt is published under the MIT license. Its generic and extensible approach allows application in a wide variety of scenarios. It enables non‑equidistant timesteps, allocation of impacts to freely definable evaluation metrics, switching between objectives, and broad flexibility in configuring the optimization. A short overview was previously published, but this paper provides a comprehensive description of the framework and introduces its novel concepts (e.g., *Effects*).

Although development has focused mainly on optimizing energy systems — especially district heating systems — Flixopt is designed to be usable across disciplines and research questions.

## Architecture and Workflow of Flixopt

The architecture and workflow of the Flixopt package consist of three main steps: initialization, modeling and solving, and postprocessing of results. Users begin by implementing a flow‑system consisting of *Element* objects, defining the network topology and all element parameters. One or more calculation setups can then be configured, including the choice of solver, calculation mode, and selected time horizon or segment.

During modeling, Flixopt creates for each calculation a sub‑flow‑system derived from the base model. It then constructs the optimization problem in a fully vectorized manner using Linopy, which handles array‑based modeling and solver communication. After solving, Flixopt provides a postprocessing environment that yields structured access to results, facilitates visualization and statistical analysis, and enables saving/restoring complete solution states.





![Overview of Flixopt´s system architecture and workflow \label{fig:architecture}](figures/architecture.svg)

## Structure and Mathematical Modeling

Flixopt models systems as flow‑systems consisting of *Flows*, *Components*, *Buses*, and *Effects*, all of which are subclasses of *Element*. While *Flows*, *Components*, and *Buses* represent structural aspects of energy or material systems — similar to many existing frameworks — the concept of *Effects* is a novel contribution. *Effects* capture evaluation metrics such as costs, CO₂ emissions, primary energy demand, floor area, number of units, or any other metric defined by the user.

Each *Element* is linked to an internal *Elementmodel* created during modeling. *Elementmodels* manage variables and constraints, while *Elements* store user parameters and serve as the user‑facing interface. Optional *Features* extend *Elementmodels* with additional functionality, such as investment decisions, on/off states, or linear segments.

Time handling in Flixopt is fully flexible: users may define equidistant or non‑equidistant timesteps, and the framework automatically derives timestep durations and total time spans.

## Effects and Objective

Effects are central to Flixopt’s abstraction of evaluation metrics. Any element can contribute operation‑ or investment‑related shares to any effect, and effects may also reference each other. This enables: (1) multi‑criteria evaluation, (2) ε‑constraint formulations, (3) weighted‑sum multiobjective optimization, (4) distinguishing operational and investment impacts, and (5) external incentives without altering existing effect definitions. One effect is chosen as the objective, and a general penalty term is added to ensure solvability and improve debugging.

## Components, Flows, and Buses

Flows represent directed transport of energy or material, defined by flow‑rates and flow‑hours. Components include sinks, sources, linear transformers, and storages. Linear transformers define ratios between incoming and outgoing flows, while storages introduce state‑of‑charge variables, efficiencies, and bounds. Buses maintain nodal balance between incoming and outgoing flows at each timestep and can include penalty terms to resolve infeasibilities.

## Features

Flixopt’s feature system extends elements with optional advanced modeling capabilities:

* Investment decisions with continuous sizes and binary selection.
* On/off states and switching behavior.
* Minimum/maximum consecutive operation or downtime.
* Prevention of simultaneous flow operation.
* Piecewise linear relations via linear segments.

## Calculation Modes

Flixopt supports several calculation modes: exact time‑resolved modeling, segmented optimization for long horizons, and aggregated/typical‑period approaches based on time‑series clustering. These modes reduce computational effort while preserving system characteristics.

## Applications

Two exemplary applications illustrate Flixopt’s capabilities: (1) investment and operational optimization of district heat generation systems, and (2) integrated heat and electricity supply for a building including renewable energy and storage technologies. In both cases, Flixopt’s flexibility allows the definition of multiple technologies, effects, constraints, and interactions with little need for custom equations.

## Conclusion

Flixopt is a highly flexible open‑source optimization framework for modeling complex systems with energy and material flows. By combining a generic architecture, vector‑based equation handling, and a powerful abstraction of evaluation metrics through *Effects*, it supports a wide range of research and practical applications. Its extensibility, detailed configuration options, and efficient postprocessing environment make it a comprehensive tool for tackling modern challenges in energy system design and operation.

# Acknowledgements

# References
