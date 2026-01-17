"""
Proof-of-concept: DCE Pattern for Vectorized Modeling

This example demonstrates how the Declaration-Collection-Execution (DCE) pattern
works with a simplified flow system. It shows:

1. How elements declare their variables and constraints
2. How the FlowSystemModel orchestrates batch creation
3. The performance benefits of vectorization

Run this file directly to see the pattern in action:
    python -m flixopt.vectorized_example
"""

from __future__ import annotations

import time

import linopy
import pandas as pd
import xarray as xr

from .structure import VariableCategory
from .vectorized import (
    ConstraintRegistry,
    ConstraintResult,
    ConstraintSpec,
    SystemConstraintRegistry,
    VariableHandle,
    VariableRegistry,
    VariableSpec,
)

# =============================================================================
# Simplified Element Classes (Demonstrating DCE Pattern)
# =============================================================================


class SimplifiedElementModel:
    """Base class for element models using the DCE pattern.

    Key methods:
        declare_variables(): Returns list of VariableSpec
        declare_constraints(): Returns list of ConstraintSpec
        on_variables_created(): Called with handles after batch creation
    """

    def __init__(self, element_id: str):
        self.element_id = element_id
        self._handles: dict[str, VariableHandle] = {}

    def declare_variables(self) -> list[VariableSpec]:
        """Override to declare what variables this element needs."""
        return []

    def declare_constraints(self) -> list[ConstraintSpec]:
        """Override to declare what constraints this element needs."""
        return []

    def on_variables_created(self, handles: dict[str, VariableHandle]) -> None:
        """Called after batch creation with handles to our variables."""
        self._handles = handles

    def get_variable(self, category: str) -> linopy.Variable:
        """Get this element's variable by category."""
        if category not in self._handles:
            raise KeyError(f"No handle for category '{category}' in element '{self.element_id}'")
        return self._handles[category].variable


class FlowModel(SimplifiedElementModel):
    """Simplified Flow model demonstrating the DCE pattern."""

    def __init__(
        self,
        element_id: str,
        min_flow: float = 0.0,
        max_flow: float = 100.0,
        with_status: bool = False,
    ):
        super().__init__(element_id)
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.with_status = with_status

    def declare_variables(self) -> list[VariableSpec]:
        specs = []

        # Main flow rate variable
        specs.append(
            VariableSpec(
                category='flow_rate',
                element_id=self.element_id,
                lower=self.min_flow if not self.with_status else 0.0,  # If status, bounds via constraint
                upper=self.max_flow,
                dims=('time',),
                var_category=VariableCategory.FLOW_RATE,
            )
        )

        # Status variable (if needed)
        if self.with_status:
            specs.append(
                VariableSpec(
                    category='status',
                    element_id=self.element_id,
                    lower=0,
                    upper=1,
                    binary=True,
                    dims=('time',),
                    var_category=VariableCategory.STATUS,
                )
            )

        return specs

    def declare_constraints(self) -> list[ConstraintSpec]:
        specs = []

        if self.with_status:
            # Flow rate upper bound: flow_rate <= status * max_flow
            specs.append(
                ConstraintSpec(
                    category='flow_rate_ub',
                    element_id=self.element_id,
                    build_fn=self._build_upper_bound,
                )
            )

            # Flow rate lower bound: flow_rate >= status * min_flow
            specs.append(
                ConstraintSpec(
                    category='flow_rate_lb',
                    element_id=self.element_id,
                    build_fn=self._build_lower_bound,
                )
            )

        return specs

    def _build_upper_bound(self, model, handles: dict[str, VariableHandle]) -> ConstraintResult:
        flow_rate = handles['flow_rate'].variable
        status = handles['status'].variable
        return ConstraintResult(
            lhs=flow_rate,
            rhs=status * self.max_flow,
            sense='<=',
        )

    def _build_lower_bound(self, model, handles: dict[str, VariableHandle]) -> ConstraintResult:
        flow_rate = handles['flow_rate'].variable
        status = handles['status'].variable
        min_bound = max(self.min_flow, 1e-6)  # Numerical stability
        return ConstraintResult(
            lhs=flow_rate,
            rhs=status * min_bound,
            sense='>=',
        )


class StorageModel(SimplifiedElementModel):
    """Simplified Storage model demonstrating the DCE pattern."""

    def __init__(
        self,
        element_id: str,
        capacity: float = 1000.0,
        max_charge_rate: float = 100.0,
        max_discharge_rate: float = 100.0,
        efficiency: float = 0.95,
    ):
        super().__init__(element_id)
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency

    def declare_variables(self) -> list[VariableSpec]:
        return [
            # State of charge
            VariableSpec(
                category='charge_state',
                element_id=self.element_id,
                lower=0,
                upper=self.capacity,
                dims=('time',),  # In practice, would use extra_timestep
                var_category=VariableCategory.CHARGE_STATE,
            ),
            # Charge rate
            VariableSpec(
                category='charge_rate',
                element_id=self.element_id,
                lower=0,
                upper=self.max_charge_rate,
                dims=('time',),
                var_category=VariableCategory.FLOW_RATE,
            ),
            # Discharge rate
            VariableSpec(
                category='discharge_rate',
                element_id=self.element_id,
                lower=0,
                upper=self.max_discharge_rate,
                dims=('time',),
                var_category=VariableCategory.FLOW_RATE,
            ),
        ]

    def declare_constraints(self) -> list[ConstraintSpec]:
        return [
            ConstraintSpec(
                category='energy_balance',
                element_id=self.element_id,
                build_fn=self._build_energy_balance,
            ),
        ]

    def _build_energy_balance(self, model, handles: dict[str, VariableHandle]) -> ConstraintResult:
        """Energy balance: soc[t] = soc[t-1] + charge*eff - discharge/eff."""
        charge_state = handles['charge_state'].variable
        charge_rate = handles['charge_rate'].variable
        discharge_rate = handles['discharge_rate'].variable

        # For simplicity, assume timestep duration = 1 hour
        # In practice, would get from model.timestep_duration
        dt = 1.0

        # soc[t] - soc[t-1] - charge*eff*dt + discharge*dt/eff = 0
        # Note: This is simplified - real implementation handles initial conditions
        lhs = (
            charge_state.isel(time=slice(1, None))
            - charge_state.isel(time=slice(None, -1))
            - charge_rate.isel(time=slice(None, -1)) * self.efficiency * dt
            + discharge_rate.isel(time=slice(None, -1)) * dt / self.efficiency
        )

        return ConstraintResult(lhs=lhs, rhs=0, sense='==')


# =============================================================================
# Simplified FlowSystemModel with DCE Support
# =============================================================================


class SimplifiedFlowSystemModel(linopy.Model):
    """Simplified model demonstrating the DCE pattern orchestration.

    This shows how FlowSystemModel would be modified to support DCE.
    """

    def __init__(self, timesteps: pd.DatetimeIndex):
        super().__init__(force_dim_names=True)
        self.timesteps = timesteps
        self.element_models: dict[str, SimplifiedElementModel] = {}
        self.variable_categories: dict[str, VariableCategory] = {}

        # DCE Registries
        self.variable_registry = VariableRegistry(self)
        self.constraint_registry: ConstraintRegistry | None = None
        self.system_constraint_registry: SystemConstraintRegistry | None = None

    def get_coords(self, dims: tuple[str, ...] | None = None) -> xr.Coordinates | None:
        """Get model coordinates (simplified version)."""
        coords = {'time': self.timesteps}
        if dims is not None:
            coords = {k: v for k, v in coords.items() if k in dims}
        return xr.Coordinates(coords) if coords else None

    def add_element(self, model: SimplifiedElementModel) -> None:
        """Add an element model."""
        self.element_models[model.element_id] = model

    def do_modeling_dce(self) -> None:
        """Build the model using the DCE pattern.

        Phase 1: Declaration - Collect all specs from elements
        Phase 2: Collection - Already done by registries
        Phase 3: Execution - Batch create variables and constraints
        """
        print('\n=== Phase 1: DECLARATION ===')
        start = time.perf_counter()

        # Collect variable declarations
        for element_id, model in self.element_models.items():
            for spec in model.declare_variables():
                self.variable_registry.register(spec)
            print(f'  Declared variables for: {element_id}')

        declaration_time = time.perf_counter() - start
        print(f'  Declaration time: {declaration_time * 1000:.2f}ms')

        print('\n=== Phase 2: COLLECTION (implicit) ===')
        print(f'  {self.variable_registry}')

        print('\n=== Phase 3: EXECUTION (Variables) ===')
        start = time.perf_counter()

        # Batch create all variables
        self.variable_registry.create_all()

        var_creation_time = time.perf_counter() - start
        print(f'  Variable creation time: {var_creation_time * 1000:.2f}ms')

        # Distribute handles to elements
        for element_id, model in self.element_models.items():
            handles = self.variable_registry.get_handles_for_element(element_id)
            model.on_variables_created(handles)
            print(f'  Distributed {len(handles)} handles to: {element_id}')

        print('\n=== Phase 3: EXECUTION (Constraints) ===')
        start = time.perf_counter()

        # Now collect and create constraints
        self.constraint_registry = ConstraintRegistry(self, self.variable_registry)

        for _element_id, model in self.element_models.items():
            for spec in model.declare_constraints():
                self.constraint_registry.register(spec)

        self.constraint_registry.create_all()

        constraint_time = time.perf_counter() - start
        print(f'  Constraint creation time: {constraint_time * 1000:.2f}ms')

        print('\n=== SUMMARY ===')
        print(f'  Variables: {len(self.variables)}')
        print(f'  Constraints: {len(self.constraints)}')
        print(f'  Categories in registry: {self.variable_registry.categories}')


# =============================================================================
# Comparison: Old Pattern vs DCE Pattern
# =============================================================================


def benchmark_old_pattern(n_elements: int, n_timesteps: int) -> float:
    """Simulate the old pattern: individual variable/constraint creation."""
    model = linopy.Model(force_dim_names=True)
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')

    start = time.perf_counter()

    # Old pattern: create variables one at a time
    for i in range(n_elements):
        model.add_variables(
            lower=0,
            upper=100,
            coords=xr.Coordinates({'time': timesteps}),
            name=f'flow_rate_{i}',
        )
        model.add_variables(
            lower=0,
            upper=1,
            coords=xr.Coordinates({'time': timesteps}),
            name=f'status_{i}',
            binary=True,
        )

    # Create constraints one at a time
    for i in range(n_elements):
        flow_rate = model.variables[f'flow_rate_{i}']
        status = model.variables[f'status_{i}']
        model.add_constraints(flow_rate <= status * 100, name=f'ub_{i}')
        model.add_constraints(flow_rate >= status * 1e-6, name=f'lb_{i}')

    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_dce_pattern(n_elements: int, n_timesteps: int) -> float:
    """Benchmark the DCE pattern: batch variable/constraint creation."""
    model = linopy.Model(force_dim_names=True)
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')

    start = time.perf_counter()

    # DCE pattern: batch create variables
    element_ids = [f'element_{i}' for i in range(n_elements)]

    # Single call for all flow_rate variables
    model.add_variables(
        lower=0,
        upper=100,
        coords=xr.Coordinates(
            {
                'element': pd.Index(element_ids),
                'time': timesteps,
            }
        ),
        name='flow_rate',
    )

    # Single call for all status variables
    model.add_variables(
        lower=0,
        upper=1,
        coords=xr.Coordinates(
            {
                'element': pd.Index(element_ids),
                'time': timesteps,
            }
        ),
        name='status',
        binary=True,
    )

    # Batch constraints (vectorized across elements)
    flow_rate = model.variables['flow_rate']
    status = model.variables['status']
    model.add_constraints(flow_rate <= status * 100, name='flow_rate_ub')
    model.add_constraints(flow_rate >= status * 1e-6, name='flow_rate_lb')

    elapsed = time.perf_counter() - start
    return elapsed


def run_benchmark():
    """Run benchmark comparing old vs DCE pattern."""
    print('\n' + '=' * 60)
    print('BENCHMARK: Old Pattern vs DCE Pattern')
    print('=' * 60)

    configs = [
        (10, 24),
        (50, 168),
        (100, 168),
        (200, 168),
        (500, 168),
    ]

    print(f'\n{"Elements":>10} {"Timesteps":>10} {"Old (ms)":>12} {"DCE (ms)":>12} {"Speedup":>10}')
    print('-' * 60)

    for n_elements, n_timesteps in configs:
        # Run each benchmark 3 times and take minimum
        old_times = [benchmark_old_pattern(n_elements, n_timesteps) for _ in range(3)]
        dce_times = [benchmark_dce_pattern(n_elements, n_timesteps) for _ in range(3)]

        old_time = min(old_times) * 1000  # Convert to ms
        dce_time = min(dce_times) * 1000
        speedup = old_time / dce_time if dce_time > 0 else float('inf')

        print(f'{n_elements:>10} {n_timesteps:>10} {old_time:>12.2f} {dce_time:>12.2f} {speedup:>10.1f}x')


def run_demo():
    """Run a demonstration of the DCE pattern."""
    print('\n' + '=' * 60)
    print('DEMO: DCE Pattern with Simplified Elements')
    print('=' * 60)

    # Create timesteps
    timesteps = pd.date_range('2024-01-01', periods=24, freq='h')

    # Create model
    model = SimplifiedFlowSystemModel(timesteps)

    # Add some flows
    model.add_element(FlowModel('Boiler_Q_th', min_flow=10, max_flow=100, with_status=True))
    model.add_element(FlowModel('HeatPump_Q_th', min_flow=5, max_flow=50, with_status=True))
    model.add_element(FlowModel('Solar_Q_th', min_flow=0, max_flow=30, with_status=False))

    # Add a storage
    model.add_element(StorageModel('ThermalStorage', capacity=500))

    # Build the model using DCE
    model.do_modeling_dce()

    # Show that elements can access their variables
    print('\n=== Element Variable Access ===')
    boiler = model.element_models['Boiler_Q_th']
    print(f'  Boiler flow_rate shape: {boiler.get_variable("flow_rate").shape}')
    print(f'  Boiler status shape: {boiler.get_variable("status").shape}')

    storage = model.element_models['ThermalStorage']
    print(f'  Storage charge_state shape: {storage.get_variable("charge_state").shape}')

    # Show batched variables
    print('\n=== Batched Variables in Registry ===')
    flow_rate_full = model.variable_registry.get_full_variable('flow_rate')
    print(f'  flow_rate full shape: {flow_rate_full.shape}')
    print(f'  flow_rate dims: {flow_rate_full.dims}')

    status_full = model.variable_registry.get_full_variable('status')
    print(f'  status full shape: {status_full.shape}')


if __name__ == '__main__':
    run_demo()
    run_benchmark()
