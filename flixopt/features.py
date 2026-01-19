"""
This module contains the features of the flixopt framework.
Features extend the functionality of Elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import linopy
import numpy as np
import xarray as xr

from .modeling import BoundingPatterns, ModelingPrimitives, ModelingUtilities
from .structure import FlowSystemModel, Submodel, VariableCategory

if TYPE_CHECKING:
    from collections.abc import Collection

    from .core import FlowSystemDimensions
    from .interface import (
        InvestParameters,
        Piecewise,
        StatusParameters,
    )
    from .types import Numeric_PS, Numeric_TPS


class InvestmentModel(Submodel):
    """Mathematical model implementation for investment decisions.

    Creates optimization variables and constraints for investment sizing decisions,
    supporting both binary and continuous sizing with comprehensive effect modeling.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/InvestParameters/>

    Args:
        model: The optimization model instance
        label_of_element: The label of the parent (Element). Used to construct the full label of the model.
        parameters: The parameters of the feature model.
        label_of_model: The label of the model. This is needed to construct the full label of the model.
        size_category: Category for the size variable (FLOW_SIZE, STORAGE_SIZE, or SIZE for generic).
    """

    parameters: InvestParameters

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: InvestParameters,
        label_of_model: str | None = None,
        size_category: VariableCategory = VariableCategory.SIZE,
    ):
        self.piecewise_effects: PiecewiseEffectsModel | None = None
        self.parameters = parameters
        self._size_category = size_category
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        super()._do_modeling()
        self._create_variables_and_constraints()
        self._add_effects()

    def _create_variables_and_constraints(self):
        size_min, size_max = (self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size)
        if self.parameters.linked_periods is not None:
            # Mask size bounds: linked_periods is a binary DataArray that zeros out non-linked periods
            size_min = size_min * self.parameters.linked_periods
            size_max = size_max * self.parameters.linked_periods

        self.add_variables(
            short_name='size',
            lower=size_min if self.parameters.mandatory else 0,
            upper=size_max,
            coords=self._model.get_coords(['period', 'scenario']),
            category=self._size_category,
        )

        if not self.parameters.mandatory:
            self.add_variables(
                binary=True,
                coords=self._model.get_coords(['period', 'scenario']),
                short_name='invested',
                category=VariableCategory.INVESTED,
            )
            BoundingPatterns.bounds_with_state(
                self,
                variable=self.size,
                state=self._variables['invested'],
                bounds=(self.parameters.minimum_or_fixed_size, self.parameters.maximum_or_fixed_size),
            )

        if self.parameters.linked_periods is not None:
            masked_size = self.size.where(self.parameters.linked_periods, drop=True)
            self.add_constraints(
                masked_size.isel(period=slice(None, -1)) == masked_size.isel(period=slice(1, None)),
                short_name='linked_periods',
            )

    def _add_effects(self):
        """Add investment effects"""
        if self.parameters.effects_of_investment:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.invested * factor if self.invested is not None else factor
                    for effect, factor in self.parameters.effects_of_investment.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_retirement and not self.parameters.mandatory:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: -self.invested * factor + factor
                    for effect, factor in self.parameters.effects_of_retirement.items()
                },
                target='periodic',
            )

        if self.parameters.effects_of_investment_per_size:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.size * factor
                    for effect, factor in self.parameters.effects_of_investment_per_size.items()
                },
                target='periodic',
            )

        if self.parameters.piecewise_effects_of_investment:
            self.piecewise_effects = self.add_submodels(
                PiecewiseEffectsModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|PiecewiseEffects',
                    piecewise_origin=(self.size.name, self.parameters.piecewise_effects_of_investment.piecewise_origin),
                    piecewise_shares=self.parameters.piecewise_effects_of_investment.piecewise_shares,
                    zero_point=self.invested,
                ),
                short_name='segments',
            )

    @property
    def size(self) -> linopy.Variable:
        """Investment size variable"""
        return self._variables['size']

    @property
    def invested(self) -> linopy.Variable | None:
        """Binary investment decision variable"""
        if 'invested' not in self._variables:
            return None
        return self._variables['invested']


class InvestmentProxy:
    """Proxy providing access to batched InvestmentsModel for a specific element.

    This class provides the same interface as InvestmentModel.size/invested
    but returns slices from the batched InvestmentsModel variables.
    """

    def __init__(self, investments_model: InvestmentsModel, element_id: str):
        self._investments_model = investments_model
        self._element_id = element_id

    @property
    def size(self):
        """Investment size variable for this element."""
        return self._investments_model.get_variable('size', self._element_id)

    @property
    def invested(self):
        """Binary investment decision variable for this element (if non-mandatory)."""
        return self._investments_model.get_variable('invested', self._element_id)


class InvestmentsModel:
    """Type-level model for batched investment decisions across multiple elements.

    Unlike InvestmentModel (one per element), InvestmentsModel handles ALL elements
    with investment in a single instance with batched variables.

    This enables:
    - Batched `size` and `invested` variables with element-type dimension (e.g., 'flow')
    - Vectorized constraint creation
    - Batched effect shares

    Variable Naming Convention:
        - '{name_prefix}|size' e.g., 'flow|size', 'storage|size'
        - '{name_prefix}|invested' e.g., 'flow|invested', 'storage|invested'

    Dimension Naming:
        - Uses element-type-specific dimension (e.g., 'flow', 'storage')
        - Prevents unwanted broadcasting when merging into solution Dataset

    The model categorizes elements by investment type:
    - mandatory: Required investment (only size variable, with bounds)
    - non_mandatory: Optional investment (size + invested variables, state-controlled bounds)

    This is a base class. Use child classes (FlowInvestmentsModel, StorageInvestmentsModel)
    that know how to access element-specific investment parameters.

    Example:
        >>> investments_model = FlowInvestmentsModel(
        ...     model=flow_system_model,
        ...     flows=flows_with_investment,
        ...     size_category=VariableCategory.FLOW_SIZE,
        ... )
        >>> investments_model.create_variables()
        >>> investments_model.create_constraints()
    """

    def __init__(
        self,
        model: FlowSystemModel,
        elements: list,
        size_category: VariableCategory = VariableCategory.SIZE,
        name_prefix: str = 'investment',
        dim_name: str = 'element',
    ):
        """Initialize the type-level investment model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            elements: List of elements with investment parameters.
            size_category: Category for size variable expansion.
            name_prefix: Prefix for variable names (e.g., 'flow', 'storage').
            dim_name: Dimension name for element grouping (e.g., 'flow', 'storage').
        """
        import logging

        import pandas as pd
        import xarray as xr

        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.elements = elements
        self._size_category = size_category
        self._name_prefix = name_prefix
        self.dim_name = dim_name

        # Storage for created variables
        self._variables: dict[str, linopy.Variable] = {}

        # Store xr and pd for use in methods
        self._xr = xr
        self._pd = pd

        self._logger.debug(
            f'InvestmentsModel initialized: {len(self.element_ids)} elements '
            f'({len(self.mandatory_ids)} mandatory, {len(self.non_mandatory_ids)} non-mandatory)'
        )

    # === Abstract methods - child classes must implement ===

    def _get_params(self, element) -> InvestParameters:
        """Get InvestParameters from an element. Override in child classes."""
        raise NotImplementedError('Child classes must implement _get_params')

    def _get_element_id(self, element) -> str:
        """Get element identifier. Override in child classes."""
        raise NotImplementedError('Child classes must implement _get_element_id')

    # === Properties for element categorization ===

    @property
    def element_ids(self) -> list[str]:
        """IDs of all elements with investment."""
        return [self._get_element_id(e) for e in self.elements]

    @property
    def mandatory_ids(self) -> list[str]:
        """IDs of mandatory elements."""
        return [self._get_element_id(e) for e in self.elements if self._get_params(e).mandatory]

    @property
    def non_mandatory_ids(self) -> list[str]:
        """IDs of non-mandatory elements."""
        return [self._get_element_id(e) for e in self.elements if not self._get_params(e).mandatory]

    def _get_element_by_id(self, element_id: str):
        """Get element by its ID."""
        for e in self.elements:
            if self._get_element_id(e) == element_id:
                return e
        return None

    # === Parameter collection helpers ===

    def _collect_param(self, attr: str, element_ids: list[str] | None = None) -> xr.DataArray:
        """Collect a scalar or DataArray parameter from elements."""
        xr = self._xr
        if element_ids is None:
            elements = self.elements
            ids = self.element_ids
        else:
            id_set = set(element_ids)
            elements = [e for e in self.elements if self._get_element_id(e) in id_set]
            ids = element_ids

        values = []
        for e in elements:
            val = getattr(self._get_params(e), attr)
            if val is None:
                values.append(np.nan)
            else:
                values.append(val)

        # Handle mixed scalar/DataArray values
        if all(np.isscalar(v) or (isinstance(v, float) and np.isnan(v)) for v in values):
            return xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: ids})
        else:
            # Convert scalars to DataArrays and concatenate
            return self._stack_bounds([xr.DataArray(v) if np.isscalar(v) else v for v in values], xr, element_ids=ids)

    def _collect_effects(self, attr: str, element_ids: list[str] | None = None) -> dict[str, xr.DataArray]:
        """Collect effects dict from elements into a dict of DataArrays."""
        xr = self._xr
        if element_ids is None:
            elements = self.elements
            ids = self.element_ids
        else:
            id_set = set(element_ids)
            elements = [e for e in self.elements if self._get_element_id(e) in id_set]
            ids = element_ids

        # Find all effect names across all elements
        all_effects: set[str] = set()
        for e in elements:
            effects = getattr(self._get_params(e), attr) or {}
            all_effects.update(effects.keys())

        if not all_effects:
            return {}

        # Build DataArray for each effect
        result = {}
        for effect_name in all_effects:
            values = []
            for e in elements:
                effects = getattr(self._get_params(e), attr) or {}
                values.append(effects.get(effect_name, np.nan))
            result[effect_name] = xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: ids})

        return result

    def _stack_bounds(self, bounds_list: list, xr, element_ids: list[str] | None = None) -> xr.DataArray:
        """Stack bounds arrays with different dimensions into single DataArray.

        Handles the case where some bounds have period/scenario dims and others don't.

        Args:
            bounds_list: List of DataArrays (one per element)
            xr: xarray module
            element_ids: Optional list of element IDs (defaults to self.element_ids)
        """
        dim = self.dim_name  # e.g., 'flow', 'storage'
        if element_ids is None:
            element_ids = self.element_ids

        # Check if all are scalars
        if all(arr.dims == () for arr in bounds_list):
            values = [float(arr.values) for arr in bounds_list]
            return xr.DataArray(values, coords={dim: element_ids}, dims=[dim])

        # Find union of all non-element dimensions and their coords
        all_dims: dict[str, any] = {}
        for arr in bounds_list:
            for d in arr.dims:
                if d != dim and d not in all_dims:
                    all_dims[d] = arr.coords[d].values

        # Expand each array to have all dimensions
        expanded = []
        for arr, eid in zip(bounds_list, element_ids, strict=False):
            # Add element dimension
            if dim not in arr.dims:
                arr = arr.expand_dims({dim: [eid]})
            # Add missing dimensions
            for d, coords in all_dims.items():
                if d not in arr.dims:
                    arr = arr.expand_dims({d: coords})
            expanded.append(arr)

        return xr.concat(expanded, dim=dim, coords='minimal')

    def _stack_bounds_for_subset(self, bounds_list: list, element_ids: list[str], xr) -> xr.DataArray:
        """Stack bounds for a subset of elements (convenience wrapper)."""
        return self._stack_bounds(bounds_list, xr, element_ids=element_ids)

    def create_variables(self) -> None:
        """Create batched investment variables with element dimension.

        Creates:
        - size: For ALL elements (with element dimension)
        - invested: For non-mandatory elements only (binary, with element dimension)
        """
        from .structure import VARIABLE_TYPE_TO_EXPANSION, VariableType

        if not self.element_ids:
            return

        xr = self._xr
        pd = self._pd

        # Get base coords (period, scenario) - may be None if neither exist
        base_coords = self.model.get_coords(['period', 'scenario'])
        base_coords_dict = dict(base_coords) if base_coords is not None else {}

        dim = self.dim_name

        # === size: ALL elements ===
        # Collect bounds from elements
        size_min = self._collect_param('minimum_or_fixed_size')
        size_max = self._collect_param('maximum_or_fixed_size')
        linked_periods = self._collect_param('linked_periods')

        # Handle linked_periods masking
        if linked_periods.notnull().any():
            linked = linked_periods.fillna(1.0)  # NaN means no linking, treat as 1
            size_min = size_min * linked
            size_max = size_max * linked

        # Build mandatory mask
        mandatory_mask = xr.DataArray(
            [self._get_params(e).mandatory for e in self.elements],
            dims=[dim],
            coords={dim: self.element_ids},
        )

        # For non-mandatory, lower bound is 0 (invested variable controls actual minimum)
        lower_bounds = xr.where(mandatory_mask, size_min, 0)
        upper_bounds = size_max

        # Build coords with element-type dimension (e.g., 'flow', 'storage')
        size_coords = xr.Coordinates(
            {
                dim: pd.Index(self.element_ids, name=dim),
                **base_coords_dict,
            }
        )

        size_var = self.model.add_variables(
            lower=lower_bounds,
            upper=upper_bounds,
            coords=size_coords,
            name=f'{self._name_prefix}|size',
        )
        self._variables['size'] = size_var

        # Register category for segment expansion
        expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.SIZE)
        if expansion_category is not None:
            self.model.variable_categories[size_var.name] = expansion_category

        # === invested: non-mandatory elements only ===
        if self.non_mandatory_ids:
            invested_coords = xr.Coordinates(
                {
                    dim: pd.Index(self.non_mandatory_ids, name=dim),
                    **base_coords_dict,
                }
            )

            invested_var = self.model.add_variables(
                binary=True,
                coords=invested_coords,
                name=f'{self._name_prefix}|invested',
            )
            self._variables['invested'] = invested_var

            # Register category
            expansion_category = VARIABLE_TYPE_TO_EXPANSION.get(VariableType.INVESTED)
            if expansion_category is not None:
                self.model.variable_categories[invested_var.name] = invested_var.name

        self._logger.debug(
            f'InvestmentsModel created variables: {len(self.element_ids)} elements '
            f'({len(self.mandatory_ids)} mandatory, {len(self.non_mandatory_ids)} non-mandatory)'
        )

    def create_constraints(self) -> None:
        """Create batched investment constraints.

        For non-mandatory investments, creates state-controlled bounds:
            invested * min_size <= size <= invested * max_size
        """
        if not self.non_mandatory_ids:
            return

        xr = self._xr
        dim = self.dim_name

        size_var = self._variables['size']
        invested_var = self._variables['invested']

        # Collect bounds for non-mandatory elements
        min_bounds = self._collect_param('minimum_or_fixed_size', self.non_mandatory_ids)
        max_bounds = self._collect_param('maximum_or_fixed_size', self.non_mandatory_ids)

        # Select size for non-mandatory elements
        size_non_mandatory = size_var.sel({dim: self.non_mandatory_ids})

        # State-controlled bounds: invested * min <= size <= invested * max
        # Lower bound with epsilon to force non-zero when invested
        from .config import CONFIG

        epsilon = CONFIG.Modeling.epsilon
        effective_min = xr.where(min_bounds > epsilon, min_bounds, epsilon)

        self.model.add_constraints(
            size_non_mandatory >= invested_var * effective_min,
            name=f'{self._name_prefix}|size|lb',
        )
        self.model.add_constraints(
            size_non_mandatory <= invested_var * max_bounds,
            name=f'{self._name_prefix}|size|ub',
        )

        # Handle linked_periods constraints
        self._add_linked_periods_constraints()

        self._logger.debug(
            f'InvestmentsModel created constraints for {len(self.non_mandatory_ids)} non-mandatory elements'
        )

    def _add_linked_periods_constraints(self) -> None:
        """Add linked periods constraints for elements that have them."""
        size_var = self._variables['size']
        dim = self.dim_name

        # Get elements with linked periods
        element_ids_with_linking = [
            self._get_element_id(e) for e in self.elements if self._get_params(e).linked_periods is not None
        ]
        if not element_ids_with_linking:
            return

        for element_id in element_ids_with_linking:
            elem = self._get_element_by_id(element_id)
            linked = self._get_params(elem).linked_periods
            element_size = size_var.sel({dim: element_id})
            masked_size = element_size.where(linked, drop=True)
            if 'period' in masked_size.dims and masked_size.sizes.get('period', 0) > 1:
                self.model.add_constraints(
                    masked_size.isel(period=slice(None, -1)) == masked_size.isel(period=slice(1, None)),
                    name=f'{element_id}|linked_periods',
                )

    # === Effect factor properties (used by EffectsModel.finalize_shares) ===

    @property
    def elements_with_per_size_effects_ids(self) -> list[str]:
        """IDs of elements with effects_of_investment_per_size."""
        return [self._get_element_id(e) for e in self.elements if self._get_params(e).effects_of_investment_per_size]

    @property
    def non_mandatory_with_fix_effects_ids(self) -> list[str]:
        """IDs of non-mandatory elements with effects_of_investment."""
        return [
            self._get_element_id(e)
            for e in self.elements
            if not self._get_params(e).mandatory and self._get_params(e).effects_of_investment
        ]

    @property
    def non_mandatory_with_retirement_effects_ids(self) -> list[str]:
        """IDs of non-mandatory elements with effects_of_retirement."""
        return [
            self._get_element_id(e)
            for e in self.elements
            if not self._get_params(e).mandatory and self._get_params(e).effects_of_retirement
        ]

    @property
    def mandatory_with_fix_effects_ids(self) -> list[str]:
        """IDs of mandatory elements with effects_of_investment."""
        return [
            self._get_element_id(e)
            for e in self.elements
            if self._get_params(e).mandatory and self._get_params(e).effects_of_investment
        ]

    # === Effect DataArray properties ===

    @property
    def effects_of_investment_per_size(self) -> xr.DataArray | None:
        """Combined effects_of_investment_per_size with (element, effect) dims.

        Stacks effects from all elements into a single DataArray.
        Returns None if no elements have effects defined.
        """
        element_ids = self.elements_with_per_size_effects_ids
        if not element_ids:
            return None
        effects_dict = self._collect_effects('effects_of_investment_per_size', element_ids)
        return self._build_factors_from_dict(effects_dict, element_ids)

    @property
    def effects_of_investment(self) -> xr.DataArray | None:
        """Combined effects_of_investment with (element, effect) dims.

        Stacks effects from non-mandatory elements into a single DataArray.
        Returns None if no elements have effects defined.
        """
        element_ids = self.non_mandatory_with_fix_effects_ids
        if not element_ids:
            return None
        effects_dict = self._collect_effects('effects_of_investment', element_ids)
        return self._build_factors_from_dict(effects_dict, element_ids)

    @property
    def effects_of_retirement(self) -> xr.DataArray | None:
        """Combined effects_of_retirement with (element, effect) dims.

        Stacks effects from non-mandatory elements into a single DataArray.
        Returns None if no elements have effects defined.
        """
        element_ids = self.non_mandatory_with_retirement_effects_ids
        if not element_ids:
            return None
        effects_dict = self._collect_effects('effects_of_retirement', element_ids)
        return self._build_factors_from_dict(effects_dict, element_ids)

    def _build_factors_from_dict(
        self, effects_dict: dict[str, xr.DataArray], element_ids: list[str]
    ) -> xr.DataArray | None:
        """Build factor array with (element, effect) dims from effects dict.

        Args:
            effects_dict: Dict mapping effect_name -> DataArray(element_dim)
            element_ids: Element IDs to include

        Returns:
            DataArray with (element, effect) dims, NaN for missing effects.
        """
        if not effects_dict:
            return None

        xr = self._xr
        dim = self.dim_name
        effect_ids = list(effects_dict.keys())

        # Stack effects into (element, effect) array
        effect_arrays = [effects_dict[eff] for eff in effect_ids]
        result = xr.concat(effect_arrays, dim='effect').assign_coords(effect=effect_ids)

        # Transpose to (element, effect) order
        return result.transpose(dim, 'effect')

    def add_constant_shares_to_effects(self, effects_model) -> None:
        """Add constant (non-variable) shares directly to effect constraints.

        This handles:
        - Mandatory fixed effects (always incurred, not dependent on invested variable)
        - Retirement constant parts (the +factor in -invested*factor + factor)
        """
        # Mandatory fixed effects
        for element_id in self.mandatory_with_fix_effects_ids:
            elem = self._get_element_by_id(element_id)
            elem_effects = self._get_params(elem).effects_of_investment or {}
            effects_dict = {}
            for effect_name, val in elem_effects.items():
                if val is not None and not (np.isscalar(val) and np.isnan(val)):
                    effects_dict[effect_name] = val
            if effects_dict:
                self.model.effects.add_share_to_effects(
                    name=f'{element_id}|invest_fix',
                    expressions=effects_dict,
                    target='periodic',
                )

        # Retirement constant parts
        for element_id in self.non_mandatory_with_retirement_effects_ids:
            elem = self._get_element_by_id(element_id)
            elem_effects = self._get_params(elem).effects_of_retirement or {}
            effects_dict = {}
            for effect_name, val in elem_effects.items():
                if val is not None and not (np.isscalar(val) and np.isnan(val)):
                    effects_dict[effect_name] = val
            if effects_dict:
                self.model.effects.add_share_to_effects(
                    name=f'{element_id}|invest_retire_const',
                    expressions=effects_dict,
                    target='periodic',
                )

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element."""
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None:
            dim = self.dim_name
            if element_id in var.coords.get(dim, []):
                return var.sel({dim: element_id})
            return None
        return var

    @property
    def size(self) -> linopy.Variable:
        """Batched size variable with element-type dimension (e.g., 'flow', 'storage')."""
        return self._variables['size']

    @property
    def invested(self) -> linopy.Variable | None:
        """Batched invested variable with element-type dimension (non-mandatory only)."""
        return self._variables.get('invested')


class FlowInvestmentsModel(InvestmentsModel):
    """Type-level investment model for flows.

    Implements the abstract methods from InvestmentsModel to access
    flow-specific investment parameters directly.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        flows: list,
        size_category: VariableCategory = VariableCategory.FLOW_SIZE,
        name_prefix: str = 'flow',
    ):
        """Initialize the flow investment model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            flows: List of Flow objects with invest_parameters.
            size_category: Category for size variable expansion.
            name_prefix: Prefix for variable names.
        """
        super().__init__(
            model=model,
            elements=flows,
            size_category=size_category,
            name_prefix=name_prefix,
            dim_name='flow',
        )

    def _get_params(self, flow) -> InvestParameters:
        """Get InvestParameters from a flow."""
        return flow.invest_parameters

    def _get_element_id(self, flow) -> str:
        """Get flow identifier (label_full)."""
        return flow.label_full


class StorageInvestmentsModel(InvestmentsModel):
    """Type-level investment model for storages.

    Implements the abstract methods from InvestmentsModel to access
    storage-specific investment parameters directly.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        storages: list,
        size_category: VariableCategory = VariableCategory.STORAGE_SIZE,
        name_prefix: str = 'storage_investment',
        dim_name: str = 'storage',
    ):
        """Initialize the storage investment model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            storages: List of Storage objects with invest_parameters.
            size_category: Category for size variable expansion.
            name_prefix: Prefix for variable names.
            dim_name: Dimension name for storage grouping.
        """
        super().__init__(
            model=model,
            elements=storages,
            size_category=size_category,
            name_prefix=name_prefix,
            dim_name=dim_name,
        )

    def _get_params(self, storage) -> InvestParameters:
        """Get InvestParameters from a storage."""
        return storage.invest_parameters

    def _get_element_id(self, storage) -> str:
        """Get storage identifier (label)."""
        return storage.label


class StatusProxy:
    """Proxy providing access to batched StatusesModel for a specific element.

    This class provides the same interface as StatusModel properties
    but returns slices from the batched StatusesModel variables.
    """

    def __init__(self, statuses_model: StatusesModel, element_id: str):
        self._statuses_model = statuses_model
        self._element_id = element_id

    @property
    def status(self):
        """Binary status variable for this element (from FlowsModel)."""
        return self._statuses_model.get_status_variable(self._element_id)

    @property
    def active_hours(self):
        """Total active hours variable for this element."""
        return self._statuses_model.get_variable('active_hours', self._element_id)

    @property
    def startup(self):
        """Startup variable for this element."""
        return self._statuses_model.get_variable('startup', self._element_id)

    @property
    def shutdown(self):
        """Shutdown variable for this element."""
        return self._statuses_model.get_variable('shutdown', self._element_id)

    @property
    def inactive(self):
        """Inactive variable for this element."""
        return self._statuses_model.get_variable('inactive', self._element_id)

    @property
    def startup_count(self):
        """Startup count variable for this element."""
        return self._statuses_model.get_variable('startup_count', self._element_id)

    @property
    def _previous_status(self):
        """Previous status for this element (from StatusesModel)."""
        return self._statuses_model.get_previous_status(self._element_id)


class StatusesModel:
    """Type-level model for batched status features across multiple elements.

    Unlike StatusModel (one per element), StatusesModel handles ALL elements
    with status in a single instance with batched variables.

    This enables:
    - Batched `active_hours`, `startup`, `shutdown` variables with element dimension
    - Vectorized constraint creation
    - Batched effect shares

    The model categorizes elements by their feature flags:
    - all: Elements that have status (always get active_hours)
    - with_startup_tracking: Elements needing startup/shutdown variables
    - with_downtime_tracking: Elements needing inactive variable
    - with_startup_limit: Elements needing startup_count variable

    This is a base class. Use child classes (FlowStatusesModel, ComponentStatusesModel)
    that know how to access element-specific status parameters.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        status: linopy.Variable,
        elements: list,
        dim_name: str = 'element',
        name_prefix: str = 'status',
    ):
        """Initialize the type-level status model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            status: Batched status variable with element dimension.
            elements: List of elements with status parameters.
            dim_name: Dimension name for the element type (e.g., 'flow', 'component').
            name_prefix: Prefix for variable names (e.g., 'status', 'component_status').
        """
        import logging

        import pandas as pd
        import xarray as xr

        self._logger = logging.getLogger('flixopt')
        self.model = model
        self.dim_name = dim_name
        self.name_prefix = name_prefix

        # Store imports for later use
        self._pd = pd
        self._xr = xr

        # Variables dict
        self._variables: dict[str, linopy.Variable] = {}

        # Store elements and status variable
        self.elements = elements
        self._batched_status_var = status

        self._logger.debug(
            f'StatusesModel initialized: {len(self.element_ids)} elements, '
            f'{len(self.startup_tracking_ids)} with startup tracking, '
            f'{len(self.downtime_tracking_ids)} with downtime tracking'
        )

    # === Abstract methods - child classes must implement ===

    def _get_params(self, element) -> StatusParameters:
        """Get StatusParameters from an element. Override in child classes."""
        raise NotImplementedError('Child classes must implement _get_params')

    def _get_element_id(self, element) -> str:
        """Get element identifier. Override in child classes."""
        raise NotImplementedError('Child classes must implement _get_element_id')

    def _get_previous_status(self, element) -> xr.DataArray | None:
        """Get previous status DataArray for an element. Override in child classes."""
        return None  # Default: no previous status

    # === Element categorization properties ===

    @property
    def element_ids(self) -> list[str]:
        """IDs of all elements with status."""
        return [self._get_element_id(e) for e in self.elements]

    @property
    def startup_tracking_ids(self) -> list[str]:
        """IDs of elements needing startup/shutdown tracking."""
        result = []
        for e in self.elements:
            params = self._get_params(e)
            needs_tracking = (
                params.effects_per_startup
                or params.min_uptime is not None
                or params.max_uptime is not None
                or params.startup_limit is not None
                or params.force_startup_tracking
            )
            if needs_tracking:
                result.append(self._get_element_id(e))
        return result

    @property
    def downtime_tracking_ids(self) -> list[str]:
        """IDs of elements needing downtime tracking (inactive variable)."""
        return [
            self._get_element_id(e)
            for e in self.elements
            if self._get_params(e).min_downtime is not None or self._get_params(e).max_downtime is not None
        ]

    @property
    def uptime_tracking_ids(self) -> list[str]:
        """IDs of elements with min_uptime or max_uptime constraints."""
        return [
            self._get_element_id(e)
            for e in self.elements
            if self._get_params(e).min_uptime is not None or self._get_params(e).max_uptime is not None
        ]

    @property
    def startup_limit_ids(self) -> list[str]:
        """IDs of elements with startup_limit constraint."""
        return [self._get_element_id(e) for e in self.elements if self._get_params(e).startup_limit is not None]

    @property
    def cluster_cyclic_ids(self) -> list[str]:
        """IDs of elements with cluster_mode == 'cyclic'."""
        return [self._get_element_id(e) for e in self.elements if self._get_params(e).cluster_mode == 'cyclic']

    # === Parameter collection helpers ===

    def _collect_param(self, attr: str, element_ids: list[str] | None = None) -> xr.DataArray:
        """Collect a scalar parameter from elements into a DataArray."""
        if element_ids is None:
            elements = self.elements
            ids = self.element_ids
        else:
            id_set = set(element_ids)
            elements = [e for e in self.elements if self._get_element_id(e) in id_set]
            ids = element_ids

        values = []
        for e in elements:
            val = getattr(self._get_params(e), attr)
            values.append(np.nan if val is None else val)

        return xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: ids})

    def _get_previous_status_batched(self) -> xr.DataArray | None:
        """Build batched previous status DataArray from elements."""
        arrays = []
        ids = []
        for e in self.elements:
            prev = self._get_previous_status(e)
            if prev is not None:
                arrays.append(prev.expand_dims({self.dim_name: [self._get_element_id(e)]}))
                ids.append(self._get_element_id(e))

        if not arrays:
            return None

        return xr.concat(arrays, dim=self.dim_name)

    def _get_element_by_id(self, element_id: str):
        """Get element by its ID."""
        for e in self.elements:
            if self._get_element_id(e) == element_id:
                return e
        return None

    def create_variables(self) -> None:
        """Create batched status feature variables with element dimension."""
        pd = self._pd
        xr = self._xr

        # Get base coordinates (period, scenario if they exist)
        base_coords = self.model.get_coords(['period', 'scenario'])
        base_coords_dict = dict(base_coords) if base_coords is not None else {}

        dim = self.dim_name
        total_hours = self.model.temporal_weight.sum(self.model.temporal_dims)

        # === active_hours: ALL elements with status ===
        # This is a per-period variable (summed over time within each period)
        active_hours_coords = xr.Coordinates(
            {
                dim: pd.Index(self.element_ids, name=dim),
                **base_coords_dict,
            }
        )

        # Build bounds DataArrays by collecting from element parameters
        active_hours_min = self._collect_param('active_hours_min')
        active_hours_max = self._collect_param('active_hours_max')
        lower_da = active_hours_min.fillna(0)
        upper_da = xr.where(active_hours_max.notnull(), active_hours_max, total_hours)

        self._variables['active_hours'] = self.model.add_variables(
            lower=lower_da,
            upper=upper_da,
            coords=active_hours_coords,
            name=f'{self.name_prefix}|active_hours',
        )

        # === startup, shutdown: Elements with startup tracking ===
        if self.startup_tracking_ids:
            temporal_coords = self.model.get_coords()
            startup_coords = xr.Coordinates(
                {
                    dim: pd.Index(self.startup_tracking_ids, name=dim),
                    **dict(temporal_coords),
                }
            )
            self._variables['startup'] = self.model.add_variables(
                binary=True,
                coords=startup_coords,
                name=f'{self.name_prefix}|startup',
            )
            self._variables['shutdown'] = self.model.add_variables(
                binary=True,
                coords=startup_coords,
                name=f'{self.name_prefix}|shutdown',
            )

        # === inactive: Elements with downtime tracking ===
        if self.downtime_tracking_ids:
            temporal_coords = self.model.get_coords()
            inactive_coords = xr.Coordinates(
                {
                    dim: pd.Index(self.downtime_tracking_ids, name=dim),
                    **dict(temporal_coords),
                }
            )
            self._variables['inactive'] = self.model.add_variables(
                binary=True,
                coords=inactive_coords,
                name=f'{self.name_prefix}|inactive',
            )

        # === startup_count: Elements with startup limit ===
        if self.startup_limit_ids:
            startup_count_coords = xr.Coordinates(
                {
                    dim: pd.Index(self.startup_limit_ids, name=dim),
                    **base_coords_dict,
                }
            )
            # Get upper bounds by collecting from elements
            startup_limit = self._collect_param('startup_limit', self.startup_limit_ids)

            self._variables['startup_count'] = self.model.add_variables(
                lower=0,
                upper=startup_limit,
                coords=startup_count_coords,
                name=f'{self.name_prefix}|startup_count',
            )

        self._logger.debug(f'StatusesModel created variables for {len(self.element_ids)} elements')

    def create_constraints(self) -> None:
        """Create batched status feature constraints.

        Uses vectorized operations where possible for better performance.
        """
        dim = self.dim_name
        status = self._batched_status_var
        previous_status_batched = self._get_previous_status_batched()

        # === active_hours tracking: sum(status * weight) == active_hours ===
        # Vectorized: single constraint for all elements
        self.model.add_constraints(
            self._variables['active_hours'] == self.model.sum_temporal(status),
            name=f'{self.name_prefix}|active_hours',
        )

        # === inactive complementary: status + inactive == 1 ===
        if self.downtime_tracking_ids:
            status_subset = status.sel({dim: self.downtime_tracking_ids})
            inactive = self._variables['inactive']
            self.model.add_constraints(
                status_subset + inactive == 1,
                name=f'{self.name_prefix}|complementary',
            )

        # === State transitions: startup, shutdown ===
        if self.startup_tracking_ids:
            status_subset = status.sel({dim: self.startup_tracking_ids})
            startup = self._variables['startup']
            shutdown = self._variables['shutdown']

            # Vectorized transition constraint for t > 0
            self.model.add_constraints(
                startup.isel(time=slice(1, None)) - shutdown.isel(time=slice(1, None))
                == status_subset.isel(time=slice(1, None)) - status_subset.isel(time=slice(None, -1)),
                name=f'{self.name_prefix}|switch|transition',
            )

            # Vectorized mutex constraint
            self.model.add_constraints(
                startup + shutdown <= 1,
                name=f'{self.name_prefix}|switch|mutex',
            )

            # Initial constraint for t = 0 (if previous_status available)
            if previous_status_batched is not None:
                # Get elements that have both startup tracking AND previous status
                prev_element_ids = list(previous_status_batched.coords[dim].values)
                elements_with_initial = [eid for eid in self.startup_tracking_ids if eid in prev_element_ids]
                if elements_with_initial:
                    prev_status_subset = previous_status_batched.sel({dim: elements_with_initial})
                    prev_state = prev_status_subset.isel(time=-1)
                    startup_subset = startup.sel({dim: elements_with_initial})
                    shutdown_subset = shutdown.sel({dim: elements_with_initial})
                    status_initial = status_subset.sel({dim: elements_with_initial}).isel(time=0)

                    self.model.add_constraints(
                        startup_subset.isel(time=0) - shutdown_subset.isel(time=0) == status_initial - prev_state,
                        name=f'{self.name_prefix}|switch|initial',
                    )

        # === startup_count: sum(startup) == startup_count ===
        if self.startup_limit_ids:
            startup = self._variables['startup'].sel({dim: self.startup_limit_ids})
            startup_count = self._variables['startup_count']
            startup_temporal_dims = [d for d in startup.dims if d not in ('period', 'scenario', dim)]
            self.model.add_constraints(
                startup_count == startup.sum(startup_temporal_dims),
                name=f'{self.name_prefix}|startup_count',
            )

        # === Uptime tracking (per-element due to previous duration complexity) ===
        for elem_id in self.uptime_tracking_ids:
            elem = self._get_element_by_id(elem_id)
            params = self._get_params(elem)
            status_elem = status.sel({dim: elem_id})
            min_uptime = params.min_uptime
            max_uptime = params.max_uptime

            # Get previous uptime if available
            previous_uptime = None
            if previous_status_batched is not None and elem_id in previous_status_batched.coords.get(dim, []):
                prev_status = previous_status_batched.sel({dim: elem_id})
                if min_uptime is not None:
                    previous_uptime = self._compute_previous_duration(
                        prev_status, target_state=1, timestep_duration=self.model.timestep_duration
                    )

            self._add_consecutive_duration_tracking(
                state=status_elem,
                name=f'{elem_id}|uptime',
                minimum_duration=min_uptime,
                maximum_duration=max_uptime,
                previous_duration=previous_uptime,
            )

        # === Downtime tracking (per-element due to previous duration complexity) ===
        for elem_id in self.downtime_tracking_ids:
            elem = self._get_element_by_id(elem_id)
            params = self._get_params(elem)
            inactive = self._variables['inactive'].sel({dim: elem_id})
            min_downtime = params.min_downtime
            max_downtime = params.max_downtime

            # Get previous downtime if available
            previous_downtime = None
            if previous_status_batched is not None and elem_id in previous_status_batched.coords.get(dim, []):
                prev_status = previous_status_batched.sel({dim: elem_id})
                if min_downtime is not None:
                    previous_downtime = self._compute_previous_duration(
                        prev_status, target_state=0, timestep_duration=self.model.timestep_duration
                    )

            self._add_consecutive_duration_tracking(
                state=inactive,
                name=f'{elem_id}|downtime',
                minimum_duration=min_downtime,
                maximum_duration=max_downtime,
                previous_duration=previous_downtime,
            )

        # === Cluster cyclic constraints ===
        if self.model.flow_system.clusters is not None:
            cyclic_ids = self.cluster_cyclic_ids
            if cyclic_ids:
                status_cyclic = status.sel({dim: cyclic_ids})
                self.model.add_constraints(
                    status_cyclic.isel(time=0) == status_cyclic.isel(time=-1),
                    name=f'{self.name_prefix}|cluster_cyclic',
                )

        self._logger.debug(f'StatusesModel created constraints for {len(self.element_ids)} elements')

    def _add_consecutive_duration_tracking(
        self,
        state: linopy.Variable,
        name: str,
        minimum_duration: float | xr.DataArray | None = None,
        maximum_duration: float | xr.DataArray | None = None,
        previous_duration: float | xr.DataArray | None = None,
    ) -> None:
        """Add consecutive duration tracking constraints for a binary state variable.

        This implements the same logic as ModelingPrimitives.consecutive_duration_tracking
        but directly on FlowSystemModel without requiring a Submodel.

        Creates:
        - duration variable: tracks consecutive time in state
        - upper bound: duration[t] <= state[t] * M
        - forward constraint: duration[t+1] <= duration[t] + dt[t]
        - backward constraint: duration[t+1] >= duration[t] + dt[t] + (state[t+1] - 1) * M
        - optional lower bound if minimum_duration provided
        """
        duration_per_step = self.model.timestep_duration
        duration_dim = 'time'

        # Big-M value
        mega = duration_per_step.sum(duration_dim) + (previous_duration if previous_duration is not None else 0)

        # Duration variable
        upper_bound = maximum_duration if maximum_duration is not None else mega
        duration = self.model.add_variables(
            lower=0,
            upper=upper_bound,
            coords=state.coords,
            name=f'{name}|duration',
        )

        # Upper bound: duration[t] <= state[t] * M
        self.model.add_constraints(duration <= state * mega, name=f'{name}|duration|ub')

        # Forward constraint: duration[t+1] <= duration[t] + duration_per_step[t]
        self.model.add_constraints(
            duration.isel({duration_dim: slice(1, None)})
            <= duration.isel({duration_dim: slice(None, -1)}) + duration_per_step.isel({duration_dim: slice(None, -1)}),
            name=f'{name}|duration|forward',
        )

        # Backward constraint: duration[t+1] >= duration[t] + dt[t] + (state[t+1] - 1) * M
        self.model.add_constraints(
            duration.isel({duration_dim: slice(1, None)})
            >= duration.isel({duration_dim: slice(None, -1)})
            + duration_per_step.isel({duration_dim: slice(None, -1)})
            + (state.isel({duration_dim: slice(1, None)}) - 1) * mega,
            name=f'{name}|duration|backward',
        )

        # Initial constraint if previous_duration provided
        if previous_duration is not None:
            # duration[0] <= (state[0] * M) if previous_duration == 0, else handle differently
            self.model.add_constraints(
                duration.isel({duration_dim: 0})
                <= state.isel({duration_dim: 0}) * (previous_duration + duration_per_step.isel({duration_dim: 0})),
                name=f'{name}|duration|initial_ub',
            )
            self.model.add_constraints(
                duration.isel({duration_dim: 0})
                >= (state.isel({duration_dim: 0}) - 1) * mega
                + previous_duration
                + state.isel({duration_dim: 0}) * duration_per_step.isel({duration_dim: 0}),
                name=f'{name}|duration|initial_lb',
            )

        # Lower bound if minimum_duration provided
        if minimum_duration is not None:
            # At shutdown (state drops to 0), duration must have reached minimum
            # This requires tracking shutdown event
            pass  # Handled by bounds naturally via backward constraint

    def _compute_previous_duration(
        self, previous_status: xr.DataArray, target_state: int, timestep_duration
    ) -> xr.DataArray:
        """Compute consecutive duration of target_state at end of previous_status."""
        xr = self._xr
        # Simple implementation: count consecutive target_state values from the end
        # This is a scalar computation, not vectorized
        values = previous_status.values
        count = 0
        for v in reversed(values):
            if (target_state == 1 and v > 0) or (target_state == 0 and v == 0):
                count += 1
            else:
                break
        # Multiply by timestep_duration (which may be time-varying)
        if hasattr(timestep_duration, 'isel'):
            # If timestep_duration is xr.DataArray, use mean or last value
            duration = float(timestep_duration.mean()) * count
        else:
            duration = timestep_duration * count
        return xr.DataArray(duration)

    # === Effect factor properties (used by EffectsModel.finalize_shares) ===

    def _collect_effects(self, attr: str, element_ids: list[str] | None = None) -> dict[str, xr.DataArray]:
        """Collect effects from elements into a dict of DataArrays.

        Args:
            attr: The attribute name on StatusParameters (e.g., 'effects_per_active_hour').
            element_ids: Optional subset of element IDs to include.

        Returns:
            Dict mapping effect_name -> DataArray with element dimension.
        """
        if element_ids is None:
            elements = self.elements
            ids = self.element_ids
        else:
            id_set = set(element_ids)
            elements = [e for e in self.elements if self._get_element_id(e) in id_set]
            ids = element_ids

        # Find all effect names across all elements
        all_effects: set[str] = set()
        for e in elements:
            effects = getattr(self._get_params(e), attr) or {}
            all_effects.update(effects.keys())

        if not all_effects:
            return {}

        # Build DataArray for each effect
        result = {}
        for effect_name in all_effects:
            values = []
            for e in elements:
                effects = getattr(self._get_params(e), attr) or {}
                values.append(effects.get(effect_name, np.nan))
            result[effect_name] = xr.DataArray(values, dims=[self.dim_name], coords={self.dim_name: ids})

        return result

    @property
    def effects_per_active_hour(self) -> xr.DataArray | None:
        """Combined effects_per_active_hour with (element, effect) dims.

        Collects effects directly from element parameters.
        Returns None if no elements have effects defined.
        """
        effects_dict = self._collect_effects('effects_per_active_hour')
        if not effects_dict:
            return None
        return self._build_factors_from_dict(effects_dict)

    @property
    def effects_per_startup(self) -> xr.DataArray | None:
        """Combined effects_per_startup with (element, effect) dims.

        Collects effects directly from element parameters.
        Returns None if no elements have effects defined.
        """
        effects_dict = self._collect_effects('effects_per_startup', self.startup_tracking_ids)
        if not effects_dict:
            return None
        # Only include elements with startup tracking
        return self._build_factors_from_dict(effects_dict, element_ids=self.startup_tracking_ids)

    def _build_factors_from_dict(
        self, effects_dict: dict[str, xr.DataArray], element_ids: list[str] | None = None
    ) -> xr.DataArray | None:
        """Build factor array with (element, effect) dims from effects dict.

        Args:
            effects_dict: Dict mapping effect_name -> DataArray with element dim.
            element_ids: Optional subset of element IDs to include.

        Returns:
            DataArray with (element, effect) dims, NaN for missing effects.
        """
        if not effects_dict:
            return None

        effects_model = getattr(self.model.effects, '_batched_model', None)
        if effects_model is None:
            return None

        effect_ids = effects_model.effect_ids
        dim = self.dim_name

        # Subset elements if specified
        if element_ids is None:
            element_ids = self.element_ids

        # Build DataArray by stacking effects
        effect_arrays = []
        for effect_name in effect_ids:
            if effect_name in effects_dict:
                arr = effects_dict[effect_name]
                # Select subset of elements if needed
                if element_ids != self.element_ids:
                    arr = arr.sel({dim: element_ids})
            else:
                # NaN for effects not defined
                arr = xr.DataArray(
                    [np.nan] * len(element_ids),
                    dims=[dim],
                    coords={dim: element_ids},
                )
            effect_arrays.append(arr)

        result = xr.concat(effect_arrays, dim='effect').assign_coords(effect=effect_ids)
        return result.transpose(dim, 'effect')

    def get_variable(self, name: str, element_id: str | None = None):
        """Get a variable, optionally selecting a specific element."""
        var = self._variables.get(name)
        if var is None:
            return None
        if element_id is not None:
            dim = self.dim_name
            if element_id in var.coords.get(dim, []):
                return var.sel({dim: element_id})
            return None
        return var

    def get_status_variable(self, element_id: str):
        """Get the binary status variable for a specific element.

        Args:
            element_id: The element identifier (e.g., 'CHP(P_el)').

        Returns:
            The binary status variable for the specified element, or None.
        """
        dim = self.dim_name
        if element_id in self._batched_status_var.coords.get(dim, []):
            return self._batched_status_var.sel({dim: element_id})
        return None

    def get_previous_status(self, element_id: str):
        """Get the previous status for a specific element.

        Args:
            element_id: The element identifier (e.g., 'CHP(P_el)').

        Returns:
            The previous status DataArray for the specified element, or None.
        """
        elem = self._get_element_by_id(element_id)
        if elem is None:
            return None
        return self._get_previous_status(elem)

    @property
    def active_hours(self) -> linopy.Variable:
        """Batched active_hours variable with element dimension."""
        return self._variables['active_hours']

    @property
    def startup(self) -> linopy.Variable | None:
        """Batched startup variable with element dimension."""
        return self._variables.get('startup')

    @property
    def shutdown(self) -> linopy.Variable | None:
        """Batched shutdown variable with element dimension."""
        return self._variables.get('shutdown')

    @property
    def inactive(self) -> linopy.Variable | None:
        """Batched inactive variable with element dimension."""
        return self._variables.get('inactive')

    @property
    def startup_count(self) -> linopy.Variable | None:
        """Batched startup_count variable with element dimension."""
        return self._variables.get('startup_count')


class FlowStatusesModel(StatusesModel):
    """Type-level status model for flows.

    Implements the abstract methods from StatusesModel to access
    flow-specific status parameters directly.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        status: linopy.Variable,
        flows: list,
        previous_status_getter: callable | None = None,
        name_prefix: str = 'status',
    ):
        """Initialize the flow status model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            status: Batched status variable with flow dimension.
            flows: List of Flow objects with status_parameters.
            previous_status_getter: Optional function (flow) -> DataArray for previous status.
            name_prefix: Prefix for variable names.
        """
        self._flows = flows
        self._previous_status_getter = previous_status_getter
        super().__init__(
            model=model,
            status=status,
            elements=flows,
            dim_name='flow',
            name_prefix=name_prefix,
        )

    def _get_params(self, flow) -> StatusParameters:
        """Get StatusParameters from a flow."""
        return flow.status_parameters

    def _get_element_id(self, flow) -> str:
        """Get flow identifier (label_full)."""
        return flow.label_full

    def _get_previous_status(self, flow) -> xr.DataArray | None:
        """Get previous status for a flow."""
        if self._previous_status_getter is not None:
            return self._previous_status_getter(flow)
        return None


class ComponentStatusFeaturesModel(StatusesModel):
    """Type-level status model for component status features.

    Implements the abstract methods from StatusesModel to access
    component-specific status parameters directly.
    """

    def __init__(
        self,
        model: FlowSystemModel,
        status: linopy.Variable,
        components: list,
        previous_status_getter: callable | None = None,
        name_prefix: str = 'component',
    ):
        """Initialize the component status features model.

        Args:
            model: The FlowSystemModel to create variables/constraints in.
            status: Batched status variable with component dimension.
            components: List of Component objects with status_parameters.
            previous_status_getter: Optional function (component) -> DataArray for previous status.
            name_prefix: Prefix for variable names.
        """
        self._components = components
        self._previous_status_getter = previous_status_getter
        super().__init__(
            model=model,
            status=status,
            elements=components,
            dim_name='component',
            name_prefix=name_prefix,
        )

    def _get_params(self, component) -> StatusParameters:
        """Get StatusParameters from a component."""
        return component.status_parameters

    def _get_element_id(self, component) -> str:
        """Get component identifier (label, not label_full)."""
        return component.label

    def _get_previous_status(self, component) -> xr.DataArray | None:
        """Get previous status for a component."""
        if self._previous_status_getter is not None:
            return self._previous_status_getter(component)
        return None


class StatusModel(Submodel):
    """Mathematical model implementation for binary status.

    Creates optimization variables and constraints for binary status modeling,
    state transitions, duration tracking, and operational effects.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/StatusParameters/>
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        parameters: StatusParameters,
        status: linopy.Variable,
        previous_status: xr.DataArray | None,
        label_of_model: str | None = None,
    ):
        """
        This feature model is used to model the status (active/inactive) state of flow_rate(s).
        It does not matter if the flow_rates are bounded by a size variable or by a hard bound.
        The used bound here is the absolute highest/lowest bound!

        Args:
            model: The optimization model instance
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            parameters: The parameters of the feature model.
            status: The variable that determines the active state
            previous_status: The previous flow_rates
            label_of_model: The label of the model. This is needed to construct the full label of the model.
        """
        self.status = status
        self._previous_status = previous_status
        self.parameters = parameters
        super().__init__(model, label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create a separate binary 'inactive' variable when needed for downtime tracking or explicit use
        # When not needed, the expression (1 - self.status) can be used instead
        if self.parameters.use_downtime_tracking:
            inactive = self.add_variables(
                binary=True,
                short_name='inactive',
                coords=self._model.get_coords(),
                category=VariableCategory.INACTIVE,
            )
            self.add_constraints(self.status + inactive == 1, short_name='complementary')

        # 3. Total duration tracking
        total_hours = self._model.temporal_weight.sum(self._model.temporal_dims)
        ModelingPrimitives.expression_tracking_variable(
            self,
            tracked_expression=self._model.sum_temporal(self.status),
            bounds=(
                self.parameters.active_hours_min if self.parameters.active_hours_min is not None else 0,
                self.parameters.active_hours_max if self.parameters.active_hours_max is not None else total_hours,
            ),
            short_name='active_hours',
            coords=['period', 'scenario'],
            category=VariableCategory.TOTAL,
        )

        # 4. Switch tracking using existing pattern
        if self.parameters.use_startup_tracking:
            self.add_variables(
                binary=True,
                short_name='startup',
                coords=self.get_coords(),
                category=VariableCategory.STARTUP,
            )
            self.add_variables(
                binary=True,
                short_name='shutdown',
                coords=self.get_coords(),
                category=VariableCategory.SHUTDOWN,
            )

            # Determine previous_state: None means relaxed (no constraint at t=0)
            previous_state = self._previous_status.isel(time=-1) if self._previous_status is not None else None

            BoundingPatterns.state_transition_bounds(
                self,
                state=self.status,
                activate=self.startup,
                deactivate=self.shutdown,
                name=f'{self.label_of_model}|switch',
                previous_state=previous_state,
                coord='time',
            )

            if self.parameters.startup_limit is not None:
                count = self.add_variables(
                    lower=0,
                    upper=self.parameters.startup_limit,
                    coords=self._model.get_coords(('period', 'scenario')),
                    short_name='startup_count',
                    category=VariableCategory.STARTUP_COUNT,
                )
                # Sum over all temporal dimensions (time, and cluster if present)
                startup_temporal_dims = [d for d in self.startup.dims if d not in ('period', 'scenario')]
                self.add_constraints(count == self.startup.sum(startup_temporal_dims), short_name='startup_count')

        # 5. Consecutive active duration (uptime) using existing pattern
        if self.parameters.use_uptime_tracking:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state=self.status,
                short_name='uptime',
                minimum_duration=self.parameters.min_uptime,
                maximum_duration=self.parameters.max_uptime,
                duration_per_step=self.timestep_duration,
                duration_dim='time',
                previous_duration=self._get_previous_uptime(),
            )

        # 6. Consecutive inactive duration (downtime) using existing pattern
        if self.parameters.use_downtime_tracking:
            ModelingPrimitives.consecutive_duration_tracking(
                self,
                state=self.inactive,
                short_name='downtime',
                minimum_duration=self.parameters.min_downtime,
                maximum_duration=self.parameters.max_downtime,
                duration_per_step=self.timestep_duration,
                duration_dim='time',
                previous_duration=self._get_previous_downtime(),
            )

        # 7. Cyclic constraint for clustered systems
        self._add_cluster_cyclic_constraint()

        self._add_effects()

    def _add_cluster_cyclic_constraint(self):
        """For 'cyclic' cluster mode: each cluster's start status equals its end status."""
        if self._model.flow_system.clusters is not None and self.parameters.cluster_mode == 'cyclic':
            self.add_constraints(
                self.status.isel(time=0) == self.status.isel(time=-1),
                short_name='cluster_cyclic',
            )

    def _add_effects(self):
        """Add operational effects (use timestep_duration only, cluster_weight is applied when summing to total)"""
        if self.parameters.effects_per_active_hour:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.status * factor * self._model.timestep_duration
                    for effect, factor in self.parameters.effects_per_active_hour.items()
                },
                target='temporal',
            )

        if self.parameters.effects_per_startup:
            self._model.effects.add_share_to_effects(
                name=self.label_of_element,
                expressions={
                    effect: self.startup * factor for effect, factor in self.parameters.effects_per_startup.items()
                },
                target='temporal',
            )

    # Properties access variables from Submodel's tracking system

    @property
    def active_hours(self) -> linopy.Variable:
        """Total active hours variable"""
        return self['active_hours']

    @property
    def inactive(self) -> linopy.Variable | None:
        """Binary inactive state variable.

        Note:
            Only created when downtime tracking is enabled (min_downtime or max_downtime set).
            For general use, prefer the expression `1 - status` instead of this variable.
        """
        return self.get('inactive')

    @property
    def startup(self) -> linopy.Variable | None:
        """Startup variable"""
        return self.get('startup')

    @property
    def shutdown(self) -> linopy.Variable | None:
        """Shutdown variable"""
        return self.get('shutdown')

    @property
    def startup_count(self) -> linopy.Variable | None:
        """Number of startups variable"""
        return self.get('startup_count')

    @property
    def uptime(self) -> linopy.Variable | None:
        """Consecutive active hours (uptime) variable"""
        return self.get('uptime')

    @property
    def downtime(self) -> linopy.Variable | None:
        """Consecutive inactive hours (downtime) variable"""
        return self.get('downtime')

    def _get_previous_uptime(self):
        """Get previous uptime (consecutive active hours).

        Returns None if no previous status is provided (relaxed mode - no constraint at t=0).
        """
        if self._previous_status is None:
            return None  # Relaxed mode
        hours_per_step = self._model.timestep_duration.isel(time=0).min().item()
        return ModelingUtilities.compute_consecutive_hours_in_state(self._previous_status, hours_per_step)

    def _get_previous_downtime(self):
        """Get previous downtime (consecutive inactive hours).

        Returns None if no previous status is provided (relaxed mode - no constraint at t=0).
        """
        if self._previous_status is None:
            return None  # Relaxed mode
        hours_per_step = self._model.timestep_duration.isel(time=0).min().item()
        return ModelingUtilities.compute_consecutive_hours_in_state(1 - self._previous_status, hours_per_step)


class PieceModel(Submodel):
    """Class for modeling a linear piece of one or more variables in parallel"""

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        dims: Collection[FlowSystemDimensions] | None,
    ):
        self.inside_piece: linopy.Variable | None = None
        self.lambda0: linopy.Variable | None = None
        self.lambda1: linopy.Variable | None = None
        self.dims = dims

        super().__init__(model, label_of_element, label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.inside_piece = self.add_variables(
            binary=True,
            short_name='inside_piece',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.INSIDE_PIECE,
        )
        self.lambda0 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda0',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.LAMBDA0,
        )

        self.lambda1 = self.add_variables(
            lower=0,
            upper=1,
            short_name='lambda1',
            coords=self._model.get_coords(dims=self.dims),
            category=VariableCategory.LAMBDA1,
        )

        # Create constraints
        # eq:  lambda0(t) + lambda1(t) = inside_piece(t)
        self.add_constraints(self.inside_piece == self.lambda0 + self.lambda1, short_name='inside_piece')


class PiecewiseModel(Submodel):
    """Mathematical model implementation for piecewise linear approximations.

    Creates optimization variables and constraints for piecewise linear relationships,
    including lambda variables, piece activation binaries, and coupling constraints.

    Mathematical Formulation:
        See <https://flixopt.github.io/flixopt/latest/user-guide/mathematical-notation/features/Piecewise/>
    """

    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_variables: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
        dims: Collection[FlowSystemDimensions] | None,
    ):
        """
        Modeling a Piecewise relation between miultiple variables.
        The relation is defined by a list of Pieces, which are assigned to the variables.
        Each Piece is a tuple of (start, end).

        Args:
            model: The FlowSystemModel that is used to create the model.
            label_of_element: The label of the parent (Element). Used to construct the full label of the model.
            label_of_model: The label of the model. Used to construct the full label of the model.
            piecewise_variables: The variables to which the Pieces are assigned.
            zero_point: A variable that can be used to define a zero point for the Piecewise relation. If None or False, no zero point is defined.
            dims: The dimensions used for variable creation. If None, all dimensions are used.
        """
        self._piecewise_variables = piecewise_variables
        self._zero_point = zero_point
        self.dims = dims

        self.pieces: list[PieceModel] = []
        self.zero_point: linopy.Variable | None = None
        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Validate all piecewise variables have the same number of segments
        segment_counts = [len(pw) for pw in self._piecewise_variables.values()]
        if not all(count == segment_counts[0] for count in segment_counts):
            raise ValueError(f'All piecewises must have the same number of pieces, got {segment_counts}')

        # Create PieceModel submodels (which creates their variables and constraints)
        for i in range(len(list(self._piecewise_variables.values())[0])):
            new_piece = self.add_submodels(
                PieceModel(
                    model=self._model,
                    label_of_element=self.label_of_element,
                    label_of_model=f'{self.label_of_element}|Piece_{i}',
                    dims=self.dims,
                ),
                short_name=f'Piece_{i}',
            )
            self.pieces.append(new_piece)

        for var_name in self._piecewise_variables:
            variable = self._model.variables[var_name]
            self.add_constraints(
                variable
                == sum(
                    [
                        piece_model.lambda0 * piece_bounds.start + piece_model.lambda1 * piece_bounds.end
                        for piece_model, piece_bounds in zip(
                            self.pieces, self._piecewise_variables[var_name], strict=False
                        )
                    ]
                ),
                name=f'{self.label_full}|{var_name}|lambda',
                short_name=f'{var_name}|lambda',
            )

            # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
            # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
            if isinstance(self._zero_point, linopy.Variable):
                self.zero_point = self._zero_point
                rhs = self.zero_point
            elif self._zero_point is True:
                self.zero_point = self.add_variables(
                    coords=self._model.get_coords(self.dims),
                    binary=True,
                    short_name='zero_point',
                    category=VariableCategory.ZERO_POINT,
                )
                rhs = self.zero_point
            else:
                rhs = 1

            # This constraint ensures at most one segment is active at a time.
            # When zero_point is a binary variable, it acts as a gate:
            # - zero_point=1: at most one segment can be active (normal piecewise operation)
            # - zero_point=0: all segments must be inactive (effectively disables the piecewise)
            self.add_constraints(
                sum([piece.inside_piece for piece in self.pieces]) <= rhs,
                name=f'{self.label_full}|{variable.name}|single_segment',
                short_name=f'{var_name}|single_segment',
            )


class PiecewiseEffectsModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        label_of_element: str,
        label_of_model: str,
        piecewise_origin: tuple[str, Piecewise],
        piecewise_shares: dict[str, Piecewise],
        zero_point: bool | linopy.Variable | None,
    ):
        origin_count = len(piecewise_origin[1])
        share_counts = [len(pw) for pw in piecewise_shares.values()]
        if not all(count == origin_count for count in share_counts):
            raise ValueError(
                f'Piece count mismatch: piecewise_origin has {origin_count} segments, '
                f'but piecewise_shares have {share_counts}'
            )
        self._zero_point = zero_point
        self._piecewise_origin = piecewise_origin
        self._piecewise_shares = piecewise_shares
        self.shares: dict[str, linopy.Variable] = {}

        self.piecewise_model: PiecewiseModel | None = None

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.shares = {
            effect: self.add_variables(coords=self._model.get_coords(['period', 'scenario']), short_name=effect)
            for effect in self._piecewise_shares
        }

        piecewise_variables = {
            self._piecewise_origin[0]: self._piecewise_origin[1],
            **{
                self.shares[effect_label].name: self._piecewise_shares[effect_label]
                for effect_label in self._piecewise_shares
            },
        }

        # Create piecewise model (which creates its variables and constraints)
        self.piecewise_model = self.add_submodels(
            PiecewiseModel(
                model=self._model,
                label_of_element=self.label_of_element,
                piecewise_variables=piecewise_variables,
                zero_point=self._zero_point,
                dims=('period', 'scenario'),
                label_of_model=f'{self.label_of_element}|PiecewiseEffects',
            ),
            short_name='PiecewiseEffects',
        )

        # Add shares to effects
        self._model.effects.add_share_to_effects(
            name=self.label_of_element,
            expressions={effect: variable * 1 for effect, variable in self.shares.items()},
            target='periodic',
        )


class ShareAllocationModel(Submodel):
    def __init__(
        self,
        model: FlowSystemModel,
        dims: list[FlowSystemDimensions],
        label_of_element: str | None = None,
        label_of_model: str | None = None,
        total_max: Numeric_PS | None = None,
        total_min: Numeric_PS | None = None,
        max_per_hour: Numeric_TPS | None = None,
        min_per_hour: Numeric_TPS | None = None,
    ):
        if 'time' not in dims and (max_per_hour is not None or min_per_hour is not None):
            raise ValueError("max_per_hour and min_per_hour require 'time' dimension in dims")

        self._dims = dims
        self.total_per_timestep: linopy.Variable | None = None
        self.total: linopy.Variable | None = None
        self.shares: dict[str, linopy.Variable] = {}
        self.share_constraints: dict[str, linopy.Constraint] = {}

        self._eq_total_per_timestep: linopy.Constraint | None = None
        self._eq_total: linopy.Constraint | None = None

        # Parameters
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

        super().__init__(model, label_of_element=label_of_element, label_of_model=label_of_model)

    def _do_modeling(self):
        """Create variables, constraints, and nested submodels"""
        super()._do_modeling()

        # Create variables
        self.total = self.add_variables(
            lower=self._total_min if self._total_min is not None else -np.inf,
            upper=self._total_max if self._total_max is not None else np.inf,
            coords=self._model.get_coords([dim for dim in self._dims if dim != 'time']),
            name=self.label_full,
            short_name='total',
            category=VariableCategory.TOTAL,
        )
        # eq: sum = sum(share_i) # skalar
        self._eq_total = self.add_constraints(self.total == 0, name=self.label_full)

        if 'time' in self._dims:
            self.total_per_timestep = self.add_variables(
                lower=-np.inf if (self._min_per_hour is None) else self._min_per_hour * self._model.timestep_duration,
                upper=np.inf if (self._max_per_hour is None) else self._max_per_hour * self._model.timestep_duration,
                coords=self._model.get_coords(self._dims),
                short_name='per_timestep',
                category=VariableCategory.PER_TIMESTEP,
            )

            self._eq_total_per_timestep = self.add_constraints(self.total_per_timestep == 0, short_name='per_timestep')

            # Add it to the total (cluster_weight handles cluster representation, defaults to 1.0)
            # Sum over all temporal dimensions (time, and cluster if present)
            weighted_per_timestep = self.total_per_timestep * self._model.weights.get('cluster', 1.0)
            self._eq_total.lhs -= weighted_per_timestep.sum(dim=self._model.temporal_dims)

    def add_share(
        self,
        name: str,
        expression: linopy.LinearExpression,
        dims: list[FlowSystemDimensions] | None = None,
    ):
        """
        Add a share to the share allocation model. If the share already exists, the expression is added to the existing share.
        The expression is added to the right hand side (rhs) of the constraint.
        The variable representing the total share is on the left hand side (lhs) of the constraint.
        var_total = sum(expressions)

        Args:
            name: The name of the share.
            expression: The expression of the share. Added to the right hand side of the constraint.
            dims: The dimensions of the share. Defaults to all dimensions. Dims are ordered automatically
        """
        if dims is None:
            dims = self._dims
        else:
            if 'time' in dims and 'time' not in self._dims:
                raise ValueError('Cannot add share with time-dim to a model without time-dim')
            if 'period' in dims and 'period' not in self._dims:
                raise ValueError('Cannot add share with period-dim to a model without period-dim')
            if 'scenario' in dims and 'scenario' not in self._dims:
                raise ValueError('Cannot add share with scenario-dim to a model without scenario-dim')

        if name in self.shares:
            self.share_constraints[name].lhs -= expression
        else:
            # Temporal shares (with 'time' dim) are segment totals that need division
            category = VariableCategory.SHARE if 'time' in dims else None
            self.shares[name] = self.add_variables(
                coords=self._model.get_coords(dims),
                name=f'{name}->{self.label_full}',
                short_name=name,
                category=category,
            )

            self.share_constraints[name] = self.add_constraints(
                self.shares[name] == expression, name=f'{name}->{self.label_full}'
            )

            if 'time' not in dims:
                self._eq_total.lhs -= self.shares[name]
            else:
                self._eq_total_per_timestep.lhs -= self.shares[name]
