"""
This module contains classes to collect Parameters for the Investment and OnOff decisions.
These are tightly connected to features.py
"""

import logging
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Optional, Union

import xarray as xr

from .config import CONFIG
from .core import NonTemporalData, NonTemporalDataUser, Scalar, TemporalDataUser
from .structure import Interface, register_class_for_io

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from .effects import NonTemporalEffectsUser, TemporalEffectsUser
    from .flow_system import FlowSystem


logger = logging.getLogger('flixopt')


@register_class_for_io
class Piece(Interface):
    def __init__(self, start: TemporalDataUser, end: TemporalDataUser):
        """
        Define a Piece, which is part of a Piecewise object.

        Args:
            start: The x-values of the piece.
            end: The end of the piece.
        """
        self.start = start
        self.end = end
        self.has_time_dim = False

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        dims = None if self.has_time_dim else ['year', 'scenario']
        self.start = flow_system.fit_to_model_coords(f'{name_prefix}|start', self.start, dims=dims)
        self.end = flow_system.fit_to_model_coords(f'{name_prefix}|end', self.end, dims=dims)


@register_class_for_io
class Piecewise(Interface):
    def __init__(self, pieces: List[Piece]):
        """
        Define a Piecewise, consisting of a list of Pieces.

        Args:
            pieces: The pieces of the piecewise.
        """
        self.pieces = pieces
        self._has_time_dim = False

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        for piece in self.pieces:
            piece.has_time_dim = value

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, index) -> Piece:
        return self.pieces[index]  # Enables indexing like piecewise[i]

    def __iter__(self) -> Iterator[Piece]:
        return iter(self.pieces)  # Enables iteration like for piece in piecewise: ...

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        for i, piece in enumerate(self.pieces):
            piece.transform_data(flow_system, f'{name_prefix}|Piece{i}')


@register_class_for_io
class PiecewiseConversion(Interface):
    def __init__(self, piecewises: Dict[str, Piecewise]):
        """
        Define a piecewise conversion between multiple Flows.
        --> "gaps" can be expressed by a piece not starting at the end of the prior piece: [(1,3), (4,5)]
        --> "points" can expressed as piece with same begin and end: [(3,3), (4,4)]

        Args:
            piecewises: Dict of Piecewises defining the conversion factors. flow labels as keys, piecewise as values
        """
        self.piecewises = piecewises
        self._has_time_dim = True
        self.has_time_dim = True  # Inital propagation

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        for piecewise in self.piecewises.values():
            piecewise.has_time_dim = value

    def items(self):
        return self.piecewises.items()

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        for name, piecewise in self.piecewises.items():
            piecewise.transform_data(flow_system, f'{name_prefix}|{name}')


@register_class_for_io
class PiecewiseEffects(Interface):
    def __init__(self, piecewise_origin: Piecewise, piecewise_shares: Dict[str, Piecewise]):
        """
        Define piecewise effects related to a variable.

        Args:
            piecewise_origin: Piecewise of the related variable
            piecewise_shares: Piecewise defining the shares to different Effects
        """
        self.piecewise_origin = piecewise_origin
        self.piecewise_shares = piecewise_shares
        self._has_time_dim = False
        self.has_time_dim = False  # Inital propagation

    @property
    def has_time_dim(self):
        return self._has_time_dim

    @has_time_dim.setter
    def has_time_dim(self, value):
        self._has_time_dim = value
        self.piecewise_origin.has_time_dim = value
        for piecewise in self.piecewise_shares.values():
            piecewise.has_time_dim = value

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        self.piecewise_origin.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|origin')
        for effect, piecewise in self.piecewise_shares.items():
            piecewise.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects|{effect}')


@register_class_for_io
class InvestParameters(Interface):
    """
    collects arguments for invest-stuff
    """

    def __init__(
        self,
        fixed_size: Optional[NonTemporalDataUser] = None,
        minimum_size: Optional[NonTemporalDataUser] = None,
        maximum_size: Optional[NonTemporalDataUser] = None,
        optional: bool = True,  # Investition ist weglassbar
        fix_effects: Optional['NonTemporalEffectsUser'] = None,
        specific_effects: Optional['NonTemporalEffectsUser'] = None,  # costs per Flow-Unit/Storage-Size/...
        piecewise_effects: Optional[PiecewiseEffects] = None,
        divest_effects: Optional['NonTemporalEffectsUser'] = None,
        investment_scenarios: Optional[Union[Literal['individual'], List[Union[int, str]]]] = None,
    ):
        """
        Args:
            fix_effects: Fixed investment costs if invested. (Attention: Annualize costs to chosen period!)
            divest_effects: Fixed divestment costs (if not invested, e.g., demolition costs or contractual penalty).
            fixed_size: Determines if the investment size is fixed.
            optional: If True, investment is not forced.
            specific_effects: Specific costs, e.g., in €/kW_nominal or €/m²_nominal.
                Example: {costs: 3, CO2: 0.3} with costs and CO2 representing an Object of class Effect
                (Attention: Annualize costs to chosen period!)
            piecewise_effects: Define the effects of the investment as a piecewise function of the size of the investment.
            minimum_size: Minimum possible size of the investment.
            maximum_size: Maximum possible size of the investment.
            investment_scenarios: For which scenarios to optimize the size for.
                - 'individual': Optimize the size of each scenario individually
                - List of scenario names: Optimize the size for the passed scenario names (equal size in all). All other scenarios will have the size 0.
                - None: Equals to a list of all scenarios (default)
        """
        self.fix_effects: 'NonTemporalEffectsUser' = fix_effects if fix_effects is not None else {}
        self.divest_effects: 'NonTemporalEffectsUser' = divest_effects if divest_effects is not None else {}
        self.fixed_size = fixed_size
        self.optional = optional
        self.specific_effects: 'NonTemporalEffectsUser' = specific_effects if specific_effects is not None else {}
        self.piecewise_effects = piecewise_effects
        self.minimum_size = minimum_size if minimum_size is not None else CONFIG.modeling.EPSILON
        self.maximum_size = maximum_size if maximum_size is not None else CONFIG.modeling.BIG  # default maximum
        self.investment_scenarios = investment_scenarios

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        self._plausibility_checks(flow_system)
        self.fix_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.fix_effects,
            label_suffix='fix_effects',
            dims=['year', 'scenario'],
        )
        self.divest_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.divest_effects,
            label_suffix='divest_effects',
            dims=['year', 'scenario'],
        )
        self.specific_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.specific_effects,
            label_suffix='specific_effects',
            dims=['year', 'scenario'],
        )
        if self.piecewise_effects is not None:
            self.piecewise_effects.has_time_dim = False
            self.piecewise_effects.transform_data(flow_system, f'{name_prefix}|PiecewiseEffects')

        self.minimum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_size', self.minimum_size, dims=['year', 'scenario']
        )
        self.maximum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_size', self.maximum_size, dims=['year', 'scenario']
        )
        if self.fixed_size is not None:
            self.fixed_size = flow_system.fit_to_model_coords(
                f'{name_prefix}|fixed_size', self.fixed_size, dims=['year', 'scenario']
            )

    def _plausibility_checks(self, flow_system):
        if isinstance(self.investment_scenarios, list):
            if not set(self.investment_scenarios).issubset(flow_system.scenarios):
                raise ValueError(
                    f'Some scenarios in investment_scenarios are not present in the time_series_collection: '
                    f'{set(self.investment_scenarios) - set(flow_system.scenarios)}'
                )
        if self.investment_scenarios is not None:
            if not self.optional:
                if self.minimum_size is not None or self.fixed_size is not None:
                    logger.warning(
                        'When using investment_scenarios, minimum_size and fixed_size should only ne used if optional is True.'
                        'Otherwise the investment cannot be 0 incertain scenarios while being non-zero in others.'
                    )

    @property
    def minimum_or_fixed_size(self) -> NonTemporalData:
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> NonTemporalData:
        return self.fixed_size if self.fixed_size is not None else self.maximum_size


# Base interface for common parameters
@register_class_for_io
class _BaseYearAwareInvestParameters(Interface):
    """
    Base parameters for year-aware investment modeling.
    Contains common sizing and effects parameters used by all variants.
    """

    def __init__(
        self,
        # Basic sizing parameters
        minimum_size: Optional[Scalar] = None,
        maximum_size: Optional[Scalar] = None,
        fixed_size: Optional[Scalar] = None,
        optional: bool = False,
        # Direct effects
        effects_of_investment_per_size: Optional['NonTemporalEffectsUser'] = None,
        effects_of_investment: Optional['NonTemporalEffectsUser'] = None,
    ):
        """
        Initialize base year-aware investment parameters.

        Args:
            minimum_size: Minimum investment size when invested. Defaults to CONFIG.modeling.EPSILON.
            maximum_size: Maximum possible investment size. Defaults to CONFIG.modeling.BIG.
            fixed_size: If specified, investment size is fixed to this value.
            effects_of_investment_per_size: Effects applied per unit of investment size for each year invested.
                Example: {'costs': 100} applies 100 * size * years_invested to total costs.
            effects_of_investment: One-time effects applied when investment decision is made.
                Example: {'costs': 1000} applies 1000 to costs in the investment year.
        """
        self.minimum_size = minimum_size if minimum_size is not None else CONFIG.modeling.EPSILON
        self.maximum_size = maximum_size if maximum_size is not None else CONFIG.modeling.BIG
        self.fixed_size = fixed_size
        self.optional = optional

        self.effects_of_investment_per_size: 'NonTemporalEffectsUser' = (
            effects_of_investment_per_size if effects_of_investment_per_size is not None else {}
        )
        self.effects_of_investment: 'NonTemporalEffectsUser' = (
            effects_of_investment if effects_of_investment is not None else {}
        )

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        """Transform all parameter data to match the flow system's coordinate structure."""
        self._plausibility_checks(flow_system)

        self.effects_of_investment_per_size = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.effects_of_investment_per_size,
            label_suffix='effects_of_investment_per_size',
            dims=['year', 'scenario'],
        )
        self.effects_of_investment = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.effects_of_investment,
            label_suffix='effects_of_investment',
            dims=['year', 'scenario'],
        )

        self.minimum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_size', self.minimum_size, dims=['year', 'scenario']
        )
        self.maximum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_size', self.maximum_size, dims=['year', 'scenario']
        )
        if self.fixed_size is not None:
            self.fixed_size = flow_system.fit_to_model_coords(
                f'{name_prefix}|fixed_size', self.fixed_size, dims=['year', 'scenario']
            )

    def _plausibility_checks(self, flow_system):
        """Validate parameter consistency and compatibility with the flow system."""
        if flow_system.years is None:
            raise ValueError("YearAwareInvestParameters requires the flow_system to have a 'years' dimension.")

    @property
    def minimum_or_fixed_size(self) -> NonTemporalData:
        """Get the effective minimum size (fixed size takes precedence)."""
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> NonTemporalData:
        """Get the effective maximum size (fixed size takes precedence)."""
        return self.fixed_size if self.fixed_size is not None else self.maximum_size

    @property
    def is_fixed_size(self) -> bool:
        """Check if investment size is fixed."""
        return self.fixed_size is not None


YearOfInvestmentData = NonTemporalDataUser
"""This datatype is used to define things related to the year of investment."""
YearOfInvestmentDataBool = Union[bool, YearOfInvestmentData]
"""This datatype is used to define things with boolean data related to the year of investment."""


@register_class_for_io
class InvestTimingParameters(Interface):
    """
    Investment with fixed start and end years.

    This is the simplest variant - investment is completely scheduled.
    No optimization variables needed for timing, just size optimization.
    """

    def __init__(
        self,
        allow_investment: YearOfInvestmentDataBool = True,
        allow_decommissioning: YearOfInvestmentDataBool = True,
        force_investment: YearOfInvestmentDataBool = False,  # TODO: Allow to simply pass the year
        force_decommissioning: YearOfInvestmentDataBool = False,  # TODO: Allow to simply pass the year
        fixed_lifetime: Optional[Scalar] = None,
        minimum_lifetime: Optional[Scalar] = None,
        maximum_lifetime: Optional[Scalar] = None,
        minimum_size: Optional[YearOfInvestmentData] = None,
        maximum_size: Optional[YearOfInvestmentData] = None,
        fixed_size: Optional[YearOfInvestmentData] = None,
        fix_effects: Optional['NonTemporalEffectsUser'] = None,
        specific_effects: Optional['NonTemporalEffectsUser'] = None,  # costs per Flow-Unit/Storage-Size/...
        fixed_effects_by_investment_year: Optional[xr.DataArray] = None,
        specific_effects_by_investment_year: Optional[xr.DataArray] = None,
        previous_lifetime: Optional[Scalar] = None,
    ):
        """
        These parameters are used to include the timing of investments in the model.
        Two out of three parameters (year_of_investment, year_of_decommissioning, duration_in_years) can be fixed.
        This has a 'year_of_investment' dimension in some parameters:
            allow_investment: Whether investment is allowed in a certain year
            allow_decommissioning: Whether divestment is allowed in a certain year
            duration_between_investment_and_decommissioning: Duration between investment and decommissioning

        Args:
            allow_investment: Allow investment in a certain year. By default, allow it in all years.
            allow_decommissioning: Allow decommissioning in a certain year. By default, allow it in all years.
            force_investment: Force the investment to occur in a certain year.
            force_decommissioning: Force the decommissioning to occur in a certain year.
            fixed_lifetime: Fix the lifetime of an investment (duration between investment and decommissioning).
            minimum_size: Minimum possible size of the investment. Can depend on the year of investment.
            maximum_size: Maximum possible size of the investment. Can depend on the year of investment.
            fixed_size: Fix the size of the investment. Can depend on the year of investment. Can still be 0 if not forced.
            specific_effects: Effects dependent on the size.
                These will occur in each year, depending on the size in that year.
            fix_effects: Effects of the Investment, independent of the size.
                These will occur in each year, depending on wether the size is greater zero in that year.

            fixed_effects_by_investment_year: Effects dependent on the year of investment.
                These effects will depend on the year of the investment. The actual effects can occur in other years,
                letting you model things like annuities, which depend on when an investment was taken.
                The passed xr.DataArray needs to match the FlowSystem dimensions (except time, but including "year_of_investment"). No internal Broadcasting!
                "year_of_investment" has the same values as the year dimension. Access it through `flow_system.year_of_investment`.
            specific_effects_by_investment_year: Effects dependent on the year of investment and the chosen size.
                These effects will depend on the year of the investment. The actual effects can occur in other years,
                letting you model things like annuities, which depend on when an investment was taken.
                The passed xr.DataArray needs to match the FlowSystem dimensions (except time, but including "year_of_investment"). No internal Broadcasting!
                "year_of_investment" has the same values as the year dimension. Access it through `flow_system.year_of_investment`.

        """
        self.minimum_size = minimum_size if minimum_size is not None else CONFIG.modeling.EPSILON
        self.maximum_size = maximum_size if maximum_size is not None else CONFIG.modeling.BIG
        self.fixed_size = fixed_size

        self.allow_investment = allow_investment
        self.allow_decommissioning = allow_decommissioning
        self.force_investment = force_investment
        self.force_decommissioning = force_decommissioning

        self.maximum_lifetime = maximum_lifetime
        self.minimum_lifetime = minimum_lifetime
        self.fixed_lifetime = fixed_lifetime
        self.previous_lifetime = previous_lifetime

        self.fix_effects: 'NonTemporalEffectsUser' = fix_effects if fix_effects is not None else {}
        self.specific_effects: 'NonTemporalEffectsUser' = specific_effects if specific_effects is not None else {}
        self.fixed_effects_by_investment_year = (
            fixed_effects_by_investment_year if fixed_effects_by_investment_year is not None else {}
        )
        self.specific_effects_by_investment_year = (
            specific_effects_by_investment_year if specific_effects_by_investment_year is not None else {}
        )

    def _plausibility_checks(self, flow_system):
        """Validate parameter consistency."""
        if flow_system.years is None:
            raise ValueError("YearAwareInvestParameters requires the flow_system to have a 'years' dimension.")

        if (self.force_investment.sum('year') > 1).any():
            raise ValueError('force_investment can only be True for a single year.')
        if (self.force_decommissioning.sum('year') > 1).any():
            raise ValueError('force_decommissioning can only be True for a single year.')

        if (self.force_investment.sum('year') == 1).any() and (self.force_decommissioning.sum('year') == 1).any():
            year_of_forced_investment = (
                self.force_investment.where(self.force_investment) * self.force_investment.year
            ).sum('year')
            year_of_forced_decommissioning = (
                self.force_decommissioning.where(self.force_decommissioning) * self.force_decommissioning.year
            ).sum('year')
            if not (year_of_forced_investment < year_of_forced_decommissioning).all():
                raise ValueError(
                    f'force_investment needs to be before force_decommissioning. Got:\n'
                    f'{self.force_investment}\nand\n{self.force_decommissioning}'
                )

        if self.previous_lifetime is not None:
            if self.fixed_size is None:
                # TODO: Might be only a warning
                raise ValueError('previous_lifetime can only be used if fixed_size is defined.')
            if self.force_investment is False:
                # TODO: Might be only a warning
                raise ValueError('previous_lifetime can only be used if force_investment is True.')

        if self.minimum_or_fixed_lifetime is not None and self.maximum_or_fixed_lifetime is not None:
            years = flow_system.years.values

            infeasible_years = []
            for i, inv_year in enumerate(years[:-1]):  # Exclude last year
                future_years = years[i + 1 :]  # All years after investment
                min_decomm = self.minimum_or_fixed_lifetime + inv_year
                max_decomm = self.maximum_or_fixed_lifetime + inv_year
                if max_decomm >= years[-1]:
                    continue

                # Check if any future year falls in decommissioning window
                future_years_da = xr.DataArray(future_years, dims=['year'])
                valid_decomm = ((min_decomm <= future_years_da) & (future_years_da <= max_decomm)).any('year')
                if not valid_decomm.all():
                    infeasible_years.append(inv_year)

            if infeasible_years:
                logger.warning(
                    f'Plausibility Check in {self.__class__.__name__}:\n'
                    f'  Investment years with no feasible decommissioning: {[int(year) for year in infeasible_years]}\n'
                    f'  Consider relaxing the lifetime constraints or including more years into your model.\n'
                    f'  Lifetime:\n'
                    f'      min={self.minimum_or_fixed_lifetime}\n'
                    f'      max={self.maximum_or_fixed_lifetime}\n'
                    f'  Model years: {list(flow_system.years)}\n'
                )

        specify_timing = (
            (self.fixed_lifetime is not None)
            + bool((self.force_investment.sum('year') > 1).any())
            + bool((self.force_decommissioning.sum('year') > 1).any())
        )

        if specify_timing in (0, 3):
            # TODO: Is there a valid use case for this? Should this be checked at all?
            logger.warning(
                'Either the the lifetime of an investment should be fixed, or the investment or decommissioning '
                'needs to be forced in a certain year.'
            )

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        """Transform all parameter data to match the flow system's coordinate structure."""
        self.fix_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.fix_effects,
            label_suffix='fix_effects',
            dims=['year', 'scenario'],
        )
        self.specific_effects = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.specific_effects,
            label_suffix='specific_effects',
            dims=['year', 'scenario'],
        )
        self.maximum_lifetime = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_lifetime', self.maximum_lifetime, dims=['scenario']
        )
        self.minimum_lifetime = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_lifetime', self.minimum_lifetime, dims=['scenario']
        )
        self.fixed_lifetime = flow_system.fit_to_model_coords(
            f'{name_prefix}|fixed_lifetime', self.fixed_lifetime, dims=['scenario']
        )

        self.force_investment = flow_system.fit_to_model_coords(
            f'{name_prefix}|force_investment', self.force_investment, dims=['year', 'scenario']
        )
        self.force_decommissioning = flow_system.fit_to_model_coords(
            f'{name_prefix}|force_decommissioning', self.force_decommissioning, dims=['year', 'scenario']
        )

        self.minimum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|minimum_size', self.minimum_size, dims=['year', 'scenario']
        )
        self.maximum_size = flow_system.fit_to_model_coords(
            f'{name_prefix}|maximum_size', self.maximum_size, dims=['year', 'scenario']
        )
        if self.fixed_size is not None:
            self.fixed_size = flow_system.fit_to_model_coords(
                f'{name_prefix}|fixed_size', self.fixed_size, dims=['year', 'scenario']
            )

        # TODO: self.previous_size to only scenarios

        # No Broadcasting! Until a safe way is established, we need to do check for this!
        self.fixed_effects_by_investment_year = flow_system.effects.create_effect_values_dict(
            self.fixed_effects_by_investment_year
        )
        for effect, da in self.fixed_effects_by_investment_year.items():
            dims = set(da.coords)
            if not {'year_of_investment', 'year'}.issubset(dims):
                raise ValueError(
                    f'fixed_effects_by_investment_year need to have a "year_of_investment" dimension and a '
                    f'"year" dimension. Got {dims} for effect {effect}'
                )
        self.specific_effects_by_investment_year = flow_system.effects.create_effect_values_dict(
            self.specific_effects_by_investment_year
        )
        for effect, da in self.specific_effects_by_investment_year.items():
            dims = set(da.coords)
            if not {'year_of_investment', 'year'}.issubset(dims):
                raise ValueError(
                    f'specific_effects_by_investment_year need to have a "year_of_investment" dimension and a '
                    f'"year" dimension. Got {dims} for effect {effect}'
                )
        self.fixed_effects_by_investment_year = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.fixed_effects_by_investment_year,
            label_suffix='fixed_effects_by_investment_year',
            dims=['year', 'scenario'],
            with_year_of_investment=True,
        )
        self.specific_effects_by_investment_year = flow_system.fit_effects_to_model_coords(
            label_prefix=name_prefix,
            effect_values=self.specific_effects_by_investment_year,
            label_suffix='specific_effects_by_investment_year',
            dims=['year', 'scenario'],
            with_year_of_investment=True,
        )

        self._plausibility_checks(flow_system)

    @property
    def minimum_or_fixed_size(self) -> NonTemporalData:
        """Get the effective minimum size (fixed size takes precedence)."""
        return self.fixed_size if self.fixed_size is not None else self.minimum_size

    @property
    def maximum_or_fixed_size(self) -> NonTemporalData:
        """Get the effective maximum size (fixed size takes precedence)."""
        return self.fixed_size if self.fixed_size is not None else self.maximum_size

    @property
    def is_fixed_size(self) -> bool:
        """Check if investment size is fixed."""
        return self.fixed_size is not None

    @property
    def minimum_or_fixed_lifetime(self) -> NonTemporalData:
        """Get the effective minimum lifetime (fixed lifetime takes precedence)."""
        return self.fixed_lifetime if self.fixed_lifetime is not None else self.minimum_lifetime

    @property
    def maximum_or_fixed_lifetime(self) -> NonTemporalData:
        """Get the effective maximum lifetime (fixed lifetime takes precedence)."""
        return self.fixed_lifetime if self.fixed_lifetime is not None else self.maximum_lifetime


@register_class_for_io
class OnOffParameters(Interface):
    def __init__(
        self,
        effects_per_switch_on: Optional['TemporalEffectsUser'] = None,
        effects_per_running_hour: Optional['TemporalEffectsUser'] = None,
        on_hours_total_min: Optional[int] = None,
        on_hours_total_max: Optional[int] = None,
        consecutive_on_hours_min: Optional[TemporalDataUser] = None,
        consecutive_on_hours_max: Optional[TemporalDataUser] = None,
        consecutive_off_hours_min: Optional[TemporalDataUser] = None,
        consecutive_off_hours_max: Optional[TemporalDataUser] = None,
        switch_on_total_max: Optional[int] = None,
        force_switch_on: bool = False,
    ):
        """
        Bundles information about the on and off state of an Element.
        If no parameters are given, the default is to create a binary variable for the on state
        without further constraints or effects and a variable for the total on hours.

        Args:
            effects_per_switch_on: cost of one switch from off (var_on=0) to on (var_on=1),
                unit i.g. in Euro
            effects_per_running_hour: costs for operating, i.g. in € per hour
            on_hours_total_min: min. overall sum of operating hours.
            on_hours_total_max: max. overall sum of operating hours.
            consecutive_on_hours_min: min sum of operating hours in one piece
                (last on-time period of timeseries is not checked and can be shorter)
            consecutive_on_hours_max: max sum of operating hours in one piece
            consecutive_off_hours_min: min sum of non-operating hours in one piece
                (last off-time period of timeseries is not checked and can be shorter)
            consecutive_off_hours_max: max sum of non-operating hours in one piece
            switch_on_total_max: max nr of switchOn operations
            force_switch_on: force creation of switch on variable, even if there is no switch_on_total_max
        """
        self.effects_per_switch_on: 'TemporalEffectsUser' = (
            effects_per_switch_on if effects_per_switch_on is not None else {}
        )
        self.effects_per_running_hour: 'TemporalEffectsUser' = (
            effects_per_running_hour if effects_per_running_hour is not None else {}
        )
        self.on_hours_total_min: Scalar = on_hours_total_min
        self.on_hours_total_max: Scalar = on_hours_total_max
        self.consecutive_on_hours_min: TemporalDataUser = consecutive_on_hours_min
        self.consecutive_on_hours_max: TemporalDataUser = consecutive_on_hours_max
        self.consecutive_off_hours_min: TemporalDataUser = consecutive_off_hours_min
        self.consecutive_off_hours_max: TemporalDataUser = consecutive_off_hours_max
        self.switch_on_total_max: Scalar = switch_on_total_max
        self.force_switch_on: bool = force_switch_on

    def transform_data(self, flow_system: 'FlowSystem', name_prefix: str = '') -> None:
        self.effects_per_switch_on = flow_system.fit_effects_to_model_coords(
            name_prefix, self.effects_per_switch_on, 'per_switch_on'
        )
        self.effects_per_running_hour = flow_system.fit_effects_to_model_coords(
            name_prefix, self.effects_per_running_hour, 'per_running_hour'
        )
        self.consecutive_on_hours_min = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_on_hours_min', self.consecutive_on_hours_min
        )
        self.consecutive_on_hours_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_on_hours_max', self.consecutive_on_hours_max
        )
        self.consecutive_off_hours_min = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_off_hours_min', self.consecutive_off_hours_min
        )
        self.consecutive_off_hours_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|consecutive_off_hours_max', self.consecutive_off_hours_max
        )
        self.on_hours_total_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|on_hours_total_max', self.on_hours_total_max, dims=['year', 'scenario']
        )
        self.on_hours_total_min = flow_system.fit_to_model_coords(
            f'{name_prefix}|on_hours_total_min', self.on_hours_total_min, dims=['year', 'scenario']
        )
        self.switch_on_total_max = flow_system.fit_to_model_coords(
            f'{name_prefix}|switch_on_total_max', self.switch_on_total_max, dims=['year', 'scenario']
        )

    @property
    def use_off(self) -> bool:
        """Determines wether the OFF Variable is needed or not"""
        return self.use_consecutive_off_hours

    @property
    def use_consecutive_on_hours(self) -> bool:
        """Determines wether a Variable for consecutive off hours is needed or not"""
        return any(param is not None for param in [self.consecutive_on_hours_min, self.consecutive_on_hours_max])

    @property
    def use_consecutive_off_hours(self) -> bool:
        """Determines wether a Variable for consecutive off hours is needed or not"""
        return any(param is not None for param in [self.consecutive_off_hours_min, self.consecutive_off_hours_max])

    @property
    def use_switch_on(self) -> bool:
        """Determines wether a Variable for SWITCH-ON is needed or not"""
        if self.force_switch_on:
            return True

        return any(
            param is not None and param != {}
            for param in [
                self.effects_per_switch_on,
                self.switch_on_total_max,
            ]
        )
