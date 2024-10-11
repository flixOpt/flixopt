# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:40:23 2020
developed by Felix Panitz* and Peter Stange*
* at Chair of Building Energy Systems and Heat Supply, Technische Universität Dresden
"""

import textwrap
from typing import List, Set, Tuple, Dict, Union, Optional, Literal, TYPE_CHECKING
import logging

import numpy as np

from flixOpt import utils
from flixOpt.math_modeling import MathModel, Variable, VariableTS, Equation
from flixOpt.core import TimeSeries, Skalar, Numeric, Numeric_TS, as_effect_dict
from flixOpt.interface import InvestParameters, OnOffParameters
from flixOpt.elements import Component
from flixOpt.components import Storage, LinearConverter

if TYPE_CHECKING:  # for type checking and preventing circular imports
    from flixOpt.elements import Flow, Effect, EffectCollection, Objective, Bus
    from flixOpt.flow_system import FlowSystem


logger = logging.getLogger('flixOpt')


class SystemModel(MathModel):
    '''
    Hier kommen die ModellingLanguage-spezifischen Sachen rein
    '''

    def __init__(self,
                 label: str,
                 modeling_language: Literal['pyomo', 'cvxpy'],
                 flow_system: FlowSystem,
                 time_indices: Union[List[int], range]):
        super().__init__(label, modeling_language)
        self.flow_system = flow_system
        self.nr_of_time_steps = len(time_indices)
        self.time_indices = range(self.nr_of_time_steps)

        # Zeitdaten generieren:
        self.time_series, self.time_series_with_end, self.dt_in_hours, self.dt_in_hours_total = (
            flow_system.get_time_data_from_indices(time_indices))

        self.effect_collection_model = EffectCollectionModel(self.flow_system.effect_collection, self)
        self.component_models: List[ComponentModel] = []
        self.bus_models: List[BusModel] = []

    def do_modeling(self):
        self.effect_collection_model.do_modeling(self)
        self.component_models = [component.create_model() for component in self.flow_system.components]
        self.bus_models = [bus.create_model() for bus in self.flow_system.all_buses]
        for component_model in self.component_models:
            component_model.do_modeling(self)
        for bus_model in self.bus_models:  # Buses after Components, because FlowModels are created in ComponentModels
            bus_model.do_modeling(self)

    def solve(self,
              mip_gap: float = 0.02,
              time_limit_seconds: int = 3600,
              solver_name: str = 'highs',
              solver_output_to_console: bool = True,
              excess_threshold: Union[int, float] = 0.1,
              logfile_name: str = 'solver_log.log',
              **kwargs):
        """
        Parameters
        ----------
        mip_gap : TYPE, optional
            DESCRIPTION. The default is 0.02.
        time_limit_seconds : TYPE, optional
            DESCRIPTION. The default is 3600.
        solver_name : TYPE, optional
            DESCRIPTION. The default is 'highs'.
        solver_output_to_console : TYPE, optional
            DESCRIPTION. The default is True.
        excess_threshold : float, positive!
            threshold for excess: If sum(Excess)>excess_threshold a warning is raised, that an excess occurs
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        main_results_str : TYPE
            DESCRIPTION.
        """

        # check valid solver options:
        if len(kwargs) > 0:
            for key in kwargs.keys():
                if key not in ['threads']:
                    raise Exception(
                        'no allowed arguments for kwargs: ' + str(key) + '(all arguments:' + str(kwargs) + ')')

        logger.info(f'{" starting solving ":#^80}')
        logger.info(f'{self.describe()}')

        super().solve(mip_gap, time_limit_seconds, solver_name, solver_output_to_console, logfile_name, **kwargs)

        if solver_name == 'gurobi':
            termination_message = self.solver_results['Solver'][0]['Termination message']
        elif solver_name == 'glpk':
            termination_message = self.solver_results['Solver'][0]['Status']
        else:
            termination_message = f'not implemented for solver "{solver_name}" yet'
        logger.info(f'Termination message: "{termination_message}"')

        # Variablen-Ergebnisse abspeichern:
        # 1. dict:
        (self.results, self.results_var) = self.flow_system.get_results_after_solve()
        # 2. struct:
        self.results_struct = utils.createStructFromDictInDict(self.results)

        def extract_main_results() -> Dict[str, Union[Skalar, Dict]]:
            main_results = {}
            effect_results = {}
            main_results['Effects'] = effect_results
            for effect in self.flow_system.effect_collection.effects:
                effect_results[f'{effect.label} [{effect.unit}]'] = {
                    'operation': float(effect.operation.model.variables['sum'].result),
                    'invest': float(effect.invest.model.variables['sum'].result),
                    'sum': float(effect.all.model.variables['sum'].result)}
            main_results['penalty'] = float(self.flow_system.effect_collection.penalty.model.variables['sum'].result)
            main_results['Result of objective'] = self.objective_result
            if self.solver_name == 'highs':
                main_results['lower bound'] = self.solver_results.best_objective_bound
            else:
                main_results['lower bound'] = self.solver_results['Problem'][0]['Lower bound']
            busesWithExcess = []
            main_results['buses with excess'] = busesWithExcess
            for aBus in self.flow_system.all_buses:
                if aBus.with_excess:
                    if (
                            np.sum(self.results[aBus.label]['excess_input']) > excess_threshold or
                            np.sum(self.results[aBus.label]['excess_output']) > excess_threshold
                    ):
                        busesWithExcess.append(aBus.label)

            invest_decisions = {'invested': {}, 'not invested': {}}
            main_results['Invest-Decisions'] = invest_decisions
            for invest_feature in self.flow_system.all_investments:
                invested_size = invest_feature.model.variables[invest_feature.name_of_investment_size].result
                invested_size = float(invested_size)  # bei np.floats Probleme bei Speichern
                label = invest_feature.owner.label_full
                if invested_size > 1e-3:
                    invest_decisions['invested'][label] = invested_size
                else:
                    invest_decisions['not invested'][label] = invested_size

            return main_results

        self.main_results_str = extract_main_results()

        logger.info(f'{" finished solving ":#^80}')
        logger.info(f'{" Main Results ":#^80}')
        for effect_name, effect_results in self.main_results_str['Effects'].items():
            logger.info(f'{effect_name}:\n'
                        f'  {"operation":<15}: {effect_results["operation"]:>10.2f}\n'
                        f'  {"invest":<15}: {effect_results["invest"]:>10.2f}\n'
                        f'  {"sum":<15}: {effect_results["sum"]:>10.2f}')

        logger.info(
            # f'{"SUM":<15}: ...todo...\n'
            f'{"penalty":<17}: {self.main_results_str["penalty"]:>10.2f}\n'
            f'{"":-^80}\n'
            f'{"Objective":<17}: {self.main_results_str["Result of objective"]:>10.2f}\n'
            f'{"":-^80}')

        logger.info(f'Investment Decisions:')
        logger.info(utils.apply_formating(data_dict={**self.main_results_str["Invest-Decisions"]["invested"],
                                                     **self.main_results_str["Invest-Decisions"]["not invested"]},
                                          key_format="<30", indent=2, sort_by='value'))

        for bus in self.main_results_str['buses with excess']:
            logger.warning(f'Excess Value in Bus {bus}!')

        if self.main_results_str["penalty"] > 10:
            logger.warning(f'A total penalty of {self.main_results_str["penalty"]} occurred.'
                           f'This might distort the results')
        logger.info(f'{" End of Main Results ":#^80}')

    @property
    def infos(self):
        infos = super().infos
        infos['str_Eqs'] = self.flow_system.description_of_equations()
        infos['str_Vars'] = self.flow_system.description_of_variables()
        infos['main_results'] = self.main_results_str   # Hauptergebnisse:
        infos.update(self._infos)
        return infos

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = {}
        for sub_model in self.sub_models:
            for key, value in sub_model.all_variables.items():
                if key in all_vars:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_vars[key] = value
        return all_vars

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = {}
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_eqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_eqs[key] = value
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.all_equations.items() if eq.eqType == 'ineq'}

    @property
    def sub_models(self) -> List['ElementModel']:
        direct_models = [self.effect_collection_model] + self.component_models
        sub_models = [sub_model for direct_model in direct_models for sub_model in direct_model.all_sub_models]
        return direct_models + sub_models


class Element:
    """ Basic Element of flixOpt"""
    def __init__(self, label: str):
        self.label = label
        self.used_time_series: List[TimeSeries] = []  # Used for better access
        self.model: Optional[ElementModel] = None

    def _plausibility_checks(self) -> None:
        """ This function is used to do some basic plausibility checks for each Element during initialization """
        raise NotImplementedError(f'Every Element needs a _plausibility_checks() method')
    
    def transform_to_time_series(self) -> None:
        """ This function is used to transform the time series data from the User to proper TimeSeries Objects """
        raise NotImplementedError(f'Every Element needs a transform_to_time_series() method')

    def create_model(self) -> None:
        raise NotImplementedError(f'Every Element needs a create_model() method')

    def get_results(self) -> Tuple[Dict, Dict]:
        """Get results after the solve"""
        return self.model.results

    def __repr__(self):
        return f"<{self.__class__.__name__}> {self.label}"

    @property
    def label_full(self) -> str:
        return self.label


class ElementModel:
    """ Interface to create the mathematical Models for Elements """

    def __init__(self, element: Element):
        logger.debug(f'Created Model for {element.label_full}')
        self.element = element
        self.variables = {}
        self.eqs = {}
        self.sub_models = []

    def add_variables(self, *variables: Variable) -> None:
        for variable in variables:
            if variable.label not in self.variables.keys():
                self.variables[variable.label] = variable
            elif variable in self.variables.values():
                raise Exception(f'Variable "{variable.label}" already exists')
            else:
                raise Exception(f'A Variable with the label "{variable.label}" already exists')

    def add_equations(self, *equations: Equation) -> None:
        for equation in equations:
            if equation.label not in self.eqs.keys():
                self.eqs[equation.label] = equation
            else:
                raise Exception(f'Equation "{equation.label}" already exists')

    @property
    def description_of_variables(self) -> List[str]:
        if self.all_variables:
            return [var.description() for var in self.all_variables.values()]
        return []

    @property
    def description_of_equations(self) -> List[str]:
        if self.all_equations:
            return [eq.description() for eq in self.all_equations.values()]
        return []

    @property
    def overview_of_model_size(self) -> Dict[str, int]:
        all_vars, all_eqs, all_ineqs = self.all_variables, self.all_equations, self.all_inequations
        return {'no eqs': len(all_eqs),
                'no eqs single': sum(eq.nr_of_single_equations for eq in all_eqs.values()),
                'no inEqs': len(all_ineqs),
                'no inEqs single': sum(ineq.nr_of_single_equations for ineq in all_ineqs.values()),
                'no vars': len(all_vars),
                'no vars single': sum(var.length for var in all_vars.values())}

    @property
    def ineqs(self) -> Dict[str, Equation]:
        return {name: eq for name, eq in self.eqs.items() if eq.eqType == 'ineq'}

    @property
    def all_variables(self) -> Dict[str, Variable]:
        all_vars = self.variables.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_variables.items():
                if key in all_vars:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_vars[key] = value
        return all_vars

    @property
    def all_equations(self) -> Dict[str, Equation]:
        all_eqs = self.eqs.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_eqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                all_eqs[key] = value
        return all_eqs

    @property
    def all_inequations(self) -> Dict[str, Equation]:
        all_ineqs = self.ineqs.copy()
        for sub_model in self.sub_models:
            for key, value in sub_model.all_equations.items():
                if key in all_ineqs:
                    raise KeyError(f"Duplicate key found: '{key}' in both main model and submodel!")
                if value.eqType == 'ineq':
                    all_ineqs[key] = value
        return all_ineqs

    @property
    def all_sub_models(self) -> List['ElementModel']:
        all_subs = []
        to_process = self.sub_models.copy()
        for model in to_process:
            all_subs.append(model)
            to_process.extend(model.sub_models)
        return all_subs


class ComponentModel(ElementModel):
    def __init__(self, element: Component):
        super().__init__(element)
        self.element: Component = element

    def do_modeling(self, system_model: SystemModel):
        """ Initiates all FlowModels """
        self.sub_models.extend([flow.create_model() for flow in self.element.inputs + self.element.outputs])
        for sub_model in self.sub_models:
            sub_model.do_modeling(system_model)


class LinearConverterModel(ComponentModel):
    def __init__(self, element: LinearConverter):
        super().__init__(element)
        self.element: LinearConverter = element
        self._on: Optional[OnOffModel] = None

    def do_modeling(self, system_model: SystemModel):
        super().do_modeling(system_model)

        if self.element.on_off_parameters:
            all_flows = self.element.inputs + self.element.outputs
            flow_rates: List[VariableTS] = [flow.model.flow_rate for flow in all_flows]
            bounds: List[Tuple[Numeric, Numeric]] = [flow.model.flow_rate_bounds for flow in all_flows]
            self._on = OnOffModel(self.element, self.element.on_off_parameters,
                                  flow_rates, bounds)

        # conversion_factors:
        if self.element.conversion_factors:
            all_input_flows = set(self.element.inputs)
            all_output_flows = set(self.element.outputs)

            # für alle linearen Gleichungen:
            for i, conversion_factor in enumerate(self.element.conversion_factors):
                # erstelle Gleichung für jedes t:
                # sum(inputs * factor) = sum(outputs * factor)
                # left = in1.flow_rate[t] * factor_in1[t] + in2.flow_rate[t] * factor_in2[t] + ...
                # right = out1.flow_rate[t] * factor_out1[t] + out2.flow_rate[t] * factor_out2[t] + ...
                # eq: left = right
                used_flows = set(conversion_factor.keys())
                used_inputs: Set = all_input_flows & used_flows
                used_outputs: Set = all_output_flows & used_flows

                eq_conversion = Equation(f'conversion_{i}', self, system_model)
                self.add_equations(eq_conversion)
                for flow in used_inputs:
                    factor = conversion_factor[flow]
                    eq_conversion.add_summand(flow.model.flow_rate, factor)  # flow1.flow_rate[t]      * factor[t]
                for flow in used_outputs:
                    factor = conversion_factor[flow]
                    eq_conversion.add_summand(flow.model.flow_rate, -1 * factor)  # output.val[t] * -1 * factor[t]

                eq_conversion.add_constant(0)  #TODO: Is this necessary?

        # (linear) segments:
        else:
            segments = {flow.model.flow_rate: self.element.segmented_conversion_factors[flow]
                        for flow in self.element.inputs + self.element.outputs}
            linear_segments = MultipleSegmentsModel(self.element, segments, None)  #TODO: Add Outside_segments Variable (On)
            linear_segments.do_modeling(system_model)
            self.sub_models.append(linear_segments)


class FlowModel(ElementModel):
    def __init__(self, element: Flow):
        super().__init__(element)
        self.element: Flow = element
        self.flow_rate: Optional[VariableTS] = None
        self.sum_flow_hours: Optional[Variable] = None

        self._on: Optional[OnOffModel] = None
        self._investment: Optional[InvestmentModel] = None

    def do_modeling(self, system_model: SystemModel):
        # eq relative_minimum(t) * size <= flow_rate(t) <= relative_maximum(t) * size
        if self.element.on_off_parameters is None and not isinstance(self.element.size, InvestParameters):
            if self.element.fixed_relative_value is None:
                fixed_flow_rate = None
            else:
                fixed_flow_rate = self.element.fixed_relative_value * self.element.size
            self.flow_rate = VariableTS('flow_rate',
                                        system_model.nr_of_time_steps, self.element.label_full, system_model,
                                        lower_bound=self.element.relative_minimum * self.element.size,
                                        upper_bound=self.element.relative_maximum * self.element.size,
                                        value=fixed_flow_rate)
        else:  # Bounds are created later and in sub_models
            self.flow_rate = VariableTS('flow_rate', system_model.nr_of_time_steps,
                                        self.element.label_full, system_model, lower_bound=0)
        self.add_variables(self.flow_rate)

        # OnOff
        if self.element.on_off_parameters is not None:
            self._on = OnOffModel(self.element, self.element.on_off_parameters,
                                  [self.flow_rate],
                                  [self.flow_rate_bounds])
            self._on.do_modeling(system_model)
            self.sub_models.append(self._on)

        # Investment
        if isinstance(self.element.size, InvestParameters):
            self._investment = InvestmentModel(self.element, self.element.size,
                                               self.flow_rate,
                                               (self.element.relative_minimum, self.element.size.minimum_size),
                                               self._on.on)
            self._investment.do_modeling(system_model)
            self.sub_models.append(self._investment)

        # sumFLowHours
        self.sum_flow_hours = Variable('sumFlowHours', 1, self.element.label_full, system_model,
                                       lower_bound=self.element.flow_hours_total_min,
                                       upper_bound=self.element.flow_hours_total_max)
        eq_sum_flow_hours = Equation('sumFlowHours', self, system_model, 'eq')
        eq_sum_flow_hours.add_summand(self.flow_rate, system_model.dt_in_hours, as_sum=True)
        eq_sum_flow_hours.add_summand(self.sum_flow_hours, -1)
        self.add_variables(self.sum_flow_hours)
        self.add_equations(eq_sum_flow_hours)

        self._create_bounds_for_load_factor(system_model)
        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        # Arbeitskosten:
        effect_collection = system_model.flow_system.effect_collection
        if self.element.effects_per_flow_hour is not None:
            effect_collection.add_share_to_operation(
                name_of_share='effects_per_flow_hour',
                owner=self.element, variable=self.flow_rate,
                effect_values=self.element.effects_per_flow_hour,
                factor=system_model.dt_in_hours
            )

    def _create_bounds_for_load_factor(self, system_model: SystemModel):
        #TODO: Add Variable load_factor for better evaluation?

        # eq: var_sumFlowHours <= size * dt_tot * load_factor_max
        if self.element.load_factor_max is not None:
            flow_hours_per_size_max = system_model.dt_in_hours_total * self.element.load_factor_max
            eq_load_factor_max = Equation('load_factor_max', self.element, system_model, 'ineq')
            self.add_equations(eq_load_factor_max)
            eq_load_factor_max.add_summand(self.sum_flow_hours, 1)
            # if investment:
            if self._investment is not None:
                eq_load_factor_max.add_summand(self._investment.size, -1 * flow_hours_per_size_max)
            else:
                eq_load_factor_max.add_constant(self.element.size * flow_hours_per_size_max)

        #  eq: size * sum(dt)* load_factor_min <= var_sumFlowHours
        if self.element.load_factor_min is not None:
            flow_hours_per_size_min = system_model.dt_in_hours_total * self.element.load_factor_min
            eq_load_factor_min = Equation('load_factor_min', self, system_model, 'ineq')
            self.add_equations(eq_load_factor_min)
            eq_load_factor_min.add_summand(self.sum_flow_hours, 1)
            if self._investment is not None:
                eq_load_factor_min.add_summand(self._investment.size, flow_hours_per_size_min)
            else:
                eq_load_factor_min.add_constant(-1 * self.element.size * flow_hours_per_size_min)

    @property
    def flow_rate_bounds(self) -> Tuple[Numeric, Numeric]:
        if not isinstance(self.element.size, InvestParameters):
            return (self.element.relative_minimum * self.element.size,
                    self.element.relative_maximum * self.element.size)
        else:
            return (self.element.relative_minimum * self.element.size.minimum_size,
                    self.element.relative_maximum * self.element.size.maximum_size)


class BusModel(ElementModel):
    def __init__(self, element: Bus):
        super().__init__(element)
        self.element: Bus
        self.excess_input: Optional[VariableTS] = None
        self.excess_output: Optional[VariableTS] = None

    def do_modeling(self, system_model: SystemModel) -> None:
        self.element: Bus
        # inputs = outputs
        eq_bus_balance = Equation('busBalance', self.element, system_model)
        self.add_equations(eq_bus_balance)
        for flow in self.element.inputs:
            eq_bus_balance.add_summand(flow.model.flow_rate, 1)
        for flow in self.element.outputs:
            eq_bus_balance.add_summand(flow.model.flow_rate, -1)

        # Fehlerplus/-minus:
        if self.element.with_excess:
            excess_penalty = np.multiply(system_model.dt_in_hours, self.element.excess_effects_per_flow_hour)
            self.excess_input = VariableTS('excess_input', system_model.nr_of_time_steps, self.element.label_full,
                                           system_model, lower_bound=0)
            self.excess_output = VariableTS('excess_output', system_model.nr_of_time_steps,
                                            self.element.label_full, system_model, lower_bound=0)
            self.add_variables(self.excess_input, self.excess_output)

            eq_bus_balance.add_summand(self.excess_output, -1)
            eq_bus_balance.add_summand(self.excess_input, 1)

            effect_collection_model: EffectCollectionModel = system_model.flow_system.effect_collection.model

            effect_collection_model.add_share_to_penalty('penalty', self.element, self.excess_input, excess_penalty)
            effect_collection_model.add_share_to_penalty('penalty', self.element, self.excess_output, excess_penalty)


class EffectModel(ElementModel):
    def __init__(self, element: Effect):
        super().__init__(element)
        self.element: Effect
        self.invest = ShareAllocationModel(self.element, False,
                                           total_max=self.element.maximum_invest,
                                           total_min=self.element.minimum_invest)
        self.operation = ShareAllocationModel(self.element, True,
                                              total_max=self.element.maximum_operation,
                                              total_min=self.element.minimum_operation,
                                              min_per_hour=self.element.minimum_operation_per_hour,
                                              max_per_hour=self.element.maximum_operation_per_hour)
        self.all = ShareAllocationModel(self.element, False,
                                        total_max=self.element.maximum_total,
                                        total_min=self.element.minimum_total)
        self.sub_models.extend([self.invest, self.operation, self.all])

    def do_modeling(self, system_model: SystemModel):
        for model in self.sub_models:
            model.do_modeling(system_model)

        self.all.add_variable_share('operation', self.element, self.operation.sum, 1, 1)
        self.all.add_variable_share('invest', self.element, self.invest.sum, 1, 1)


class EffectCollectionModel(ElementModel):
    #TODO: Maybe all EffectModels should be sub_models of this Model? Including Objective and Penalty?
    def __init__(self, element: EffectCollection, system_model: SystemModel):
        super().__init__(element)
        self.element = element
        self._system_model = system_model
        self._effect_models: Dict[Effect, EffectModel] = {}
        self.penalty: Optional[ShareAllocationModel] = None
        self.objective: Optional[Equation] = None

    def do_modeling(self, system_model: SystemModel):
        self._effect_models = {effect: EffectModel(effect) for effect in self.element.effects}
        self.penalty = ShareAllocationModel(self.element.penalty, True)
        self.sub_models.extend(list(self._effect_models.values()) + [self.penalty])
        for model in self.sub_models:
            model.do_modeling(system_model)

        self.objective = Equation('OBJECTIVE', self, system_model, 'objective')
        self.add_equations(self.objective)
        self.objective.add_summand(self._objective_effect_model.operation.sum, 1)
        self.objective.add_summand(self._objective_effect_model.invest.sum, 1)
        self.objective.add_summand(self.penalty.sum, 1)

    @property
    def _objective_effect_model(self) -> EffectModel:
        return self._effect_models[self.element.objective_effect]

    def add_share_to_effects(self,
                             name: str,
                             target: Literal['operation', 'invest'],
                             effect_values: Union[Numeric, Dict[Optional[Effect], TimeSeries]],
                             factor: Numeric,
                             variable: Optional[Variable] = None) -> None:
        effect_values_dict = as_effect_dict(effect_values)

        # an alle Effects, die einen Wert haben, anhängen:
        for effect, value in effect_values_dict.items():
            if effect is None:  # Falls None, dann Standard-effekt nutzen:
                effect = self.element.standard_effect
            assert effect in self.element.effects, f'Effect {effect.label} was used but not added to model!'

            if target == 'operation':
                model = self._effect_models[effect].operation
            elif target == 'invest':
                model = self._effect_models[effect].invest
            else:
                raise ValueError(f'Target {target} not supported!')
            
            if variable is None:
                model.add_constant_share(self._system_model, name, effect, value, factor)
            elif isinstance(variable, Variable):
                model.add_variable_share(self._system_model, name, effect, variable, value, factor)
            else:
                raise TypeError

    def add_share_to_penalty(self,
                             name: str,
                             share_holder: Element,
                             variable: Optional[Variable],
                             factor: Numeric,
                             ) -> None:
        assert variable is not None
        if variable is None:
            self.penalty.add_constant_share(self._system_model, name, share_holder, factor, 1)
        elif isinstance(variable, Variable):
            self.penalty.add_variable_share(self._system_model, name, share_holder, variable, factor, 1)
        else:
            raise TypeError

    def add_share_between_effects(self):
        for origin_effect in self.element.effects:
            # 1. operation: -> hier sind es Zeitreihen (share_TS)
            name_of_share = 'specific_share_to_other_effects_operation'  # + effectType.label
            for target_effect, factor in origin_effect.specific_share_to_other_effects_operation.items():
                target_model = self._effect_models[target_effect].operation
                origin_model = self._effect_models[origin_effect].operation
                target_model.add_variable_share(self._system_model, name_of_share, origin_effect, origin_model.sum_TS, factor, 1)
            # 2. invest:    -> hier ist es Skalar (share)
            name_of_share = 'specificShareToOtherEffects_invest_'  # + effectType.label
            for target_effect, factor in origin_effect.specific_share_to_other_effects_invest.items():
                target_model = self._effect_models[target_effect].invest
                origin_model = self._effect_models[origin_effect].invest
                target_model.add_variable_share(self._system_model, name_of_share, origin_effect, origin_model.sum, factor, 1)


def _extract_sample_points(data: List[Skalar]) -> List[Tuple[Skalar, Skalar]]:
    assert len(data) % 2 == 0, f'Segments must have an even number of start-/endpoints'
    return [(data[i], data[i+1]) for i in range(0, len(data), 2)]


class InvestmentModel(ElementModel):
    """Class for modeling an investment"""
    def __init__(self, element: Union[Flow, Storage],
                 invest_parameters: InvestParameters,
                 defining_variable: [VariableTS],
                 relative_bounds_of_defining_variable: Tuple[Numeric, Numeric],
                 on_variable: [VariableTS]):
        """
        if relative_bounds are both equal, then its like a fixed relative value
        """
        super().__init__(element)
        self.element: Union[Flow, Storage] = element
        self.size: Optional[Union[Skalar, Variable]] = None
        self.is_invested: Optional[Variable] = None
        
        self._segments: Optional[SegmentedSharesModel] = None

        self._on_variable = on_variable
        self._defining_variable = defining_variable
        self._relative_bounds_of_defining_variable = relative_bounds_of_defining_variable
        self._invest_parameters = invest_parameters

    def do_modeling(self, system_model: SystemModel):
        invest_parameters = self._invest_parameters
        if invest_parameters.fixed_size:
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 value=invest_parameters.fixed_size)
        else:
            lower_bound = 0 if invest_parameters.optional else invest_parameters.minimum_size
            self.size = Variable('size', 1, self.element.label_full, system_model,
                                 lower_bound=lower_bound,
                                 upper_bound=invest_parameters.maximum_size)
        self.add_variables(self.size)
        # Optional
        if invest_parameters.optional:
            self.is_invested = Variable('isInvested', 1, self.element.label_full, system_model,
                                        is_binary=True)
            self.add_variables(self.is_invested)
            self._create_bounds_for_optional_investment(system_model)

        # Bounds for defining variable
        self._create_bounds_for_defining_variable(system_model)

        self._create_shares(system_model)

    def _create_shares(self, system_model: SystemModel):
        effect_collection = system_model.flow_system.effect_collection
        invest_parameters = self._invest_parameters

        # fix_effects:
        fix_effects = invest_parameters.fix_effects
        if fix_effects is not None and fix_effects != 0:
            if invest_parameters.optional:  # share: + isInvested * fix_effects
                effect_collection.add_share_to_invest('fix_effects', self.element,
                                                      self.is_invested, fix_effects, 1)
            else:  # share: + fix_effects
                effect_collection.add_constant_share_to_invest('fix_effects', self.element,
                                                               fix_effects,1)
        # divest_effects:
        divest_effects = invest_parameters.divest_effects
        if divest_effects is not None and divest_effects != 0:
            if invest_parameters.optional:  # share: [divest_effects - isInvested * divest_effects]
                # 1. part of share [+ divest_effects]:
                effect_collection.add_constant_share_to_invest('divest_effects', self.element,
                                                               divest_effects, 1)
                # 2. part of share [- isInvested * divest_effects]:
                effect_collection.add_share_to_invest('divest_cancellation_effects', self.element,
                                                      self.is_invested, divest_effects, -1)
                # TODO : these 2 parts should be one share! -> SingleShareModel...?

        # # specific_effects:
        specific_effects = invest_parameters.specific_effects
        if specific_effects is not None:
            # share: + investment_size (=var)   * specific_effects
            effect_collection.add_share_to_invest('specific_effects', self.element,
                                                  self.size, specific_effects, 1)
        # segmented Effects
        invest_segments = invest_parameters.effects_in_segments
        if invest_segments:
            self._segments = SegmentedSharesModel(self.element,
                                                  (self.size, invest_segments[0]),
                                                  invest_segments[1], self.is_invested)
            self.sub_models.append(self._segments)
            self._segments.do_modeling(system_model)

    def _create_bounds_for_optional_investment(self, system_model: SystemModel):
        if self._invest_parameters.fixed_size:
            # eq: investment_size = isInvested * fixed_size
            eq_is_invested = Equation('is_invested', self.element, system_model, 'eq')
            eq_is_invested.add_summand(self.size, -1)
            eq_is_invested.add_summand(self.is_invested, self._invest_parameters.fixed_size)
            self.add_equations(eq_is_invested)
        else:
            # eq1: P_invest <= isInvested * investSize_max
            eq_is_invested_ub = Equation('is_invested_ub', self.element, system_model, 'ineq')
            eq_is_invested_ub.add_summand(self.size, 1)
            eq_is_invested_ub.add_summand(self.is_invested, np.multiply(-1, self._invest_parameters.maximum_size))

            # eq2: P_invest >= isInvested * max(epsilon, investSize_min)
            eq_is_invested_lb = Equation('is_invested_lb', self.element, system_model, 'ineq')
            eq_is_invested_lb.add_summand(self.size, -1)
            eq_is_invested_lb.add_summand(self.is_invested, np.max(system_model.epsilon, self._invest_parameters.minimum_size))
            self.add_equations(eq_is_invested_ub, eq_is_invested_lb)

    def _create_bounds_for_defining_variable(self, system_model: SystemModel):
        label = self._defining_variable.label
        relative_minimum, relative_maximum = self._relative_bounds_of_defining_variable
        # fixed relative value
        if np.array_equal(relative_minimum, relative_maximum):
            #TODO: Allow Off? Currently not...
            eq_fixed = Equation(f'fixed_{label}', self.element, system_model)
            eq_fixed.add_summand(self._defining_variable, 1)
            eq_fixed.add_summand(self.size, np.multiply(-1, relative_maximum))
            self.add_equations(eq_fixed)
        else:
            eq_upper = Equation(f'ub_{label}', self.element, system_model, 'ineq')
            # eq: defining_variable(t)  <= size * upper_bound(t)
            eq_upper.add_summand(self._defining_variable, 1)
            eq_upper.add_summand(self.size, np.multiply(-1, relative_maximum))

            ## 2. Gleichung: Minimum durch Investmentgröße ##
            eq_lower = Equation(f'lb_{label}', self, system_model, 'ineq')
            if self._on_variable is None:
                # eq: defining_variable(t) >= investment_size * relative_minimum(t)
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self.size, relative_minimum)
            else:
                ## 2. Gleichung: Minimum durch Investmentgröße und On
                # eq: defining_variable(t) >= mega * (On(t)-1) + size * relative_minimum(t)
                #     ... mit mega = relative_maximum * maximum_size
                # äquivalent zu:.
                # eq: - defining_variable(t) + mega * On(t) + size * relative_minimum(t) <= + mega
                mega = relative_maximum * self._invest_parameters.maximum_size
                eq_lower.add_summand(self._defining_variable, -1)
                eq_lower.add_summand(self._on_variable, mega)
                eq_lower.add_summand(self.size, relative_minimum)
                eq_lower.add_constant(mega)
                # Anmerkung: Glg bei Spezialfall relative_minimum = 0 redundant zu OnOff ??
            self.add_equations(eq_lower, eq_upper)


class OnOffModel(ElementModel):
    """
    Class for modeling the on and off state of a variable
    If defining_bounds are given, creates sufficient lower bounds
    """
    def __init__(self, element: Element,
                 on_off_parameters: OnOffParameters,
                 defining_variables: List[VariableTS],
                 defining_bounds: List[Tuple[Numeric, Numeric]]):
        """
        defining_bounds: a list of Numeric, that can be  used to create the bound for On/Off more efficiently
        """
        super().__init__(element)
        self.element = element
        self.on: Optional[VariableTS] = None
        self.total_on_hours: Optional[Variable] = None

        self.consecutive_on_hours: Optional[VariableTS] = None
        self.consecutive_off_hours: Optional[VariableTS] = None

        self.off: Optional[VariableTS] = None

        self.switch_on: Optional[VariableTS] = None
        self.switch_off: Optional[VariableTS] = None
        self.nr_switch_on: Optional[VariableTS] = None

        self._on_off_parameters = on_off_parameters
        self._defining_variables = defining_variables
        self._defining_bounds = defining_bounds
        assert len(defining_variables) == len(defining_bounds), f'Every defining Variable needs bounds to Model OnOff'

    def do_modeling(self, system_model: SystemModel):
        if self._on_off_parameters.use_on:
            self.on = VariableTS('on', system_model.nr_of_time_steps, self.element.label_full, system_model,
                                 is_binary=True, before_value=self._on_off_parameters.on_values_before_begin[0])
            self.total_on_hours = Variable('onHoursSum', 1, self.element.label_full, system_model,
                                           lower_bound=self._on_off_parameters.on_hours_total_min,
                                           upper_bound=self._on_off_parameters.on_hours_total_max)
            self.add_variables(self.on, self.total_on_hours)

            self._add_on_constraints(system_model, system_model.time_indices)

        if self._on_off_parameters.use_off:
            self.off = VariableTS('off', system_model.nr_of_time_steps, self.element.label_full, system_model,
                                  is_binary=True)
            self.add_variables(self.off)

            self._add_off_constraints(system_model, system_model.time_indices)

        if self._on_off_parameters.use_on_hours:
            self.consecutive_on_hours = VariableTS('onHours', system_model.nr_of_time_steps,
                                                   self.element.label_full, system_model,
                                                   lower_bound=0,
                                                   upper_bound=self._on_off_parameters.consecutive_on_hours_max)
            self.add_variables(self.consecutive_on_hours)
            self._add_duration_constraints(self.consecutive_on_hours, self.on,
                                           self._on_off_parameters.consecutive_on_hours_min,
                                           system_model, system_model.time_indices)
        # offHours:
        if self._on_off_parameters.use_off_hours:
            self.consecutive_off_hours = VariableTS('offHours', system_model.nr_of_time_steps,
                                                    self.element.label_full, system_model,
                                                    lower_bound=0,
                                                    upper_bound=self._on_off_parameters.consecutive_off_hours_max)
            self.add_variables(self.consecutive_off_hours)
            self._add_duration_constraints(self.consecutive_off_hours, self.off,
                                           self._on_off_parameters.consecutive_off_hours_min,
                                           system_model, system_model.time_indices)
        # Var SwitchOn
        if self._on_off_parameters.use_switch_on:
            self.switch_on = VariableTS('switchOn', system_model.nr_of_time_steps, self.element.label_full, system_model, is_binary=True)
            self.switch_off = VariableTS('switchOff', system_model.nr_of_time_steps, self.element.label_full, system_model, is_binary=True)
            self.nr_switch_on = Variable('nrSwitchOn', 1, self.element.label_full, system_model,
                                         upper_bound=self._on_off_parameters.switch_on_total_max)
            self.add_variables(self.switch_on, self.switch_off, self.nr_switch_on)
            self._add_switch_constraints(system_model, system_model.time_indices)

        self._create_shares(system_model)

    def _add_on_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.on is not None, f'On variable of {self.element} must be defined to add constraints'
        # % Bedingungen 1) und 2) müssen erfüllt sein:

        # % Anmerkung: Falls "abschnittsweise linear" gewählt, dann ist eigentlich nur Bedingung 1) noch notwendig
        # %            (und dann auch nur wenn erstes Segment bei Q_th=0 beginnt. Dann soll bei Q_th=0 (d.h. die Maschine ist Aus) On = 0 und segment1.onSeg = 0):)
        # %            Fazit: Wenn kein Performance-Verlust durch mehr Gleichungen, dann egal!

        nr_of_defining_variables = len(self._defining_variables)
        assert nr_of_defining_variables > 0, 'Achtung: mindestens 1 Flow notwendig'

        eq_on_1 = Equation('On_Constraint_1', self.element, system_model, eqType='ineq')
        eq_on_2 = Equation('On_Constraint_2', self.element, system_model, eqType='ineq')
        self.add_equations(eq_on_1, eq_on_2)
        if nr_of_defining_variables == 1:
            variable = self._defining_variables[0]
            lower_bound, upper_bound = self._defining_bounds
            #### Bedingung 1) ####
            # eq: On(t) * max(epsilon, lower_bound) <= -1 * Q_th(t)
            eq_on_1.add_summand(variable, -1, time_indices)
            eq_on_1.add_summand(self.on, np.maximum(system_model.epsilon, lower_bound), time_indices)

            #### Bedingung 2) ####
            # eq: Q_th(t) <= Q_th_max * On(t)
            eq_on_2.add_summand(variable, 1, time_indices)
            eq_on_2.add_summand(self.on, upper_bound, time_indices)

        else:  # Bei mehreren Leistungsvariablen:
            #### Bedingung 1) ####
            # When all defining variables are 0, On is 0
            # eq: - sum(alle Leistungen(t)) + Epsilon * On(t) <= 0
            for variable in self._defining_variables:
                eq_on_1.add_summand(variable, -1, time_indices)
            eq_on_1.add_summand(self.on, system_model.epsilon, time_indices)

            #### Bedingung 2) ####
            ## sum(alle Leistung) >0 -> On = 1 | On=0 -> sum(Leistung)=0
            #  eq: sum( Leistung(t,i))              - sum(Leistung_max(i))             * On(t) <= 0
            #  --> damit Gleichungswerte nicht zu groß werden, noch durch nr_of_flows geteilt:
            #  eq: sum( Leistung(t,i) / nr_of_flows ) - sum(Leistung_max(i)) / nr_of_flows * On(t) <= 0
            absolute_maximum: Numeric = 0
            for variable, bounds in zip(self._defining_variables, self._defining_bounds):
                eq_on_2.add_summand(variable, 1 / nr_of_defining_variables, time_indices)
                absolute_maximum += bounds[1]  # der maximale Nennwert reicht als Obergrenze hier aus. (immer noch math. günster als BigM)

            upper_bound = absolute_maximum / nr_of_defining_variables
            eq_on_2.add_summand(self.on, -1 * upper_bound, time_indices)

        if np.max(upper_bound) > 1000:
            logger.warning(f'!!! ACHTUNG in {self.element.label_full}  Binärdefinition mit großem Max-Wert ('
                           f'{np.max(upper_bound)}). Ggf. falsche Ergebnisse !!!')

    def _add_off_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.off is not None, f'Off variable of {self.element} must be defined to add constraints'
        # Definition var_off:
        # eq: var_on(t) + var_off(t) = 1
        eq_off = Equation('var_off', self, system_model, eqType='eq')
        self.add_equations(eq_off)
        eq_off.add_summand(self.off, 1, time_indices)
        eq_off.add_summand(self.on, 1, time_indices)
        eq_off.add_constant(1)

    def _add_duration_constraints(self,
                                  duration_variable: VariableTS,
                                  binary_variable: VariableTS,
                                  minimum_duration: Optional[Numeric_TS],
                                  system_model: SystemModel,
                                  time_indices: Union[list[int], range]):
        """
        i.g.
        binary_variable        = [0 0 1 1 1 1 0 1 1 1 0 ...]
        duration_variable =      [0 0 1 2 3 4 0 1 2 3 0 ...] (bei dt=1)
                                            |-> min_onHours = 3!

        if you want to count zeros, define var_bin_off: = 1-binary_variable before!
        """
        assert duration_variable is not None, f'Duration Variable of {self.element} must be defined to add constraints'
        assert binary_variable is not None, f'Duration Variable of {self.element} must be defined to add constraints'
        # TODO: Einfachere Variante von Peter umsetzen!

        # 1) eq: onHours(t) <= On(t)*Big | On(t)=0 -> onHours(t) = 0
        # mit Big = dt_in_hours_total
        label_prefix = duration_variable.label
        mega = system_model.dt_in_hours_total
        constraint_1 = Equation(f'{label_prefix}_constraint_1', self.element, system_model, eqType='ineq')
        self.add_equations(constraint_1)
        constraint_1.add_summand(duration_variable, 1)
        constraint_1.add_summand(binary_variable, -1 * mega)

        # 2a) eq: onHours(t) - onHours(t-1) <= dt(t)
        #    on(t)=1 -> ...<= dt(t)
        #    on(t)=0 -> onHours(t-1)>=
        constraint_2a = Equation(f'{label_prefix}_constraint_2a', self.element, system_model, eqType='ineq')
        self.add_equations(constraint_2a)
        constraint_2a.add_summand(duration_variable, 1, time_indices[1:])  # onHours(t)
        constraint_2a.add_summand(duration_variable, -1, time_indices[0:-1])  # onHours(t-1)
        constraint_2a.add_constant(system_model.dt_in_hours[1:])  # dt(t)

        # 2b) eq:  onHours(t) - onHours(t-1)             >=  dt(t) - Big*(1-On(t)))
        #    eq: -onHours(t) + onHours(t-1) + On(t)*Big <= -dt(t) + Big
        # with Big = dt_in_hours_total # (Big = maxOnHours, should be usable, too!)
        constraint_2b = Equation(f'{label_prefix}_constraint_2b', self.element, system_model, eqType='ineq')
        self.add_equations(constraint_2b)
        constraint_2b.add_summand(duration_variable, -1, time_indices[1:])  # onHours(t)
        constraint_2b.add_summand(duration_variable, 1, time_indices[0:-1])  # onHours(t-1)
        constraint_2b.add_summand(binary_variable, mega, time_indices[1:])  # on(t)
        constraint_2b.add_constant(-1 + system_model.dt_in_hours[1:] + mega)  # dt(t)

        # 3) check minimum_duration before switchOff-step
        # (last on-time period of timeseries is not checked and can be shorter)
        if minimum_duration is not None:
            # Note: switchOff-step is when: On(t)-On(t-1) == -1
            # eq:  onHours(t-1) >= minOnHours * -1 * [On(t)-On(t-1)]
            # eq: -onHours(t-1) - minimum_duration * On(t) + minimum_duration*On(t-1) <= 0
            eq_min_duration = Equation(f'{label_prefix}_minimum_duration', self.element, system_model, eqType='ineq')
            self.add_equations(eq_min_duration)
            eq_min_duration.add_summand(duration_variable, -1, time_indices[0:-1])  # onHours(t-1)
            eq_min_duration.add_summand(binary_variable, -1 * minimum_duration, time_indices[1:])  # on(t)
            eq_min_duration.add_summand(binary_variable, minimum_duration, time_indices[0:-1])  # on(t-1)

        # TODO: Maximum Duration?? Is this not modeled yet?!!

        # 4) first index:
        #    eq: onHours(t=0)= dt(0) * On(0)
        first_index = time_indices[0]  # only first element
        eq_first = Equation(f'{label_prefix}_firstTimeStep', self.element, system_model)
        self.add_equations(eq_first)
        eq_first.add_summand(duration_variable, 1, first_index)
        eq_first.add_summand(binary_variable, -1 * system_model.dt_in_hours[first_index], first_index)

    def _add_switch_constraints(self, system_model: SystemModel, time_indices: Union[list[int], range]):
        assert self.switch_on is not None, f'Switch On Variable of {self.element} must be defined to add constraints'
        assert self.switch_off is not None, f'Switch Off Variable of {self.element} must be defined to add constraints'
        assert self.nr_switch_on is not None, f'Nr of Switch On Variable of {self.element} must be defined to add constraints'
        assert self.on is not None, f'On Variable of {self.element} must be defined to add constraints'
        # % Schaltänderung aus On-Variable
        # % SwitchOn(t)-SwitchOff(t) = On(t)-On(t-1)
        eq_switch = Equation('Switch', self.element, system_model)
        self.add_equations(eq_switch)
        eq_switch.add_summand(self.switch_on, 1, time_indices[1:])  # SwitchOn(t)
        eq_switch.add_summand(self.switch_off, -1, time_indices[1:])  # SwitchOff(t)
        eq_switch.add_summand(self.on, -1, time_indices[1:])  # On(t)
        eq_switch.add_summand(self.on, +1, time_indices[0:-1])  # On(t-1)

        ## Ersten Wert SwitchOn(t=1) bzw. SwitchOff(t=1) festlegen
        # eq: SwitchOn(t=1)-SwitchOff(t=1) = On(t=1)- ValueBeforeBeginOfTimeSeries;
        eq_initial_switch = Equation('Initial_Switch', self.element, system_model)
        first_index = time_indices[0]
        eq_initial_switch.add_summand(self.switch_on, 1, first_index)
        eq_initial_switch.add_summand(self.switch_off, -1, first_index)
        eq_initial_switch.add_summand(self.on, -1, first_index)
        eq_initial_switch.add_constant(-1 * self.on.before_value)  # letztes Element der Before-Werte nutzen,  Anmerkung: wäre besser auf lhs aufgehoben

        ## Entweder SwitchOff oder SwitchOn
        # eq: SwitchOn(t) + SwitchOff(t) <= 1
        eq_switch_on_or_off = Equation('Switch_On_or_Off', self.element, system_model, eqType='ineq')
        eq_switch_on_or_off.add_summand(self.switch_on, 1)
        eq_switch_on_or_off.add_summand(self.switch_off, 1)
        eq_switch_on_or_off.add_constant(1)

        ## Anzahl Starts:
        # eq: nrSwitchOn = sum(SwitchOn(t))
        eq_nr_switch_on = Equation('NrSwitchOn', self.element, system_model)
        eq_nr_switch_on.add_summand(self.nr_switch_on, 1)
        eq_nr_switch_on.add_summand(self.switch_on, -1, as_sum=True)

    def _create_shares(self, system_model: SystemModel):
        # Anfahrkosten:
        effect_collection = system_model.effect_collection_model
        effects_per_switch_on = self._on_off_parameters.effects_per_switch_on
        if effects_per_switch_on is not None:
            effect_collection.add_share_to_effects('switch_on_effects', 'operation',
                                                   effects_per_switch_on, 1, self.switch_on)

        # Betriebskosten:
        effects_per_running_hour = self._on_off_parameters.effects_per_running_hour
        if effects_per_running_hour is not None:
            effect_collection.add_share_to_effects('running_hour_effects', 'operation',
                                                   effects_per_running_hour, system_model.dt_in_hours, self.on)


class SegmentModel(ElementModel):
    """Class for modeling a linear segment of one or more variables in parallel"""
    def __init__(self, element: Element, segment_index: Union[int, str],
                 sample_points: Dict[Variable, Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]):
        super().__init__(element)
        self.element = element
        self.in_segment: Optional[VariableTS] = None
        self.lambda0: Optional[VariableTS] = None
        self.lambda1: Optional[VariableTS] = None

        self._segment_index = segment_index
        self._sample_points = sample_points

    def do_modeling(self, system_model: SystemModel):
        length = system_model.nr_of_time_steps
        self.in_segment = VariableTS(f'onSeg_{self._segment_index}', length, self.element.label_full, system_model,
                                     is_binary=True)  # Binär-Variable
        self.lambda0 = VariableTS(f'lambda0_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.lambda1 = VariableTS(f'lambda1_{self._segment_index}', length, self.element.label_full, system_model,
                                  lower_bound=0, upper_bound=1)  # Wertebereich 0..1
        self.add_variables(self.in_segment)
        self.add_variables(self.lambda0)
        self.add_variables(self.lambda1)

        # eq: -aSegment.onSeg(t) + aSegment.lambda1(t) + aSegment.lambda2(t)  = 0
        equation = Equation(f'Lambda_onSeg_{self._segment_index}', self, system_model)
        self.add_equations(equation)

        equation.add_summand(self.in_segment, -1)
        equation.add_summand(self.lambda0, 1)
        equation.add_summand(self.lambda1, 1)

        #  eq: - v(t) + (v_0 * lambda_0 + v_1 * lambda_1) = 0       -> v_0, v_1 = Stützstellen des Segments
        for variable, sample_points in self._sample_points.items():
            sample_0, sample_1 = sample_points
            if isinstance(sample_0, TimeSeries):
                sample_0 = sample_0.active_data
                sample_1 = sample_1.active_data
            else:
                sample_0 = sample_0
                sample_1 = sample_1

            lambda_eq = Equation(f'{variable.label_full}_lambda', self, system_model)
            lambda_eq.add_summand(variable, -1)
            lambda_eq.add_summand(self.lambda0, sample_0)
            lambda_eq.add_summand(self.lambda1, sample_1)
            self.add_equations(lambda_eq)


class MultipleSegmentsModel(ElementModel):
    def __init__(self, element: Element,
                 sample_points: Dict[Variable, List[Tuple[Union[Numeric, TimeSeries], Union[Numeric, TimeSeries]]]],
                 outside_segments: Optional[Variable] = None):
        super().__init__(element)
        self.element = element

        self.outside_segments: Optional[VariableTS] = outside_segments  # Variable to allow being outside segments = 0

        self._sample_points = sample_points
        self._segment_models: List[SegmentModel] = []

    def do_modeling(self, system_model: SystemModel):
        restructured_variables_with_segments: List[Dict[Variable, Tuple[Numeric, Numeric]]] = [
            {key: values[i] for key, values in self._sample_points.items()}
            for i in range(self._nr_of_segments)
        ]

        for i, sample_points in enumerate(restructured_variables_with_segments):
            self._segment_models.append(SegmentModel(self.element, i, sample_points))

        for segment_model in self._segment_models:
            segment_model.do_modeling(system_model)

        # Outside of Segments
        if self.outside_segments is None:  # TODO: Make optional
            self.outside_segments = VariableTS(f'outside_segments', system_model.nr_of_time_steps, self.element.label_full,
                                          system_model, is_binary=True)
            self.add_variables(self.outside_segments)

        # a) eq: Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 1                Aufenthalt nur in Segmenten erlaubt
        # b) eq: -On(t) + Segment1.onSeg(t) + Segment2.onSeg(t) + ... = 0       zusätzlich kann alles auch Null sein
        in_single_segment = Equation('in_single_Segment', self, system_model)
        for segment_model in self._segment_models:
            in_single_segment.add_summand(segment_model.in_segment, 1)
        if self.outside_segments is None:
            in_single_segment.add_constant(1)
        else:
            in_single_segment.add_summand(self.outside_segments, -1)

    @property
    def _nr_of_segments(self):
        return len(next(iter(self._sample_points.values())))


class ShareAllocationModel(ElementModel):
    def __init__(self,
                 element: Element,
                 shares_are_time_series: bool,
                 total_max: Optional[Skalar] = None,
                 total_min: Optional[Skalar] = None,
                 max_per_hour: Optional[TimeSeries] = None,
                 min_per_hour: Optional[TimeSeries] = None):
        super().__init__(element)
        if not shares_are_time_series:  # If the condition is True
            assert max_per_hour is None and min_per_hour is None, \
                "Both max_per_hour and min_per_hour cannot be used when shares_are_time_series is False"
        self.element = element
        self.sum_TS: Optional[VariableTS] = None
        self.sum: Optional[Variable] = None

        self._eq_time_series: Optional[Equation] = None
        self._eq_sum: Optional[Equation] = None

        # Parameters
        self._shares_are_time_series = shares_are_time_series
        self._total_max = total_max
        self._total_min = total_min
        self._max_per_hour = max_per_hour
        self._min_per_hour = min_per_hour

    def do_modeling(self, system_model: SystemModel):
        self.sum = Variable('sum', 1, self.element.label_full, system_model,
                            lower_bound=self._total_min, upper_bound=self._total_max)
        self.add_variables(self.sum)
        # eq: sum = sum(share_i) # skalar
        self._eq_sum = Equation('sum', self, system_model)
        self._eq_sum.add_summand(self.sum, -1)
        self.add_equations(self._eq_sum)

        if self._shares_are_time_series:
            lb_TS = None if (self._min_per_hour is None) else np.multiply(self._min_per_hour.active_data, system_model.dt_in_hours)
            ub_TS = None if (self._max_per_hour is None) else np.multiply(self._max_per_hour.active_data, system_model.dt_in_hours)
            self.sum_TS = VariableTS('sum_TS', system_model.nr_of_time_steps, self.element.label_full, system_model,
                                     lower_bound=lb_TS, upper_bound=ub_TS)
            self.add_variables(self.sum_TS)

            # eq: sum_TS = sum(share_TS_i) # TS
            self._eq_time_series = Equation('time_series', self, system_model)
            self._eq_time_series.add_summand(self.sum_TS, -1)
            self.add_equations(self._eq_time_series)

            # eq: sum = sum(sum_TS(t)) # additionaly to self.sum
            self._eq_sum.add_summand(self.sum_TS, 1, as_sum=True)

    def add_variable_share(self,
                           system_model: SystemModel,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           variable: Variable,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):  # if variable = None, then fix Share
        if variable is None:
            raise Exception('add_variable_share() needs variable as input. Use add_constant_share() instead')
        self._add_share(system_model, name_of_share, share_holder, variable, factor1, factor2)

    def add_constant_share(self,
                           system_model: SystemModel,
                           name_of_share: Optional[str],
                           share_holder: Element,
                           factor1: Numeric_TS,
                           factor2: Numeric_TS):
        variable = None
        self._add_share(system_model, name_of_share, share_holder, variable, factor1, factor2)

    def _add_share(self,
                   system_model: SystemModel,
                   name_of_share: Optional[str],
                   share_holder: Element,
                   variable: Optional[Variable],
                   factor1: Numeric_TS,
                   factor2: Numeric_TS):
        #TODO: accept only one factor or accept unlimited factors -> *factors

        # Falls TimeSeries, Daten auslesen:
        if isinstance(factor1, TimeSeries):
            factor1 = factor1.active_data
        if isinstance(factor2, TimeSeries):
            factor2 = factor2.active_data
        total_factor = np.multiply(factor1, factor2)

        # var and eq for publishing share-values in results:
        if name_of_share is not None:  #TODO: is this check necessary?
            new_share = SingleShareModel(share_holder, self._shares_are_time_series, name_of_share)
            new_share.do_modeling(system_model)
            new_share.add_summand_to_share(variable, total_factor)

        # Check to which equation the share should be added
        if self._shares_are_time_series:
            target_eq = self._eq_time_series
        else:
            assert total_factor.shape[0] == 1, (f'factor1 und factor2 müssen Skalare sein, '
                                                f'da shareSum {self.element.label} skalar ist')
            target_eq = self._eq_sum

        if variable is None:  # constant share
            target_eq.add_constant(-1 * total_factor)
        else:  # variable share
            target_eq.add_summand(variable, total_factor)
        #TODO: Instead use new_share.single_share: Variable ?


class SingleShareModel(ElementModel):
    def __init__(self, element: Element, shares_are_time_series: bool, name_of_share: str):
        super().__init__(element)
        self.single_share: Optional[Variable] = None
        self._equation: Optional[Equation] = None
        self._full_name_of_share = f'{element.label_full}_{name_of_share}'
        self._shares_are_time_series = shares_are_time_series
        self._name_of_share = name_of_share

    def do_modeling(self, system_model: SystemModel):
        self.single_share = Variable(self._full_name_of_share, 1, self.element.label_full, system_model)
        self.add_variables(self.single_share)

        self._equation = Equation(self._full_name_of_share, self, system_model)
        self._equation.add_summand(self.single_share, -1)
        self.add_equations(self._equation)

    def add_summand_to_share(self, variable: Optional[Variable], factor: Numeric):
        """share to a sum"""
        if variable is None:  # if constant share:
            constant_value = np.sum(factor) if self._shares_are_time_series else factor
            self._equation.add_constant(-1 * constant_value)
        else:  # if variable share - always as a skalar -> as_sum=True if shares are timeseries
            self._equation.add_summand(variable, factor, as_sum=self._shares_are_time_series)


class SegmentedSharesModel(ElementModel):
    def __init__(self,
                 element: Element,
                 variable_segments: Tuple[Variable, List[Tuple[Skalar, Skalar]]],
                 share_segments: Dict[Effect, List[Tuple[Skalar, Skalar]]],
                 outside_segments: Optional[Variable]):
        super().__init__(element)
        assert len(variable_segments[1]) == len(list(share_segments.values())[0]), \
            f'Segment length of variable_segments and share_segments must be equal'
        self.element: Element
        self._outside_segments = outside_segments
        self._variable_segments = variable_segments
        self._share_segments = share_segments
        self._shares: Optional[Dict[Effect, SingleShareModel]] = None
        self._segments_model: Optional[MultipleSegmentsModel] = None

    def do_modeling(self, system_model: SystemModel):
        self._shares = {effect: SingleShareModel(self.element, False, f'segmented')
                        for effect in self._share_segments}
        for single_share in self._shares.values():
            single_share.do_modeling(system_model)

        segments: Dict[Variable, List[Tuple[Skalar, Skalar]]] = {
            self._shares[effect].single_share: segment for effect, segment in self._share_segments.values()}
        self._segments_model = MultipleSegmentsModel(self.element, self._outside_segments, segments)
        self._segments_model.do_modeling(system_model)

        # Shares
        effect_collection = system_model.flow_system.effect_collection
        for effect, single_share_model in self._shares.items():
            effect_collection.add_share_to_invest(
                name_of_share='segmented_effects',
                owner=self.element, variable=single_share_model.single_share,
                effect_values={effect: 1},
                factor=1
            )
