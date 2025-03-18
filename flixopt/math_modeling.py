"""
This module contains the mathematical core of the flixOpt framework.
THe module is designed to be used by other modules than flixOpt itself.
It holds all necessary classes and functions to create a mathematical model, consisting of Varaibles and constraints,
and translate it into a ModelingLanguage like Pyomo, and the solve it through a solver.
Multiple solvers are supported.
"""

import logging
import re
import timeit
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pyomo.environ as pyo

from . import utils
from .core import Numeric

logger = logging.getLogger('flixOpt')


class Variable:
    """
    Variable class
    """

    def __init__(
        self,
        label: str,
        length: int,
        label_short: Optional[str] = None,
        is_binary: bool = False,
        fixed_value: Optional[Numeric] = None,
        lower_bound: Optional[Numeric] = None,
        upper_bound: Optional[Numeric] = None,
    ):
        """
        label: full label of the variable
        label_short: short label of the variable

        # TODO: Allow for None values in fixed_value. If None, the index gets not fixed!
        """
        self.label = label
        self.label_short = label_short or label
        self.length = length
        self.is_binary = is_binary
        self.fixed_value = fixed_value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.indices = range(self.length)
        self.fixed = False

        self.result = None  # Ergebnis-Speicher

        if self.fixed_value is not None:  # Check if value is within bounds, element-wise
            above = self.lower_bound is None or np.all(np.asarray(self.fixed_value) >= np.asarray(self.lower_bound))
            below = self.upper_bound is None or np.all(np.asarray(self.fixed_value) <= np.asarray(self.upper_bound))
            if not (above and below):
                raise Exception(
                    f'Fixed value of Variable {self.label} not inside set bounds:'
                    f'\n{self.fixed_value=};\n{self.lower_bound=};\n{self.upper_bound=}'
                )

            # Mark as fixed
            self.fixed = True

        logger.debug('Variable created: ' + self.label)

    def description(self, max_length_ts=60) -> str:
        bin_type = 'bin' if self.is_binary else '   '

        header = f'Var {bin_type} x {self.length:<6} "{self.label}"'
        if self.fixed:
            description = f'{header:<40}: fixed={str(self.fixed_value)[:max_length_ts]:<10}'
        else:
            description = (
                f'{header:<40}: min={str(self.lower_bound)[:max_length_ts]:<10}, '
                f'max={str(self.upper_bound)[:max_length_ts]:<10}'
            )
        return description

    def reset_result(self):
        self.result = None


class VariableTS(Variable):
    """
    Timeseries-Variable, optionally with previous_values. class for Variables that are related by time
    """

    def __init__(
        self,
        label: str,
        length: int,
        label_short: Optional[str] = None,
        is_binary: bool = False,
        fixed_value: Optional[Numeric] = None,
        lower_bound: Optional[Numeric] = None,
        upper_bound: Optional[Numeric] = None,
        previous_values: Optional[Numeric] = None,
    ):
        assert length > 1, 'length is one, that seems not right for VariableTS'
        super().__init__(
            label,
            length,
            label_short,
            is_binary=is_binary,
            fixed_value=fixed_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self.previous_values = previous_values


class _Constraint:
    """
    Abstract Class for Constraints. Use Child classes!

    """

    def __init__(self, label: str, label_short: Optional[str] = None):
        """
        Equation of the form: ∑(<summands>) = <constant>        type: 'eq'
        Equation of the form: ∑(<summands>) <= <constant>       type: 'ineq'
        Equation of the form: ∑(<summands>) = <constant>        type: 'objective'

        Parameters
        ----------
            label: full label of the variable
            label_short: short label of the variable. If None, the the full label is used
        """
        self.label = label
        self.label_short = label_short or label
        self.summands: List[SumOfSummand] = []
        self.parts_of_constant: List[Numeric] = []
        self.constant: Numeric = 0  # Total of right side

        self.length = 1  # Anzahl der Gleichungen

        logger.debug(f'Equation created: {self.label}')

    def add_summand(
        self,
        variable: Variable,
        factor: Numeric,
        indices_of_variable: Optional[Union[int, np.ndarray, range, List[int]]] = None,
        as_sum: bool = False,
    ) -> None:
        """
        Adds a summand to the left side of the equation.

        This method creates a summand from the given variable and factor, optionally summing over all given indices.
        The summand is then added to the summands of the equation, which represent the left side.

        Parameters:
        -----------
        variable : Variable
            The variable to be used in the summand.
        factor : Numeric
            The factor by which the variable is multiplied.
        indices_of_variable : Optional[Numeric], optional
            Specific indices of the variable to be used. If not provided, all indices are used.
        as_sum : bool, optional
            If True, the summand is treated as a sum over all indices of the variable.

        Raises:
        -------
        TypeError
            If the provided variable is not an instance of the Variable class.
        ValueError
            If the variable is None and as_sum is True.
        ValueError
            If the length doesnt match the Equation's length.
        """
        # TODO: Functionality to create A Sum of Summand over a specified range of indices? For Limiting stuff per one year...?
        if not isinstance(variable, Variable):
            raise TypeError(f'Error in Equation "{self.label}": no variable given (variable = "{variable}")')
        if variable is None and as_sum:
            raise ValueError(f'Error in Equation "{self.label}": Variable can not be None and be summed up!')

        if np.isscalar(indices_of_variable):  # Wenn nur ein Wert, dann Liste mit einem Eintrag drausmachen:
            indices_of_variable = [indices_of_variable]

        if as_sum:
            summand = SumOfSummand(variable, factor, indices=indices_of_variable)
        else:
            summand = Summand(variable, factor, indices=indices_of_variable)

        try:
            self._update_length(summand.length)  # Check Variablen-Länge:
        except ValueError as e:
            raise ValueError(
                f'Length of Summand with variable "{variable.label}" does not fit equation "{self.label}": {e}'
            ) from e
        self.summands.append(summand)

    def add_constant(self, value: Numeric) -> None:
        """
        Adds a constant value to the rigth side of the equation

        Parameters
        ----------
        value : float or array
          constant-value of equation [A*x = constant] or [A*x <= constant]

        Returns
        -------
        None.

        Raises:
        -------
        ValueError
            If the length doesnt match the Equation's length.

        """
        self.constant = np.add(self.constant, value)  # Adding to current constant
        self.parts_of_constant.append(value)  # Adding to parts of constants

        length = 1 if np.isscalar(self.constant) else len(self.constant)
        try:
            self._update_length(length)
        except ValueError as e:
            raise ValueError(f'Length of Constant {value=} does not fit: {e}') from e

    def description(self, at_index: int = 0) -> str:
        raise NotImplementedError('Not implemented for Abstract class <_Constraint>')

    def _update_length(self, new_length: int) -> None:
        """
        Passes if the new_length is 1, the current length is 1 or new_length matches the existing length of the Equation
        """
        if self.length == 1:  # First Summand sets length
            self.length = new_length
        elif new_length == 1 or new_length == self.length:  # Length 1 is always possible
            pass
        else:
            raise ValueError(
                f'The length of the new element {new_length=} doesnt match the existing '
                f'length of the Equation {self.length=}!'
            )

    @property
    def constant_vector(self) -> Numeric:
        return utils.as_vector(self.constant, self.length)


class Equation(_Constraint):
    """
    Equation of the form: ∑(<summands>) = <constant>
    Can be the Objective of a MathModel.

    Parameters
    ----------
    label : str
        Full label of the variable.
    label_short : str, optional
        Short label of the variable. If None, the full label is used.
    is_objective : bool, optional
        Indicates if this equation is the objective of the model (default is False).
    """

    def __init__(self, label, label_short=None, is_objective=False):
        super().__init__(label, label_short)
        self.is_objective = is_objective

    def description(self, at_index: int = 0) -> str:
        equation_nr = min(at_index, self.length - 1)

        # Name and index as str
        if self.is_objective == 'objective':
            name, index_str = 'OBJ', ''
        else:
            name, index_str = f'EQ {self.label}', f'[{equation_nr + 1}/{self.length}]'

        # Summands:
        summand_strings = [summand.description(at_index) for summand in self.summands]
        all_summands_string = ' + '.join(summand_strings)

        constant = self.constant_vector[equation_nr]

        # String formating
        header_width = 30
        header = f'{name:<{header_width - len(index_str) - 1}} {index_str}'
        return f'{header:<{header_width}}: {constant:>8} = {all_summands_string}'


class Inequation(_Constraint):
    """
    Equation of the form: <constant> >= ∑(<summands>)

    Parameters
    ----------
        label: full label of the variable
        label_short: short label of the variable. If None, the full label is used
    """

    def __init__(self, label, label_short=None):
        super().__init__(label, label_short)

    def description(self, at_index: int = 0) -> str:
        equation_nr = min(at_index, self.length - 1)

        # Name and index as str
        name, index_str = f'INEQ {self.label}', f'[{equation_nr + 1}/{self.length}]'

        # Summands:
        summand_strings = [summand.description(at_index) for summand in self.summands]
        all_summands_string = ' + '.join(summand_strings)

        constant = self.constant_vector[equation_nr]

        # String formating
        header_width = 30
        header = f'{name:<{header_width - len(index_str) - 1}} {index_str}'
        return f'{header:<{header_width}}: {constant:>8} >= {all_summands_string}'


class Summand:
    """
    Represents a part of a Constraint , consisting of a variable (or a time-series variable) and a factor.

    Parameters
    ----------
    variable : Variable
        The variable associated with this summand.
    factor : Numeric
        The factor by which the variable is multiplied in the equation.
    indices : int, np.ndarray, range, List[int], optional
        Specifies which indices of the variable to use. If None, all indices of the variable are used.
    """

    def __init__(
        self, variable: Variable, factor: Numeric, indices: Optional[Union[int, np.ndarray, range, List[int]]] = None
    ):  # indices_of_variable default : alle
        self.variable = variable
        self.factor = factor
        self.indices = indices if indices is not None else variable.indices  # wenn nicht definiert, dann alle Indexe

        self.length = self._check_length()  # Länge ermitteln:

        self.factor_vec = utils.as_vector(factor, self.length)  # Faktor als Vektor:

    def description(self, at_index=0):
        i = 0 if self.length == 1 else at_index
        index = self.indices[i]
        factor = self.factor_vec[i]
        factor_str = f'{factor:.6}' if isinstance(factor, (float, np.floating)) else str(factor)
        return f'{factor_str} * {self.variable.label}[{index}]'

    def _check_length(self):
        """
        Determines and returns the length of the summand by comparing the lengths of the factor and the variable indices.
        Sets the attribute .length to this value.

        Returns:
        --------
        int
            The length of the summand, which is the length of the indices if they match the length of the factor,
            or the length of the longer one if one of them is a scalar.

        Raises:
        -------
        Exception
            If the lengths of the factor and the variable indices do not match and neither is a scalar.
        """
        length_of_factor = 1 if np.isscalar(self.factor) else len(self.factor)
        length_of_indices = len(self.indices)
        if length_of_indices == length_of_factor:
            return length_of_indices
        elif length_of_factor == 1:
            return length_of_indices
        elif length_of_indices == 1:
            return length_of_factor
        else:
            raise Exception(
                f'Variable {self.variable.label} (length={length_of_indices}) und '
                f'Faktor (length={length_of_factor}) müssen gleiche Länge haben oder Skalar sein'
            )


class SumOfSummand(Summand):
    """
    Represents a part of an Equation that sums all components of a regular Summand over specified indices.

    Parameters
    ----------
    variable : Variable
        The variable associated with this summand.
    factor : Numeric
        The factor by which the variable is multiplied.
    indices : int, np.ndarray, range, List[int], optional
        Specifies which indices of the variable to use for the sum. If None, all indices are summed.
    """

    def __init__(
        self, variable: Variable, factor: Numeric, indices: Optional[Union[int, np.ndarray, range, List[int]]] = None
    ):  # indices_of_variable default : alle
        super().__init__(variable, factor, indices)
        self.length = 1

    def description(self, at_index=0):
        index = self.indices[at_index]
        factor = self.factor_vec[0]
        factor_str = str(factor) if isinstance(factor, int) else f'{factor:.6}'
        single_summand_str = f'{factor_str} * {self.variable.label}[{index}]'
        return f'∑({("..+" if index > 0 else "")}{single_summand_str}{("+.." if index < self.variable.length else "")})'


class MathModel:
    """
    A mathematical model for defining equations and constraints of the form:

        a1 * x1 + a2 + x2  = y
        and
        a1 * x1 + a2 + x2 <= y

    where 'a1', 'a2' and y can be vectors or scalars, while 'x1' and 'x2' are variables with an appropriate length.


    This class provides methods to add variables, equations, and inequality constraints to the model and supports
    translation to a specified modeling language like pyomo.

    The expression 'a1 * x1' is referred to as a 'Summand'. Supported summand formats are:
    - 'Variable[j] * Factor[i]'     : Multiplication of vector variables and vector factors.
    - 'Variable[j] * Factor'        : Vector variable with scalar factor.
    - 'Variable    * Factor'        : Scalar variable with scalar factor.
    - 'Factor'                      : Scalar constant.


    Parameters
    ----------
    label : str
        A descriptive label for the model.
    modeling_language : {'pyomo', 'cvxpy'}, optional
        Specifies the modeling language used for translation (default is 'pyomo').

    Attributes
    ----------
    label : str
        The label assigned to the model.
    modeling_language : str
        The modeling language to which the model will be translated.
    epsilon : float
        Small tolerance value used in model calculations, defaulting to `1e-5`.
    solver : Optional[Solver]
        The solver instance assigned to solve the model.
    model : Optional[ModelingLanguage]
        The model instance in the specified modeling language.
    _variables : List[Variable]
        List of variables added to the model.
    _constraints : List[Union[Equation, Inequation]]
        List of equations and inequality constraints in the model.
    _objective : Optional[Equation]
        The objective function, if defined as an equation.
    duration : dict
        Dictionary tracking the time taken for translation and solving steps.

    Methods
    -------
    add(*args)
        Adds variables, equations, or inequations to the model.
    describe_size()
        Provides a summary of the number of equations, inequations, and variables.
    translate_to_modeling_language()
        Translates the model to the specified modeling language.
    solve(solver)
        Solves the model using the specified solver instance.
    results()
        Returns a dictionary of variable results after solving.
    """

    def __init__(self, label: str, modeling_language: Literal['pyomo', 'cvxpy'] = 'pyomo'):
        self._infos = {}
        self.label = label
        self.modeling_language: str = modeling_language

        self.solver: Optional[Solver] = None
        self.model: Optional[ModelingLanguage] = None

        self._variables: List[Variable] = []
        self._constraints: List[Union[Equation, Inequation]] = []
        self._objective: Optional[Equation] = None
        self.result_of_objective: Optional[float] = None

        self.duration = {}

    def add(self, *args: Union[Variable, Equation, Inequation]) -> None:
        if not isinstance(args, list):
            args = list(args)
        for arg in args:
            if isinstance(arg, Variable):
                self._variables.append(arg)
            elif isinstance(arg, (Equation, Inequation)):
                if isinstance(arg, Equation) and arg.is_objective:
                    self._objective = arg
                else:
                    self._constraints.append(arg)
            else:
                raise Exception(f'{arg} cant be added this way!')

    def describe_size(self) -> str:
        return (
            f'No. of Equations   (single): {self.nr_of_equations} ({self.nr_of_single_equations})\n'
            f'No. of Inequations (single): {self.nr_of_inequations} ({self.nr_of_single_inequations})\n'
            f'No. of Variables   (single): {self.nr_of_variables} ({self.nr_of_single_variables})'
        )

    def translate_to_modeling_language(self) -> None:
        t_start = timeit.default_timer()
        if self.modeling_language == 'pyomo':
            self.model = PyomoModel()
            self.model.translate_model(self)
        else:
            raise NotImplementedError('Modeling Language cvxpy is not yet implemented')
        self.duration['Translation'] = round(timeit.default_timer() - t_start, 2)

    def solve(self, solver: 'Solver') -> None:
        self.solver = solver
        t_start = timeit.default_timer()
        for variable in self.variables:
            variable.reset_result()  # altes Ergebnis löschen (falls vorhanden)
        self.model.solve(self, solver)
        self.duration['Solving'] = round(timeit.default_timer() - t_start, 2)

    def results(self) -> Dict[str, Numeric]:
        return {variable.label: variable.result for variable in self.variables}

    @property
    def infos(self) -> Dict:
        return {
            'Solver': repr(self.solver),
            'Model Size': {
                'No. of Eqs.': self.nr_of_equations,
                'No. of Eqs. (single)': self.nr_of_single_equations,
                'No. of Ineqs.': self.nr_of_inequations,
                'No. of Ineqs. (single)': self.nr_of_single_inequations,
                'No. of Vars.': self.nr_of_variables,
                'No. of Vars. (single)': self.nr_of_single_variables,
                'No. of Vars. (TS)': len(self.ts_variables),
            },
            'Solver Log': self.solver.log.infos if isinstance(self.solver.log, SolverLog) else self.solver.log,
        }

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @property
    def equations(self) -> List[Equation]:
        return [eq for eq in self._constraints if isinstance(eq, Equation)]

    @property
    def inequations(self):
        return [eq for eq in self._constraints if isinstance(eq, Inequation)]

    @property
    def objective(self) -> Equation:
        return self._objective

    @property
    def ts_variables(self) -> List[VariableTS]:
        return [variable for variable in self.variables if isinstance(variable, VariableTS)]

    @property
    def nr_of_variables(self) -> int:
        return len(self.variables)

    @property
    def nr_of_constraints(self) -> int:
        return len(self._constraints)

    @property
    def nr_of_equations(self) -> int:
        return len(self.equations)

    @property
    def nr_of_inequations(self) -> int:
        return len(self.inequations)

    @property
    def nr_of_single_variables(self) -> int:
        return sum([var.length for var in self.variables])

    @property
    def nr_of_single_equations(self) -> int:
        return sum([eq.length for eq in self.equations])

    @property
    def nr_of_single_inequations(self) -> int:
        return sum([eq.length for eq in self.inequations])


class SolverLog:
    """
    Parses and holds solver log information for specific solvers.

    Attributes:
        solver_name (str): Name of the solver (e.g., 'gurobi', 'cbc').
        log (str): Content of the log file.
        presolved_rows (Optional[int]): Number of rows after presolving.
        presolved_cols (Optional[int]): Number of columns after presolving.
        presolved_nonzeros (Optional[int]): Number of nonzeros after presolving.
        presolved_continuous (Optional[int]): Number of continuous variables after presolving.
        presolved_integer (Optional[int]): Number of integer variables after presolving.
        presolved_binary (Optional[int]): Number of binary variables after presolving.
    """

    def __init__(self, solver_name: str, filename: str):
        with open(filename, 'r') as file:
            self.log = file.read()

        self.solver_name = solver_name

        self.presolved_rows = None
        self.presolved_cols = None
        self.presolved_nonzeros = None

        self.presolved_continuous = None
        self.presolved_integer = None
        self.presolved_binary = None
        self.parse_infos()

    @property
    def infos(self) -> Dict[str, Dict[str, int]]:
        return {
            'presolved': {
                'cols': self.presolved_cols,
                'continuous': self.presolved_continuous,
                'integer': self.presolved_integer,
                'binary': self.presolved_binary,
                'rows': self.presolved_rows,
                'nonzeros': self.presolved_nonzeros,
            }
        }

    # Suche infos aus log:
    def parse_infos(self):
        if self.solver_name == 'gurobi':
            # string-Schnipsel 1:
            """
            Optimize a model with 285 rows, 292 columns and 878 nonzeros
            Model fingerprint: 0x1756ffd1
            Variable types: 202 continuous, 90 integer (90 binary)
            """
            # string-Schnipsel 2:
            """
            Presolve removed 154 rows and 172 columns
            Presolve time: 0.00s
            Presolved: 131 rows, 120 columns, 339 nonzeros
            Variable types: 53 continuous, 67 integer (67 binary)
            """
            # string: Presolved: 131 rows, 120 columns, 339 nonzeros\n
            match = re.search(
                r'Presolved: (\d+) rows, (\d+) columns, (\d+) nonzeros\n'
                r'Variable types: (\d+) continuous, (\d+) integer \((\d+) binary\)',
                self.log,
            )
            if match:
                # string: Presolved: 131 rows, 120 columns, 339 nonzeros\n
                self.presolved_rows = int(match.group(1))
                self.presolved_cols = int(match.group(2))
                self.presolved_nonzeros = int(match.group(3))
                # string: Variable types: 53 continuous, 67 integer (67 binary)
                self.presolved_continuous = int(match.group(4))
                self.presolved_integer = int(match.group(5))
                self.presolved_binary = int(match.group(6))

        elif self.solver_name == 'cbc':
            # string: Presolve 1623 (-1079) rows, 1430 (-1078) columns and 4296 (-3306) elements
            match = re.search(r'Presolve (\d+) \((-?\d+)\) rows, (\d+) \((-?\d+)\) columns and (\d+)', self.log)
            if match is not None:
                self.presolved_rows = int(match.group(1))
                self.presolved_cols = int(match.group(3))
                self.presolved_nonzeros = int(match.group(5))

            # string: Presolved problem has 862 integers (862 of which binary)
            match = re.search(r'Presolved problem has (\d+) integers \((\d+) of which binary\)', self.log)
            if match is not None:
                self.presolved_integer = int(match.group(1))
                self.presolved_binary = int(match.group(2))
                self.presolved_continuous = self.presolved_cols - self.presolved_integer

        elif self.solver_name == 'glpk':
            logger.warning(f'{"":#^80}\n')
            logger.warning(f'{" No solver-log parsing implemented for glpk yet! ":#^80}\n')
        else:
            raise Exception('SolverLog.parse_infos() is not defined for solver ' + self.solver_name)


class Solver(ABC):
    """
    Abstract base class for solvers.

    Attributes:
        mip_gap (float): Solver's mip gap setting. The MIP gap describes the accepted (MILP) objective,
            and the lower bound, which is the theoretically optimal solution (LP)
        solver_output_to_console (bool): Whether to display solver output.
        logfile_name (str): Filename for saving the solver log.
        objective (Optional[float]): Objective value from the solution.
        best_bound (Optional[float]): Best bound from the solver.
        termination_message (Optional[str]): Solver's termination message.
    """

    def __init__(
        self,
        mip_gap: float,
        solver_output_to_console: bool,
        logfile_name: str,
    ):
        self.mip_gap = mip_gap
        self.solver_output_to_console = solver_output_to_console
        self.logfile_name = logfile_name

        self.objective: Optional[float] = None
        self.best_bound: Optional[float] = None
        self.termination_message: Optional[str] = None
        self.log: Optional[str, SolverLog] = None

        self._solver = None
        self._results: Optional[float, str] = None

    @abstractmethod
    def solve(self, modeling_language: 'ModelingLanguage'):
        raise NotImplementedError(' Solving is not possible with this Abstract class')

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'mip_gap={self.mip_gap}, '
            f'solver_output_to_console={self.solver_output_to_console}, '
            f"logfile_name='{self.logfile_name}', "
            f'objective={self.objective!r}, '
            f'best_bound={self.best_bound!r}, '
            f'termination_message={self.termination_message!r})'
        )


class GurobiSolver(Solver):
    """
    Solver implementation for Gurobi.
    Also Look in class Solver for more details

    Attributes:
        time_limit_seconds (int): Time limit for the solver. After this time, the solver takes the currently
        best solution, ignoring the mip_gap.
    """

    def __init__(
        self,
        mip_gap: float = 0.01,
        time_limit_seconds: int = 300,
        logfile_name: str = 'gurobi.log',
        solver_output_to_console: bool = True,
    ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyo.SolverFactory('gurobi')
            self._results = self._solver.solve(
                modeling_language.model,
                tee=self.solver_output_to_console,
                keepfiles=True,
                logfile=self.logfile_name,
                options={'mipgap': self.mip_gap, 'TimeLimit': self.time_limit_seconds},
            )

            self.objective = modeling_language.model.objective.expr()
            self.termination_message = self._results.solver.termination_message
            self.best_bound = self._results.problem.lower_bound

            from pyomo.opt import SolverStatus, TerminationCondition

            if not (
                self._results.solver.status == SolverStatus.ok
                and self._results.solver.termination_condition == TerminationCondition.optimal
            ):
                logger.warning(
                    f'Solver ended with status {self._results.solver.status} and '
                    f'termination condition {self._results.solver.termination_condition}'
                )
            try:
                self.log = SolverLog('gurobi', self.logfile_name)
            except Exception as e:
                self.log = None
                logger.warning(f'SolverLog could not be loaded. {e}')

            try:
                import gurobi_logtools

                self.log = gurobi_logtools.get_dataframe([str(self.logfile_name)]).T.to_dict()[0]
            except ImportError:
                logger.info(
                    'Evaluationg the gurobi log after the solve was not possible, due to a missing dependency '
                    '"gurobi_logtools". For further details of the solving process, '
                    'install the dependency via "pip install gurobi_logtools".'
                )
        else:
            raise NotImplementedError('Only Pyomo is implemented for GUROBI solver.')


class CplexSolver(Solver):
    """
    Solver implementation for CPLEX.
    Also Look in class Solver for more details

    Attributes:
        time_limit_seconds (int): Time limit for the solver. After this time, the solver takes the currently
        best solution, ignoring the mip_gap.
    """

    def __init__(
        self,
        mip_gap: float = 0.01,
        time_limit_seconds: int = 300,
        logfile_name: str = 'cplex.log',
        solver_output_to_console: bool = True,
    ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyo.SolverFactory('cplex')
            self._results = self._solver.solve(
                modeling_language.model,
                tee=self.solver_output_to_console,
                keepfiles=True,
                logfile=self.logfile_name,
                options={'mipgap': self.mip_gap, 'timelimit': self.time_limit_seconds},
            )

            self.objective = modeling_language.model.objective.expr()
            self.termination_message: Optional[str] = f'Not Implemented for {self.__class__.__name__} yet'
            self.best_bound = self._results['Problem'][0]['Lower bound']
            self.log = f'Not Implemented for {self.__class__.__name__} yet'
        else:
            raise NotImplementedError('Only Pyomo is implemented for CPLEX solver.')


class HighsSolver(Solver):
    """
    Solver implementation for HIGHS.
    Also Look in class Solver for more details

    Attributes:
        time_limit_seconds (int): Time limit for the solver. After this time, the solver takes the currently
        best solution, ignoring the mip_gap.
        threads (int): Number of threads to use for the solver.
    """

    def __init__(
        self,
        mip_gap: float = 0.01,
        time_limit_seconds: int = 300,
        logfile_name: str = 'highs.log',
        solver_output_to_console: bool = True,
        threads: int = 4,
    ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds
        self.threads = threads

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            from pyomo.contrib import appsi

            self._solver = appsi.solvers.Highs()
            self._solver.highs_options = {
                'mip_rel_gap': self.mip_gap,
                'time_limit': self.time_limit_seconds,
                'log_file': str(self.logfile_name),
                # "log_to_console": self.solver_output_to_console,
                'threads': self.threads,
                'parallel': 'on',
                'presolve': 'on',
                'output_flag': True,
            }
            self._solver.config.stream_solver = True

            self._results = self._solver.solve(
                modeling_language.model
            )  # HiGHS writes logs to stdout/stderr, so we capture them here

            self.objective = modeling_language.model.objective.expr()
            self.termination_message: Optional[str] = self._results.termination_condition.name
            if not self.termination_message == 'optimal':
                logger.warning(f'Solution is not optimal. Termination Message: "{self.termination_message}"')
            self.best_bound = self._results.best_objective_bound
            self.log = f'Not Implemented for {self.__class__.__name__} yet'
        else:
            raise NotImplementedError('Only Pyomo is implemented for HIGHS solver.')


class CbcSolver(Solver):
    """
    Solver implementation for CBC.
    Also Look in class Solver for more details

    Attributes:
        time_limit_seconds (int): Time limit for the solver. After this time, the solver takes the currently
        best solution, ignoring the mip_gap.
    """

    def __init__(
        self,
        mip_gap: float = 0.01,
        time_limit_seconds: int = 300,
        logfile_name: str = 'cbc.log',
        solver_output_to_console: bool = True,
    ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)
        self.time_limit_seconds = time_limit_seconds

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyo.SolverFactory('cbc')
            self._results = self._solver.solve(
                modeling_language.model,
                tee=self.solver_output_to_console,
                keepfiles=True,
                logfile=self.logfile_name,
                options={'ratio': self.mip_gap, 'sec': self.time_limit_seconds},
            )
            self.objective = modeling_language.model.objective.expr()
            self.termination_message: Optional[str] = f'Not Implemented for {self.__class__.__name__} yet'
            self.best_bound = self._results['Problem'][0]['Lower bound']
            self.log = f'Not Implemented for {self.__class__.__name__} yet'
        else:
            raise NotImplementedError('Only Pyomo is implemented for Cbc solver.')


class GlpkSolver(Solver):
    """Solver implementation for Glpk. Also Look in class Solver for more details"""

    def __init__(
        self,
        mip_gap: float = 0.01,
        logfile_name: str = 'glpk.log',
        solver_output_to_console: bool = True,
    ):
        super().__init__(mip_gap, solver_output_to_console, logfile_name)

    def solve(self, modeling_language: 'ModelingLanguage'):
        if isinstance(modeling_language, PyomoModel):
            self._solver = pyo.SolverFactory('glpk')
            self._results = self._solver.solve(
                modeling_language.model,
                tee=self.solver_output_to_console,
                keepfiles=True,
                logfile=self.logfile_name,
                options={'mipgap': self.mip_gap},
            )

            self.objective = modeling_language.model.objective.expr()
            self.termination_message = self._results['Solver'][0]['Status']
            self.best_bound = self._results['Problem'][0]['Lower bound']
            try:
                self.log = SolverLog('glpk', self.logfile_name)
            except Exception as e:
                self.log = None
                logger.warning(f'SolverLog could not be loaded. {e}')
        else:
            raise NotImplementedError('Only Pyomo is implemented for Cbc solver.')


class ModelingLanguage(ABC):
    """
    Abstract base class for modeling languages.

    Methods:
        translate_model(model): Translates a math model into a solveable form.
    """

    @abstractmethod
    def translate_model(self, model: MathModel):
        raise NotImplementedError

    def solve(self, math_model: MathModel, solver: Solver):
        raise NotImplementedError


class PyomoModel(ModelingLanguage):
    """
    Pyomo-based modeling language for constructing and solving optimization models.
    Translates a MathModel into a PyomoModel.

    Attributes:
        model: Pyomo model instance.
        mapping (dict): Maps variables and equations to Pyomo components.
        _counter (int): Counter for naming Pyomo components.
    """

    def __init__(self):
        logger.debug('Loaded pyomo modules')

        self.model = pyo.ConcreteModel(name='(Minimalbeispiel)')

        self.mapping: Dict[Union[Variable, Equation], Any] = {}  # Mapping to Pyomo Units
        self._counter = 0

    def solve(self, math_model: MathModel, solver: Solver):
        if self._counter == 0:
            raise Exception(' First, call .translate_model(). Else PyomoModel cant solve()')
        solver.solve(self)

        # write results
        math_model.result_of_objective = self.model.objective.expr()
        for variable in math_model.variables:
            raw_results = self.mapping[variable].get_values().values()  # .values() of dict, because {0:0.1, 1:0.3,...}
            if variable.is_binary:
                dtype = np.int8  # geht das vielleicht noch kleiner ???
            else:
                dtype = float
            # transform to np-array (fromiter() is 5-7x faster than np.array(list(...)) )
            result = np.fromiter(raw_results, dtype=dtype)
            # Falls skalar:
            if len(result) == 1:
                variable.result = result[0]
            else:
                variable.result = result

    def translate_model(self, math_model: MathModel):
        for variable in math_model.variables:  # Variablen erstellen
            logger.debug(f'VAR {variable.label} gets translated to Pyomo')
            self.translate_variable(variable)
        for eq in math_model.equations:  # Gleichungen erstellen
            logger.debug(f'EQ {eq.label} gets translated to Pyomo')
            self.translate_equation(eq)
        for ineq in math_model.inequations:  # Ungleichungen erstellen:
            logger.debug(f'INEQ {ineq.label} gets translated to Pyomo')
            self.translate_inequation(ineq)

        obj = math_model.objective
        logger.debug(f'{obj.label} gets translated to Pyomo')
        self.translate_objective(obj)

    def translate_variable(self, variable: Variable):
        assert isinstance(variable, Variable), 'Wrong type of variable'

        if variable.is_binary:
            pyomo_comp = pyo.Var(variable.indices, domain=pyo.Binary)
        else:
            pyomo_comp = pyo.Var(variable.indices, within=pyo.Reals)
        self.mapping[variable] = pyomo_comp

        # Register in pyomo-model:
        self._register_pyomo_comp(pyomo_comp, variable)

        lower_bound_vector = utils.as_vector(variable.lower_bound, variable.length)
        upper_bound_vector = utils.as_vector(variable.upper_bound, variable.length)
        fixed_value_vector = utils.as_vector(variable.fixed_value, variable.length)
        for i in variable.indices:
            # Wenn Vorgabe-Wert vorhanden:
            if variable.fixed and (fixed_value_vector[i] is not None):
                # Fixieren:
                pyomo_comp[i].value = fixed_value_vector[i]
                pyomo_comp[i].fix()
            else:
                # Boundaries:
                pyomo_comp[i].setlb(lower_bound_vector[i])  # min
                pyomo_comp[i].setub(upper_bound_vector[i])  # max

    def translate_equation(self, equation: Equation):
        if not isinstance(equation, Equation):
            raise TypeError(f'Wrong Class: {equation.__class__.__name__}')

        # constant_vector hier erneut erstellen, da Anz. Glg. vorher noch nicht bekannt:
        constant_vector = equation.constant_vector

        def linear_sum_pyomo_rule(model, i):
            """This function is needed for pyomoy internal construction of Constraints."""
            lhs = 0
            summand: Summand
            for summand in equation.summands:
                lhs += self._summand_math_expression(summand, i)  # i-te Gleichung (wenn Skalar, dann wird i ignoriert)
            rhs = constant_vector[i]
            return lhs == rhs

        pyomo_comp = pyo.Constraint(range(equation.length), rule=linear_sum_pyomo_rule)  # Nebenbedingung erstellen

        self._register_pyomo_comp(pyomo_comp, equation)

    def translate_inequation(self, inequation: Inequation):
        if not isinstance(inequation, Inequation):
            raise TypeError(f'Wrong Class: {inequation.__class__.__name__}')

        # constant_vector hier erneut erstellen, da Anz. Glg. vorher noch nicht bekannt:
        constant_vector = inequation.constant_vector

        def linear_sum_pyomo_rule(model, i):
            """This function is needed for pyomoy internal construction of Constraints."""
            lhs = 0
            summand: Summand
            for summand in inequation.summands:
                lhs += self._summand_math_expression(summand, i)  # i-te Gleichung (wenn Skalar, dann wird i ignoriert)
            rhs = constant_vector[i]

            return lhs <= rhs

        pyomo_comp = pyo.Constraint(range(inequation.length), rule=linear_sum_pyomo_rule)  # Nebenbedingung erstellen

        self._register_pyomo_comp(pyomo_comp, inequation)

    def translate_objective(self, objective: Equation):
        if not isinstance(objective, Equation):
            raise TypeError(f'Class {objective.__class__.__name__} Can not be the objective!')
        if not objective.is_objective:
            raise TypeError(
                f'Objective Equation is not marked as objective, {objective.is_objective=}, '
                f'but was sent to translate to objective!'
            )
        if objective.length != 1:
            raise Exception('Length of Objective must be 0')

        def _rule_linear_sum_skalar(model):
            skalar = 0
            for summand in objective.summands:
                skalar += self._summand_math_expression(summand)
            return skalar

        self.model.objective = pyo.Objective(rule=_rule_linear_sum_skalar, sense=pyo.minimize)
        self.mapping[objective] = self.model.objective

    def _summand_math_expression(self, summand: Summand, at_index: int = 0) -> 'pyo.Expression':
        pyomo_variable = self.mapping[summand.variable]
        if isinstance(summand, SumOfSummand):
            return sum(pyomo_variable[summand.indices[j]] * summand.factor_vec[j] for j in summand.indices)

        # Ausdruck für i-te Gleichung (falls Skalar, dann immer gleicher Ausdruck ausgegeben)
        if summand.length == 1:
            # ignore argument at_index, because Skalar is used for every single equation
            return pyomo_variable[summand.indices[0]] * summand.factor_vec[0]
        if len(summand.indices) == 1:
            return pyomo_variable[summand.indices[0]] * summand.factor_vec[at_index]
        return pyomo_variable[summand.indices[at_index]] * summand.factor_vec[at_index]

    def _register_pyomo_comp(self, pyomo_comp, part: Union[Variable, Equation, Inequation]) -> None:
        self._counter += 1  # Counter to guarantee unique names
        self.model.add_component(f'{part.label}__{self._counter}', pyomo_comp)
        self.mapping[part] = pyomo_comp
