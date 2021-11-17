# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Constrained Quadratic Model class.
"""

import collections.abc as abc
import json
import re
import tempfile
import uuid
import warnings
import zipfile

from numbers import Number
from typing import Hashable, Optional, Union, BinaryIO, ByteString, Iterable, Collection, Dict
from typing import Callable, MutableMapping, Iterator, Tuple, Mapping, Any, NamedTuple

import numpy as np

from dimod.core.bqm import BQM as BQMabc
from dimod.binary.binary_quadratic_model import BinaryQuadraticModel, Binary, Spin, as_bqm
from dimod.discrete.discrete_quadratic_model import DiscreteQuadraticModel
from dimod.quadratic import QuadraticModel
from dimod.sampleset import as_samples
from dimod.sym import Comparison, Eq, Le, Ge, Sense
from dimod.serialization.fileview import SpooledTemporaryFile, _BytesIO
from dimod.serialization.fileview import load, read_header, write_header
from dimod.typing import Bias, Variable
from dimod.utilities import new_label
from dimod.variables import Variables, serialize_variable, deserialize_variable
from dimod.vartypes import Vartype, as_vartype, VartypeLike
from dimod.serialization.lp import make_lp_grammar, get_variables_from_parsed_lp, constraint_symbols, obj_senses

__all__ = ['ConstrainedQuadraticModel', 'CQM', 'cqm_to_bqm']


CQM_MAGIC_PREFIX = b'DIMODCQM'


class ConstraintData(NamedTuple):
    label: Hashable
    lhs_energy: float
    rhs_energy: float
    sense: Sense
    activity: float
    violation: float


class ConstrainedQuadraticModel:
    r"""A constrained quadratic model.

    Constrained quadratic models are problems of the form:

    .. math::

        \begin{align}
            \text{Minimize an objective:} & \\
            & \sum_{i} a_i x_i + \sum_{i<j} b_{ij} x_i x_j + c, \\
            \text{Subject to constraints:} & \\
            & \sum_i a_i^{(c)} x_i + \sum_{i<j} b_{ij}^{(c)} x_i x_j+ c^{(c)} \le 0,
            \quad c=1, \dots, C_{\rm ineq.}, \\
            & \sum_i a_i^{(d)} x_i + \sum_{i<j} b_{ij}^{(d)} x_i x_j + c^{(d)} = 0,
            \quad d=1, \dots, C_{\rm eq.},
        \end{align}

    where :math:`\{ x_i\}_{i=1, \dots, N}` can be binary or integer
    variables, :math:`a_{i}, b_{ij}, c` are real values and
    :math:`C_{\rm ineq.}, C_{\rm eq,}` are the number of inequality and
    equality constraints respectively.

    The objective and constraints are encoded as either :class:`.QuadraticModel`
    or :class:`.BinaryQuadraticModel` depending on the variable types used.

    Example:

        Solve a simple `bin packing problem <https://w.wiki/3jz4>`_. In this
        problem we wish to pack a set of items of different weights into
        the smallest number of bins possible.

        See :func:`~dimod.generators.bin_packing` for a general function to
        generate bin packing problems. We follow the same naming conventions
        in this example.

        Let's start with four object weights and assume that each bin has a
        capacity of 1.

        >>> weights = [.9, .7, .2, .1]
        >>> capacity = 1

        Let :math:`y_j` indicate that we used bin :math:`j`. We know that we
        will use four or fewer total bins.

        >>> y = [dimod.Binary(f'y_{j}') for j in range(len(weights))]

        Let :math:`x_{i,j}` indicate that we put item :math:`i` in bin
        :math:`j`.

        >>> x = [[dimod.Binary(f'x_{i}_{j}') for j in range(len(weights))]
        ...      for i in range(len(weights))]

        Create an empty constrained quadratic model with no objective or
        constraints.

        >>> cqm = dimod.ConstrainedQuadraticModel()

        We wish to minimize the number of bins used. Therefore our objective
        is to minimize the value of :math:`\sum_j y_j`.

        >>> cqm.set_objective(sum(y))

        We also need to enforce the constraint that each item can only go
        in one bin. We can express this constraint, for a given item :math:`i`,
        with :math:`\sum_j x_{i, j} == 1`. Note that the label of each
        constraint is returned so that we can access them in the future if
        desired.

        >>> for i in range(len(weights)):
        ...     cqm.add_constraint(sum(x[i]) == 1, label=f'item_placing_{i}')
        'item_placing_0'
        'item_placing_1'
        'item_placing_2'
        'item_placing_3'

        Finally, we need to enforce the limits on each bin. We can express
        this constraint, for a given bin :math:`j`, with
        :math:`\sum_i x_{i, j} * w_i <= c` where :math:`w_i` is the weight
        of item :math:`i` and :math:`c` is the capacity.

        >>> for j in range(len(weights)):
        ...     cqm.add_constraint(
        ...         sum(weights[i] * x[i][j] for i in range(len(weights))) - y[j] * capacity <= 0,
        ...         label=f'capacity_bin_{j}')
        'capacity_bin_0'
        'capacity_bin_1'
        'capacity_bin_2'
        'capacity_bin_3'

    """
    def __init__(self):
        # discrete variable tracking, we probably can do this with less memory
        # but for now let's keep it simple
        self.discrete: Set[Hashable] = set()  # collection of discrete constraints
        self._discrete: Set[Variable] = set()  # collection of all variables used in discrete

        self._objective = QuadraticModel()

    @property
    def constraints(self) -> Dict[Hashable, Comparison]:
        """The constraints as a dictionary.

        This dictionary and its contents should not be modified.
        """
        try:
            return self._constraints
        except AttributeError:
            pass

        self._constraints: Dict[Hashable, Comparison] = {}
        return self._constraints

    @property
    def objective(self) -> QuadraticModel:
        """The objective to be minimized."""
        return self._objective

    @property
    def variables(self) -> Variables:
        """The variables in use over the objective and all constraints."""
        try:
            return self._variables
        except AttributeError:
            pass

        self._variables = variables = self.objective.variables

        # to support backwards compatibility (0.10.0 - 0.10.5), we annotate
        # this object with some attributes. All of these will be removed in
        # 0.11.0
        def vartype(v):
            warnings.warn(
                "cqm.variables.vartype(v) is deprecated and will be removed in dimod 0.11.0, "
                "use cqm.vartype(v) instead.", DeprecationWarning, stacklevel=2)
            return self.vartype(v)

        variables.vartype = vartype  # method
        variables.vartypes = _Vartypes(self)
        variables.lower_bounds = _LowerBounds(self)
        variables.upper_bounds = _UpperBounds(self)

        return variables

    def _add_variables_from(self, model: Union[BinaryQuadraticModel, QuadraticModel]):
        # todo: singledispatchmethod in 3.8+
        if isinstance(model, (BinaryQuadraticModel, BQMabc)):
            vartype = model.vartype

            for v in model.variables:
                self.objective.add_variable(vartype, v)

        elif isinstance(model, QuadraticModel):
            for v in model.variables:
                # for spin, binary variables the bounds are ignored anyway
                self.objective.add_variable(model.vartype(v), v,
                                            lower_bound=model.lower_bound(v),
                                            upper_bound=model.upper_bound(v))
        else:
            raise TypeError("model should be a QuadraticModel or a BinaryQuadraticModel")

    def add_constraint(self, data, *args, **kwargs) -> Hashable:
        """A convenience wrapper for other methods that add constraints.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integers
            >>> i, j = Integers(['i', 'j'])
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_constraint(i + j <= 3, label='Constrained i-j range')
            'Constrained i-j range'

        See also:
            :meth:`~.ConstrainedQuadraticModel.add_constraint_from_model`

            :meth:`~.ConstrainedQuadraticModel.add_constraint_from_comparison`

            :meth:`~.ConstrainedQuadraticModel.add_constraint_from_iterable`

        """
        # in python 3.8+ we can use singledispatchmethod
        if isinstance(data, (BinaryQuadraticModel, QuadraticModel, BQMabc)):
            return self.add_constraint_from_model(data, *args, **kwargs)
        elif isinstance(data, Comparison):
            return self.add_constraint_from_comparison(data, *args, **kwargs)
        elif isinstance(data, Iterable):
            return self.add_constraint_from_iterable(data, *args, **kwargs)
        else:
            raise TypeError("unexpected data format")

    def add_constraint_from_model(self,
                                  qm: Union[BinaryQuadraticModel, QuadraticModel],
                                  sense: Union[Sense, str],
                                  rhs: Bias = 0,
                                  label: Optional[Hashable] = None,
                                  copy: bool = True) -> Hashable:
        """Add a constraint from a quadratic model.

        Args:
            qm: A quadratic model or binary quadratic model.

            sense: One of `<=', '>=', '=='.

            rhs: The right hand side of the constraint.

            label: A label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

            copy: If `True`, the BQM is copied. This can be set to `False` to
                improve performance, but subsequently mutating the bqm can
                cause issues.

        Returns:
            The label of the added constraint.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Binary
            >>> cqm = ConstrainedQuadraticModel()
            >>> x = Binary('x')
            >>> cqm.add_constraint_from_model(x, '>=', 0, 'Min x')
            'Min x'
        """
        variables = self.variables

        # get sense as an enum
        if isinstance(sense, str):
            sense = Sense(sense)

        if label is None:
            # we support up to 100k constraints and :6 gives us 16777216
            # possible so pretty safe
            label = uuid.uuid4().hex[:6]
            while label in self.constraints:
                label = uuid.uuid4().hex[:6]
        elif label in self.constraints:
            raise ValueError("a constraint with that label already exists")

        if isinstance(qm, BQMabc):
            qm = as_bqm(qm)  # handle legacy BQMs

        self._add_variables_from(qm)

        if copy:
            qm = qm.copy()

        if sense is Sense.Le:
            self.constraints[label] = Le(qm, rhs)
        elif sense is Sense.Ge:
            self.constraints[label] = Ge(qm, rhs)
        elif sense is Sense.Eq:
            self.constraints[label] = Eq(qm, rhs)
        else:
            raise RuntimeError("unexpected sense")

        return label

    def add_constraint_from_comparison(self,
                                       comp: Comparison,
                                       label: Optional[Hashable] = None,
                                       copy: bool = True) -> Hashable:
        """Add a constraint from a comparison.

        Args:
            comp: A comparison object.

            label: A label for the constraint. Must be unique. If no label
                is provided, one is generated using :mod:`uuid`.

            copy: If `True`, the model is copied. You can set to `False` to
                improve performance, but subsequently mutating the model can
                cause issues.

        Returns:
            Label of the added constraint.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer
            >>> i = Integer('i')
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_constraint_from_comparison(i <= 3, label='Max i')
            'Max i'
        """
        if not isinstance(comp.rhs, Number):
            raise TypeError("comparison should have a numeric rhs")

        if isinstance(comp.lhs, (BinaryQuadraticModel, QuadraticModel)):
            return self.add_constraint_from_model(comp.lhs, comp.sense, rhs=comp.rhs,
                                                  label=label, copy=copy)
        else:
            raise ValueError("comparison should have a binary quadratic model "
                             "or quadratic model lhs.")

    def add_constraint_from_iterable(self, iterable: Iterable,
                                     sense: Union[Sense, str],
                                     rhs: Bias = 0,
                                     label: Optional[Hashable] = None,
                                     ) -> Hashable:
        """Add a constraint from an iterable of tuples.

        Args:
            iterable: An iterable of terms as tuples. The variables must
                have already been added to the object.

            sense: One of `<=', '>=', '=='.

            rhs: The right hand side of the constraint.

            label: A label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

        Returns:
            The label of the added constraint.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer, Binary
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_variable('i', 'INTEGER')   # doctest: +IGNORE_RESULT
            >>> cqm.add_variable('j', 'INTEGER')   # doctest: +IGNORE_RESULT
            >>> cqm.add_variable('x', 'BINARY')    # doctest: +IGNORE_RESULT
            >>> cqm.add_variable('y', 'BINARY')    # doctest: +IGNORE_RESULT
            >>> label1 = cqm.add_constraint_from_iterable([('x', 'y', 1), ('i', 2), ('j', 3),
            ...                                           ('i', 'j', 1)], '<=', rhs=1)

        """
        qm = self._iterable_to_qm(iterable)

        # use quadratic model in the future
        return self.add_constraint_from_model(
            qm, sense, rhs=rhs, label=label, copy=False)

    def add_discrete(self, variables: Iterable[Variable],
                     label: Optional[Hashable] = None) -> Hashable:
        """Add an iterable of binary variables as a disjoint one-hot constraint.

        Adds a special kind of one-hot constraint. These one-hot constraints
        must be disjoint, that is they must not have any overlapping variables.

        Args:
            variables: An iterable of variables.

            label: Label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

        Returns:
            Label of the added constraint.

        Raises:
            ValueError: If any of the given variables have already been added
                to the model with any vartype other than `BINARY`.

            ValueError: If any of the given variables are already used in
                another discrete variable.

        """
        if label is not None and label in self.constraints:
            raise ValueError("a constraint with that label already exists")

        if isinstance(variables, Iterator):
            variables = list(variables)

        for v in variables:
            if v in self._discrete:
                # todo: language around discrete variables?
                raise ValueError(f"variable {v!r} is already used in a discrete variable")
            elif v in self.variables and self.vartype(v) != Vartype.BINARY:
                raise ValueError(f"variable {v!r} has already been added but is not BINARY")

        # we can! So add them
        bqm = BinaryQuadraticModel('BINARY', dtype=np.float32)
        bqm.add_variables_from((v, 1) for v in variables)
        label = self.add_constraint(bqm == 1, label=label)
        self.discrete.add(label)
        self._discrete.update(variables)
        return label

    def add_variable(self, v: Variable, vartype: VartypeLike,
                     *, lower_bound: int = 0, upper_bound: Optional[int] = None):
        """Add a variable to the model.

        Args:
            variable: A variable label.

            vartype:
                Variable type. One of:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`.Vartype.INTEGER`, ``'INTEGER'``

            lower_bound:
                A lower bound on the variable. Ignored when the variable is
                not :class:`Vartype.INTEGER`.

            upper_bound:
                An upper bound on the variable. Ignored when the variable is
                not :class:`Vartype.INTEGER`.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_variable('i', 'INTEGER')   # doctest: +IGNORE_RESULT

        """
        if self.variables.count(v):
            if as_vartype(vartype, extended=True) != self.vartype(v):
                raise ValueError("given variable has already been added with a different vartype")
        else:
            return self.objective.add_variable(vartype, v, lower_bound=lower_bound, upper_bound=upper_bound)

    def check_feasible(self, sample_like, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
        r"""Return the feasibility of the given sample.

        A sample is feasible if all constraints are satisfied. A constraint's
        satisfaction is tested using the following equation:

        .. math::

            violation <= (atol + rtol * | rhs\_energy | )

        where ``violation`` and ``rhs_energy`` are as returned by :meth:`.iter_constraint_data`.

        Args:
            sample_like: A sample.
            rtol: The relative tolerance.
            atol: the absolute tolerance.

        Returns:
            True if the sample is feasible (given the tolerances).

        """
        return all(datum.violation <= atol + rtol*abs(datum.rhs_energy)
                   for datum in self.iter_constraint_data(sample_like))

    @classmethod
    def from_bqm(cls, bqm: BinaryQuadraticModel) -> 'ConstrainedQuadraticModel':
        """Alias for :meth:`from_quadratic_model`."""
        return cls.from_quadratic_model(bqm)

    @classmethod
    def from_discrete_quadratic_model(cls, dqm: DiscreteQuadraticModel, *,
                                      relabel_func: Callable[[Variable, int], Variable] = lambda v, c: (v, c),
                                      ) -> 'ConstrainedQuadraticModel':
        """Construct a constrained quadratic model from a discrete quadratic model.

        Args:
            dqm: a discrete quadratic model.

            relabel_func (optional): A function that takes two arguments, the
                variable label and the case label, and returns a new variable
                label to be used in the CQM. By default generates a 2-tuple
                `(variable, case)`.

        Returns:
            A constrained quadratic model.

        """
        cqm = cls()

        objective = BinaryQuadraticModel(Vartype.BINARY)

        seen = set()
        for v in dqm.variables:
            seen.add(v)

            # convert v, case to a flat set of variables
            v_vars = list(relabel_func(v, case) for case in dqm.get_cases(v))

            # add the one-hot constraint
            cqm.add_discrete(v_vars, label=v)

            # add to the objective
            objective.add_linear_from(zip(v_vars, dqm.get_linear(v)))

            for u in dqm.adj[v]:
                if u in seen:  # only want upper-triangle
                    continue

                u_vars = list(relabel_func(u, case) for case in dqm.get_cases(u))

                objective.add_quadratic_from(
                    (u_vars[cu], v_vars[cv], bias)
                    for (cu, cv), bias
                    in dqm.get_quadratic(u, v).items()
                    )

        objective.offset = dqm.offset

        cqm.set_objective(objective)

        return cqm

    from_dqm = from_discrete_quadratic_model

    @classmethod
    def from_quadratic_model(cls, qm: Union[QuadraticModel, BinaryQuadraticModel]
                             ) -> 'ConstrainedQuadraticModel':
        """Construct a constrained quadratic model from a quadratic model or
        binary quadratic model.

        The specified model is set as the objective to be minimzed in the constructed
        constrained quadratic model (CQM). You can then add constraints that any feasible
        solutions should meet.

        Args:
            qm: Binary quadratic model (BQM) or quadratic model (QM).

        Examples:
            This example creates a CQM to minimize a triangular problem with the added
            constraint that one of the variables must have value 1 in feasible solutions.

            >>> from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel
            >>> bqm = BinaryQuadraticModel.from_ising({}, {'ab': 1, 'bc': 1, 'ac': 1})
            >>> cqm = ConstrainedQuadraticModel().from_bqm(bqm)
            >>> cqm.objective.linear
            {'a': 0.0, 'b': 0.0, 'c': 0.0}
            >>> cqm.objective.quadratic
            {('b', 'a'): 1.0, ('c', 'a'): 1.0, ('c', 'b'): 1.0}
            >>> label1 = cqm.add_constraint_from_model(BinaryQuadraticModel({'a': 0}, {}, 0, 'SPIN'), '>=', 0)
        """
        cqm = cls()
        cqm.set_objective(qm)
        return cqm

    @classmethod
    def from_qm(cls, qm: QuadraticModel) -> 'ConstrainedQuadraticModel':
        """Alias for :meth:`from_quadratic_model`."""
        return cls.from_quadratic_model(qm)

    @classmethod
    def from_file(cls, fp: Union[BinaryIO, ByteString]) -> "ConstrainedQuadraticModel":
        """Construct from a file-like object.

        The inverse of :meth:`~ConstrainedQuadraticModel.to_file`.
        """
        if isinstance(fp, ByteString):
            file_like: BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
        else:
            file_like = fp

        header_info = read_header(file_like, CQM_MAGIC_PREFIX)

        if header_info.version >= (2, 0):
            raise ValueError("cannot load a BQM serialized with version "
                             f"{header_info.version!r}, try upgrading your "
                             "dimod version")

        # we don't actually need the data

        cqm = CQM()

        with zipfile.ZipFile(file_like, mode='r') as zf:
            cqm.set_objective(load(zf.read("objective")))

            constraint_labels = set()
            for arch in zf.namelist():
                # even on windows zip uses /
                match = re.match("constraints/([^/]+)/", arch)
                if match is not None:
                    constraint_labels.add(match.group(1))

            for constraint in constraint_labels:
                lhs = load(zf.read(f"constraints/{constraint}/lhs"))
                rhs = np.frombuffer(zf.read(f"constraints/{constraint}/rhs"), np.float64)[0]
                sense = zf.read(f"constraints/{constraint}/sense").decode('ascii')
                discrete = any(zf.read(f"constraints/{constraint}/discrete"))
                label = deserialize_variable(json.loads(constraint))
                cqm.add_constraint(lhs, rhs=rhs, sense=sense, label=label)
                if discrete:
                    cqm.discrete.add(label)

        return cqm

    def iter_constraint_data(self, sample_like) -> Iterator[ConstraintData]:
        """Yield information about the constraints for the given sample.

        Args:
            sample_like: A sample.

        Yields:
            A :class:`collections.namedtuple` with ``label``, ``lhs_energy``,
            ``rhs_energy``, ``sense``, ``activity``, and ``violation`` fields.
            ``label`` is the constraint label.
            ``lhs_energy`` is the energy of the left hand side of the constraint.
            ``rhs_energy`` is the energy of the right hand side of the constraint.
            ``sense`` is the :class:`dimod.sym.Sense` of the constraint.
            ``activity`` is ``lhs_energy - rhs_energy``
            ``violation`` is determined by the type of constraint. If ``violation``
            is positive, that means that the constraint has been violated by
            that amount. If it is negative, that means that the constraint has
            been satisfied by the amount.

        """

        sample, labels = as_samples(sample_like)

        if sample.shape[0] != 1:
            raise ValueError("sample_like should be a single sample, "
                             f"received {sample.shape[0]} samples")

        for label, constraint in self.constraints.items():
            lhs = constraint.lhs.energy((sample, labels))
            rhs = constraint.rhs
            sense = constraint.sense

            activity = lhs - rhs

            if sense is Sense.Eq:
                violation = abs(activity)
            elif sense is Sense.Ge:
                violation = -activity
            elif sense is Sense.Le:
                violation = activity
            else:
                raise RuntimeError("unexpected sense")

            yield ConstraintData(
                activity=activity,
                sense=sense,
                violation=violation,
                lhs_energy=lhs,
                rhs_energy=rhs,
                label=label,
                )

    def iter_violations(self, sample_like, *, skip_satisfied: bool = False, clip: bool = False,
                        ) -> Iterator[Tuple[Hashable, Bias]]:
        """Yield violations for all constraints.

        Args:
            sample_like: A sample over the CQM variables.
            skip_satisfied: If True, does not yield constraints that are satisfied.
            clip: If True, negative violations are rounded up to 0.

        Yields:
            A 2-tuple containing the constraint label and the amount of
            constraints violation.

        Example:

            Construct a constrained quadratic model.

            >>> i, j, k = dimod.Binaries(['i', 'j', 'k'])
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_constraint(i + j + k == 10, label='equal')
            'equal'
            >>> cqm.add_constraint(i + j <= 15, label='less equal')
            'less equal'
            >>> cqm.add_constraint(j - k >= 0, label='greater equal')
            'greater equal'

            Check the violations of a sample that satisfies all constraints.

            >>> sample = {'i': 3, 'j': 5, 'k': 2}
            >>> for label, violation in cqm.iter_violations(sample, clip=True):
            ...     print(label, violation)
            equal 0.0
            less equal 0.0
            greater equal 0.0

            Check the violations for a sample that does not satisfy all of the
            constraints.

            >>> sample = {'i': 3, 'j': 2, 'k': 5}
            >>> for label, violation in cqm.iter_violations(sample, clip=True):
            ...     print(label, violation)
            equal 0.0
            less equal 0.0
            greater equal 3.0

            >>> sample = {'i': 3, 'j': 2, 'k': 5}
            >>> for label, violation in cqm.iter_violations(sample, skip_satisfied=True):
            ...     print(label, violation)
            greater equal 3.0

        """
        if skip_satisfied:
            # clip doesn't matter in this case
            # todo: feasibility tolerance?
            for datum in self.iter_constraint_data(sample_like):
                if datum.violation > 0:
                    yield datum.label, datum.violation
        elif clip:
            for datum in self.iter_constraint_data(sample_like):
                yield datum.label, max(datum.violation, 0.0)
        else:
            for datum in self.iter_constraint_data(sample_like):
                yield datum.label, datum.violation

    def is_almost_equal(self, other: 'ConstrainedQuadraticModel',
                        places: int = 7) -> bool:
        """Return True if the given model's objective and constraints are almost equal."""
        def constraint_eq(c0: Comparison, c1: Comparison) -> bool:
            return (c0.sense is c1.sense
                    and c0.lhs.is_almost_equal(c1.lhs, places=places)
                    and not round(c0.rhs - c1.rhs, places))

        return (self.objective.is_almost_equal(other.objective, places=places)
                and self.constraints.keys() == other.constraints.keys()
                and all(constraint_eq(constraint, other.constraints[label])
                        for label, constraint in self.constraints.items()))

    def is_equal(self, other: 'ConstrainedQuadraticModel') -> bool:
        """Return True if the given model has the same objective and constraints."""
        def constraint_eq(c0: Comparison, c1: Comparison) -> bool:
            return (c0.sense is c1.sense
                    and c0.lhs.is_equal(c1.lhs)
                    and c0.rhs == c1.rhs)

        return (self.objective.is_equal(other.objective)
                and self.constraints.keys() == other.constraints.keys()
                and all(constraint_eq(constraint, other.constraints[label])
                        for label, constraint in self.constraints.items()))

    def lower_bound(self, v: Variable) -> Bias:
        """Return the lower bound on the specified variable."""
        return self.objective.lower_bound(v)

    def num_biases(self) -> int:
        """The number of biases accross the objective and constraints."""
        num_biases = len(self.objective.linear) + len(self.objective.quadratic)
        num_biases += sum(len(const.lhs.linear) + len(const.lhs.quadratic)
                          for const in self.constraints.values())
        return num_biases

    def num_quadratic_variables(self) -> int:
        """Return the total number of variables with at least one quadratic
        interaction accross all constraints."""
        count = 0
        for const in self.constraints.values():
            lhs = const.lhs
            count += sum(lhs.degree(v) > 0 for v in lhs.variables)
        return count

    def set_objective(self, objective: Union[BinaryQuadraticModel,
                                             QuadraticModel, Iterable]):
        """Set the objective of the constrained quadratic model.

        Args:
            objective: Binary quadratic model (BQM) or quadratic model (QM) or
            an iterable of tuples.

        Examples:
            >>> from dimod import Integer, ConstrainedQuadraticModel
            >>> i = Integer('i')
            >>> j = Integer('j')
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.set_objective(2*i - 0.5*i*j + 10)

        """
        if isinstance(objective, Iterable):
            objective = self._iterable_to_qm(objective)
        # clear out current objective, keeping only the variables
        self.objective.quadratic.clear()  # there may be a more performant way...
        for v in self.objective.variables:
            self.objective.set_linear(v, 0)
        # offset is overwritten later

        # now add everything from the new objective
        self._add_variables_from(objective)

        for v in objective.variables:
            self.objective.set_linear(v, objective.get_linear(v))
        self.objective.add_quadratic_from(objective.iter_quadratic())
        self.objective.offset = objective.offset

    def _iterable_to_qm(self, iterable: Iterable) -> QuadraticModel:
        qm = QuadraticModel()

        def _add_variable(v):
            # handles vartype, and bounds
            vartype = self.vartype(v)

            if vartype is not Vartype.SPIN and vartype is not Vartype.BINARY:
                # need to worry about bounds
                qm.add_variable(vartype, v,
                                lower_bound=self.lower_bound(v),
                                upper_bound=self.upper_bound(v))
            else:
                qm.add_variable(vartype, v)

        for *variables, bias in iterable:
            if len(variables) == 0:
                qm.offset += bias
            elif len(variables) == 1:
                v, = variables
                _add_variable(v)
                qm.add_linear(v, bias)
            elif len(variables) == 2:
                u, v = variables
                _add_variable(u)
                _add_variable(v)
                qm.add_quadratic(u, v, bias)
            else:
                raise ValueError("terms must be constant, linear or quadratic")
        return qm

    def _substitute_self_loops_from_model(self, qm: Union[BinaryQuadraticModel, QuadraticModel],
                                          mapping: MutableMapping[Variable, Variable]):
        if isinstance(qm, BinaryQuadraticModel):
            # bqms never have self-loops
            return

        for u in qm.variables:
            vartype = qm.vartype(u)

            # integer and binary variables never have self-loops
            if vartype is Vartype.SPIN or vartype is Vartype.BINARY:
                continue

            try:
                bias = qm.get_quadratic(u, u)
            except ValueError:
                # no self-loop
                continue

            lb = qm.lower_bound(u)
            ub = qm.upper_bound(u)

            if u not in mapping:
                # we've never seen this integer before
                new: Variable = new_label()

                # on the off chance there are conflicts. Luckily self.variables
                # is global accross all constraints/objective so we don't need
                # to worry about accidentally picking something we'll regret
                while new in self.constraints or new in self.variables:
                    new = new_label()

                mapping[u] = new

                self.objective.add_variable(vartype, new, lower_bound=lb, upper_bound=ub)

                # we don't add the constraint yet because we don't want
                # to modify self.constraints
            else:
                new = mapping[u]

            qm.add_variable(vartype, new, lower_bound=lb, upper_bound=ub)

            qm.add_quadratic(u, new, bias)
            qm.remove_interaction(u, u)

    def substitute_self_loops(self) -> Dict[Variable, Variable]:
        """Replace any integer self-loops in the objective or constraints.

        Self-loop :math:`i^2` is removed by introducing a new variable
        :math:`j` with interaction :math:`i*j` and adding constraint
        :math:`j == i`.

        Acts on the objective and constraints in-place.

        Returns:
            Mapping from the integer variable labels to their introduced
            counterparts. The constraint enforcing :math:`j == i` uses
            the same label.

        Examples:
            >>> from dimod import Integer, ConstrainedQuadraticModel
            >>> i = Integer('i')
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_constraint(i*i <=3, label='i squared')
            'i squared'
            >>> cqm.substitute_self_loops()                      # doctest: +IGNORE_RESULT
            >>> cqm.constraints   # doctest: +IGNORE_RESULT
            {'i squared': QuadraticModel({'i': 0.0, 'cf651f3d-bdf8-4735-9139-eee0a32e217f': 0.0}, {('cf651f3d-bdf8-4735-9139-eee0a32e217f', 'i'): 1.0}, 0.0, {'i': 'INTEGER', 'cf651f3d-bdf8-4735-9139-eee0a32e217f': 'INTEGER'}, dtype='float64') <= 3,
            'cf651f3d-bdf8-4735-9139-eee0a32e217f': QuadraticModel({'i': 1.0, 'cf651f3d-bdf8-4735-9139-eee0a32e217f': -1.0}, {}, 0.0, {'i': 'INTEGER', 'cf651f3d-bdf8-4735-9139-eee0a32e217f': 'INTEGER'}, dtype='float64') == 0}
        """
        mapping: Dict[Variable, Variable] = dict()

        self._substitute_self_loops_from_model(self.objective, mapping)

        for comparison in self.constraints.values():
            self._substitute_self_loops_from_model(comparison.lhs, mapping)

        # finally add the constraints for the variables
        for v, new in mapping.items():
            self.add_constraint([(v, 1), (new, -1)], rhs=0, sense='==', label=new)

        return mapping

    def to_file(self, *, spool_size: int = int(1e9)) -> tempfile.SpooledTemporaryFile:
        """Serialize to a file-like object.

        Args:
            spool_size: Defines the `max_size` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents will be kept on disk or in
                memory.

        Format Specification (Version 1.1):

            This format is inspired by the `NPY format`_

            The first 8 bytes are a magic string: exactly "DIMODCQM".

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data HEADER_LEN.

            The next HEADER_LEN bytes form the header data. This is a
            json-serialized dictionary. The dictionary is exactly:

            .. code-block:: python

                dict(num_variables=len(cqm.variables),
                     num_constraints=len(cqm.constraints),
                     num_biases=cqm.num_biases(),
                     num_quadratic_variables=cqm.num_quadratic_variables(),
                     )

            it is terminated by a newline character and padded with spaces to
            make the entire length of the entire header divisible by 64.

            The constraint quadratic model data comes after the header. It is
            encoded as a zip file. The zip file will contain one file
            named `objective`, containing the objective as encoded as a file
            view. It will also contain a directory called `constraints`. The
            `constraints` directory will contain one subdirectory for each
            constraint, each containing `lhs`, `rhs` and `sense` encoding
            the `lhs` as a fileview, the `rhs` as a float and the sense
            as a string. Each directory will also contain a `discrete` file,
            encoding whether the constraint represents a discrete variable.

        Format Specification (Version 1.0):

            This format is the same as Version 1.1, except that the data dict
            does not have `num_quadratic_variables`.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

        """
        file = SpooledTemporaryFile(max_size=spool_size)

        data = dict(num_variables=len(self.variables),
                    num_constraints=len(self.constraints),
                    num_biases=self.num_biases(),
                    num_quadratic_variables=self.num_quadratic_variables(),
                    )

        write_header(file, CQM_MAGIC_PREFIX, data, version=(1, 1))

        # write the values
        with zipfile.ZipFile(file, mode='a') as zf:
            try:
                zf.writestr(
                    'objective', self.objective.to_file(spool_size=int(1e12))._file.getbuffer())
            except AttributeError:
                # no objective to write
                pass

            for label, constraint in self.constraints.items():
                # put everything in a constraints/label/ directory
                lstr = json.dumps(serialize_variable(label))

                lhs = constraint.lhs.to_file(spool_size=int(1e12))._file.getbuffer()
                zf.writestr(f'constraints/{lstr}/lhs', lhs)

                rhs = np.float64(constraint.rhs).tobytes()
                zf.writestr(f'constraints/{lstr}/rhs', rhs)

                sense = bytes(constraint.sense.value, 'ascii')
                zf.writestr(f'constraints/{lstr}/sense', sense)

                discrete = bytes((label in self.discrete,))
                zf.writestr(f'constraints/{lstr}/discrete', discrete)

        file.seek(0)
        return file

    def upper_bound(self, v: Variable) -> Bias:
        """Return the upper bound on the specified variable."""
        return self.objective.upper_bound(v)

    def vartype(self, v: Variable) -> Vartype:
        """The vartype of the given variable."""
        return self.objective.vartype(v)

    def violations(self, sample_like, *, skip_satisfied: bool = False, clip: bool = False,
                   ) -> Dict[Hashable, Bias]:
        """Return a dictionary mapping constraint labels to the amount the constraints are violated.

        This method is a shortcut for ``dict(cqm.iter_violations(sample))``.
        """
        return dict(self.iter_violations(sample_like, skip_satisfied=skip_satisfied, clip=clip))

    @classmethod
    def from_lp_file(cls,
                     fp: Union[BinaryIO, ByteString],
                     lower_bound_default: Optional[int] = None,
                     upper_bound_default: Optional[int] = None) -> "ConstrainedQuadraticModel":
        """Create a CQM model from a LP file.

        Args:
            fp: file-like object in the LP format
            lower_bound_default: in case lower bounds for integer are not given, this will be the default
            upper_bound_default: in case upper bounds for integer are not given, this will be the default

        Returns:
            The model encoded in the LP file as a :class:`ConstrainedQuadraticModel`.
        """
        grammar = make_lp_grammar()
        parse_output = grammar.parseFile(fp)
        obj = get_variables_from_parsed_lp(parse_output, lower_bound_default, upper_bound_default)

        cqm = ConstrainedQuadraticModel()

        # parse and set the objective
        obj_coefficient = 1
        for oe in parse_output.objective:

            if isinstance(oe, str):
                if oe in obj_senses.keys():
                    obj_coefficient = obj_senses[oe]

            else:
                if len(oe) == 2:
                    if oe.name != "":
                        # linear term
                        obj.add_linear(oe.name[0], obj_coefficient * oe.coef)

                    else:
                        # this is a pure squared term
                        varname = oe.squared_name[0]
                        vartype = obj.vartype(varname)

                        if vartype is Vartype.BINARY:
                            obj.add_linear(varname, obj_coefficient * oe.coef * 0.5)
                        elif vartype is Vartype.INTEGER:
                            obj.add_quadratic(varname, varname,
                                              obj_coefficient * oe.coef * 0.5)
                        else:
                            raise TypeError("Unexpected variable type: {}".format(vartype))

                elif len(oe) == 3:
                    # bilinear term
                    var1 = oe.name[0]
                    var2 = oe.second_var_name[0]
                    obj.add_quadratic(var1, var2, obj_coefficient * oe.coef * 0.5)

        cqm.set_objective(obj)

        # adding constraints
        for c in parse_output.constraints:

            try:
                cname = c.name[0]
            except IndexError:
                # the constraint is nameless, set this to None now
                cname = None

            csense = constraint_symbols[c.sense]
            ccoef = c.rhs
            constraint = QuadraticModel()

            if not c.lin_expr and not c.quad_expr:
                # empty constraint
                warnings.warn('The LP file contained an empty constraint and it will be ignored',
                              stacklevel=2)
                continue

            if c.lin_expr:

                for le in c.lin_expr:
                    var = le.name[0]
                    vartype = obj.vartype(var)
                    lb = obj.lower_bound(var)
                    ub = obj.upper_bound(var)
                    if vartype is Vartype.BINARY:
                        constraint.add_variable(Vartype.BINARY, var)
                    elif vartype is Vartype.INTEGER:
                        constraint.add_variable(Vartype.INTEGER, var, lower_bound=lb, upper_bound=ub)
                    else:
                        raise ValueError("Unexpected vartype: {}".format(vartype))
                    constraint.add_linear(var, le.coef)

            if c.quad_expr:

                for qe in c.quad_expr:
                    if qe.name != "":
                        var1 = qe.name[0]
                        vartype1 = obj.vartype(var1)
                        lb1 = obj.lower_bound(var1)
                        ub1 = obj.upper_bound(var1)

                        if vartype1 is Vartype.BINARY:
                            constraint.add_variable(vartype1, var1)
                        elif vartype1 is Vartype.INTEGER:
                            constraint.add_variable(vartype1, var1, lower_bound=lb1, upper_bound=ub1)
                        else:
                            raise ValueError("Unexpected vartype: {}".format(vartype1))

                        var2 = qe.second_var_name[0]
                        vartype2 = obj.vartype(var2)
                        lb2 = obj.lower_bound(var2)
                        ub2 = obj.upper_bound(var2)

                        if vartype2 is Vartype.BINARY:
                            constraint.add_variable(vartype2, var2)
                        elif vartype2 is Vartype.INTEGER:
                            constraint.add_variable(vartype2, var2, lower_bound=lb2, upper_bound=ub2)

                        constraint.add_quadratic(var1, var2, qe.coef)

                    else:
                        # this is a pure squared term
                        var = qe.squared_name[0]
                        vartype = obj.vartype(var)
                        lb = obj.lower_bound(var)
                        ub = obj.upper_bound(var)
                        if vartype is Vartype.BINARY:
                            constraint.add_variable(vartype, var)
                            constraint.add_linear(var, qe.coef)
                        elif vartype is Vartype.INTEGER:
                            constraint.add_variable(vartype, var, lower_bound=lb, upper_bound=ub)
                            constraint.add_quadratic(var, var, qe.coef)
                        else:
                            raise TypeError("Unexpected variable type: {}".format(vartype))

            # finally mode the RHS to the LHS with a minus sign
            constraint.offset = - ccoef

            cqm.add_constraint(constraint, label=cname, sense=csense)

        return cqm


CQM = ConstrainedQuadraticModel


class _Vartypes(abc.Sequence):
    """Support deprecated attribute on ``CQM.variables``"""
    def __init__(self, cqm: ConstrainedQuadraticModel):
        self.cqm: ConstrainedQuadraticModel = cqm

    def __getitem__(self, index: int) -> Vartype:
        warnings.warn(
            "cqm.variables.vartypes[i] is deprecated and will be removed in dimod 0.11.0, "
            "use cqm.vartype(cqm.variables[i]) instead.", DeprecationWarning, stacklevel=3)
        return self.cqm.vartype(self.cqm.variables[index])

    def __len__(self) -> int:
        warnings.warn(
            "cqm.variables.vartypes is deprecated and will be removed in dimod 0.11.0",
            DeprecationWarning, stacklevel=3)
        return len(self.cqm.variables)


class _LowerBounds(abc.Mapping):
    """Support deprecated attribute on ``CQM.variables``"""
    def __init__(self, cqm: ConstrainedQuadraticModel):
        self.cqm: ConstrainedQuadraticModel = cqm

    def __getitem__(self, key: Variable) -> float:
        warnings.warn(
            "cqm.variables.lower_bounds[v] is deprecated and will be removed in dimod 0.11.0, "
            "use cqm.lower_bound(v) instead.", DeprecationWarning, stacklevel=3)
        return self.cqm.lower_bound(key)

    def __iter__(self) -> Iterator[Variable]:
        warnings.warn(
            "cqm.variables.lower_bounds is deprecated and will be removed in dimod 0.11.0",
            DeprecationWarning, stacklevel=3)
        yield from self.cqm.variables

    def __len__(self) -> int:
        warnings.warn(
            "cqm.variables.lower_bounds is deprecated and will be removed in dimod 0.11.0",
            DeprecationWarning, stacklevel=3)
        return len(self.cqm.variables)


class _UpperBounds(abc.Mapping):
    """Support deprecated attribute on ``CQM.variables``"""
    def __init__(self, cqm: ConstrainedQuadraticModel):
        self.cqm: ConstrainedQuadraticModel = cqm

    def __getitem__(self, key: Variable) -> float:
        warnings.warn(
            "cqm.variables.upper_bounds[v] is deprecated and will be removed in dimod 0.11.0, "
            "use cqm.upper_bound(v) instead.", DeprecationWarning, stacklevel=3)
        return self.cqm.upper_bound(key)

    def __iter__(self) -> Iterator[Variable]:
        warnings.warn(
            "cqm.variables.upper_bounds is deprecated and will be removed in dimod 0.11.0",
            DeprecationWarning, stacklevel=3)
        yield from self.cqm.variables

    def __len__(self) -> int:
        warnings.warn(
            "cqm.variables.upper_bounds is deprecated and will be removed in dimod 0.11.0",
            DeprecationWarning, stacklevel=3)
        return len(self.cqm.variables)


def _qm_to_bqm(qm: QuadraticModel, integers: MutableMapping[Variable, BinaryQuadraticModel],
               ) -> BinaryQuadraticModel:
    # dev note: probably we'll want to make this function or something similar
    # public facing at some point, but right now the interface is pretty weird
    # and it only returns BINARY bqms

    if any(qm.vartype(v) is Vartype.SPIN for v in qm.variables):
        # bqm is BINARY so we want to handle these
        qm = qm.spin_to_binary(inplace=False)

    bqm = BinaryQuadraticModel(Vartype.BINARY)

    for v in qm.variables:
        if v in integers:
            bqm += qm.get_linear(v) * integers[v]
        else:
            bqm.add_linear(v, qm.get_linear(v))

    for u, v, bias in qm.iter_quadratic():
        if u in integers:
            if v in integers:
                bqm += integers[u] * integers[v] * bias
            else:
                bqm += Binary(v) * integers[u] * bias
        elif v in integers:
            bqm += Binary(u) * integers[v] * bias
        else:
            bqm.add_quadratic(u, v, bias)
    bqm.offset += qm.offset

    return bqm


class CQMToBQMInverter:
    """Invert a sample from a binary quadratic model constructed by :func:`cqm_to_bqm`."""
    __slots__ = ('_binary', '_integers')

    def __init__(self,
                 binary: Mapping[Variable, Vartype],
                 integers: Mapping[Variable, BinaryQuadraticModel]):
        self._binary = binary
        self._integers = integers

    def __call__(self, sample: Mapping[Variable, int]) -> Mapping[Variable, int]:
        new = {}

        for v, vartype in self._binary.items():
            if vartype is Vartype.BINARY:
                new[v] = sample[v]
            elif vartype is Vartype.SPIN:
                new[v] = 2*sample[v] - 1
            else:
                raise RuntimeError("unexpected vartype")

        for v, bqm in self._integers.items():
            new[v] = 0
            for u in bqm.variables:
                new[v] += sample[u] * u[1]

        return new

    @classmethod
    def from_dict(cls, doc: Dict[str, Dict[Variable, Any]]) -> 'CQMToBQMInverter':
        """Construct an inverter from a serialized representation."""

        integers = {}
        for v, variables in doc['integers'].items():
            v = deserialize_variable(v)

            bqm = BinaryQuadraticModel(Vartype.BINARY)
            bqm.add_linear_from((deserialize_variable(u), u[1]) for u in variables)

            integers[v] = bqm

        return cls(
            dict((deserialize_variable(v), as_vartype(vartype))
                 for v, vartype in doc['binary'].items()),
            integers,
            )

    def to_dict(self) -> Dict[str, Dict[Variable, Any]]:
        """Return a json-serializable encoding of the inverter."""
        # todo: in 3.8 we can used TypedDict for the typing
        return dict(
            binary=dict((serialize_variable(v), vartype.name)
                        for v, vartype in self._binary.items()),
            integers=dict((serialize_variable(v), bqm.variables.to_serializable())
                          for v, bqm in self._integers.items()),
            )


# Developer note: This function is *super* ad hoc. In the future, we may want
# A BQM.from_cqm method or similar, but for now I think it makes sense to
# expose that functionality as a function for easier later deprecation.
def cqm_to_bqm(cqm: ConstrainedQuadraticModel, lagrange_multiplier: Optional[Bias] = None,
               ) -> Tuple[BinaryQuadraticModel, CQMToBQMInverter]:
    """Construct a binary quadratic model from a constrained quadratic model.

    Args:
        cqm: A constrained quadratic model. All constraints must be linear
            and all integer variables must have a lower bound of 0.

        lagrange_multiplier: The penalty strength used when converting
            constraints into penalty models. Defaults to 10x the largest
            bias in the objective.

    Returns:
        A 2-tuple containing:

            A binary quadratic model

            A function that converts samples over the binary quadratic model
            back into samples for the constrained quadratic model.

    Example:

        Start with a constrained quadratic model

        >>> num_widget_a = dimod.Integer('num_widget_a', upper_bound=7)
        >>> num_widget_b = dimod.Integer('num_widget_b', upper_bound=3)
        >>> cqm = dimod.ConstrainedQuadraticModel()
        >>> cqm.set_objective(-3 * num_widget_a - 4 * num_widget_b)
        >>> cqm.add_constraint(num_widget_a + num_widget_b <= 5, label='total widgets')
        'total widgets'

        Convert it to a binary quadratic model and solve it using
        :class:`dimod.ExactSolver`.

        >>> bqm, invert = dimod.cqm_to_bqm(cqm)
        >>> sampleset = dimod.ExactSolver().sample(bqm)

        Interpret the answer in the original variable classes

        >>> invert(sampleset.first.sample)
        {'num_widget_a': 2, 'num_widget_b': 3}

        Note that the inverter is also serializable.

        >>> import json
        >>> newinvert = dimod.constrained.CQMToBQMInverter.from_dict(
        ...     json.loads(json.dumps(invert.to_dict())))
        >>> newinvert(sampleset.first.sample)
        {'num_widget_a': 2, 'num_widget_b': 3}

    """

    from dimod.generators.integer import binary_encoding  # avoid circular import

    bqm = BinaryQuadraticModel(Vartype.BINARY)
    binary: Dict[Variable, Vartype] = {}
    integers: Dict[Variable, BinaryQuadraticModel] = {}

    # add the variables
    for v in cqm.variables:
        vartype = cqm.vartype(v)

        if vartype is Vartype.SPIN or vartype is Vartype.BINARY:
            binary[v] = vartype
        elif vartype is Vartype.INTEGER:
            if cqm.lower_bound(v) != 0:
                raise ValueError("integer variables must have a lower bound of 0, "
                                 f"variable {v} has a lower bound of {cqm.lower_bound(v)}")
            v_bqm = integers[v] = binary_encoding(v, int(cqm.upper_bound(v)))

            if not v_bqm.variables.isdisjoint(bqm.variables):
                # this should be pretty unusual, so let's not bend over backwards
                # to accommodate it.
                raise ValueError("given CQM has conflicting variables with ones "
                                 "generated by dimod.generators.binary_encoding")

            bqm.add_variables_from((v, 0) for v in v_bqm.variables)
        else:
            raise RuntimeError("unexpected vartype")

    # objective, we know it's always a QM
    bqm += _qm_to_bqm(cqm.objective, integers)

    if lagrange_multiplier is None:
        if cqm.constraints and bqm.num_variables:
            max_bias = max(-bqm.linear.min(), bqm.linear.max())
            if not bqm.is_linear():
                max_bias = max(-bqm.quadratic.min(), bqm.quadratic.max(), max_bias)
            lagrange_multiplier = 10 * max_bias
        else:
            lagrange_multiplier = 0  # doesn't matter

    for constraint in cqm.constraints.values():
        lhs = constraint.lhs
        rhs = constraint.rhs
        sense = constraint.sense

        if isinstance(lhs, QuadraticModel):
            lhs = _qm_to_bqm(lhs, integers)

        if not lhs.is_linear():
            raise ValueError("CQM must not have any quadratic constraints")

        if lhs.vartype is Vartype.SPIN:
            lhs = lhs.change_vartype(Vartype.BINARY, inplace=True)

        # at this point we know we have a BINARY bqm

        if sense is Sense.Eq:
            bqm.add_linear_equality_constraint(
                ((v, lhs.get_linear(v)) for v in lhs.variables),
                lagrange_multiplier,
                lhs.offset - rhs,
                )
        elif sense is Sense.Ge:
            bqm.add_linear_inequality_constraint(
                ((v, lhs.get_linear(v)) for v in lhs.variables),
                lagrange_multiplier,
                new_label(),
                constant=lhs.offset,
                lb=rhs,
                ub=np.iinfo(np.int64).max,
                )
        elif sense is Sense.Le:
            bqm.add_linear_inequality_constraint(
                ((v, lhs.get_linear(v)) for v in lhs.variables),
                lagrange_multiplier,
                new_label(),
                constant=lhs.offset,
                lb=np.iinfo(np.int64).min,
                ub=rhs,
                )
        else:
            raise RuntimeError("unexpected sense")

    return bqm, CQMToBQMInverter(binary, integers)


# register fileview loader
load.register(CQM_MAGIC_PREFIX, ConstrainedQuadraticModel.from_file)
