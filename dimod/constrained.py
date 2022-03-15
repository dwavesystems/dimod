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

from __future__ import annotations

import collections.abc as abc
import copy
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
from dimod.exceptions import InfeasibileModelError
from dimod.quadratic import QuadraticModel
from dimod.sampleset import as_samples
from dimod.sym import Comparison, Eq, Le, Ge, Sense
from dimod.serialization.fileview import SpooledTemporaryFile, _BytesIO
from dimod.serialization.fileview import load, read_header, write_header
from dimod.typing import Bias, Variable, SamplesLike
from dimod.utilities import iter_safe_relabels, new_label
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
            & \sum_{i} a_i x_i + \sum_{i \le j} b_{ij} x_i x_j + c, \\
            \text{Subject to constraints:} & \\
            & \sum_i a_i^{(c)} x_i + \sum_{i \le j} b_{ij}^{(c)} x_i x_j+ c^{(c)} \le 0,
            \quad c=1, \dots, C_{\rm ineq.}, \\
            & \sum_i a_i^{(d)} x_i + \sum_{i \le j} b_{ij}^{(d)} x_i x_j + c^{(d)} = 0,
            \quad d=1, \dots, C_{\rm eq.},
        \end{align}

    where :math:`\{ x_i\}_{i=1, \dots, N}` can be binary\ [#]_ or integer
    variables, :math:`a_{i}, b_{ij}, c` are real values and
    :math:`C_{\rm ineq.}, C_{\rm eq,}` are the number of inequality and
    equality constraints respectively.

    .. [#]
        For binary variables, the range of the quadratic-term summation is
        :math:`i < j` because :math:`x^2 = x` for binary values :math:`\{0, 1\}`
        and :math:`s^2 = 1` for spin values :math:`\{-1, 1\}`.

    The objective and constraints are encoded as either :class:`.QuadraticModel`
    or :class:`.BinaryQuadraticModel` depending on the variable types used.

    Example:

        This example solves the simple `bin packing problem <https://w.wiki/3jz4>`_
        of packing a set of items of different weights into the smallest
        possible number of bins.

        `dimod` provides a general :func:`~dimod.generators.random_bin_packing`
        function to generate bin packing problems, and this example follows the
        same naming conventions.

        Consider four objects with weights between 0 and 1, and assume that each
        bin has a capacity to hold up to a total weight of 1.

        >>> weights = [.9, .7, .2, .1]
        >>> capacity = 1

        Variable :math:`y_j` indicates that bin :math:`j` is used. Clearly, no
        more than four bins are needed.

        >>> y = [dimod.Binary(f'y_{j}') for j in range(len(weights))]

        Variable :math:`x_{i,j}` indicates that item :math:`i` is put in bin
        :math:`j`.

        >>> x = [[dimod.Binary(f'x_{i}_{j}') for j in range(len(weights))]
        ...      for i in range(len(weights))]

        Create an empty constrained quadratic model ("empty" meaning that no
        objective or constraints have set).

        >>> cqm = dimod.ConstrainedQuadraticModel()

        The problem is to minimize the number of bins used. Therefore the objective
        is to minimize the value of :math:`\sum_j y_j`.

        >>> cqm.set_objective(sum(y))

        Any feasible solution must meet the constraint that each item can only go
        in one bin. You can express this constraint, for a given item :math:`i`,
        with :math:`\sum_j x_{i, j} == 1`. Note that the label of each
        constraint is returned so that you can access them in the future if
        desired.

        >>> for i in range(len(weights)):
        ...     cqm.add_constraint(sum(x[i]) == 1, label=f'item_placing_{i}')
        'item_placing_0'
        'item_placing_1'
        'item_placing_2'
        'item_placing_3'

        Finally, enforce the limits on each bin. You can express this constraint,
        for a given bin :math:`j`, with :math:`\sum_i x_{i, j} * w_i <= c` where
        :math:`w_i` is the weight of item :math:`i` and :math:`c` is the capacity.

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

        self._objective = QuadraticModel()

    @property
    def constraints(self) -> Dict[Hashable, Comparison]:
        """Constraints as a dictionary.

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
        """Objective to be minimized."""
        return self._objective

    @property
    def variables(self) -> Variables:
        """Variables in use over the objective and all constraints."""
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
            qm: Quadratic model or binary quadratic model.

            sense: One of `<=', '>=', '=='.

            rhs: Right hand side of the constraint.

            label: Label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

            copy: If `True`, model ``qm`` is copied. This can be set to `False`
                to improve performance, but subsequently mutating ``qm`` can
                cause issues.

        Returns:
            Label of the added constraint.

        Examples:
            This example adds a constraint from the single-variable binary
            quadratic model ``x``.

            >>> from dimod import ConstrainedQuadraticModel, Binary
            >>> cqm = ConstrainedQuadraticModel()
            >>> x = Binary('x')
            >>> cqm.add_constraint_from_model(x, '>=', 0, 'Min x')
            'Min x'
            >>> print(cqm.constraints["Min x"].to_polystring())
            x >= 0

            Adding a constraint without copying the model requires caution:

            >>> cqm.add_constraint_from_model(x, "<=", 3, "Risky constraint", copy=False)
            'Risky constraint'
            >>> x *= 2
            >>> print(cqm.constraints["Risky constraint"].to_polystring())
            2*x <= 3
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
            comp: Comparison object.

            label: Label for the constraint. Must be unique. If no label
                is provided, one is generated using :mod:`uuid`.

            copy: If `True`, the model used in the comparison is copied. You can
                set to `False` to improve performance, but subsequently mutating
                the model can cause issues.

        Returns:
            Label of the added constraint.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer
            >>> i = Integer('i')
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_constraint_from_comparison(i <= 3, label='Max i')
            'Max i'
            >>> print(cqm.constraints["Max i"].to_polystring())
            i <= 3

            Adding a constraint without copying the comparison's model requires
            caution:

            >>> cqm.add_constraint_from_comparison(i >= 1, label="Risky constraint", copy=False)
            'Risky constraint'
            >>> i *= 2
            >>> print(cqm.constraints["Risky constraint"].to_polystring())
            2*i >= 1
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
            iterable: Iterable of terms as tuples. The variables must
                have already been added to the object.

            sense: One of `<=', '>=', '=='.

            rhs: The right hand side of the constraint.

            label: Label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

        Returns:
            Label of the added constraint.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer, Binary
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_variable('i', 'INTEGER')   # doctest: +IGNORE_RESULT
            >>> cqm.add_variable('j', 'INTEGER')   # doctest: +IGNORE_RESULT
            >>> cqm.add_variable('x', 'BINARY')    # doctest: +IGNORE_RESULT
            >>> cqm.add_variable('y', 'BINARY')    # doctest: +IGNORE_RESULT
            >>> label1 = cqm.add_constraint_from_iterable([('x', 'y', 1), ('i', 2), ('j', 3),
            ...                                           ('i', 'j', 1)], '<=', rhs=1)
            >>> print(cqm.constraints[label1].to_polystring())
            2*i + 3*j + x*y + i*j <= 1

        """
        qm = self._iterable_to_qm(iterable)

        # use quadratic model in the future
        return self.add_constraint_from_model(
            qm, sense, rhs=rhs, label=label, copy=False)

    def add_discrete(self, data, *args, **kwargs) -> Hashable:
        """A convenience wrapper for other methods that add one-hot constraints.

        One-hot constraints can represent discrete variables (for example a
        ``color`` variable that has values ``{"red", "blue", "green"}``) by
        requiring that only one of a set of two or more binary variables is
        assigned a value of 1.

        These constraints support only :class:`~dimod.Vartype.BINARY` variables
        and must be disjoint; that is, variables in one such constraint must not
        be used in others in the model.

        Constraints added by the methods wrapped by :meth:`add_discrete` are
        guaranteed to be satisfied in solutions returned
        by the :class:`~dwave.system.samplers.LeapHybridCQMSampler` hybrid sampler.

        See also:
            :meth:`~.ConstrainedQuadraticModel.add_discrete_from_model`

            :meth:`~.ConstrainedQuadraticModel.add_discrete_from_comparison`

            :meth:`~.ConstrainedQuadraticModel.add_discrete_from_iterable`

        Examples:

            >>> cqm = dimod.ConstrainedQuadraticModel()

            Add a discrete constraint over variables ``x, y, z`` from an
            iterable.

            >>> iterable = ['x', 'y', 'z']
            >>> for v in iterable:
            ...      cqm.add_variable(v, 'BINARY')
            'x'
            'y'
            'z'
            >>> cqm.add_discrete(iterable, label='discrete-xyz')
            'discrete-xyz'

            Add a discrete constraint over variables ``a, b, c`` from a
            model.

            >>> a, b, c = dimod.Binaries(['a', 'b', 'c'])
            >>> cqm.add_discrete(sum([a, b, c]), label='discrete-abc')
            'discrete-abc'

            Add a discrete constraint over variables ``d, e, f`` from a
            comparison.

            >>> d, e, f = dimod.Binaries(['d', 'e', 'f'])
            >>> cqm.add_discrete(d + e + f == 1, label='discrete-def')
            'discrete-def'

        """
        # in python 3.8+ we can use singledispatchmethod
        if isinstance(data, (BinaryQuadraticModel, QuadraticModel, BQMabc)):
            return self.add_discrete_from_model(data, *args, **kwargs)
        elif isinstance(data, Comparison):
            return self.add_discrete_from_comparison(data, *args, **kwargs)
        elif isinstance(data, Iterable):
            return self.add_discrete_from_iterable(data, *args, **kwargs)
        else:
            raise TypeError("unexpected data format")

    def add_discrete_from_comparison(self,
                                     comp: Comparison,
                                     label: Optional[Hashable] = None,
                                     copy: bool = True) -> Hashable:
        """Add a one-hot constraint from a comparison.

        One-hot constraints can represent discrete variables (for example a
        ``color`` variable that has values ``{"red", "blue", "green"}``) by
        requiring that only one of a set of two or more binary variables is
        assigned a value of 1.

        These constraints support only :class:`~dimod.Vartype.BINARY` variables
        and must be disjoint; that is, variables in such a constraint must not be
        used elsewhere in the model.

        Constraints added by this method are guaranteed to be satisfied in
        solutions returned by the :class:`~dwave.system.samplers.LeapHybridCQMSampler`
        hybrid sampler.

        Args:
            comp: Comparison object. The comparison must be a linear
                equality constraint with all of the linear biases on the
                left-hand side equal to one and the right hand side equal
                to one.

            label: Label for the constraint. Must be unique. If no label
                is provided, one is generated using :mod:`uuid`.

            copy: If `True`, the model used in the comparison is copied. You can
                set to `False` to improve performance, but subsequently mutating
                the model can cause issues.

        Returns:
            Label of the added constraint.

        Examples:

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> r, b, g = dimod.Binaries(["red", "blue", "green"])
            >>> cqm.add_discrete_from_comparison(r + b + g == 1, label="One color")
            'One color'

        """
        if comp.sense is not Sense.Eq:
            raise ValueError("discrete constraints must be equality constraints")
        if comp.rhs != 1:
            # could scale, but let's keep it simple for now
            raise ValueError("the right hand side of a discrete constraint must be 1")
        return self.add_discrete_from_model(comp.lhs, label=label, copy=copy)

    def add_discrete_from_iterable(self,
                                   variables: Iterable[Variable],
                                   label: Optional[Hashable] = None) -> Hashable:
        """Add a one-hot constraint from an iterable.

        One-hot constraints can represent discrete variables (for example a
        ``color`` variable that has values ``{"red", "blue", "green"}``) by
        requiring that only one of a set of two or more binary variables is
        assigned a value of 1.

        These constraints support only :class:`~dimod.Vartype.BINARY` variables
        and must be disjoint; that is, variables in such a constraint must not be
        used elsewhere in the model.

        Constraints added by this method are guaranteed to be satisfied in
        solutions returned by the :class:`~dwave.system.samplers.LeapHybridCQMSampler`
        hybrid sampler.

        Args:
            variables: An iterable of variables.

            label: Label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

        Returns:
            Label of the added constraint.

        Examples:

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> color = ["red", "blue", "green"]
            >>> for v in color:
            ...      cqm.add_variable(v, 'BINARY')
            'red'
            'blue'
            'green'
            >>> cqm.add_discrete(color, label='one-color')
            'one-color'
            >>> print(cqm.constraints['one-color'].to_polystring())
            red + blue + green == 1

        """
        if label is not None and label in self.constraints:
            raise ValueError("a constraint with that label already exists")

        bqm = BinaryQuadraticModel(Vartype.BINARY, dtype=np.float32)

        for v in variables:
            if v in self.variables:
                # it already exists, let's make sure it's not already used
                if any(v in self.constraints[label].lhs.variables for label in self.discrete):
                    raise ValueError(f"variable {v!r} is already used in a discrete variable")
                if self.vartype(v) is not Vartype.BINARY:
                    raise ValueError(f"variable {v!r} has already been added but is not BINARY")

            bqm.set_linear(v, 1)

        if bqm.num_variables < 2:
            raise ValueError("discrete constraints must have at least two variables")

        label = self.add_constraint_from_comparison(bqm == 1, label=label, copy=False)
        self.discrete.add(label)
        return label

    def add_discrete_from_model(self,
                                qm: Union[BinaryQuadraticModel, QuadraticModel],
                                label: Optional[Hashable] = None,
                                copy: bool = True) -> Hashable:
        """Add a one-hot constraint from a model.

        One-hot constraints can represent discrete variables (for example a
        ``color`` variable that has values ``{"red", "blue", "green"}``) by
        requiring that only one of a set of two or more binary variables is
        assigned a value of 1.

        These constraints support only :class:`~dimod.Vartype.BINARY` variables
        and must be disjoint; that is, variables in such a constraint must not be
        used elsewhere in the model.

        Constraints added by this method are guaranteed to be satisfied in
        solutions returned by the :class:`~dwave.system.samplers.LeapHybridCQMSampler`
        hybrid sampler.

        Args:
            qm: A quadratic model or binary quadratic model.
                The model must be linear with all of the linear biases on the
                left-hand side equal to one and the right hand side equal
                to one.

            label: A label for the constraint. Must be unique. If no label
                is provided, one is generated using :mod:`uuid`.

            copy: If `True`, the model is copied. You can set to `False` to
                improve performance, but subsequently mutating the model can
                cause issues.

        Returns:
            Label of the added constraint.

        Examples:

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> r, b, g = dimod.Binaries(["red", "blue", "green"])
            >>> cqm.add_discrete(sum([r, g, b]), label="One color")
            'One color'
            >>> print(cqm.constraints["One color"].to_polystring())
            red + green + blue == 1

        """
        vartype = qm.vartype if isinstance(qm, QuadraticModel) else lambda v: qm.vartype

        if qm.num_interactions:
            raise ValueError("discrete constraints must be linear")

        if qm.num_variables < 2:
            raise ValueError("discrete constraints must have at least two variables")

        for v, bias in qm.iter_linear():
            if v in self.variables:
                # it already exists, let's make sure it's not already used
                if any(v in self.constraints[label].lhs.variables for label in self.discrete):
                    raise ValueError(f"variable {v!r} is already used in a discrete variable")
                if self.vartype(v) is not Vartype.BINARY:
                    raise ValueError(f"variable {v!r} has already been added but is not BINARY")
            elif not vartype(v) is Vartype.BINARY:
                raise ValueError("all variables in a discrete constraint must be binary, "
                                 f"{v!r} is {vartype(v).name!r}")
            # we could maybe do a scaling, but let's just keep it simple for now
            if bias != 1:
                raise ValueError("all linear biases in a discrete constraint must be 1")

        label = self.add_constraint_from_model(qm, sense='==', rhs=1, label=label, copy=copy)
        self.discrete.add(label)
        return label

    def add_variable(self, v: Variable, vartype: VartypeLike,
                     *,
                     lower_bound: Optional[float] = None,
                     upper_bound: Optional[float] = None,
                     ) -> Variable:
        """Add a variable to the model.

        Args:
            variable: Variable label.

            vartype:
                Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``

            lower_bound:
                Lower bound on the variable. Ignored when the variable is
                :class:`~dimod.Vartype.BINARY` or :class:`~dimod.Vartype.SPIN`.

            upper_bound:
                Upper bound on the variable. Ignored when the variable is
                :class:`~dimod.Vartype.BINARY` or :class:`~dimod.Vartype.SPIN`.

        Returns:
            Variable label.

        Raises:
            ValueError: If ``v`` is already a variable in the model and
                the ``vartype``, ``lower_bound``, or ``upper_bound`` are
                inconsistent.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_variable('i', 'INTEGER')
            'i'

        """
        return self.objective.add_variable(
            vartype, v, lower_bound=lower_bound, upper_bound=upper_bound)

    def check_feasible(self, sample_like: SamplesLike, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
        r"""Return the feasibility of the given sample.

        A sample is feasible if all constraints are satisfied. A constraint's
        satisfaction is tested using the following equation:

        .. math::

            violation <= (atol + rtol * | rhs\_energy | )

        where ``violation`` and ``rhs_energy`` are as returned by :meth:`.iter_constraint_data`.

        Args:
            sample_like: A sample. `sample-like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            True if the sample is feasible (given the tolerances).

        Examples:
            This example violates a constraint that :math:`i \le 4` by `0.2`,
            which is greater than the absolute tolerance set by ``atol = 0.1``
            but within the relative tolerance of :math:`0.1 * 4 = 0.4`.

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> i = dimod.Integer("i")
            >>> cqm.add_constraint_from_comparison(i <= 4, label="Max i")
            'Max i'
            >>> cqm.check_feasible({"i": 4.2}, atol=0.1)
            False
            >>> next(cqm.iter_constraint_data({"i": 4.2})).rhs_energy
            4
            >>> cqm.check_feasible({"i": 4.2}, rtol=0.1)
            True

            Note that the :func:`next` function is used here because the model
            has just a single constraint.
        """
        return all(datum.violation <= atol + rtol*abs(datum.rhs_energy)
                   for datum in self.iter_constraint_data(sample_like))

    def fix_variable(self, v: Variable, value: float, *,
                     cascade: bool = False,
                     ) -> Dict[Variable, float]:
        """Fix the value of a variable in the model.

        Note that this function does not test feasibility.

        Args:
            v: Variable label for a variable in the model.

            value: Value to assign variable ``v``.

            cascade: If ``True``, additional variables may be removed from the
                model based on the assignment of ``v``.
                Currently handles the following cases:

                * Discrete constraints (see :meth:`add_discrete`)

                  Fixing one of the binary variables to `1` fixes the remaining
                  to `0`. Fixing all but one of the binary variables to `0`
                  fixes the remaining to `1`.

                * Equality constraint

                  Fixing one of two variables fixes the other. For example fixing
                  ``i`` to `3` in ``i + j == 7`` also fixes ``j`` to `4`

        Returns:
            Assignments of any additional variables fixed.
            For ``cascade==False``, this is always ``{}``.
            If you set ``cascade==True``, additional variables may be fixed.
            See above.

        Raises:
            ValueError: If ``v`` is not the label of a variable in the model.

        Examples:

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> r, b, g = dimod.Binaries(["red", "blue", "green"])
            >>> cqm.add_discrete_from_comparison(r + b + g == 1, label="One color")
            'One color'
            >>> cqm.fix_variable("red", 1, cascade=True)
            {'green': 0, 'blue': 0}

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> r, b, g = dimod.Binaries(["red", "blue", "green"])
            >>> cqm.add_discrete_from_comparison(r + b + g == 1, label="One color")
            'One color'
            >>> cqm.fix_variable("red", 0, cascade=True)
            {}
            >>> cqm.fix_variable("blue", 0, cascade=True)
            {'green': 1}

        """
        if v not in self.variables:
            raise ValueError(f"unknown variable {v!r}")

        new: Dict[Variable, float] = {}
        for label, comparison in self.constraints.items():
            if v in comparison.lhs.variables:
                comparison.lhs.fix_variable(v, value)

                if label in self.discrete:
                    # we've fixed a variable in a discrete constraint, so it
                    # might not be one any more.

                    if value not in Vartype.BINARY.value:
                        raise ValueError(
                            f"variable {v} is part of a discrete constraint so "
                            "it can only be fixed to 0 or 1")

                    if value == 1:  # we've found the one-hot!
                        self.discrete.remove(label)

                        if cascade:
                            # everything else gets set to 0
                            new.update((v, 0) for v in comparison.lhs.variables)

                    elif len(comparison.lhs.variables) < 2:  # we've made it too small
                        self.discrete.remove(label)

                        if cascade and len(comparison.lhs.variables) == 1:
                            new[comparison.lhs.variables[0]] = 1

                elif cascade and comparison.sense is Sense.Eq and len(comparison.lhs.variables) == 1:
                    # we have a constraint like i == 7, so can just go ahead and set it
                    new[comparison.lhs.variables[0]] = comparison.rhs - comparison.lhs.offset

        self.objective.fix_variable(v, value)

        for assignment in new.items():
            self.fix_variable(*assignment)

        return new

    def fix_variables(self,
                      fixed: Union[Mapping[Variable, float], Iterable[Tuple[Variable, float]]],
                      *,
                      cascade: bool = False,
                      ) -> Dict[Variable, float]:
        """Fix the value of the variables and remove them.

        Args:
            fixed: Dictionary or iterable of 2-tuples of variable assignments.
            cascade: See :meth:`.fix_variable`.

        Returns:
            Assignments of any additional variables fixed.
            For ``cascade==False``, this is always ``{}``.
            If you set ``cascade==True``, additional variables may be fixed.
            See :meth:`.fix_variable`.

        Raises:
            ValueError: If given a label for a variable not in the model.

            :exc:`~dimod.exceptions.InfeasibileModelError`: If fixing the
                given variables results in an infeasible model. Raising this
                exception is currently supported for only some simple cases;
                variable fixes may create an infeasible model without raising
                this error.

        Examples:
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> r, b, g = dimod.Binaries(["red", "blue", "green"])
            >>> cqm.add_discrete_from_comparison(r + b + g == 1, label="One color")
            'One color'
            >>> cqm.fix_variables({"red": 0, "green": 0}, cascade=True)
            {'blue': 1}

        """
        if isinstance(fixed, Mapping):
            fixed = fixed.items()

        fix_variable = self.fix_variable

        new: Dict[Variable, float] = {}
        for v, val in fixed:
            if v in new:
                if new[v] != val:
                    raise InfeasibileModelError()
            else:
                new.update(fix_variable(v, val, cascade=cascade))

        return new

    def flip_variable(self, v: Variable):
        """Flip the specified binary variable in the objective and constraints.

        Note that this may terminate a constraint's status as a discrete constraint
        (see :meth:`add_discrete`). Subsequently flipping the variable again does
        not restore that status.

        Args:
            v: Variable label of a :class:`~dimod.Vartype.BINARY` or
                :class:`~dimod.Vartype.SPIN` variable.

        Raises:
            ValueError: If given a non-binary variable to flip.

        Examples:
            This example flips :math:`x`` in an objective, :math:`2xy-2x`, which
            is equivalent for binary variables with values :math:`\{0, 1\}` to
            the substitution :math:`x \Rightarrow 1-x`, creating a new objective,
            :math:`2xy-2x \Rightarrow 2(1-x)y -2(1-x) = 2y -2xy -2 -2x`.

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> x, y = dimod.Binaries(["x", "y"])
            >>> cqm.set_objective(2 * x * y - 2 * x)
            >>> cqm.flip_variable("x")
            >>> print(cqm.objective.to_polystring())
            -2 + 2*x + 2*y - 2*x*y

            The next example flips a variable in a one-hot constraint. Subsequently
            fixing one of this discrete constaint's binary variables to `1`
            does not ensure the remaining variables are assigned `0`.

            >>> r, b, g = dimod.Binaries(["red", "blue", "green"])
            >>> cqm.add_discrete_from_comparison(r + b + g == 1, label="One color")
            'One color'
            >>> cqm.flip_variable("red")
            >>> cqm.fix_variable("blue", 1, cascade=True)
            {}

        """
        self.objective.flip_variable(v)  # checks that it exists and is the correct vartype

        for label, comparison in self.constraints.items():
            lhs = comparison.lhs
            if v in lhs.variables:
                comparison.lhs.flip_variable(v)

                self.discrete.discard(label)  # no longer a discrete variable

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
            dqm: Discrete quadratic model.

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
        constrained quadratic model (CQM). You can then add constraints that
        feasible solutions should meet.

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

        Args:
            fp: File pointer to a readable, seekable file-like object.

        The inverse of :meth:`~ConstrainedQuadraticModel.to_file`.

        Examples:
            >>> cqm1 = dimod.ConstrainedQuadraticModel()
            >>> x, y = dimod.Binaries(["x", "y"])
            >>> cqm1.set_objective(2 * x * y - 2 * x)
            >>> cqm_file = cqm1.to_file()
            >>> cqm2 = dimod.ConstrainedQuadraticModel.from_file(cqm_file)
            >>> print(cqm2.objective.to_polystring())
            -2*x + 2*x*y

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

    def iter_constraint_data(self, sample_like: SamplesLike) -> Iterator[ConstraintData]:
        """Yield information about the constraints for the given sample.

        Note that this method iterates over constraints in the same order as
        they appear in :attr:`.constraints`.

        Args:
            sample_like: A sample. `sample-like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

        Yields:
            A :class:`collections.namedtuple` with the following fields.

            * ``label``: Constraint label.
            * ``lhs_energy``:  Energy of the left hand side of the constraint.
            * ``rhs_energy``: Energy of the right hand side of the constraint.
            * ``sense``: :class:`dimod.sym.Sense` of the constraint.
            * ``activity``: Equals ``lhs_energy - rhs_energy``.
            * ``violation``: Ammount by which the constraint is violated, if
              positive, or satisfied, if negative. Determined by the type of
              constraint.

        Examples:
            The sample in this example sets a value of ``2`` for two constraints,
            :math:`i \le 3` and :math:`j \ge 3`, which satisfies the first and
            violates the second by the same ammount, flipping the sign of the
            ``violation`` field.

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> i, j = dimod.Integers(["i", "j"])
            >>> cqm.add_constraint_from_comparison(i <= 3, label="Upper limit")
            'Upper limit'
            >>> cqm.add_constraint_from_comparison(j >= 3, label="Lower limit")
            'Lower limit'
            >>> for constraint in cqm.iter_constraint_data({"i": 2, "j": 2}):
            ...     print(constraint.label, constraint.violation)
            Upper limit -1.0
            Lower limit 1.0
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

    def iter_violations(self, sample_like: SamplesLike, *,
                        skip_satisfied: bool = False,
                        clip: bool = False,
                        ) -> Iterator[Tuple[Hashable, Bias]]:
        """Yield violations for all constraints.

        Args:
            sample_like: A sample. `sample-like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`..
            skip_satisfied: If True, does not yield constraints that are satisfied.
            clip: If True, negative violations are rounded up to 0.

        Yields:
            A 2-tuple containing the constraint label and the amount of the
            constraint's violation.

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
        """Test for near equality to a given constrained quadratic model.

        All biases of the objective and constraints are compared.

        Args:
            other:
                Constrained quadratic model with which to compare biases.
            places:
                Number of decimal places to which the Python :func:`round`
                function calculates approximate equality.

        Examples:
            >>> cqm1 = dimod.ConstrainedQuadraticModel.from_quadratic_model(
            ...   dimod.BinaryQuadraticModel({0: 0.1234}, {(0, 1): -1.1234}, 0, "BINARY"))
            >>> cqm2 = dimod.ConstrainedQuadraticModel.from_quadratic_model(
            ...   dimod.BinaryQuadraticModel({0: 0.1232}, {(0, 1): -1.1229}, 0, "BINARY"))
            >>> cqm1.is_almost_equal(cqm2, 4)
            False
            >>> cqm1.is_almost_equal(cqm2, 3)
            True
        """
        def constraint_eq(c0: Comparison, c1: Comparison) -> bool:
            return (c0.sense is c1.sense
                    and c0.lhs.is_almost_equal(c1.lhs, places=places)
                    and not round(c0.rhs - c1.rhs, places))

        return (self.objective.is_almost_equal(other.objective, places=places)
                and self.constraints.keys() == other.constraints.keys()
                and all(constraint_eq(constraint, other.constraints[label])
                        for label, constraint in self.constraints.items()))

    def is_equal(self, other: 'ConstrainedQuadraticModel') -> bool:
        """Test for equality to a given constrained quadratic model.

        All biases of the objective and constraints are compared.

        Args:
            other:
                Constrained quadratic model with which to compare biases.
        """
        def constraint_eq(c0: Comparison, c1: Comparison) -> bool:
            return (c0.sense is c1.sense
                    and c0.lhs.is_equal(c1.lhs)
                    and c0.rhs == c1.rhs)

        return (self.objective.is_equal(other.objective)
                and self.constraints.keys() == other.constraints.keys()
                and all(constraint_eq(constraint, other.constraints[label])
                        for label, constraint in self.constraints.items()))

    def lower_bound(self, v: Variable) -> Bias:
        """Return the lower bound on the specified variable.

        Args:
            v: Variable label for a variable in the model.

        Examples:
            >>> i = dimod.Integer("i", lower_bound=3)
            >>> j = dimod.Integer("j", upper_bound=3)
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_constraint_from_comparison(i + j >= 4, label="Lower limit")
            'Lower limit'
            >>> cqm.lower_bound("i")
            3.0

        """
        return self.objective.lower_bound(v)

    def num_biases(self) -> int:
        """Number of biases across the objective and constraints.

        Examples:
            This example counts the three linear biases (including a linear bias
            implicitly set to zero for variable ``y``) and two quadratic baises.

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> i = dimod.Integer("i")
            >>> x, y = dimod.Binaries(["x", "y"])
            >>> cqm.set_objective(2*x + 3*i - x*y + i*x)
            >>> cqm.num_biases()
            5
        """
        num_biases = len(self.objective.linear) + len(self.objective.quadratic)
        num_biases += sum(len(const.lhs.linear) + len(const.lhs.quadratic)
                          for const in self.constraints.values())
        return num_biases

    def num_quadratic_variables(self) -> int:
        """Return the total number of variables with at least one quadratic
        interaction across all constraints.

        Examples:
            This example counts the two variables participating in interaction
            ``3*i*k`` in constraint ``Constraint1`` but not variable ``j``'s
            interaction ``2*i*j`` in the objective.

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> i, j, k = dimod.Integers(["i", "j", "k"])
            >>> cqm.set_objective(i - 2*i*j)
            >>> cqm.add_constraint_from_comparison(3*i*k - 2*j <= 4, label="Constraint1")
            'Constraint1'
            >>> cqm.num_quadratic_variables()
            2
        """
        count = 0
        for const in self.constraints.values():
            lhs = const.lhs
            count += sum(lhs.degree(v) > 0 for v in lhs.variables)
        return count

    def relabel_constraints(self, mapping: Mapping[Hashable, Hashable]):
        """Relabel the constraints.

        Note that this method does not maintain the constraint order.

        Args:
            mapping: Mapping from the old constraint labels to the new.

        Examples:
            >>> x, y, z = dimod.Binaries(['x', 'y', 'z'])
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_constraint(x + y == 1, label='c0')
            'c0'
            >>> cqm.add_constraint(y*z == 0, label='c1')
            'c1'
            >>> cqm.relabel_constraints({'c1': 'c2'})
            >>> list(cqm.constraints)
            ['c0', 'c2']

        """
        for submap in iter_safe_relabels(mapping, self.constraints):
            for old, new in submap.items():
                try:
                    self.constraints[new] = self.constraints[old]
                except KeyError:
                    continue  # do nothing
                del self.constraints[old]

                if old in self.discrete:
                    self.discrete.add(new)
                    self.discrete.remove(old)

    def relabel_variables(self,
                          mapping: Mapping[Variable, Variable],
                          inplace: bool = True,
                          ) -> 'ConstrainedQuadraticModel':
        """Relabel the variables of the objective and constraints.

        Args:
            mapping: Mapping from the old variable labels to the new.
            inplace: Relabels the model's variables in-place if True, or returns
                a copy of the model with relabeled variables if False.

        Returns:
            A constrained quadratic model. Itself by default, or a copy if
            ``inplace`` is set to False.

        """
        if not inplace:
            return copy.deepcopy(self).relabel_variables(mapping, inplace=True)

        self.objective.relabel_variables(mapping, inplace=True)
        for constraint in self.constraints.values():
            constraint.lhs.relabel_variables(mapping, inplace=True)

        return self

    def remove_constraint(self, label: Hashable, *, cascade: bool = False):
        """Remove a constraint from the model.

        Args:
            label: Label of the constraint to remove.
            cascade: If set to True, also removes any variables found only in the
                removed constraint that contribute no energy to the objective.

        Examples:
            This example also removes variable ``k`` from the model when removing
            the constraint while keeping the variables used in the objective.

            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> i, j, k = dimod.Integers(["i", "j", "k"])
            >>> cqm.set_objective(2 * i - 3 * i * j)
            >>> cqm.add_constraint_from_comparison(i * k - 2 * j <= 4, label="C1")
            'C1'
            >>> cqm.remove_constraint("C1", cascade=True)
            >>> cqm.variables
            Variables(['i', 'j'])
        """
        try:
            comparison = self.constraints.pop(label)
        except KeyError:
            raise ValueError(f"{label!r} is not a constraint") from None
        self.discrete.discard(label)  # if it's discrete

        if cascade:
            for v in comparison.lhs.variables:
                if self.objective.degree(v) or self.objective.get_linear(v):
                    # it's used somewhere in the objective
                    continue

                if any(v in comp.lhs.variables for comp in self.constraints.values()):
                    # it's used in at least one constraint
                    continue

                self.objective.remove_variable(v)

    def set_lower_bound(self, v: Variable, lb: float):
        """Set the lower bound for a variable.

        Args:
            v: Variable label of a variable in the constrained quadratic model.
            lb: Lower bound to set for variable ``v``.

        Raises:
            ValueError: If ``v`` is a :class:`~dimod.Vartype.SPIN`
                or :class:`~dimod.Vartype.BINARY` variable.

        Examples:
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_variable("j", "INTEGER", upper_bound=5)
            'j'
            >>> cqm.set_lower_bound("j", 2)
        """
        self.objective.set_lower_bound(v, lb)
        for comp in self.constraints.values():
            qm = comp.lhs
            if v in qm.variables:
                qm.set_lower_bound(v, lb)

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

    def set_upper_bound(self, v: Variable, ub: float):
        """Set the upper bound for a variable.

        Args:
            v: Variable label of a variable in the constrained quadratic model.
            ub: Upper bound to set for variable ``v``.

        Raises:
            ValueError: If ``v`` is a :class:`~dimod.Vartype.SPIN`
                or :class:`~dimod.Vartype.BINARY` variable.

        Examples:
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_variable("j", "INTEGER", lower_bound=2)
            'j'
            >>> cqm.set_upper_bound("j", 5)
        """
        self.objective.set_upper_bound(v, ub)
        for comp in self.constraints.values():
            qm = comp.lhs
            if v in qm.variables:
                qm.set_upper_bound(v, ub)

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
        """Replace any self-loops in the objective or constraints.

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

        Examples:
            >>> cqm1 = dimod.ConstrainedQuadraticModel()
            >>> x, y = dimod.Binaries(["x", "y"])
            >>> cqm1.set_objective(2 * x * y - 2 * x)
            >>> cqm_file = cqm1.to_file()
            >>> cqm2 = dimod.ConstrainedQuadraticModel.from_file(cqm_file)
            >>> print(cqm2.objective.to_polystring())
            -2*x + 2*x*y
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
        """Return the upper bound on the specified variable.

        Args:
            v: Variable label for a variable in the model.

        Examples:
            >>> i = dimod.Integer("i", upper_bound=3)
            >>> j = dimod.Integer("j", upper_bound=3)
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_constraint_from_comparison(i + j >= 1, label="Upper limit")
            'Upper limit'
            >>> cqm.set_upper_bound("i", 5)
            >>> cqm.upper_bound("i")
            5.0
            >>> cqm.upper_bound("j")
            3.0

        """
        return self.objective.upper_bound(v)

    def vartype(self, v: Variable) -> Vartype:
        """Vartype of the given variable.

        Args:
            v: Variable label for a variable in the model.

        """
        return self.objective.vartype(v)

    def violations(self, sample_like: SamplesLike, *,
                   skip_satisfied: bool = False,
                   clip: bool = False,) -> Dict[Hashable, Bias]:
        """Return a dict of violations for all constraints.

        The dictionary maps constraint labels to the amount each constraint is
        violated. This method is a shortcut for ``dict(cqm.iter_violations(sample))``.

        Args:
            sample_like: A sample. `sample-like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`..
            skip_satisfied: If True, does not yield constraints that are satisfied.
            clip: If True, negative violations are rounded up to 0.

        Returns:
            A dict of 2-tuples containing the constraint label and the amount of
            the constraint's violation.

        Examples:

            This example meets one constraint exactly, is well within the
            requirement of a second, and violates a third.

            >>> i, j, k = dimod.Binaries(['i', 'j', 'k'])
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_constraint(i + j + k == 10, label='equal')
            'equal'
            >>> cqm.add_constraint(i + j <= 15, label='less equal')
            'less equal'
            >>> cqm.add_constraint(j - k >= 0, label='greater equal')
            'greater equal'
            >>> sample = {"i": 3, "j": 2, "k": 5}
            >>> cqm.violations(sample)
            {'equal': 0.0, 'less equal': -10.0, 'greater equal': 3.0}
        """
        return dict(self.iter_violations(sample_like, skip_satisfied=skip_satisfied, clip=clip))

    @classmethod
    def from_lp_file(cls,
                     fp: Union[BinaryIO, ByteString],
                     lower_bound_default: Optional[int] = None,
                     upper_bound_default: Optional[int] = None) -> "ConstrainedQuadraticModel":
        """Create a constrained quadratic model from an LP file.

        Args:
            fp: file-like object in LP (linear program) format.
            lower_bound_default: Default lower bound for integer variables.
            upper_bound_default: Default upper bound for integer variables.

        Returns:
            :class:`ConstrainedQuadraticModel` representing the model encoded in
            the LP file.
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
