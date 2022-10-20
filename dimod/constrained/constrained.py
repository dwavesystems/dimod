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

import collections.abc
import copy
import io
import json
import os.path
import re
import tempfile
import typing
import uuid
import warnings
import zipfile

from numbers import Number

import numpy as np

import dimod.typing

from dimod.binary.binary_quadratic_model import BinaryQuadraticModel, Binary
from dimod.constrained.cyconstrained import cyConstrainedQuadraticModel, ConstraintView, ObjectiveView
from dimod.quadratic.quadratic_model import QuadraticModel
from dimod.sampleset import as_samples
from dimod.serialization.fileview import SpooledTemporaryFile, _BytesIO
from dimod.serialization.fileview import load, read_header, write_header
from dimod.sym import Comparison, Sense
from dimod.utilities import new_variable_label
from dimod.variables import serialize_variable, deserialize_variable, Variables
from dimod.vartypes import as_vartype, Vartype

__all__ = ['ConstrainedQuadraticModel', 'CQM', 'cqm_to_bqm']


CQM_MAGIC_PREFIX = b'DIMODCQM'


class ConstraintData(typing.NamedTuple):
    label: typing.Hashable
    lhs_energy: float
    rhs_energy: float
    sense: Sense
    activity: float
    violation: float


class DiscreteView(collections.abc.MutableSet):
    def __init__(self, parent):
        self.parent = parent

    def __contains__(self, key):
        # warnings.warn("ConstrainedQuadraticModel.discrete attribute is deprecated",
        #               DeprecationWarning, stacklevel=3)
        return key in self.parent.constraint_labels and self.parent.constraints[key].lhs.is_discrete()

    def __iter__(self):
        # warnings.warn("ConstrainedQuadraticModel.discrete attribute is deprecated",
        #               DeprecationWarning, stacklevel=3)
        for lbl, comp in self.parent.constraints.items():
            if comp.lhs.is_discrete():
                yield lbl

    def __len__(self):
        # warnings.warn("ConstrainedQuadraticModel.discrete attribute is deprecated",
        #               DeprecationWarning, stacklevel=3)
        return sum(1 for _ in self)

    def add(self, key):
        # warnings.warn("ConstrainedQuadraticModel.discrete attribute is deprecated",
        #               DeprecationWarning, stacklevel=3)
        self.parent.constraints[key].lhs.mark_discrete(True)

    def difference(self, *others):
        # warnings.warn("ConstrainedQuadraticModel.discrete attribute is deprecated",
        #               DeprecationWarning, stacklevel=3)
        return set(self).difference(*others)

    def discard(self, key):
        # warnings.warn("ConstrainedQuadraticModel.discrete attribute is deprecated",
        #               DeprecationWarning, stacklevel=3)
        if key in self.parent.constraint_labels:
            self.parent.constraints[key].lhs.mark_discrete(False)


class SoftConstraint(typing.NamedTuple):
    weight: float
    penalty: str


# todo: remove
class SoftView(collections.abc.Mapping):
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        constraint = self.parent.constraints[key].lhs

        if not constraint.is_soft():
            raise KeyError

        return SoftConstraint(constraint.weight(), constraint.penalty())

    def __iter__(self):
        for label, comp in self.parent.constraints.items():
            if comp.lhs.is_soft():
                yield label

    def __len__(self):
        return sum(1 for _ in self)


class ConstrainedQuadraticModel(cyConstrainedQuadraticModel):
    r"""A constrained quadratic model.

    Constrained quadratic models are problems of the form:

    .. math::

        \begin{align}
            \text{Minimize an objective:} & \\
            & \sum_{i} a_i x_i + \sum_{i \le j} b_{ij} x_i x_j + c, \\
            \text{Subject to constraints:} & \\
            & \sum_i a_i^{(m)} x_i + \sum_{i \le j} b_{ij}^{(m)} x_i x_j+ c^{(m)} \circ 0,
            \quad m=1, \dots, M,
        \end{align}

    where :math:`\{ x_i\}_{i=1, \dots, N}` can be binary\ [#]_ or integer
    variables, :math:`a_{i}, b_{ij}, c` are real values,
    :math:`\circ \in \{ \ge, \le, = \}` and  :math:`M` is the total number of constraints.

    .. [#]
        For binary variables, the range of the quadratic-term summation is
        :math:`i < j` because :math:`x^2 = x` for binary values :math:`\{0, 1\}`
        and :math:`s^2 = 1` for spin values :math:`\{-1, 1\}`.

    Constraints are often categorized as either "hard" or "soft". Any hard constraint
    must be satisfied for a solution of the model to qualify as feasible. Soft
    constraints may be violated to achieve an overall good solution. By setting
    appropriate weights to soft constraints in comparison to the objective
    and to other soft constraints, you can express the relative importance of such
    constraints.

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
        super().__init__()

        self.discrete: typing.Set[typing.Hashable] = DiscreteView(self)
        self._soft = SoftView(self)  # todo: remove

    def add_constraint(self, data, *args, **kwargs) -> typing.Hashable:
        """Add a constraint to the model.

        This method dispatches to one of several specific methods based on
        the type of the first argument.
        For a detailed description of the accepted argument types, see
        :meth:`~.ConstrainedQuadraticModel.add_constraint_from_model`,
        :meth:`~.ConstrainedQuadraticModel.add_constraint_from_comparison`,
        and :meth:`~.ConstrainedQuadraticModel.add_constraint_from_iterable`.

        """
        # in python 3.8+ we can use singledispatchmethod
        if isinstance(data, (BinaryQuadraticModel, QuadraticModel)):
            return self.add_constraint_from_model(data, *args, **kwargs)
        elif isinstance(data, Comparison):
            return self.add_constraint_from_comparison(data, *args, **kwargs)
        elif isinstance(data, typing.Iterable):
            return self.add_constraint_from_iterable(data, *args, **kwargs)
        else:
            raise TypeError("unexpected data format")

    def add_constraint_from_comparison(self,
                                       comp: Comparison,
                                       label: typing.Optional[typing.Hashable] = None,
                                       copy: bool = True,
                                       weight: typing.Optional[float] = None,
                                       penalty: str = 'linear',
                                       ) -> typing.Hashable:
        r"""Add a constraint from a symbolic comparison.

        For a more detailed discussion of symbolic model manipulation, see
        :ref:`intro_symbolic_math`.

        Args:
            comp: Comparison object, generally constructed using symbolic math.
                The right-hand side of any symbolic equation must be an integer
                or float.

            label: Label for the constraint. Must be unique. If no label
                is provided, one is generated using :mod:`uuid`.

            copy: If `True`, the model used in the comparison is copied. You can
                set to `False` to improve performance, but subsequently mutating
                the model can cause issues.

            weight: weight for soft-constraint. If None, the constraints
                is interpreted as hard.

            penalty: Penalty type for soft-constraint. Must be one of
                'linear', 'quadratic'. Ignored if weight is None. 'quadratic'
                only accepted if the constraint has binary variables.

        Returns:
            Label of the added constraint.

        Example:

            Encode a constraint.

            .. math::

                x + y + xy <= 1

            First create the relevant variables and the model.

            >>> x, y = dimod.Binaries(['x', 'y'])
            >>> cqm = dimod.ConstrainedQuadraticModel()

            And add the constraint to the model.

            >>> cqm.add_constraint(x + y + x*y <= 1, label='c0')
            'c0'
            >>> cqm.constraints['c0'].to_polystring()
            'x + y + x*y <= 1'

        Example:

            Encode a constraint with a symbolic right-hand side.


            .. math::

                x + y \le x y


            First create the relevant variables and the model.

            >>> x, y = dimod.Binaries(['x', 'y'])
            >>> cqm = dimod.ConstrainedQuadraticModel()

            Trying to directly compare the left-hand and right-hand side
            of the equation will raise an error, because the right-hand side
            is not an integer or float.

            >>> try:
            ...     cqm.add_constraint(x + y <= x * y)
            ... except TypeError:
            ...     print("Not allowed!")
            Not allowed!

            To avoid this, simply subtract the right-hand side from both sides.

            .. math::

                x + y - xy \le 0

            The constraint can then be added.

            >>> cqm.add_constraint(x + y - x*y <= 0, label="c0")
            'c0'

        """
        if not isinstance(comp.rhs, Number):
            raise TypeError("comparison should have a numeric rhs")

        if isinstance(comp.lhs, (BinaryQuadraticModel, QuadraticModel)):
            return self.add_constraint_from_model(comp.lhs, comp.sense, rhs=comp.rhs,
                                                  label=label, copy=copy, weight=weight,
                                                  penalty=penalty)
        else:
            raise ValueError("comparison should have a binary quadratic model "
                             "or quadratic model lhs.")

    def add_constraint_from_iterable(self, iterable: typing.Iterable,
                                     sense: typing.Union[Sense, str],
                                     rhs: dimod.typing.Bias = 0,
                                     label: typing.Optional[typing.Hashable] = None,
                                     weight: typing.Optional[float] = None,
                                     penalty: str = 'linear',
                                     ) -> typing.Hashable:
        """Add a constraint from an iterable of tuples.

        Args:
            iterable: Iterable of terms as tuples. The variables must
                have already been added to the object.

            sense: One of `<=`, `>=`, `==`.

            rhs: The right-hand side of the constraint.

            label: Label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

            weight: weight for soft-constraint. If None, the constraints
                is interpreted as hard.

            penalty: Penalty type for soft-constraint. Must be one of
                'linear', 'quadratic'. Ignored if weight is None. 'quadratic'
                only accepted if the constraint has binary variables.

        Returns:
            Label of the added constraint.

        Examples:
            >>> from dimod import ConstrainedQuadraticModel, Integer, Binary
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_variable('INTEGER', 'i')
            'i'
            >>> cqm.add_variable('INTEGER', 'j')
            'j'
            >>> cqm.add_variable('BINARY', 'x')
            'x'
            >>> cqm.add_variable('BINARY', 'y')
            'y'
            >>> label1 = cqm.add_constraint_from_iterable([('x', 'y', 1), ('i', 2), ('j', 3),
            ...                                           ('i', 'j', 1)], '<=', rhs=1)
            >>> print(cqm.constraints[label1].to_polystring())
            2*i + 3*j + x*y + i*j <= 1

        """
        if label is None:
            label = self._new_constraint_label()
        elif label in self.constraint_labels:
            raise ValueError("a constraint with that label already exists")

        return super().add_constraint_from_iterable(iterable, sense, rhs, label, weight, penalty)

    def add_constraint_from_model(self,
                                  qm: typing.Union[BinaryQuadraticModel, QuadraticModel],
                                  sense: typing.Union[Sense, str],
                                  rhs: dimod.typing.Bias = 0,
                                  *,
                                  label: typing.Optional[typing.Hashable] = None,
                                  copy: bool = True,
                                  weight: typing.Optional[float] = None,
                                  penalty: str = 'linear',
                                  ) -> typing.Hashable:
        """Add a constraint from a quadratic model.

        Args:
            qm: Quadratic model or binary quadratic model.

            sense: One of `<=`, `>=`, `==`.

            rhs: Right-hand side of the constraint.

            label: Label for the constraint. Must be unique. If no label
                is provided, then one is generated using :mod:`uuid`.

            copy: If `True`, model ``qm`` is copied. This can be set to `False`
                to improve performance, but subsequently mutating ``qm`` can
                cause issues.

            weight: weight for soft-constraint. If None, the constraints
                is interpreted as hard.

            penalty: Penalty type for soft-constraint. Must be one of
                'linear', 'quadratic'. Ignored if weight is None. 'quadratic'
                only accepted if the constraint has binary variables.

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
        if label is None:
            label = self._new_constraint_label()
        elif label in self.constraint_labels:
            raise ValueError("a constraint with that label already exists")

        if isinstance(qm, BinaryQuadraticModel) and qm.dtype == object:
            qm = BinaryQuadraticModel(qm)

        return super().add_constraint_from_model(
            qm.data,
            sense,
            rhs,
            label,
            bool(copy),
            weight,
            penalty,
            )

    def add_discrete(self, data, *args, **kwargs) -> typing.Hashable:
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
            ...      cqm.add_variable('BINARY', v)
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
        if isinstance(data, (BinaryQuadraticModel, QuadraticModel)):
            return self.add_discrete_from_model(data, *args, **kwargs)
        elif isinstance(data, Comparison):
            return self.add_discrete_from_comparison(data, *args, **kwargs)
        elif isinstance(data, typing.Iterable):
            return self.add_discrete_from_iterable(data, *args, **kwargs)
        else:
            raise TypeError("unexpected data format")

    def add_discrete_from_comparison(self,
                                     comp: Comparison,
                                     label: typing.Optional[typing.Hashable] = None,
                                     copy: bool = True) -> typing.Hashable:
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
                left-hand side equal to one and the right-hand side equal
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
            raise ValueError("the right-hand side of a discrete constraint must be 1")
        return self.add_discrete_from_model(comp.lhs, label=label, copy=copy)

    def add_discrete_from_iterable(self,
                                   variables: typing.Iterable[dimod.typing.Variable],
                                   label: typing.Optional[typing.Hashable] = None,
                                   ) -> typing.Hashable:
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
            ...      cqm.add_variable('BINARY', v)
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
                                qm: typing.Union[BinaryQuadraticModel, QuadraticModel],
                                label: typing.Optional[typing.Hashable] = None,
                                copy: bool = True,
                                ) -> typing.Hashable:
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
                left-hand side equal to one and the right-hand side equal
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

    def add_variable(self, vartype: dimod.typing.VartypeLike,
                     v: typing.Optional[dimod.typing.Variable] = None,
                     *,
                     lower_bound: typing.Optional[float] = None,
                     upper_bound: typing.Optional[float] = None,
                     ) -> dimod.typing.Variable:
        """Add a variable to the model.

        Args:
            vartype:
                Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
                * :class:`~dimod.Vartype.REAL`, ``'REAL'``

            v:
                Label for the variable. Defaults to the length of the
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

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
            >>> cqm.add_variable('INTEGER', 'i')
            'i'

        """
        try:
            vartype = as_vartype(vartype, extended=True)
        except TypeError:
            # in dimod<0.11 the argument order was v, vartype so let's allow that case
            warnings.warn(
                "Parameter order CQM.add_variable(v, vartype) "
                "is deprecated since dimod 0.11.0 and will be removed in 0.13.0. "
                "Use CQM.add_variable(vartype, v) instead.",
                DeprecationWarning, stacklevel=2)
            v, vartype = vartype, v

        return super().add_variable(vartype, v, lower_bound=lower_bound, upper_bound=upper_bound)

    def add_variables(self,
                      vartype: dimod.typing.VartypeLike,
                      variables: typing.Union[int, typing.Sequence[dimod.typing.Variable]],
                      *,
                      lower_bound: typing.Optional[float] = None,
                      upper_bound: typing.Optional[float] = None,
                      ) -> None:
        if isinstance(variables, typing.Iterable):
            for v in variables:
                self.add_variable(vartype, v, lower_bound=lower_bound, upper_bound=upper_bound)
        else:
            for _ in range(variables):
                self.add_variable(vartype, lower_bound=lower_bound, upper_bound=upper_bound)

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

    def fix_variable(self, v: dimod.typing.Variable, value: float, *,
                     cascade: bool = False,
                     ) -> typing.Dict[dimod.type.Variable, float]:
        """Fix the value of a variable in the model.

        Note that this function does not test feasibility.

        Args:
            v: Variable label for a variable in the model.

            value: Value to assign variable ``v``.

            cascade: Deprecated. Does nothing.

        Returns:
            An empty dictionary, for legacy reasons.

        Raises:
            ValueError: If ``v`` is not the label of a variable in the model.

        """
        # todo: fix above docstring
        # todo: deprecation warning

        # get the discrete constaints that are possibly affected
        discrete = dict()
        for label in self.discrete:
            lhs = self.constraints[label].lhs
            if v in lhs.variables:
                discrete[label] = lhs

        super().fix_variable(v, value)

        if discrete:
            if value == 1:
                self.discrete -= discrete
            else:
                for label, lhs in discrete.items():
                    if lhs.num_variables <= 1:
                        self.discrete.discard(label)
        return {}

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
        # todo: fix above docstring
        # todo: deprecation warning

        if isinstance(fixed, typing.Mapping):
            fixed = fixed.items()

        for v, val in fixed:
            self.fix_variable(v, val)

        return {}

    def flip_variable(self, v):
        r"""Flip the specified binary variable in the objective and constraints.

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
        super().flip_variable(v)

        for label in list(self.discrete):
            lhs = self.constraints[label].lhs
            if v in lhs.variables:
                self.discrete.discard(label)  # no longer a discrete variable

    @classmethod
    def from_bqm(cls, bqm: BinaryQuadraticModel) -> ConstrainedQuadraticModel:
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
    def from_qm(cls, qm: QuadraticModel) -> 'ConstrainedQuadraticModel':
        """Alias for :meth:`from_quadratic_model`."""
        return cls.from_quadratic_model(qm)

    @classmethod
    def from_quadratic_model(cls, qm: typing.Union[QuadraticModel, BinaryQuadraticModel]
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
    def from_file(cls,
                  fp: typing.Union[typing.BinaryIO, typing.ByteString],
                  *,
                  check_header: bool = True,
                  ) -> ConstrainedQuadraticModel:
        """Construct from a file-like object.

        Args:
            fp: File pointer to a readable, seekable file-like object.

            check_header: If True, the header is checked for consistency
                against the deserialized model. Otherwise it is ignored.

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
        if isinstance(fp, typing.ByteString):
            file_like: typing.BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
        else:
            file_like = fp

        header_info = read_header(file_like, CQM_MAGIC_PREFIX)

        if not (1, 0) <= header_info.version <= (1, 3):
            raise ValueError("cannot load a CQM serialized with version "
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

                try:
                    weight = np.frombuffer(zf.read(f"constraints/{constraint}/weight"), np.float64)[0]
                    penalty = zf.read(f"constraints/{constraint}/penalty").decode('ascii')
                except KeyError:
                    weight = None
                    penalty = None

                cqm.add_constraint(lhs, rhs=rhs, sense=sense, label=label, weight=weight, penalty=penalty, copy=False)
                if discrete:
                    cqm.discrete.add(label)

        if check_header:
            expected = dict(num_variables=len(cqm.variables),
                            num_constraints=len(cqm.constraints),
                            num_biases=cqm.num_biases(),
                            )
            if header_info.version >= (1, 1):
                expected.update(
                    num_quadratic_variables=cqm.num_quadratic_variables(include_objective=False))
            if header_info.version >= (1, 2):
                expected.update(
                    num_quadratic_variables_real=cqm.num_quadratic_variables(Vartype.REAL, include_objective=True),
                    num_linear_biases_real=cqm.num_biases(Vartype.REAL, linear_only=True),
                    )
            if header_info.version >= (1, 3):
                expected.update(
                    num_weighted_constraints=cqm.num_soft_constraints(),
                    )

            if expected != header_info.data:
                raise ValueError(
                    "header data does not match the deserialized CQM. "
                    f"Expected {expected!r}, recieved {header_info.data!r}"
                    )

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
            * ``lhs_energy``:  Energy of the left-hand side of the constraint.
            * ``rhs_energy``: Energy of the right-hand side of the constraint.
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
        # We go ahead and coerce to Variables for performance, since .energies() prefers
        # that format
        sample, labels = as_samples(sample_like, labels_type=Variables)

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

    def iter_violations(self, sample_like: dimod.typing.SamplesLike, *,
                        skip_satisfied: bool = False,
                        clip: bool = False,
                        ) -> typing.Iterator[typing.Tuple[typing.Hashable, dimod.typing.Bias]]:
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

    def is_linear(self) -> bool:
        """Return True if the model has no quadratic interactions."""
        return (self.objective.is_linear() and
                all(comp.lhs.is_linear() for comp in self.constraints.values()))

    def _new_constraint_label(self) -> str:
        # we support up to 100k constraints and :6 gives us 16777216
        # possible so pretty safe
        label = 'c' + uuid.uuid4().hex[:6]
        while label in self.constraints:
            label = 'c' + uuid.uuid4().hex[:6]
        return label

    def num_biases(self, vartype: typing.Optional[dimod.typing.VartypeLike] = None, *,
                   linear_only: bool = False,
                   ) -> int:
        """Number of biases in the constrained quadratic model.

        Includes biases in both the objective and any constraints.

        Args:
            vartype: Count only variables of the specified :class:`~dimod.Vartype`.

            linear_only: Count only linear biases.

        Returns:
            The number of biases.

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
        if vartype is None:
            def count(qm: typing.Union[QuadraticModel, BinaryQuadraticModel]) -> int:
                return qm.num_variables + (0 if linear_only else qm.num_interactions)
        else:
            vartype = as_vartype(vartype, extended=True)

            def count(qm: typing.Union[QuadraticModel, BinaryQuadraticModel]) -> int:
                if isinstance(qm, BinaryQuadraticModel):
                    if qm.vartype is not vartype:
                        return 0
                    return qm.num_variables + (0 if linear_only else qm.num_interactions)
                else:
                    num_biases = sum(qm.vartype(v) is vartype for v in qm.variables)
                    if not linear_only:
                        num_biases += sum(qm.vartype(u) is vartype or qm.vartype(v) is vartype
                                          for u, v, _ in qm.iter_quadratic())
                    return num_biases

        return count(self.objective) + sum(count(const.lhs) for const in self.constraints.values())

    def num_quadratic_variables(self, vartype: typing.Optional[dimod.typing.VartypeLike] = None, *,
                                include_objective: typing.Optional[bool] = None,
                                ) -> int:
        """Number of variables with at least one quadratic interaction in the constrained quadratic model.

        Includes interactions in both the objective and any constraints.

        Args:
            vartype: Count only variables of the specified :class:`~dimod.Vartype`.

            include_objective: Count also variables in the objective. Currently defaults to false.

        Return:
            The number of variables.

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

        .. deprecated:: 0.10.14

            In dimod 0.12.0 ``include_objective`` will default to true.

        """
        if include_objective is None:
            warnings.warn(
                "in dimod 0.12.0 the default value of include_objective will change to true. "
                "To suppress this warning while keeping the existing behavior, set include_objective=False. "
                "To get the new behavior, set include_objective=True",
                DeprecationWarning,
                stacklevel=2,
                )

        if vartype is None:
            def count(qm: typing.Union[QuadraticModel, BinaryQuadraticModel]) -> int:
                return sum(qm.degree(v) > 0 for v in qm.variables)
        else:
            vartype = as_vartype(vartype, extended=True)

            def count(qm: typing.Union[QuadraticModel, BinaryQuadraticModel]) -> int:
                if isinstance(qm, BinaryQuadraticModel):
                    return sum(qm.degree(v) > 0 for v in qm.variables) if qm.vartype is vartype else 0
                else:
                    return sum(qm.vartype(v) is vartype and qm.degree(v) > 0 for v in qm.variables)

        n = sum(count(const.lhs) for const in self.constraints.values())

        if include_objective:
            n += count(self.objective)

        return n

    def relabel_constraints(self, mapping: typing.Mapping[typing.Hashable, typing.Hashable]):
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
        discrete_indices = [self.constraint_labels.index(c) for c in self.discrete]
        self.constraint_labels._relabel(mapping)
        self.discrete.clear()
        self.discrete |= (self.constraint_labels[i] for i in discrete_indices)

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

        self.variables._relabel(mapping)

        return self

    def remove_constraint(self, label: typing.Hashable, *, cascade: bool = False):
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
        if cascade:
            # Identify which variables are unique to the constraint we're about
            # to remove.
            # We could pull this down to the C++ level, but the performance
            # is pretty bad regardless so let's keep it simple and do it here.
            # There are also some potential Python performance improvements here
            to_remove: typing.Set[dimod.typing.Variable] = set()
            to_remove.update(self.constraints[label].lhs.variables)
            to_remove.difference_update(self.objective.variables)
            for lbl, comp in self.constraints.items():
                if not to_remove:
                    break
                if lbl != label:
                    to_remove.difference_update(comp.lhs.variables)

        super().remove_constraint(label)
        self.discrete.discard(label)

        if cascade and to_remove:
            for v in to_remove:
                self.remove_variable(v)

    def remove_variable(self, v: dimod.typing.Variable):
        for label in self.discrete:
            if v in self.constraints[label].variables:
                # todo: support this
                raise ValueError("cannot remove a variable used in a discrete constraint")

        super().remove_variable(v)

    def spin_to_binary(self, inplace: bool = False) -> ConstrainedQuadraticModel:
        """Convert any spin-valued variables to binary-valued.

        Args:
            inplace: If set to False, returns a new constrained quadratic model.
                Otherwise, the constrained quadratic model is modified in-place.

        """
        if not inplace:
            return copy.deepcopy(self).spin_to_binary(inplace=True)

        for v in self.variables:
            if self.vartype(v) is Vartype.SPIN:
                self.change_vartype(Vartype.BINARY, v)

        return self

    def _substitute_self_loops_from_model(self, qm: typing.Union[ConstraintView, ObjectiveView],
                                          mapping: typing.MutableMapping[dimod.typing.Variable,
                                                                         dimod.typing.Variable]):
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
                new: dimod.typing.Variable = new_variable_label()

                # on the off chance there are conflicts. Luckily self.variables
                # is global accross all constraints/objective so we don't need
                # to worry about accidentally picking something we'll regret
                while new in self.constraints or new in self.variables:
                    new = new_variable_label()

                mapping[u] = new

                self.add_variable(vartype, new, lower_bound=lb, upper_bound=ub)

                # we don't add the constraint yet because we don't want
                # to modify self.constraints
            else:
                new = mapping[u]

            self.add_variable(vartype, new, lower_bound=lb, upper_bound=ub)

            qm.add_quadratic(u, new, bias)
            qm.remove_interaction(u, u)

    def substitute_self_loops(self) -> typing.Dict[dimod.typing.Variable, dimod.typing.Variable]:
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
            Note that the ellipses in the outputs below are stand-ins for the
            generated variable labels, which differ between runs.

            >>> from dimod import Integer, ConstrainedQuadraticModel
            >>> i = Integer('i')
            >>> cqm = ConstrainedQuadraticModel()
            >>> cqm.add_constraint(i*i <=3, label='i squared')
            'i squared'
            >>> cqm.substitute_self_loops() #doctest: +ELLIPSIS
            {...}
            >>> cqm.constraints #doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
            {'i squared': Le(QuadraticModel({'i': 0.0, '...': 0.0}, {('...', 'i'): 1.0}, 0.0, {'i': 'INTEGER', '...': 'INTEGER'}, dtype='float64'), 3),
            '...': Eq(QuadraticModel({'i': 1.0, '...': -1.0}, {}, 0.0, {'i': 'INTEGER', '...': 'INTEGER'}, dtype='float64'), 0)}
        """
        # dev note: we can cythonize this for better performance

        mapping: typing.Dict[dimod.typing.Variable, dimod.typing.Variable] = dict()

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

        Format Specification (Version 1.3):

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
                     num_quadratic_variables_real=cqm.num_quadratic_variables('REAL', include_objective=True),
                     num_linear_biases_real=cqm.num_biases('REAL', linear_only=True),
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
            Weighted constraints also contain a `weight` and `penalty` file,
            encoding the weight and the penalty type for the constraint.

        Format Specification (Version 1.2):
            This format is the same as Version 1.3, except that there are no
            weighted constraints, and thus the `weight` and `penalty` files
            are not present.

        Format Specification (Version 1.1):

            This format is the same as Version 1.2, except that the data dict
            does not have ``num_quadratic_variables_real`` and
            ``num_linear_biases_real``.

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
                    num_quadratic_variables=self.num_quadratic_variables(include_objective=False),
                    num_quadratic_variables_real=self.num_quadratic_variables(Vartype.REAL, include_objective=True),
                    num_linear_biases_real=self.num_biases(Vartype.REAL, linear_only=True),
                    num_weighted_constraints=sum(comp.lhs.is_soft() for comp in self.constraints.values()),
                    )

        write_header(file, CQM_MAGIC_PREFIX, data, version=(1, 3))

        # write the values
        with zipfile.ZipFile(file, mode='a') as zf:
            with self.objective.to_file(spool_size=int(1e12)) as f:
                # we can avoid a copy by trying to read from the underlying buffer
                obj = f._file.getbuffer() if isinstance(f._file, io.BytesIO) else f.read()
                zf.writestr('objective', obj)
                # in the case we got the underlying buffer, we need to delete our
                # reference to it, otherwise we get an error when the file is closed
                del obj

            for label, constraint in self.constraints.items():
                # put everything in a constraints/label/ directory
                lstr = json.dumps(serialize_variable(label))

                with constraint.lhs.to_file(spool_size=int(1e12)) as f:
                    # we can avoid a copy by trying to read from the underlying buffer
                    lhs = f._file.getbuffer() if isinstance(f._file, io.BytesIO) else f.read()
                    zf.writestr(f'constraints/{lstr}/lhs', lhs)
                    # in the case we got the underlying buffer, we need to delete our
                    # reference to it, otherwise we get an error when the file is closed
                    del lhs

                rhs = np.float64(constraint.rhs).tobytes()
                zf.writestr(f'constraints/{lstr}/rhs', rhs)

                sense = bytes(constraint.sense.value, 'ascii')
                zf.writestr(f'constraints/{lstr}/sense', sense)

                discrete = bytes((label in self.discrete,))
                zf.writestr(f'constraints/{lstr}/discrete', discrete)

                # soft constraints
                if constraint.lhs.is_soft():
                    weight = np.float64(constraint.lhs.weight()).tobytes()
                    penalty = bytes(constraint.lhs.penalty(), 'ascii')
                    zf.writestr(f'constraints/{lstr}/weight', weight)
                    zf.writestr(f'constraints/{lstr}/penalty', penalty)

        file.seek(0)
        return file

    def violations(self, sample_like: dimod.typing.SamplesLike, *,
                   skip_satisfied: bool = False,
                   clip: bool = False,
                   ) -> typing.Dict[typing.Hashable, dimod.typing.Bias]:
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
                     fp: typing.Union[typing.BinaryIO, typing.ByteString],
                     lower_bound_default: typing.Optional[float] = None,
                     upper_bound_default: typing.Optional[float] = None,
                     ) -> ConstrainedQuadraticModel:
        """Create a constrained quadratic model from an LP file.

        Args:
            fp: A file-like or a binary string.
            lower_bound_default: Deprecated. Does nothing.
            upper_bound_default: Deprecated. Does nothing.

        Returns:
            The constrained quadratic model defined by the LP file.

        .. deprecated:: 0.11.0

            This method will be removed in dimod 0.13.0.
            Use :func:`~dimod.lp.load` or :func:`~dimod.lp.loads` instead.

        .. deprecated:: 0.11.0

            The ``lower_bound_default`` and ``upper_bound_default`` keyword
            arguments are deprecated and do nothing.

        """
        from dimod.lp import load, loads

        warnings.warn(
            "this method is deprecated as of dimod 0.11.0 "
            "and will be removed in 0.13.0. "
            "Use dimod.lp.load() or dimod.lp.load() instead.",
            DeprecationWarning, stacklevel=2)

        if isinstance(fp, (str, bytes)) and not os.path.isfile(fp):
            obj = loads(fp)
        else:
            obj = load(fp)

        return obj

    _STR_MAX_DISPLAY_ITEMS = 10

    def __str__(self) -> str:
        vartype_name = {Vartype.SPIN: 'Spin',
                        Vartype.BINARY: 'Binary',
                        Vartype.INTEGER: 'Integer',
                        Vartype.REAL: 'Real'}

        def var_encoder(v):
            return f'{vartype_name[self.vartype(v)]}({v!r})'

        sio = io.StringIO()

        def render_limited_number(iterable, render_element):
            tail_limit = self._STR_MAX_DISPLAY_ITEMS // 2
            head_limit = self._STR_MAX_DISPLAY_ITEMS - tail_limit
            limited = False
            tail = []

            for k, x in enumerate(iterable):
                assert x is not None

                if k < head_limit:
                    render_element(x)
                else:
                    tail.append(x)

                    if len(tail) > tail_limit:
                        if not limited:
                            sio.write('  ...\n')
                            limited = True
                        tail.pop(0)

            while tail:
                render_element(tail.pop(0))

        def render_constraint(item):
            label, c = item
            sio.write(f'  {label}: ')
            sio.write(c.to_polystring(encoder=var_encoder))
            sio.write('\n')

        def render_bound(v):
            sio.write(f'  {self.lower_bound(v)} <= {var_encoder(v)} <= {self.upper_bound(v)}\n')

        sio.write('Constrained quadratic model: ')
        sio.write(f'{len(self.variables)} variables, ')
        sio.write(f'{len(self.constraints)} constraints, ')
        sio.write(f'{self.num_biases()} biases\n\n')

        sio.write('Objective\n')
        sio.write('  ')
        sio.write(self.objective.to_polystring(encoder=var_encoder))
        sio.write('\n')

        sio.write('\n')
        sio.write('Constraints\n')
        render_limited_number(self.constraints.items(), render_constraint)

        sio.write('\n')
        sio.write('Bounds\n')
        bound_vars = (v for v in self.variables
                      if self.vartype(v) in (Vartype.INTEGER, Vartype.REAL))
        render_limited_number(bound_vars, render_bound)

        return sio.getvalue()


CQM = ConstrainedQuadraticModel


def _qm_to_bqm(qm: QuadraticModel,
               integers: typing.MutableMapping[dimod.typing.Variable, BinaryQuadraticModel],
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
                 binary: typing.Mapping[dimod.typing.Variable, Vartype],
                 integers: typing.Mapping[dimod.typing.Variable, BinaryQuadraticModel]):
        self._binary = binary
        self._integers = integers

    def __call__(self, sample: typing.Mapping[dimod.typing.Variable, int],
                 ) -> typing.Mapping[dimod.typing.Variable, int]:
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
    def from_dict(cls, doc: typing.Dict[str, typing.Dict[dimod.typing.Variable, typing.Any]],
                  ) -> CQMToBQMInverter:
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

    def to_dict(self) -> typing.Dict[str, typing.Dict[dimod.typing.Variable, typing.Any]]:
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
def cqm_to_bqm(cqm: ConstrainedQuadraticModel,
               lagrange_multiplier: typing.Optional[dimod.typing.Bias] = None,
               ) -> typing.Tuple[BinaryQuadraticModel, CQMToBQMInverter]:
    """Construct a binary quadratic model from a constrained quadratic model.

    Args:
        cqm: A constrained quadratic model. All constraints must be linear
            and all integer variables must have a lower bound of 0.

        lagrange_multiplier: The penalty strength used when converting
            constraints into penalty models. Defaults to 10x the largest
            bias in the objective.

    Returns:
        A 2-tuple containing a binary quadratic model and a function that converts
        samples over the binary quadratic model back into samples for the
        constrained quadratic model.

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
    binary: typing.Dict[dimod.typing.Variable, Vartype] = {}
    integers: typing.Dict[dimod.typing.Variable, BinaryQuadraticModel] = {}

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

    for v in binary:
        bqm.add_variable(v)

    # objective, we know it's always a QM
    bqm += _qm_to_bqm(cqm.objective, integers)

    if lagrange_multiplier is None:
        if cqm.constraints and bqm.num_variables:
            max_bias = max(-bqm.linear.min(), bqm.linear.max())
            if not bqm.is_linear():
                max_bias = max(-bqm.quadratic.min(), bqm.quadratic.max(), max_bias)
            if max_bias:
                lagrange_multiplier = 10 * max_bias
            else:
                lagrange_multiplier = 1
        else:
            lagrange_multiplier = 0  # doesn't matter

    for constraint in cqm.constraints.values():
        lhs = constraint.lhs
        rhs = constraint.rhs
        sense = constraint.sense

        if not isinstance(lhs, BinaryQuadraticModel):
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
                new_variable_label(),
                constant=lhs.offset,
                lb=rhs,
                ub=np.iinfo(np.int64).max,
                )
        elif sense is Sense.Le:
            bqm.add_linear_inequality_constraint(
                ((v, lhs.get_linear(v)) for v in lhs.variables),
                lagrange_multiplier,
                new_variable_label(),
                constant=lhs.offset,
                lb=np.iinfo(np.int64).min,
                ub=rhs,
                )
        else:
            raise RuntimeError("unexpected sense")

    return bqm, CQMToBQMInverter(binary, integers)


# register fileview loader
load.register(CQM_MAGIC_PREFIX, ConstrainedQuadraticModel.from_file)
