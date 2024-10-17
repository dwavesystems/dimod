# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Un_lt required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import collections.abc
import io
import itertools
import numbers
import typing

from copy import deepcopy

cimport cython
import numpy as np

from cython.operator cimport preincrement as inc, dereference as deref
from libc.math cimport ceil, floor
from libcpp.cast cimport reinterpret_cast
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport move
from libcpp.vector cimport vector

import dimod

from dimod.binary import BinaryQuadraticModel
from dimod.constrained.expression import ObjectiveView, ConstraintView
from dimod.cyqmbase cimport cyQMBase
from dimod.cyqmbase.cyqmbase_float64 import BIAS_DTYPE, INDEX_DTYPE
from dimod.cyutilities cimport as_numpy_float
from dimod.cyutilities cimport cppvartype
from dimod.discrete.cydiscrete_quadratic_model cimport cyDiscreteQuadraticModel
from dimod.libcpp.abc cimport QuadraticModelBase as cppQuadraticModelBase
from dimod.libcpp.constrained_quadratic_model cimport Sense as cppSense, Penalty as cppPenalty, Constraint as cppConstraint
from dimod.libcpp.vartypes cimport Vartype as cppVartype, vartype_info as cppvartype_info
from dimod.sym import Sense, Eq, Ge, Le
from dimod.typing cimport int8_t
from dimod.variables import Variables
from dimod.vartypes import as_vartype, Vartype
from dimod.views.quadratic import QuadraticViewsMixin


# todo: move to cyutilities?
cdef cppSense cppsense(object sense) except? cppSense.GE:
    if isinstance(sense, str):
        sense = Sense(sense)

    if sense is Sense.Eq:
        return cppSense.EQ
    elif sense is Sense.Le:
        return cppSense.LE
    elif sense is Sense.Ge:
        return cppSense.GE
    else:
        raise RuntimeError(f"unexpected sense: {sense!r}")


cdef class cyConstraintsView:
    cdef cyConstrainedQuadraticModel parent

    def __init__(self, cyConstrainedQuadraticModel parent):
        self.parent = parent

    def __delitem__(self, key) -> None:
        # todo: warn
        raise NotImplementedError

    def __getitem__(self, key) -> ConstraintView:
        cdef Py_ssize_t vi
        try:
            vi = self.parent.constraint_labels.index(key)
        except ValueError as err:
            raise KeyError(repr(key)) from None  # todo: better message

        lhs = ConstraintView(self.parent, key)
        rhs = as_numpy_float(self.parent.cppcqm.constraint_ref(vi).rhs())

        if self.parent.cppcqm.constraint_ref(vi).sense() == cppSense.EQ:
            return Eq(lhs, rhs)
        elif self.parent.cppcqm.constraint_ref(vi).sense() == cppSense.LE:
            return Le(lhs, rhs)
        elif self.parent.cppcqm.constraint_ref(vi).sense() == cppSense.GE:
            return Ge(lhs, rhs)
        else:
            raise RuntimeError("unexpected Sense")

    def __iter__(self):
        yield from self.parent.constraint_labels

    def __len__(self):
        return len(self.parent.constraint_labels)



class ConstraintsView(cyConstraintsView, collections.abc.Mapping):
    def __str__(self) -> str:
        stream = io.StringIO()
        stream.write('{')
        last = len(self) - 1
        for i, (key, value) in enumerate(self.items()):
            stream.write(f'{key!r}: {value!r}')
            if i != last:
                stream.write(', ')
        stream.write('}')
        return stream.getvalue()



cdef class cyConstrainedQuadraticModel:
    def __cinit__(self):
        self.constraint_labels = Variables()
        self.variables = Variables()

        self.dtype = BIAS_DTYPE
        self.index_dtype = INDEX_DTYPE

    def __init__(self):
        self.objective = ObjectiveView(self)

        self.constraints = ConstraintsView(self)

        self.REAL_INTERACTIONS = dimod.REAL_INTERACTIONS

    def __deepcopy__(self, memo):
        cdef cyConstrainedQuadraticModel new = type(self)()

        new.cppcqm = self.cppcqm  # copy assignment operator
        new.constraint_labels = deepcopy(self.constraint_labels, memo)
        new.variables = deepcopy(self.variables, memo)

        memo[id(self)] = new

        return new

    def add_constraint_from_iterable(self, iterable, sense, bias_type rhs, label, weight, penalty):
        # get a fresh constraint        
        constraint = self.cppcqm.new_constraint()

        cdef Py_ssize_t ui, vi
        cdef bias_type bias
        for *variables, bias in iterable:
            if len(variables) == 0:
                constraint.add_offset(bias)
            elif len(variables) == 1:
                vi = self.variables.index(variables[0])  # must already be a variable
                constraint.add_linear(vi, bias)
            elif len(variables) == 2:
                ui = self.variables.index(variables[0])  # must already be a variable
                vi = self.variables.index(variables[1])
                constraint.add_quadratic(ui, vi, bias)
            else:
                raise ValueError("terms must be constant, linear or quadratic")

        constraint.set_sense(cppsense(sense))
        constraint.set_rhs(rhs)

        self.cppcqm.add_constraint(move(constraint))
        label = self.constraint_labels._append(label)
        assert(self.cppcqm.num_constraints() == self.constraint_labels.size())

        if weight is not None:
            ConstraintView(self, label).set_weight(weight, penalty=penalty)

        return label

    def add_constraint_from_model(self, cyQMBase model, sense, bias_type rhs, label, bint copy, weight, penalty):
        # get a mapping from the model's variables to ours
        cdef vector[index_type] mapping
        mapping.reserve(model.num_variables())
        cdef Py_ssize_t vi
        for vi in range(model.num_variables()):
            v = model.variables.at(vi)
            if self.variables.count(v):
                # there is a variable already
                mapping.push_back(self.variables.index(v))

                if self.cppcqm.vartype(mapping[vi]) != model.base.vartype(vi):
                    raise ValueError(f"conflicting vartypes: {v!r}")

                if self.cppcqm.lower_bound(mapping[vi]) != model.base.lower_bound(vi):
                    raise ValueError(f"conflicting lower bounds: {v!r}")

                if self.cppcqm.upper_bound(mapping[vi]) != model.base.upper_bound(vi):
                    raise ValueError(f"conflicting upper bounds: {v!r}")
            else:
                # not yet present, let's just track that fact for now
                # in case there is a mismatch so we don't modify our object yet
                mapping.push_back(-1)
        
        for vi in range(mapping.size()):
            if mapping[vi] != -1:
                continue  # already added and checked

            mapping[vi] = self.cppcqm.num_variables()  # we're about to add a new one

            v = model.variables.at(vi)
            vartype = model.vartype(v)

            self.add_variable(vartype, v,
                              lower_bound=model.base.lower_bound(vi),
                              upper_bound=model.base.upper_bound(vi),
                              )

        if copy:
            self.cppcqm.add_constraint(deref(model.base), cppsense(sense), rhs, mapping)
        else:
            self.cppcqm.add_constraint(move(deref(model.base)), cppsense(sense), rhs, mapping)
            model.clear()

        label = self.constraint_labels._append(label)
        assert(self.cppcqm.num_constraints() == self.constraint_labels.size())

        if weight is not None:
            ConstraintView(self, label).set_weight(weight, penalty=penalty)

        return label

    def add_variables(self, vartype, variables, *, lower_bound=None, upper_bound=None):
        """Add variables to the model.

        Args:
            vartype:
                Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
                * :class:`~dimod.Vartype.REAL`, ``'REAL'``

            variables:
                An iterable of variable labels or an integer. An integer ``n``
                is interpreted as ``range(n)``.

            lower_bound:
                Lower bound on the variable. Ignored when the variable is
                :class:`~dimod.Vartype.BINARY` or :class:`~dimod.Vartype.SPIN`.

            upper_bound:
                Upper bound on the variable. Ignored when the variable is
                :class:`~dimod.Vartype.BINARY` or :class:`~dimod.Vartype.SPIN`.

        Exceptions:
            ValueError: If a variable is added with a different ``vartype``,
                ``lower_bound``, or ``upper_bound``.
                Note that the variables before the inconsistent variable will
                be added to the model.

        """
        cdef cppVartype vt = cppvartype(as_vartype(vartype, extended=True))

        # for BINARY and SPIN the bounds are ignored
        if vt == cppVartype.SPIN:
            lower_bound = -1
            upper_bound = +1
        elif vt == cppVartype.BINARY:
            lower_bound = 0
            upper_bound = 1
        elif vt != cppVartype.INTEGER and vt != cppVartype.REAL:
            raise RuntimeError("unexpected vartype")  # catch some future issues

        # bound parsing, we'll also want to track whether the bound was specified or not
        cdef bint lb_given = lower_bound is not None
        cdef bias_type lb = lower_bound if lb_given else cppvartype_info[bias_type].default_min(vt)
        if lb < cppvartype_info[bias_type].min(vt):
            raise ValueError(f"lower_bound cannot be less than {cppvartype_info[bias_type].min(vt)}")
        
        cdef bint ub_given = upper_bound is not None
        cdef bias_type ub = upper_bound if ub_given else cppvartype_info[bias_type].default_max(vt)
        if ub > cppvartype_info[bias_type].max(vt):
            raise ValueError(f"upper_bound cannot be greater than {cppvartype_info[bias_type].max(vt)}")

        if lb > ub:
            raise ValueError("lower_bound must be less than or equal to upper_bound")        

        # parse the variables
        if isinstance(variables, int):
            variables = range(variables)

        cdef Py_ssize_t count = self.variables.size()
        for v in variables:
            self.variables._append(v, permissive=True)

            if count == self.variables.size():
                # the variable already existed
                vi = self.variables.index(v)

                if vt != self.cppcqm.vartype(vi):
                    raise ValueError(f"variable {v!r} already exists with a different vartype")

                if lb_given and lb != self.cppcqm.lower_bound(vi):
                    raise ValueError(
                        f"the specified lower bound, {lower_bound}, for "
                        f"variable {v!r} is different than the existing lower "
                        f"bound, {self.cppcqm.lower_bound(vi)}")

                if ub_given and ub != self.cppcqm.upper_bound(vi):
                    raise ValueError(
                        f"the specified upper bound, {upper_bound}, for "
                        f"variable {v!r} is different than the existing upper "
                        f"bound, {self.cppcqm.upper_bound(vi)}")

            elif count == self.variables.size() - 1:
                # we added a new variable
                self.cppcqm.add_variable(vt, lb, ub)
                count += 1

            else:
                raise RuntimeError("something went wrong")

    def change_vartype(self, vartype, v):
        vartype = as_vartype(vartype, extended=True)
        cdef cppVartype vt = cppvartype(vartype)
        cdef Py_ssize_t vi = self.variables.index(v)
        try:
            self.cppcqm.change_vartype(vt, vi)
        except RuntimeError as err:
            # c++ logic_error
            raise TypeError(f"cannot change vartype {self.vartype(v).name!r} "
                            f"to {vartype.name!r}") from None

    def clear(self):
        self.variables._clear()
        self.constraint_labels._clear()
        self.cppcqm.clear()

    def fix_variable(self, v, bias_type assignment):
        cdef Py_ssize_t vi = self.variables.index(v)

        if self.cppcqm.vartype(vi) == cppVartype.BINARY and assignment:
            # we may be affecting discrete constraints, so let's update the markers
            for i in range(self.cppcqm.num_constraints()):
                constraint = self.cppcqm.constraint_ref(i)

                if constraint.marked_discrete() and constraint.has_variable(vi):
                    constraint.mark_discrete(False)

        self.cppcqm.fix_variable(vi, assignment)
        self.variables._remove(v)

    def fix_variables(self, fixed, *, bint inplace = True):
        if isinstance(fixed, collections.abc.Mapping):
            fixed = fixed.items()

        if inplace:
            for v, assignment in fixed:
                self.fix_variable(v, assignment)
            return self

        cdef vector[index_type] variables
        cdef vector[bias_type] assignments
        labels = set()

        for v, bias in fixed:
            variables.push_back(self.variables.index(v))
            assignments.push_back(bias)
            labels.add(v)

        cqm = make_cqm(self.cppcqm.fix_variables(variables.begin(), variables.end(), assignments.begin()))

        # relabel variables
        mapping = dict()
        i = 0
        for v in self.variables:
            if v not in labels:
                mapping[i] = v
                i += 1
        cqm.relabel_variables(mapping)

        # relabel constraints
        cqm.relabel_constraints(dict(enumerate(self.constraint_labels)))

        return cqm

    def flip_variable(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        if self.cppcqm.vartype(vi) == cppVartype.SPIN:
            self.cppcqm.substitute_variable(vi, -1, 0)
        elif self.cppcqm.vartype(vi) == cppVartype.BINARY:
            self.cppcqm.substitute_variable(vi, -1, 1)
        else:
            raise ValueError(f"can only flip SPIN and BINARY variables")

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def from_discrete_quadratic_model(cls, dqm, relabel_func=lambda v, c: (v, c)):
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
        cdef cyConstrainedQuadraticModel cqm = cls()

        cdef cyDiscreteQuadraticModel cydqm = dqm._cydqm

        cqm.cppcqm.set_objective(cydqm.cppbqm)
        cqm.variables._extend(relabel_func(v, case) for v in dqm.variables for case in dqm.get_cases(v))

        cdef Py_ssize_t vi, ci
        for vi in range(cydqm.num_variables()):
            constraint = cqm.cppcqm.new_constraint()
            for ci in range(cydqm.case_starts_[vi], cydqm.case_starts_[vi+1]):
                constraint.add_linear(ci, 1)
            constraint.set_sense(cppSense.EQ)
            constraint.set_rhs(1)
            constraint.mark_discrete()

            cqm.cppcqm.add_constraint(move(constraint))

        cqm.constraint_labels._extend(dqm.variables)  # adjust the labels to match

        assert(cqm.cppcqm.num_variables() == cqm.variables.size())
        assert(cqm.cppcqm.num_constraints() == cqm.constraint_labels.size())

        return cqm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ivarinfo(self):
        cdef Py_ssize_t num_variables = self.cppcqm.num_variables()

        # We choose the field names to mirror the internals of the ConstrainedQuadraticModel
        dtype = np.dtype([('vartype', np.int8), ('lb', self.dtype), ('ub', self.dtype)],
                         align=False)
        varinfo = np.empty(num_variables, dtype)

        cdef int8_t[:] vartype_view = varinfo['vartype']
        cdef bias_type[:] lb_view = varinfo['lb']
        cdef bias_type[:] ub_view = varinfo['ub']

        cdef Py_ssize_t vi
        for vi in range(self.num_variables()):
            vartype_view[vi] = self.cppcqm.vartype(vi)
            lb_view[vi] = self.cppcqm.lower_bound(vi)
            ub_view[vi] = self.cppcqm.upper_bound(vi)

        return varinfo

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ivarinfo_load(self, buff, Py_ssize_t num_variables):
        """Inverse of ._ivarinfo()"""
        if self.cppcqm.num_variables():
            raise RuntimeError("varinfo can only be loaded into an empty model")

        dtype = np.dtype([('vartype', np.int8), ('lb', self.dtype), ('ub', self.dtype)],
                         align=False)

        arr = np.frombuffer(buff[:dtype.itemsize*num_variables], dtype=dtype)
        cdef const int8_t[:] vartype_view = arr['vartype']
        cdef const bias_type[:] lb_view = arr['lb']
        cdef const bias_type[:] ub_view = arr['ub']

        cdef Py_ssize_t vi
        cdef cppVartype cpp_vartype
        for vi in range(num_variables):
            self.cppcqm.add_variable(<cppVartype>(vartype_view[vi]), lb_view[vi], ub_view[vi])

        self.variables._extend(range(num_variables))

    def lower_bound(self, v):
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
        return as_numpy_float(self.cppcqm.lower_bound(self.variables.index(v)))

    def num_constraints(self):
        return self.cppcqm.num_constraints()

    def num_soft_constraints(self):
        cdef Py_ssize_t count = 0
        for c in range(self.cppcqm.num_constraints()):
            if self.cppcqm.constraint_ref(c).is_soft():
                count += 1 
        return count

    def num_variables(self):
        return self.cppcqm.num_variables()

    def remove_constraint(self, label):
        cdef Py_ssize_t ci = self.constraint_labels.index(label)
        self.cppcqm.remove_constraint(ci)
        self.constraint_labels._remove(label)

    def remove_variable(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        self.cppcqm.remove_variable(vi)
        self.variables._remove(v)

    def set_lower_bound(self, v, bias_type lb):
        """Set the lower bound for a variable.

        Args:
            v: Variable label of a variable in the constrained quadratic model.
            lb: Lower bound to set for variable ``v``.

        Raises:
            ValueError: If ``v`` is a :class:`~dimod.Vartype.SPIN`
                or :class:`~dimod.Vartype.BINARY` variable.

        Examples:
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_variable('INTEGER', 'j', upper_bound=5)
            'j'
            >>> cqm.set_lower_bound('j', 2)
        """
        cdef Py_ssize_t vi = self.variables.index(v)

        cdef cppVartype vt = self.cppcqm.vartype(vi)
        if vt == cppVartype.BINARY or vt == cppVartype.SPIN:
            raise ValueError(
                "cannot set the lower bound for BINARY or SPIN variables, "
                f"{v!r} is a {self.vartype(v).name} variable")

        if lb < cppvartype_info[bias_type].min(vt):
            raise ValueError(f"lower_bound cannot be less than {cppvartype_info[bias_type].min(vt)}")

        if lb > self.cppcqm.upper_bound(vi):
            raise ValueError(
                f"the specified lower bound, {lb}, cannot be set greater than the "
                f"current upper bound, {self.cppcqm.upper_bound(vi)}"
                )

        if vt == cppVartype.INTEGER:
            if ceil(lb) > floor(self.cppcqm.upper_bound(vi)):
                raise ValueError(
                    "there must be at least one integer value between "
                    f"the specified lower bound, {lb} and the "
                    f"current upper bound, {self.cppcqm.upper_bound(vi)}"
                    )

        self.cppcqm.set_lower_bound(vi, lb)

    def _set_objective_from_cyqm(self, cyQMBase objective):

        # get a mapping from the objective's variables to ours
        cdef vector[Py_ssize_t] mapping
        mapping.reserve(objective.num_variables())
        cdef Py_ssize_t vi
        for vi in range(objective.num_variables()):
            v = objective.variables.at(vi)
            if self.variables.count(v):
                # there is a variable already
                mapping.push_back(self.variables.index(v))

                if self.cppcqm.vartype(mapping[vi]) != objective.base.vartype(vi):
                    raise ValueError(f"conflicting vartypes: {v!r}")

                if self.cppcqm.lower_bound(mapping[vi]) != objective.base.lower_bound(vi):
                    raise ValueError(f"conflicting lower bounds: {v!r}")

                if self.cppcqm.upper_bound(mapping[vi]) != objective.base.upper_bound(vi):
                    raise ValueError(f"conflicting upper bounds: {v!r}")
            else:
                # not yet present, let's just track that fact for now
                # in case there is a mismatch so we don't modify our object yet
                mapping.push_back(-1)
        
        for vi in range(mapping.size()):
            if mapping[vi] != -1:
                continue  # already added and checked

            mapping[vi] = self.cppcqm.num_variables()  # we're about to add a new one

            v = objective.variables.at(vi)
            vartype = objective.vartype(v)

            self.add_variable(vartype, v,
                              lower_bound=objective.base.lower_bound(vi),
                              upper_bound=objective.base.upper_bound(vi),
                              )

        self.cppcqm.set_objective(deref(objective.base), mapping)

    def set_objective(self, objective):
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
        if isinstance(objective, collections.abc.Iterable):
            terms = objective
        else:
            if isinstance(objective, BinaryQuadraticModel) and objective.dtype == object:
                objective = BinaryQuadraticModel(objective, dtype=self.dtype)

            return self._set_objective_from_cyqm(objective.data)

            # # assume that we're some sort of QM
            # terms = itertools.chain(
            #     objective.iter_linear(),
            #     objective.iter_quadratic(),
            #     ((objective.offset,),)
            #     )

        # clear out anything currently in there. This actually introduces
        # an issue where if the terms later raise an error that we get only
        # a partly formed model. We could mitigate this by dumping the terms
        # to a set of vectors or something, but this avoids that intermediate
        # object
        self.cppcqm.objective.clear()  

        cdef Py_ssize_t ui
        cdef Py_ssize_t vi
        cdef bias_type bias
        for *variables, bias in terms:
            if len(variables) == 0:
                self.cppcqm.objective.add_offset(bias)
            elif len(variables) == 1:
                ui = self.variables.index(variables[0])
                self.cppcqm.objective.add_linear(ui, bias)
            elif len(variables) == 2:
                # quadratic
                ui = self.variables.index(variables[0])
                vi = self.variables.index(variables[1])
                self.cppcqm.objective.add_quadratic(ui, vi, bias)
            else:
                raise ValueError("terms must be a tuple of length 1, 2, or 3")

    def set_upper_bound(self, v, bias_type ub):
        """Set the upper bound for a variable.

        Args:
            v: Variable label of a variable in the constrained quadratic model.
            ub: Upper bound to set for variable ``v``.

        Raises:
            ValueError: If ``v`` is a :class:`~dimod.Vartype.SPIN`
                or :class:`~dimod.Vartype.BINARY` variable.

        Examples:
            >>> cqm = dimod.ConstrainedQuadraticModel()
            >>> cqm.add_variable('INTEGER', 'j', lower_bound=2)
            'j'
            >>> cqm.set_upper_bound('j', 5)
        """
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef cppVartype vt = self.cppcqm.vartype(vi)

        if vt == cppVartype.BINARY or vt == cppVartype.SPIN:
            raise ValueError(
                "cannot set the upper bound for BINARY or SPIN variables, "
                f"{v!r} is a {self.vartype(v).name} variable")

        if ub > cppvartype_info[bias_type].max(vt):
            raise ValueError(f"upper_bound cannot be more than {cppvartype_info[bias_type].max(vt)}")
            
        if ub < self.cppcqm.lower_bound(vi):
            raise ValueError(
                f"the specified upper bound, {ub}, cannot be set less than the "
                f"current lower bound, {self.cppcqm.lower_bound(vi)}"
                )

        if vt == cppVartype.INTEGER:
            if ceil(self.cppcqm.lower_bound(vi)) > floor(ub):
                raise ValueError(
                    "there must be at least one integer value between "
                    f"the specified upper bound, {ub} and the "
                    f"current lower bound, {self.cppcqm.lower_bound(vi)}"
                    )

        self.cppcqm.set_upper_bound(vi, ub)

    def upper_bound(self, v):
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
        return as_numpy_float(self.cppcqm.upper_bound(self.variables.index(v)))

    def vartype(self, v):
        """Vartype of the given variable.
        
        Args:
            v: Variable label for a variable in the model.
        """
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef cppVartype cppvartype = self.cppcqm.vartype(vi)

        if cppvartype == cppVartype.BINARY:
            return Vartype.BINARY
        elif cppvartype == cppVartype.SPIN:
            return Vartype.SPIN
        elif cppvartype == cppVartype.INTEGER:
            return Vartype.INTEGER
        elif cppvartype == cppVartype.REAL:
            return Vartype.REAL
        else:
            raise RuntimeError("unexpected vartype")


cdef object make_cqm(cppConstrainedQuadraticModel[bias_type, index_type] cppcqm):
    cdef cyConstrainedQuadraticModel cqm = dimod.ConstrainedQuadraticModel()

    cqm.variables._extend(range(cppcqm.num_variables()))
    cqm.constraint_labels._extend(range(cppcqm.num_constraints()))
    cqm.cppcqm = move(cppcqm)

    return cqm
