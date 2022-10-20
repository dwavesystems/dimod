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

from cython.operator cimport preincrement as inc, dereference as deref
from libc.math cimport ceil, floor
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport move
from libcpp.vector cimport vector

# import numpy as np

import dimod

from dimod.binary import BinaryQuadraticModel
from dimod.constrained.expression import ObjectiveView, ConstraintView
from dimod.cyqmbase cimport cyQMBase
from dimod.cyqmbase.cyqmbase_float64 import BIAS_DTYPE, INDEX_DTYPE
from dimod.cyutilities cimport as_numpy_float
from dimod.cyutilities cimport cppvartype
from dimod.libcpp.abc cimport QuadraticModelBase as cppQuadraticModelBase
from dimod.libcpp.constrained_quadratic_model cimport Sense as cppSense, Penalty as cppPenalty, Constraint as cppConstraint
from dimod.libcpp.vartypes cimport Vartype as cppVartype, vartype_info as cppvartype_info
# from dimod.quadratic cimport cyQM

# from dimod.sampleset import as_samples
from dimod.sym import Sense, Eq, Ge, Le
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

        if weight is not None:
            raise NotImplementedError  # todo

        self.cppcqm.add_constraint(move(constraint))
        label = self.constraint_labels._append(label)

        assert(self.cppcqm.num_constraints() == self.constraint_labels.size())

        return label

    def add_constraint_from_model(self, cyQMBase model, sense, bias_type rhs, label, bint copy, weight, penalty):
        # get a mapping from the model's variables to ours
        cdef vector[Py_ssize_t] mapping
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

        cdef Py_ssize_t ci = self.cppcqm.add_constraint(deref(model.base),
                                   cppsense(sense),
                                   rhs,
                                   mapping)

        if weight is not None:
            if penalty == 'linear':
                self.cppcqm.constraint_ref(ci).set_penalty(cppPenalty.LINEAR)
            elif penalty == 'quadratic':
                for vi in range(model.num_variables()):
                    if model.base.vartype(vi) != cppVartype.BINARY and model.base.vartype(vi) != cppVartype.SPIN:
                        raise ValueError("quadratic penalty only allowed if the constraint has binary variables")
                self.cppcqm.constraint_ref(ci).set_penalty(cppPenalty.QUADRATIC)
            elif penalty == 'constant':
                raise NotImplementedError('penalty should be "linear" or "quadratic"')
                # self.cppcqm.constraint_ref(ci).set_penalty(cppPenalty.CONSTANT)
            else:
                raise NotImplementedError('penalty should be "linear" or "quadratic"')

            self.cppcqm.constraint_ref(ci).set_weight(weight)
            

        return self.constraint_labels._append(label)

    def add_variable(self, vartype, v=None, *, lower_bound=None, upper_bound=None):
        cdef cppVartype vt = cppvartype(as_vartype(vartype, extended=True))

        cdef Py_ssize_t vi
        cdef bias_type lb
        cdef bias_type ub

        if v is not None and self.variables.count(v):
            # variable is already present
            vi = self.variables.index(v)
            if self.cppcqm.vartype(vi) != vt:
                raise TypeError(f"variable {v!r} already exists with a different vartype")
            if vt != cppVartype.BINARY and vt != cppVartype.SPIN:
                if lower_bound is not None:
                    lb = lower_bound
                    if lb != self.cppcqm.lower_bound(vi):
                        raise ValueError(
                            f"the specified lower bound, {lower_bound}, for "
                            f"variable {v!r} is different than the existing lower "
                            f"bound, {self.cppcqm.lower_bound(vi)}")
                if upper_bound is not None:
                    ub = upper_bound
                    if ub != self.cppcqm.upper_bound(vi):
                        raise ValueError(
                            f"the specified upper bound, {upper_bound}, for "
                            f"variable {v!r} is different than the existing upper "
                            f"bound, {self.cppcqm.upper_bound(vi)}")

            return v

        # ok, we have a shiny new variable

        if vt == cppVartype.SPIN or vt == cppVartype.BINARY:
            # we can ignore bounds
            v = self.variables._append(v)
            self.cppcqm.add_variable(vt)
            return v

        if vt != cppVartype.INTEGER and vt != cppVartype.REAL:
            raise RuntimeError("unexpected vartype")  # catch some future issues


        
        if lower_bound is None:
            lb = cppvartype_info[bias_type].default_min(vt)
        else:
            lb = lower_bound
            if lb < cppvartype_info[bias_type].min(vt):
                raise ValueError(f"lower_bound cannot be less than {cppvartype_info[bias_type].min(vt)}")

        if upper_bound is None:
            ub = cppvartype_info[bias_type].default_max(vt)
        else:
            ub = upper_bound
            if ub > cppvartype_info[bias_type].max(vt):
                raise ValueError(f"upper_bound cannot be greater than {cppvartype_info[bias_type].max(vt)}")

        v = self.variables._append(v)
        self.cppcqm.add_variable(vt, lb, ub)
        return v

    def change_vartype(self, vartype, v):
        vartype = as_vartype(vartype, extended=True)
        cdef cppVartype vt = cppvartype(vartype)
        cdef Py_ssize_t vi = self.variables.index(v)
        try:
            self.cppcqm.change_vartype(vt, vi)
        except RuntimeError as err:
            # c++ logic_error
            print(err)

            raise TypeError(f"cannot change vartype {self.vartype(v).name!r} "
                            f"to {vartype.name!r}") from None

    def fix_variable(self, v, bias_type assignment):
        self.cppcqm.fix_variable(self.variables.index(v), assignment)
        self.variables._remove(v)

    def flip_variable(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        if self.cppcqm.vartype(vi) == cppVartype.SPIN:
            self.cppcqm.substitute_variable(vi, -1, 0)
        elif self.cppcqm.vartype(vi) == cppVartype.BINARY:
            self.cppcqm.substitute_variable(vi, -1, 1)
        else:
            raise ValueError(f"can only flip SPIN and BINARY variables")

    def lower_bound(self, v):
        return as_numpy_float(self.cppcqm.lower_bound(self.variables.index(v)))

    def num_constraints(self):
        return self.cppcqm.num_constraints()

    def num_soft_constraints(self):
        cdef Py_ssize_t count = 0
        for c in range(self.cppcqm.num_constraints()):
            if self.cppcqm.constraint_ref(c).is_soft():
                count += 1 
        return count

    def remove_constraint(self, label):
        cdef Py_ssize_t ci = self.constraint_labels.index(label)
        self.cppcqm.remove_constraint(ci)
        self.constraint_labels._remove(label)

    def remove_variable(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        self.cppcqm.remove_variable(vi)
        self.variables._remove(v)

    def set_lower_bound(self, v, bias_type lb):
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
        """

        note: bad inputs = bad models
        """
        if isinstance(objective, typing.Iterable):
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
        return as_numpy_float(self.cppcqm.upper_bound(self.variables.index(v)))

    def vartype(self, v):
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
