# Copyright 2022 D-Wave Systems Inc.
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

import os

from cython.operator cimport preincrement as inc, dereference as deref
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from dimod.constrained import ConstrainedQuadraticModel
from dimod.libcpp cimport cppread_lp, cppLPModel, cppQuadraticModel
from dimod.quadratic cimport cyQM_float64
from dimod.quadratic.cyqm.cyqm_float64 cimport bias_type, index_type
from dimod.quadratic.cyqm.cyqm_float64 import BIAS_DTYPE
from dimod.quadratic import QuadraticModel


cdef void _swap_qm(cyQM_float64 qm, cppQuadraticModel[bias_type, index_type]& cppqm, unordered_map[string, index_type] labels) except +:
    assert cppqm.num_variables() == labels.size()
    assert qm.num_variables() == 0

    qm.cppqm.swap(cppqm)

    cdef vector[string] variables
    variables.resize(labels.size())
    it = labels.begin()
    while it != labels.end():
        variables[deref(it).second] = deref(it).first
        inc(it)

    # cdef object v
    vit = variables.begin()
    while vit != variables.end():
        qm.variables._append(deref(vit).decode())
        inc(vit)


def cyread_lp_file(object filename):
    """Create a constrained quadratic model from the given LP file."""

    if not os.path.isfile(filename):
        raise ValueError(f"no file named {filename}")

    cdef string _filename = filename.encode()
    cdef cppLPModel[bias_type, index_type] lpmodel = cppread_lp[bias_type, index_type](_filename)

    cqm = ConstrainedQuadraticModel()

    if cqm.objective.dtype != BIAS_DTYPE:
        raise RuntimeError("unexpected dtype")

    _swap_qm(cqm.objective.data, lpmodel.objective.model, lpmodel.objective.labels)

    cdef Py_ssize_t i
    for i in range(lpmodel.constraints.size()):
        lhs = QuadraticModel(dtype=BIAS_DTYPE)
        _swap_qm(lhs.data, lpmodel.constraints[i].lhs.model, lpmodel.constraints[i].lhs.labels)

        cqm.add_constraint_from_model(
            lhs,
            sense=lpmodel.constraints[i].sense.decode(),
            rhs=lpmodel.constraints[i].rhs,
            label=lpmodel.constraints[i].lhs.name.decode(),
            copy=False,
            )

    return cqm
