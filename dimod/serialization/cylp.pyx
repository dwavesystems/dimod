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
from dimod.quadratic import QuadraticModel


ctypedef np.float64_t bias_type
ctypedef np.int32_t index_type
dtype = np.float64


cdef void swap_qm(cyQM_float64 qm, cppQuadraticModel[bias_type, index_type]& cppqm, unordered_map[string, index_type] labels) except +:
    assert cppqm.num_variables() == labels.size()

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


# todo: annotations (in Python)
# todo: dtype
def read_lp_file(object filename):

    if not os.path.isfile(filename):
        raise NotImplementedError

    cdef string _filename = filename.encode()
    cdef cppLPModel[bias_type, index_type] lpmodel = cppread_lp[bias_type, index_type](_filename)

    cqm = ConstrainedQuadraticModel()

    swap_qm(cqm.objective.data, lpmodel.objective.model, lpmodel.objective.labels)

    cdef Py_ssize_t i
    for i in range(lpmodel.constraints.size()):
        lhs = QuadraticModel()
        swap_qm(lhs.data, lpmodel.constraints[i].lhs.model, lpmodel.constraints[i].lhs.labels)

        cqm.add_constraint_from_model(
            lhs,
            sense=lpmodel.constraints[i].sense.decode(),
            rhs=lpmodel.constraints[i].rhs,
            label=lpmodel.constraints[i].lhs.name.decode(),
            copy=False,
            )

    return cqm
