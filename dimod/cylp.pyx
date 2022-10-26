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

import os.path

from libcpp.string cimport string

from dimod.constrained import ConstrainedQuadraticModel
from dimod.constrained.cyconstrained cimport cyConstrainedQuadraticModel, make_cqm
from dimod.libcpp.lp cimport read as cppread_lp, LPModel as cppLPModel
from dimod.quadratic.cyqm.cyqm_float64 cimport bias_type, index_type


def cyread_lp_file(object filename):
    """Create a constrained quadratic model from the given LP file."""

    if not os.path.isfile(filename):
        raise ValueError(f"no file named {filename}")

    cdef string _filename = filename.encode()
    cdef cppLPModel[bias_type, index_type] lpmodel = cppread_lp[bias_type, index_type](_filename)

    cdef cyConstrainedQuadraticModel cqm = make_cqm(lpmodel.model)

    # relabel the variables and constraints
    cdef Py_ssize_t i

    variable_mapping = dict()
    for i in range(lpmodel.variable_labels.size()):
        variable_mapping[i] = lpmodel.variable_labels[i].decode()
    assert(len(variable_mapping) == cqm.num_variables())
    cqm.relabel_variables(variable_mapping)

    constraint_mapping = dict()
    for i in range(lpmodel.constraint_labels.size()):
        if lpmodel.constraint_labels[i].size():
            constraint_mapping[i] = lpmodel.constraint_labels[i].decode()
    cqm.relabel_constraints(constraint_mapping)

    return cqm
