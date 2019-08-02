# distutils: language = c++
# cython: language_level=3
#
# Copyright 2018 D-Wave Systems Inc.
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
#
# =============================================================================


from libcpp.map cimport map
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from dimod.vartypes import Vartype

cdef extern from "fix_variables.hpp" namespace "fix_variables_":
    vector[pair[int, int]] fixQuboVariablesMap(map[pair[int, int], double] QMap,
                                               int QSize, int method) except +


def fix_variables_wrapper(bqm, method):
    """bqm should be binary and linear indexed

    method 1 -> roof-duality & strongly connected components
    method 2 -> roof-duality only

    """

    if bqm.vartype is not Vartype.BINARY:
        raise ValueError("bqm must be BINARY")
    if not all(v in bqm.linear for v in range(len(bqm))):
        raise ValueError("bqm must be integer-labeled")
    if not isinstance(method, int):
        raise TypeError("method should be an int")
    if method < 1 or method > 2:
        raise ValueError("method should 1 or 2")

    cdef map[pair[int, int], double] QMap
    for (u, v), bias in bqm.quadratic.items():
        QMap[pair[int, int](u, v)] = bias
    for v, bias in bqm.linear.items():
        QMap[pair[int, int](v, v)] = bias

    fixed = fixQuboVariablesMap(QMap, len(bqm), int(method))

    return {int(v - 1): int(val) for v, val in fixed}
