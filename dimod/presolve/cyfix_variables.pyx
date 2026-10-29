# distutils: language = c++
# cython: language_level=3
#
# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cython.operator cimport dereference as deref

from libcpp.utility cimport pair
from libcpp.vector cimport vector

import dimod
import numpy as np

from dimod.libcpp cimport BinaryQuadraticModel as cppBinaryQuadraticModel
from dimod.binary.cybqm cimport cyBQM_float64


cdef extern from "include/dwave-preprocessing/fix_variables.hpp" namespace "fix_variables_":
    pair[double, vector[pair[int, int]]] fixQuboVariables[V, B](cppBinaryQuadraticModel[B, V]& refBQM,
                                                                bint strict, 
                                                                double offset) except +

def fix_variables_wrapper(bqm, strict):
    """Cython wrapper for fix_variables().

    Args:
        bqm (:class:`.BinaryQuadraticModel`):
            A binary quadratic model with binary-valued variables, indexed 
            linearly from zero.

        strict (bool):
            If True, only fixes variables for which assignments are true for all 
            minimizing points (strong persistency). If False, also fixes variables 
            for which the assignments are true for some but not all minimizing 
            points (weak persistency).
    """
    bqm = dimod.as_bqm(bqm, dtype=np.float64)

    if bqm.vartype is not dimod.BINARY:
        raise ValueError("bqm must be BINARY")
    if not all(v in bqm.linear for v in range(len(bqm))):
        raise ValueError("bqm must be linearly indexed")

    cdef cyBQM_float64 cybqm = bqm.data
    lower_bound, fixed = fixQuboVariables(deref(cybqm.cppbqm), bool(strict), bqm.offset)
    return lower_bound, {int(v): int(val) for v, val in fixed}
