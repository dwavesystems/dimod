# distutils: language = c++
# cython: language_level=3
#
# Copyright 2020 D-Wave Systems Inc.
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

from libcpp.vector cimport vector

from dimod.bqm.common cimport Bias, VarIndex
from dimod.bqm.cppbqm cimport AdjVectorBQM as cppAdjVectorBQM


cdef packed struct Range:
    Py_ssize_t start
    Py_ssize_t stop


cdef class cyDiscreteQuadraticModel:

    # we use a BQM to store the biases. Each case of the DQM is a variable in
    # the BQM
    cdef cppAdjVectorBQM[VarIndex, Bias] bqm_

    cdef vector[Range] variables_
