# distutils: language = c++
# cython: language_level=3
#
# Copyright 2019 D-Wave Systems Inc.
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

from libcpp cimport bool
from libcpp.map cimport map as cppmap
from libcpp.pair cimport pair
from libcpp.vector cimport vector


cdef extern from "src/adjarray.cc":
    pass

cdef extern from "src/adjarray.h" namespace "dimod" nogil:

    size_t num_variables[V, B](const vector[pair[size_t, B]]&,
                               const vector[pair[V, B]]&)
    size_t num_interactions[V, B](const vector[pair[size_t, B]]&,
                                  const vector[pair[V, B]]&)

    B get_linear[V, B](const vector[pair[size_t, B]]&,
                       const vector[pair[V, B]]&,
                       V)

    B get_quadratic[V, B](const vector[pair[size_t, B]]&,
                          const vector[pair[V, B]]&,
                          V, V)


cdef extern from "src/adjmap.cc":
    pass

cdef extern from "src/adjmap.h" namespace "dimod" nogil:

    # some of these should be `const vector[pair[cppmap[V, B], B]]&` but
    # cython seems to have trouble with const vector of mutable objects,
    # so for now just leave them as-is

    size_t num_variables[V, B](vector[pair[cppmap[V, B], B]]&)
    size_t num_interactions[V, B](vector[pair[cppmap[V, B], B]]&)

    B get_linear[V, B](vector[pair[cppmap[V, B], B]]&, V)
    B get_quadratic[V, B](vector[pair[cppmap[V, B], B]]&, V, V)

    void set_linear[V, B](vector[pair[cppmap[V, B], B]]&, V, B)
    void set_quadratic[V, B](vector[pair[cppmap[V, B], B]]&, V, V, B)

    V add_variable[V, B](vector[pair[cppmap[V, B], B]]&)
    bool add_interaction[V, B](vector[pair[cppmap[V, B], B]]&, V, V)

    V pop_variable[V, B](vector[pair[cppmap[V, B], B]]&)
