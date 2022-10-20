# distutils: language = c++
# cython: language_level=3

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

cimport numpy as np

from dimod.cyqmbase.cyqmbase_float64 cimport cyQMBase_float64 as cyQMBase, bias_type, index_type

__all__ = ['cyBQM_float64']

include "cybqm_template.pxd.pxi"

cdef class cyBQM_float64(cyBQM_template):
    pass
