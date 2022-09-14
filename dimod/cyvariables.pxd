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

__all__ = ['cyVariables']

cdef class cyVariables:
    cdef object _index_to_label
    cdef object _label_to_index
    cdef Py_ssize_t _stop

    cdef object at(self, Py_ssize_t)
    cdef Py_ssize_t size(self)

    cpdef object _append(self, object v=*, bint permissive=*)
    cpdef void _clear(self)
    cpdef object _extend(self, object iterable, bint permissive=*)
    cpdef bint _is_range(self)
    cpdef object _pop(self)
    cpdef cyVariables copy(self)
    cdef Py_ssize_t _count_int(self, object) except -1
    cpdef Py_ssize_t count(self, object) except -1
    cpdef Py_ssize_t index(self, object, bint permissive=*) except -1
    cpdef _remove(self, object)