# distutils: language = c++
# cython: language_level = 3
#
# NOTE: This is a procedurally generated file. It should not be edited. See vector.pyx.template
#

from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from dimod.bqm.vectors.abc import Vector


ctypedef np.npy_float64 dtype


cdef class _Vector_npy_float64:
    cdef vector[dtype] biases

    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    def __cinit__(self):
        self.biases = vector[dtype]()


    def __init__(self, iterable=tuple()):

        if not isinstance(iterable, np.ndarray):
            for v in iterable:
                self.biases.push_back(v)
            return

        cdef dtype[:] iterable_view = iterable
        cdef Py_ssize_t num_biases = len(iterable)
        cdef Py_ssize_t i
        for i in range(num_biases):
            self.biases.push_back(iterable_view[i])

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(self.biases[0])

        self.shape[0] = self.biases.size()

        self.strides[0] = sizeof(self.biases[0])

        buffer.buf = <char *>&(self.biases[0])
        buffer.format = 'd'  # needs to correspond to dtype
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.biases.size() * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __len__(self):
        return self.biases.size()

    def __getitem__(self, Py_ssize_t i):
        if i >= self.biases.size():
            raise IndexError('index out of range')
        return self.biases[i]

    def __delitem__(self, Py_ssize_t i):
        if i >= self.biases.size():
            raise IndexError('assignment index out of range')
        self.biases.erase(self.biases.begin() + i)

    def __setitem__(self, Py_ssize_t i, dtype bias):
        if i >= self.biases.size():
            raise IndexError('assignment index out of range')
        self.biases[i] = bias

    def insert(self, int i, const dtype bias):
        if i < 0:
            raise IndexError("assignment index out of range")
        if i >= len(self):
            self.biases.push_back(bias)
        else:
            self.biases.insert(self.biases.begin() + i, bias)

class Vector_npy_float64(_Vector_npy_float64, Vector):
    __slots__ = ()
