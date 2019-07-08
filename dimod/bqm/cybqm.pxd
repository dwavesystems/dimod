# distutils: language = c++
# cython: language_level=3

from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport numpy as np

ctypedef np.float64_t Bias
ctypedef np.uint32_t VarIndex
ctypedef size_t InteractionIndex

ctypedef pair[InteractionIndex, Bias]  InVar
ctypedef pair[VarIndex, Bias] OutVar

ctypedef fused Sample:
    np.npy_int8
    np.npy_int16
    np.npy_int32
    np.npy_int64
    np.npy_float32
    np.npy_float64

# cdef Bias energy(vector[InVar], vector[OutVar], Sample[:]) nogil


cdef class AdjArrayBQM:
    # developer note: we really just want an array but in cython you can't
    # do that, so we use vectors. This does potentially create issues because
    # it might be reallocated. Other than in the __cinit__, we will try to
    # only use methods available to std::array
    cdef vector[InVar] invars_  # cdef InVar[:] invars_
    cdef vector[OutVar] outvars_  # cdef OutVar[:] outvars_
