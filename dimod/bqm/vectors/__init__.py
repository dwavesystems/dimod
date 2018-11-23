import numpy as np

from dimod.bqm.vectors.vector_npy_float32 import Vector_npy_float32
from dimod.bqm.vectors.vector_npy_float64 import Vector_npy_float64
from dimod.bqm.vectors.vector_npy_int8 import Vector_npy_int8
from dimod.bqm.vectors.vector_npy_int16 import Vector_npy_int16
from dimod.bqm.vectors.vector_npy_int32 import Vector_npy_int32
from dimod.bqm.vectors.vector_npy_int64 import Vector_npy_int64


__all__ = 'vector',


def vector(data=tuple(), dtype=None):

    if dtype is None:
        dtype = np.float64
    else:
        # handle strings etc
        dtype = np.dtype(dtype).type

    if dtype is np.float64:
        return Vector_npy_float64(data)
    elif dtype is np.float32:
        return Vector_npy_float32(data)
    elif dtype is np.int8:
        return Vector_npy_int8(data)
    elif dtype is np.int16:
        return Vector_npy_int16(data)
    elif dtype is np.int32:
        return Vector_npy_int32(data)
    elif dtype is np.int64:
        return Vector_npy_int64(data)
    else:
        raise ValueError("unsupported dtype")
