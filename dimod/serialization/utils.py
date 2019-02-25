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
import io

import numpy as np


__all__ = 'array2bytes', 'bytes2array'


def array2bytes(arr, bytes_type=bytes):
    """Wraps NumPy's save function to return bytes.

    We use :func:`numpy.save` rather than :meth:`numpy.ndarray.tobytes` because
    it encodes endianness and order.

    Args:
        arr (:obj:`numpy.ndarray`):
            Array to be saved.

        bytes_type (class, optional, default=bytes):
            This class will be used to wrap the bytes objects in the
            serialization if `use_bytes` is true. Useful for when using
            Python 2 and using BSON encoding, which will not accept the raw
            `bytes` type, so `bson.Binary` can be used instead.


    Returns:
        bytes_type

    """
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bytes_type(bio.getvalue())


def bytes2array(bytes):
    """Inverse of :func:`array2bytes`"""
    return np.load(io.BytesIO(bytes))
