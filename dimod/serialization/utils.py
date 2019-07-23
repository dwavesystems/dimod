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
from numbers import Integral, Number

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from six import string_types


def serialize_ndarray(arr, use_bytes=False, bytes_type=bytes):
    """Serializes a NumPy array.

    Args:
        arr (array-like):
            An array.

        use_bytes (bool, optional, default=False):
            If True, a compact representation of the biases as bytes is used.

        bytes_type (class, optional, default=bytes):
            If `use_bytes` is True, this class is used to wrap the bytes
            objects in the serialization. Useful for Python 2 using BSON
            encoding, which does not accept the raw `bytes` type;
            `bson.Binary` can be used instead.

    Returns:
        dict: A serializable object.

    """
    arr = np.asarray(arr)  # support array-like
    if use_bytes:
        data = bytes_type(arr.tobytes(order='C'))
    else:
        data = arr.tolist()
    return dict(type='array',
                data=data,
                data_type=arr.dtype.name,
                shape=arr.shape,
                use_bytes=bool(use_bytes))


def deserialize_ndarray(obj):
    """Inverse of serialize_ndarray.

    Args:
        obj (dict):
            As constructed by :func:`.serialize_ndarray`.

    Returns:
        :obj:`numpy.ndarray`

    """
    if obj['use_bytes']:
        arr = np.frombuffer(obj['data'], dtype=obj['data_type'])
    else:
        arr = np.asarray(obj['data'], dtype=obj['data_type'])
    arr = arr.reshape(obj['shape'])  # makes a view generally, but that's fine
    return arr


def serialize_ndarrays(obj, use_bytes=False, bytes_type=bytes):
    """Looks through the object, serializing numpy arrays.

    Developer note: this function was written for serializing info fields
    in the sample set and binary quadratic model objects. This is not a general
    serialization function.

    Notes:
        Lists and dicts are copies in the returned object. Does not attempt to
        only copy-on-write, even though that would be more performant.

        Does not check for recursive references.

    """
    if isinstance(obj, np.ndarray):
        return serialize_ndarray(obj, use_bytes=use_bytes, bytes_type=bytes_type)
    elif isinstance(obj, abc.Mapping):
        return {serialize_ndarrays(key): serialize_ndarrays(val)
                for key, val in obj.items()}
    elif isinstance(obj, abc.Sequence) and not isinstance(obj, string_types):
        return list(map(serialize_ndarrays, obj))
    if isinstance(obj, Integral):
        return int(obj)
    elif isinstance(obj, Number):
        return float(obj)
    return obj


def deserialize_ndarrays(obj):
    """Inverse of dfs_serialize_ndarray."""
    if isinstance(obj, abc.Mapping):
        if obj.get('type', '') == 'array':
            return deserialize_ndarray(obj)
        return {key: deserialize_ndarrays(val) for key, val in obj.items()}
    elif isinstance(obj, abc.Sequence) and not isinstance(obj, string_types):
        return list(map(deserialize_ndarrays, obj))
    return obj


def pack_samples(states):
    # ensure that they are stored big-endian order
    if not states.size:
        return np.empty(states.shape, dtype=np.uint32)
    pad_len = 31 - (states.shape[-1]+31) % 32
    pad_sizes = ((0, 0),)*(states.ndim - 1) + ((0, pad_len),)
    shape = states.shape[:-1] + (-1, 4, 8)
    padded = np.pad(states, pad_sizes, "constant").reshape(shape)[..., ::-1]
    return np.packbits(padded).view(np.uint32).reshape(*(states.shape[:-1]+(-1,)))


def unpack_samples(packed, n, dtype=np.uint32):
    if not packed.size:
        return np.empty((packed.shape[0], n), dtype=dtype)
    bytewise_shape = packed.shape[:-1] + (-1, 4, 8)
    unpacked = np.unpackbits(packed.view(np.uint8)).reshape(bytewise_shape)
    unpacked = unpacked[..., ::-1].reshape(packed.shape[:-1] + (-1,))
    return unpacked[..., :n].astype(dtype)
