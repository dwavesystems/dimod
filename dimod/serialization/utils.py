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
import numpy as np


def pack_samples(states):
    # ensure that they are stored big-endian order
    if not states.size:
        return np.empty(states.shape, dtype=np.uint32)
    pad_len = 31 - (states.shape[-1]+31) % 32
    pad_sizes = ((0, 0),)*(states.ndim - 1) + ((0, pad_len),)
    shape = states.shape[:-1] + (-1, 4, 8)
    padded = np.pad(states, pad_sizes, "constant").reshape(shape)[..., ::-1]
    return np.packbits(padded).view(np.uint32).reshape(*states.shape[:-1], -1)


def unpack_samples(packed, n, dtype=np.uint32):
    if not packed.size:
        return np.empty((packed.shape[0], n), dtype=dtype)
    bytewise_shape = packed.shape[:-1] + (-1, 4, 8)
    unpacked = np.unpackbits(packed.view(np.uint8)).reshape(bytewise_shape)
    unpacked = unpacked[..., ::-1].reshape(packed.shape[:-1] + (-1,))
    return unpacked[..., :n].astype(dtype)
