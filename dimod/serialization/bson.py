# Copyright 2018 D-Wave Systems Inc.
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
# ================================================================================================
import itertools

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel


def bqm_bson_encoder(bqm):
    """todo"""
    num_variables = len(bqm)

    index_dtype = np.uint32
    if num_variables <= 2**16:
        index_dtype = np.uint16

    variable_order = sorted(bqm.linear)
    num_possible_edges = num_variables*(num_variables - 1) // 2
    density = len(bqm.quadratic) / num_possible_edges
    as_complete = density >= 0.5

    lin, (i, j, _vals), off = bqm.to_numpy_vectors(
        dtype=np.float32,
        index_dtype=index_dtype,
        sort_indices=as_complete,
        variable_order=variable_order)

    if as_complete:
        vals = np.zeros(num_possible_edges, dtype=np.float32)

        def mul(a, b):
            return np.multiply(a, b, dtype=np.int64)

        edge_idxs = (mul(i, num_variables - 1) - mul(i, (i+1)//2) -
                     ((i+1) % 2)*(i//2) + j - 1)
        vals[edge_idxs] = _vals

    else:
        vals = _vals

    doc = {
        "as_complete": as_complete,
        "linear": lin.tobytes(),
        "quadratic_vals": vals.tobytes(),
        "variable_type": "SPIN" if bqm.vartype == bqm.SPIN else "BINARY",
        "offset": off,
        "variable_order": variable_order,
        "index_dtype": np.dtype(index_dtype).str,
    }

    if not as_complete:
        doc["quadratic_head"] = i.tobytes()
        doc["quadratic_tail"] = j.tobytes()

    return doc


def bqm_bson_decoder(doc, cls=BinaryQuadraticModel):
    lin = np.frombuffer(doc["linear"], dtype=np.float32)
    num_variables = len(lin)
    vals = np.frombuffer(doc["quadratic_vals"], dtype=np.float32)
    index_dtype = doc["index_dtype"]
    if doc["as_complete"]:
        i, j = zip(*itertools.combinations(range(num_variables), 2))
    else:
        i = np.frombuffer(doc["quadratic_head"], dtype=index_dtype)
        j = np.frombuffer(doc["quadratic_tail"], dtype=index_dtype)

    off = doc["offset"]

    return cls.from_numpy_vectors(lin, (i, j, vals), off,
                                  str(doc["variable_type"]),
                                  variable_order=doc["variable_order"])
