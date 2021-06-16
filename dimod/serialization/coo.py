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
# =============================================================================
"""
A simple text encoding of dimod BQMs.

The COOrdinate_ list is a sparse matrix representation which can be used to
store binary quadratic models. This format is best used when readability is
important.

Note that this format only works for BQMs that are labelled with positive
integers.

.. _COOrdinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

Examples:

    >>> from dimod.serialization import coo

    Serialize a QUBO.

    >>> Q = {(0, 0): -1, (0, 1): 1, (1, 2): -4.5}
    >>> bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    >>> print(coo.dumps(bqm))
    0 0 -1.000000
    0 1 1.000000
    1 2 -4.500000

    We can also include the vartype as a header.

    >>> Q = {(0, 0): -1, (0, 1): 1, (1, 2): -4.5}
    >>> bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    >>> print(coo.dumps(bqm, vartype_header=True))
    # vartype=BINARY
    0 0 -1.000000
    0 1 1.000000
    1 2 -4.500000

    Loading from a COO string. Note that you must specify a vartype.

    >>> coo_string = '''
    ... 0 0 -1
    ... 0 1 1
    ... 1 2 -4.5
    ... '''
    >>> coo.loads(coo_string, vartype=dimod.BINARY)
    BinaryQuadraticModel({0: -1.0, 1: 0.0, 2: 0.0}, {(1, 0): 1.0, (2, 1): -4.5}, 0.0, 'BINARY')

    Or provide the vartype as a header.

    >>> coo_string = '''
    ... # vartype=BINARY
    ... 0 0 -1
    ... 0 1 1
    ... 1 2 -4.5
    ... '''
    >>> coo.loads(coo_string)
    BinaryQuadraticModel({0: -1.0, 1: 0.0, 2: 0.0}, {(1, 0): 1.0, (2, 1): -4.5}, 0.0, 'BINARY')

"""

import re

from numbers import Integral

from dimod import Vartype
from dimod.binary_quadratic_model import BinaryQuadraticModel

_LINE_REGEX = r'^\s*(\d+)\s+(\d+)\s+([+-]?\d*(?:\.\d+)?)\s*$'
"""
Each line should look like

0 1 2.000000

where 0, 1 are the variable labels and 2.0 is the bias.
"""

_VARTYPE_HEADER_REGEX = r'^[ \t\f]*#.*?vartype[:=][ \t]*([-_.a-zA-Z0-9]+)'
"""
The header should be in the first line and look like

# vartype=SPIN

"""


def dumps(bqm, vartype_header=False):
    """Dump a binary quadratic model to a string in COOrdinate format."""
    return '\n'.join(_iter_triplets(bqm, vartype_header))


def dump(bqm, fp, vartype_header=False):
    """Dump a binary quadratic model to a file in COOrdinate format."""
    for triplet in _iter_triplets(bqm, vartype_header):
        fp.write('%s\n' % triplet)


def loads(s, cls=BinaryQuadraticModel, vartype=None):
    """Load a COOrdinate formatted binary quadratic model from a string."""
    return load(s.split('\n'), cls=cls, vartype=vartype)


def load(fp, cls=BinaryQuadraticModel, vartype=None):
    """Load a COOrdinate formatted binary quadratic model from a file."""
    pattern = re.compile(_LINE_REGEX)
    vartype_pattern = re.compile(_VARTYPE_HEADER_REGEX)

    triplets = []
    for line in fp:
        triplets.extend(pattern.findall(line))

        vt = vartype_pattern.findall(line)
        if vt:
            if vartype is None:
                vartype = vt[0]
            else:
                if isinstance(vartype, str):
                    vartype = Vartype[vartype]
                else:
                    vartype = Vartype(vartype)
                if Vartype[vt[0]] != vartype:
                    raise ValueError("vartypes from headers and/or inputs do not match")

    if vartype is None:
        raise ValueError("vartype must be provided either as a header or as an argument")

    bqm = cls.empty(vartype)

    for u, v, bias in triplets:
        if u == v:
            bqm.add_variable(int(u), float(bias))
        else:
            bqm.add_interaction(int(u), int(v), float(bias))

    return bqm


def _iter_triplets(bqm, vartype_header):
    if not all(isinstance(v, Integral) and v >= 0 for v in bqm.linear):
        raise ValueError("only positive index-labeled binary quadratic models can be dumped to COOrdinate format")

    if vartype_header:
        yield '# vartype=%s' % bqm.vartype.name

    # developer note: we could (for some threshold sparseness) sort the neighborhoods,
    # but this is simple and probably sufficient

    variables = sorted(bqm.variables)  # integer labeled so we can sort in py3

    for idx, u in enumerate(variables):
        for v in variables[idx:]:
            if u == v and bqm.linear[u]:
                yield '%d %d %f' % (u, u, bqm.linear[u])
            elif u in bqm.adj[v]:
                yield '%d %d %f' % (u, v, bqm.adj[u][v])
