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

import math

from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
from dimod.typing import Variable
from dimod.vartypes import Vartype

__all__ = ['binary_encoding']


def binary_encoding(v: Variable, upper_bound: int) -> BinaryQuadraticModel:
    """Return a binary quadratic model encoding an integer.

    Args:
        v: The integer variable label.
        upper_bound: The upper bound on the integer value (inclusive).

    Returns:
        A binary quadratic model. The variables in the BQM will be labelled
        with tuples of length two or three. The first value of the tuple will
        be the variable label ``v`` provided. The second value will be the
        coefficient in the integer encoding. One of the variables will
        have a third value in the tuple, ``'msb'``. This is the variable
        occupying the position of the most significant bit. Though it may
        actually be a smaller number in order to enforce the ``upper_bound``.

    Example:

        >>> bqm = dimod.generators.binary_encoding('i', 6)
        >>> bqm
        BinaryQuadraticModel({('i', 1): 1.0, ('i', 2): 2.0, ('i', 3, 'msb'): 3.0}, {}, 0.0, 'BINARY')

        We can use a sample to restore the original integer value.

        >>> sample = {('i', 1): 1, ('i', 2): 0, ('i', 3, 'msb'): 1}
        >>> bqm.energy(sample)
        4.0
        >>> sum(v[1]*val for v, val in sample.items()) + bqm.offset
        4.0

        If you wish to encode integers with a lower bound, you can use the
        binary quadratic model's :attr:`~BinaryQuadraticModel.offset` attribute.

        >>> i = dimod.generators.binary_encoding('i', 10) + 5  # integer in [5, 15]

    References:
        [1]: Sahar Karimi, Pooya Ronagh (2017), Practical Integer-to-Binary
        Mapping for Quantum Annealers. arxiv.org:1706.01945.

    """
    # note: the paper above also gives a nice way to handle bounded coefficients
    # if we want to do that in the future.

    if upper_bound <= 1:
        raise ValueError("upper_bound must be greater than or equal to 1, "
                         f"received {upper_bound}")
    upper_bound = math.floor(upper_bound)

    bqm = BinaryQuadraticModel(Vartype.BINARY)

    max_pow = math.floor(math.log2(upper_bound))
    for exp in range(max_pow):
        val = 1 << exp
        bqm.set_linear((v, val), val)
    else:
        val = upper_bound - ((1 << max_pow) - 1)
        bqm.set_linear((v, val, 'msb'), val)

    return bqm
