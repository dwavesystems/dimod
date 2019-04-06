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
import itertools

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument, vartype_argument
from dimod.vartypes import BINARY

__all__ = 'combinations',


def combinations(n, k, strength=1, vartype=BINARY):
    r"""Generate a bqm that is minimized when k of n variables are selected.

    More fully, we wish to generate a binary quadratic model which is minimized
    for each of the k-combinations of its variables.

    The energy for the binary quadratic model is given by
    :math:`(\sum_{i} x_i - k)^2`.

    Args:
        n (int/list/set):
            If n is an integer, variables are labelled [0, n-1]. If n is list or
            set then the variables are labelled accordingly.

        k (int):
            The generated binary quadratic model will have 0 energy when any k
            of the variables are 1.

        strength (number, optional, default=1):
            The energy of the first excited state of the binary quadratic model.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Returns:
        :obj:`.BinaryQuadraticModel`

    Examples:

        >>> bqm = dimod.generators.combinations(['a', 'b', 'c'], 2)
        >>> bqm.energy({'a': 1, 'b': 0, 'c': 1})
        0.0
        >>> bqm.energy({'a': 1, 'b': 1, 'c': 1})
        1.0

        >>> bqm = dimod.generators.combinations(5, 1)
        >>> bqm.energy({0: 0, 1: 0, 2: 1, 3: 0, 4: 0})
        0.0
        >>> bqm.energy({0: 0, 1: 0, 2: 1, 3: 1, 4: 0})
        1.0

        >>> bqm = dimod.generators.combinations(['a', 'b', 'c'], 2, strength=3.0)
        >>> bqm.energy({'a': 1, 'b': 0, 'c': 1})
        0.0
        >>> bqm.energy({'a': 1, 'b': 1, 'c': 1})
        3.0

    """
    if isinstance(n, abc.Sized) and isinstance(n, abc.Iterable):
        # what we actually want is abc.Collection but that doesn't exist in
        # python2
        variables = n
    else:
        try:
            variables = range(n)
        except TypeError:
            raise TypeError('n should be a collection or an integer')

    if k > len(variables) or k < 0:
        raise ValueError("cannot select k={} from {} variables".format(k, len(variables)))

    # (\sum_i x_i - k)^2
    #     = \sum_i x_i \sum_j x_j - 2k\sum_i x_i + k^2
    #     = \sum_i,j x_ix_j + (1 - 2k)\sim_i x_i + k^2
    lbias = float(strength*(1 - 2*k))
    qbias = float(2*strength)

    bqm = BinaryQuadraticModel.empty(vartype)
    bqm.add_variables_from(((v, lbias) for v in variables), vartype=BINARY)
    bqm.add_interactions_from(((u, v, qbias)
                               for u, v in itertools.combinations(variables, 2)),
                              vartype=BINARY)
    bqm.add_offset(strength*(k**2))

    return bqm
