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

from typing import Collection, Union

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.typing import Variable, VartypeLike
from dimod.vartypes import BINARY

__all__ = ['combinations']


def combinations(n: Union[int, Collection[Variable]], k: int,
                 strength: float = 1, vartype: VartypeLike = BINARY) -> BinaryQuadraticModel:
    r"""Generate a binary quadratic model that is minimized when ``k`` of ``n``
    variables are selected.

    More fully, generates a binary quadratic model (BQM) that is minimized for
    each of the ``k``-combinations of its variables.

    The energy for the BQM is given by
    :math:`(\sum_{i} x_i - k)^2`.

    Args:
        n: Variable labels. If ``n`` is an integer, variables are labelled `[0, n)`.

        k: The number of selected variables (variables assigned value `1`) that
            minimizes the generated BQM, resulting in an energy of `0`.

        strength: Energy of the first excited state of the BQM. The first excited
            state occurs when the number of selected variables is :math:`k \pm 1`.

        vartype: Variable type for the BQM. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Returns:
        A binary quadratic model.

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
    if isinstance(n, Collection):
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
    #     = \sum_{i,j} x_ix_j + (1 - 2k)\sum_i x_i + k^2
    lbias = float(strength*(1 - 2*k))
    qbias = float(2*strength)

    if not isinstance(n, int):
        num_vars = len(n)
        variables = n
    else:
        num_vars = n
        try:
            variables = range(n)
        except TypeError:
            raise TypeError('n should be a collection or an integer')

    Q = np.triu(np.ones((num_vars, num_vars))*qbias, k=1)
    np.fill_diagonal(Q, lbias)
    bqm = BinaryQuadraticModel.from_qubo(Q, offset=strength*(k**2))

    if not isinstance(n, int):
        bqm.relabel_variables(dict(zip(range(len(n)), n)))

    return bqm.change_vartype(vartype, inplace=True)
