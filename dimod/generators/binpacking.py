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

import typing

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel
from dimod.vartypes import BINARY

__all__ = ['random_bin_packing']


def random_bin_packing(num_items: int,
                       seed: typing.Optional[int] = None,
                       weight_range: typing.Tuple[int, int] = (10, 30),
                       ) -> ConstrainedQuadraticModel:
    """Generate a bin packing problem as a constrained quadratic model.

    The weights for each item are integers uniformly drawn from in the
    ``weight_range``. The bin capacity is set to ``num_items * mean(weights) / 5``.

    Args:
        num_items: Number of items to choose from.

        seed: Seed for NumPy random number generator.

        weight_range: The range of the randomly generated weights for each item.

    Returns:
        The constrained quadratic model encoding the bin packing problem.
        Variables are labeled as ``y_{j}`` where ``y_{j} == 1`` means that bin
        ``j`` has been used and ``x_{i}_{j}`` where ``x_{i}_{j} == 1`` means
        that item ``i`` has been placed in bin ``j``.

    """

    rng = np.random.RandomState(seed)

    max_num_bins = num_items
    weights = list(rng.randint(*weight_range, num_items))
    bin_capacity = int(num_items * np.mean(weights) / 5)
    model = ConstrainedQuadraticModel()

    obj = BinaryQuadraticModel(BINARY)
    y = {j: obj.add_variable(f'y_{j}') for j in range(max_num_bins)}

    for j in range(max_num_bins):
        obj.set_linear(y[j], 1)

    model.set_objective(obj)

    x = {(i, j): model.add_variable(BINARY, f'x_{i}_{j}') for i in range(num_items) for
         j in range(max_num_bins)}

    # Each item goes to one bin
    for i in range(num_items):
        model.add_constraint([(x[(i, j)], 1) for j in range(max_num_bins)] + [(-1,)], sense="==",
                             label='item_placing_{}'.format(i))

    # Bin capacity constraint
    for j in range(max_num_bins):
        model.add_constraint(
            [(x[(i, j)], weights[i]) for i in range(num_items)] + [(y[j], -bin_capacity)],
            sense="<=", label='capacity_bin_{}'.format(j))

    return model
