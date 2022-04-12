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

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel
from typing import Tuple

__all__ = ['random_multi_knapsack']


def random_multi_knapsack(num_items: int,
                          num_bins: int,
                          seed: int = 32,
                          value_range: Tuple[int, int] = (10, 50),
                          weight_range: Tuple[int, int] = (10, 50),
                          ) -> ConstrainedQuadraticModel:
    """Generate a constrained quadratic model encoding a multiple-knapsack
    problem.

    Given the number of items and the number of bins, generates a
    multiple-knapsack problem, formulated as a :class:`~dimod.ConstrainedQuadraticModel`.
    Values and weights for each item are uniformly sampled within the specified
    ranges. Capacities of bins are randomly assigned.

    Args:
        num_items: Number of items.

        num_bins: Number of bins.

        seed: Seed for RNG.

        value_range: Range of the randomly generated values for each item.

        weight_range: Range of the randomly generated weights for each item.

    Returns:

        A constrained quadratic model encoding the multiple-knapsack problem.
        Variables are labelled as ``x_{i}_{j}``, where ``x_{i}_{j} == 1`` means
        that item ``i`` is placed in bin ``j``.

    """

    rng = np.random.RandomState(seed)

    weights = rng.randint(*weight_range, num_items)
    values = rng.randint(*value_range, num_items)

    cap_low = int(weight_range[0] * num_items / num_bins)
    cap_high = int(weight_range[1] * num_items / num_bins)
    capacities = rng.randint(cap_low, cap_high, num_bins)

    model = ConstrainedQuadraticModel()

    obj = BinaryQuadraticModel(vartype='BINARY')
    x = {(i, j): obj.add_variable(f'x_{i}_{j}') for i in range(num_items) for j in range(num_bins)}

    for i in range(num_items):
        for j in range(num_bins):
            obj.set_linear(x[(i, j)], -values[i])

    model.set_objective(obj)

    # Each item at most goes to one bin.
    for i in range(num_items):
        model.add_constraint([(x[(i, j)], 1) for j in range(num_bins)] + [(-1,)], sense="<=",
                             label='item_placing_{}'.format(i))

    # Build knapsack capacity constraints
    for j in range(num_bins):
        model.add_constraint(
            [(x[(i, j)], weights[i]) for i in range(num_items)] + [(-capacities[j],)],
            sense="<=", label='capacity_bin_{}'.format(j))

    return model
