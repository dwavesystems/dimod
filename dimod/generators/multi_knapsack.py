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

import warnings

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel
from typing import Tuple

__all__ = ['multi_knapsack', 'random_multi_knapsack']


def random_multi_knapsack(num_items: int,
                          num_bins: int,
                          seed: int = 32,
                          value_range: Tuple[int, int] = (10, 50),
                          weight_range: Tuple[int, int] = (10, 50),
                          ) -> ConstrainedQuadraticModel:
    """Return a constrained quadratic model encoding a multiple knapsack
    problem.

    Given the number of items and the number of bins, the code generates a
    multiple-knapsack problem, formulated as a Constrained Quadratic Model.
    Values and weights  for each item are uniformly sampled within the provided
    ranges. Capacities of bins are randomly assigned.

    Args:
        num_items: Number of items.

        num_bins: Number of bins.

        seed: seed for RNG.

        value_range: The range of the randomly generated values for each item.

        weight_range: The range of the randomly generated weights for each item.

    Returns:

        A constrained quadratic model encoding the multiple knapsack problem.
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


# We want to use multi_knapsack in the future for problems with specified weights/
# values, so we'll deprecate it and use the more explicit random_knapsack.
# Once the deprecation period is over we can use the multi_knapsack with a different
# api.
def multi_knapsack(*args, **kwargs) -> ConstrainedQuadraticModel:
    warnings.warn("multi_knapsack was deprecated after 0.10.6 and will be removed in 0.11.0, "
                  "use random_bin_packing instead.",
                  DeprecationWarning,
                  stacklevel=2)
    return random_multi_knapsack(*args, **kwargs)
