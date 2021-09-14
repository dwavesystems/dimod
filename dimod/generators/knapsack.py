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

__all__ = ['knapsack', 'random_knapsack']


def random_knapsack(num_items: int,
                    seed: int = 32,
                    value_range: Tuple[int, int] = (10, 30),
                    weight_range: Tuple[int, int] = (10, 30),
                    tightness_ratio: float = 0.5,
                    ) -> ConstrainedQuadraticModel:
    """Returns a Constrained Quadratic Model encoding a knapsack problem.

    Given the number of items, the code generates a random knapsack problem,
    formulated as a Constrained Quadratic model. The capacity of the bin is set
    to be ``tightness_ratio`` times the sum of the weights.

    Args:
        num_items: Number of items to choose from.

        seed: Seed for numpy random number generator.

        value_range: The range of the randomly generated values for each item.

        weight_range: The range of the randomly generated weights for each item.

        tightness_ratio: ratio of capacity over sum of weights.

    Returns:
        The quadratic model encoding the knapsack problem. Variables are
        denoted as ``x_{i}`` where ``x_{i} == 1`` means that the item ``i`` has
        been placed in the knapsack.

    """

    rng = np.random.RandomState(seed)

    value = {i: rng.randint(*value_range) for i in range(num_items)}
    weight = {i: rng.randint(*weight_range) for i in range(num_items)}
    capacity = int(sum(weight.values()) * tightness_ratio)

    model = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    x = {i: obj.add_variable(f'x_{i}') for i in range(num_items)}

    for i in range(num_items):
        obj.set_linear(x[i], -value[i])

    model.set_objective(obj)
    constraint = [(x[i], weight[i]) for i in range(num_items)] + [(-capacity, )]
    model.add_constraint(constraint, sense="<=", label='capacity')

    return model


# We want to use knapsack in the future for problems with specified weights/
# values, so we'll deprecate it and use the more explicit random_knapsack.
# Once the deprecation period is over we can use the knapsack with a different
# api.
def knapsack(*args, **kwargs) -> ConstrainedQuadraticModel:
    warnings.warn("knapsack was deprecated after 0.10.6 and will be removed in 0.11.0, "
                  "use random_bin_packing instead.",
                  DeprecationWarning,
                  stacklevel=2)
    return random_knapsack(*args, **kwargs)
