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

__all__ = ['random_knapsack']


def random_knapsack(num_items: int,
                    seed: typing.Optional[int] = None,
                    value_range: typing.Tuple[int, int] = (10, 30),
                    weight_range: typing.Tuple[int, int] = (10, 30),
                    tightness_ratio: float = 0.5,
                    ) -> ConstrainedQuadraticModel:
    """Generates a constrained quadratic model encoding a knapsack problem.

    Given the number of items, generates a random knapsack problem, formulated as
    a :class:`~dimod.ConstrainedQuadraticModel`. The capacity of bins is set
    to be ``tightness_ratio`` times the sum of the weights.

    Args:
        num_items: Number of items to choose from.

        seed: Seed for NumPy random number generator.

        value_range: Range of the randomly generated values for each item.

        weight_range: Range of the randomly generated weights for each item.

        tightness_ratio: Ratio of capacity over sum of weights.

    Returns:
        The quadratic model encoding the knapsack problem. Variables are
        denoted as ``x_{i}`` where ``x_{i} == 1`` means that item ``i`` is
        placed in the knapsack.

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
