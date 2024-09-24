# Copyright 2024 D-Wave
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
from typing import Tuple
from dimod.typing import ArrayLike

__all__ = ['quadratic_knapsack', 'random_quadratic_knapsack']


def quadratic_knapsack(
    values: ArrayLike,
    weights: ArrayLike,
    profits: ArrayLike,
    capacity: float,
) -> ConstrainedQuadraticModel:
    """Generates a constrained quadratic model encoding a quadratic knapsack problem.

    The quadratic knapsack problem,
    `QKP <https://en.wikipedia.org/wiki/Quadratic_knapsack_problem>`_,
    seeks to fit the most value into a knapsack of weight less than or equal to
    capacity for a given list of items with associated values and weights.

    Args:
        values: A list of each item's value.
        weights: A list of each item's associated weight.
        profits: A matrix where entry (i, j) is the value of adding items i and j together.
        capacity: The maximum weight a knapsack can hold.

    Returns:
        A constrained quadratic model encoding the quadratic knapsack problem.
        Variables are denoted as ``x_{i}`` where ``x_{i} == 1`` means that item ``i`` is
        placed in the knapsack.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    profits = np.asarray(profits)

    if values.shape != weights.shape:
        raise ValueError("`values` and `weights` must have the same shape")

    if not np.array_equal(profits, profits.T):
        raise ValueError("`profits` must be symmetric")

    if values.shape[0] != profits.shape[0]:
        raise ValueError("`profits` must have an entry for each pair of items")

    model = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    x = [obj.add_variable(f'x_{i}') for i in range(profits.shape[0])]

    for i, value in enumerate(values):
        obj.set_linear(x[i], -value)

    for i, profit in np.ndenumerate(profits):
        if i[0] < i[1]:
            obj.set_quadratic(x[i[0]], x[i[1]], -profit)

    model.set_objective(obj)
    constraint = [(x[i], weight) for i, weight in enumerate(weights)] + [(-capacity, )]
    model.add_constraint(constraint, sense='<=', label='capacity')

    return model


def random_quadratic_knapsack(
    num_items: int,
    seed: typing.Optional[int] = None,
    value_range: Tuple[int, int] = (10, 50),
    weight_range: Tuple[int, int] = (10, 50),
    profits_range: Tuple[int, int] = (10, 50),
    tightness_ratio: float = 0.5,
) -> ConstrainedQuadraticModel:
    """Generate a constrained quadratic model encoding a quadratic knapsack
    problem.

    Given the number of items and the number of bins, generates a
    quadratic knapsack problem, formulated as a :class:`~dimod.ConstrainedQuadraticModel`.
    Values, weights, and profits are uniformly sampled within the specified
    ranges. The capacity of bins is set to be ``tightness_ratio`` times the sum of the weights.

    Args:
        num_items: Number of items to choose from.
        seed: Seed for NumPy random number generator.
        value_range: Range of the randomly generated values for each item.
        weight_range: Range of the randomly generated weights for each item.
        profits_range: Range of the randomly generated profits for each pair.
        tightness_ratio: Ratio of capacity over sum of weights.

    Returns:
        A constrained quadratic model encoding the quadratic knapsack problem.
        Variables are denoted as ``x_{i}`` where ``x_{i} == 1`` means that item
        ``i`` is placed in the knapsack.
    """

    rng = np.random.default_rng(seed)

    values = rng.integers(*value_range, num_items)
    weights = rng.integers(*weight_range, num_items)
    profits = np.triu(rng.integers(*profits_range, (num_items, num_items)))
    profits = profits + profits.T - np.diag(np.diag(profits))

    capacity = int(np.sum(weights) * tightness_ratio)

    model = quadratic_knapsack(weights, values, profits, capacity)

    return model
