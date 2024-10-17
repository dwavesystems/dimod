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
from dimod.typing import ArrayLike

__all__ = ['knapsack', 'quadratic_knapsack', 'random_knapsack']


def knapsack(values: ArrayLike,
            weights: ArrayLike,
            capacity: float,
            ) -> ConstrainedQuadraticModel:
    """Generates a constrained quadratic model encoding a knapsack problem.

    The knapsack problem,
    `KP <https://en.wikipedia.org/wiki/Knapsack_problem>`_,
    seeks to fit the most value into a knapsack of weight less than or equal to
    capacity for a given list of items with associated values and weights.

    Args:
        values: A list of each item's value.
        weights: A list of each item's associated weight.
        capacity: The maximum weight a knapsack can hold.

    Returns:
        The quadratic model encoding the knapsack problem. Variables are
        denoted as ``x_{i}`` where ``x_{i} == 1`` means that item ``i`` is
        placed in the knapsack.

    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if values.shape != weights.shape:
        raise ValueError("`values` and `weights` must have the same shape")

    model = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    x = [obj.add_variable(f'x_{i}') for i in range(values.shape[0])]

    for i, value in enumerate(values):
        obj.set_linear(x[i], -value)

    model.set_objective(obj)
    constraint = [(x[i], weight) for i, weight in enumerate(weights)] + [(-capacity, )]
    model.add_constraint(constraint, sense="<=", label='capacity')

    return model


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


def random_knapsack(num_items: int,
                    seed: typing.Optional[int] = None,
                    value_range: tuple[int, int] = (10, 30),
                    weight_range: tuple[int, int] = (10, 30),
                    tightness_ratio: float = 0.5,
                    ) -> ConstrainedQuadraticModel:
    """Generates a constrained quadratic model encoding a random knapsack problem.

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

    rng = np.random.default_rng(seed)

    values = list(rng.integers(*value_range, num_items))
    weights = list(rng.integers(*weight_range, num_items))
    capacity = int(np.sum(weights) * tightness_ratio)

    model = knapsack(values, weights, capacity)

    return model
