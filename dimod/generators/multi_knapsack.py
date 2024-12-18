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

from typing import Tuple

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel
from dimod.typing import ArrayLike

__all__ = ['multi_knapsack', 'quadratic_multi_knapsack', 'random_multi_knapsack']


def multi_knapsack(values: ArrayLike,
                   weights: ArrayLike,
                   capacities: ArrayLike,
                   ) -> ConstrainedQuadraticModel:
    """Generate a constrained quadratic model encoding a multiple knapsack problem.

    The multiple knapsack problem seeks to fit the most value into each knapsack
    of weight less than or equal to each knapsack's capacity for a given list of
    items with associated values and weights.

    Args:
        values: A list of each item's value.
        weights: A list of each item's associated weight.
        capacities: A list of the maximum weights each knapsack can hold.

    Returns:

        A constrained quadratic model encoding the multiple-knapsack problem.
        Variables are labelled as ``x_{i}_{j}``, where ``x_{i}_{j} == 1`` means
        that item ``i`` is placed in knapsack ``j``.

    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if values.shape != weights.shape:
        raise ValueError("`values` and `weights` must have the same shape")

    model = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    x = {(i, j): obj.add_variable(f'x_{i}_{j}') for i in range(values.shape[0]) for j in range(len(capacities))}

    for i, value in enumerate(values):
        for j in range(len(capacities)):
            obj.set_linear(x[(i, j)], -value)

    model.set_objective(obj)

    # Each item at most goes to one bin.
    for i in range(values.shape[0]):
        model.add_constraint(
            [(x[(i, j)], 1) for j in range(len(capacities))] + [(-1,)],
            sense="<=", label='item_placing_{}'.format(i))

    # Build knapsack capacity constraints
    for j, capacity in enumerate(capacities):
        model.add_constraint(
            [(x[(i, j)], weight) for i, weight in enumerate(weights)] + [(-capacity,)],
            sense="<=", label='capacity_bin_{}'.format(j))

    return model


def quadratic_multi_knapsack(values: ArrayLike,
                             weights: ArrayLike,
                             profits: ArrayLike,
                             capacities: ArrayLike,
                             ) -> ConstrainedQuadraticModel:
    """Generate a constrained quadratic model encoding a quadratic multiple
    knapsack problem.

    The quadratic multiple knapsack problem seeks to fit the most value into each
    knapsack of weight less than or equal to each knapsack's capacity and maximize
    profits associated with adding any two items to the same knapsack.

    Args:
        values: A list of each item's value.
        weights: A list of each item's associated weight.
        profits: A matrix where entry (i, j) is the value of adding items i and j together.
        capacities: A list of the maximum weights each knapsack can hold.

    Returns:

        A constrained quadratic model encoding the quadratic multiple knapsack problem.
        Variables are labelled as ``x_{i}_{j}``, where ``x_{i}_{j} == 1`` means
        that item ``i`` is placed in knapsack ``j``.

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
    x = {(i, j): obj.add_variable(f'x_{i}_{j}') for i in range(values.shape[0]) for j in range(len(capacities))}

    for i, value in enumerate(values):
        for j in range(len(capacities)):
            obj.set_linear(x[(i, j)], -value)

    for i, profit in np.ndenumerate(profits):
        if i[0] < i[1]:
            for j in range(len(capacities)):
                obj.set_quadratic(x[i[0], j], x[i[1], j], -profit)

    model.set_objective(obj)

    # Each item at most goes to one bin.
    for i in range(values.shape[0]):
        model.add_constraint(
            [(x[(i, j)], 1) for j in range(len(capacities))] + [(-1,)],
            sense="<=", label='item_placing_{}'.format(i))

    # Build knapsack capacity constraints
    for j, capacity in enumerate(capacities):
        model.add_constraint(
            [(x[(i, j)], weight) for i, weight in enumerate(weights)] + [(-capacity,)],
            sense="<=", label='capacity_bin_{}'.format(j))

    return model


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
        seed: Seed for random number generator.
        value_range: Range of the randomly generated values for each item.
        weight_range: Range of the randomly generated weights for each item.

    Returns:

        A constrained quadratic model encoding the multiple-knapsack problem.
        Variables are labelled as ``x_{i}_{j}``, where ``x_{i}_{j} == 1`` means
        that item ``i`` is placed in knapsack ``j``.

    """

    rng = np.random.default_rng(seed)

    values = rng.integers(*value_range, num_items)
    weights = rng.integers(*weight_range, num_items)

    cap_low = int(weight_range[0] * num_items / num_bins)
    cap_high = int(weight_range[1] * num_items / num_bins)
    capacities = rng.integers(cap_low, cap_high, num_bins)

    model = multi_knapsack(values, weights, capacities)

    return model
