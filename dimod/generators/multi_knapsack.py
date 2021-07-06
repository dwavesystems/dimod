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
#
# =============================================================================

from dimod import BQM, CQM
import numpy as np


__all__ = ['multi_knapsack']


def multi_knapsack(num_items, num_bins, seed=32, value_range=(10, 50), weight_range=(10, 50)):
    """
    Given the number of items and the number of bins, the code generates a multiple-knapsack
    problem, formulated as a Constrained Quadratic model.

    Args:
        num_items (int):
            Number of items

        num_bins (int):
            Number of bins

        seed (int, optional, default=32):
            Seed of random number generator

        value_range (tuple, optional, default=(10, 50)):
            The range of the randomly generated values for each item

        weight_range (tuple, optional, default=(10, 50)):
            The range of the randomly generated weights for each item

    Returns:
        :obj:`.ConstrainedQuadraticModel`.

    """
    np.random.seed(seed)
    items = list(range(num_items))
    weights = np.random.randint(*weight_range, num_items)
    values = np.random.randint(*value_range, num_items)
    bins = list(range(num_bins))

    cap_low = int(weight_range[0] * num_items / num_bins)
    cap_high = int(weight_range[1] * num_items / num_bins)
    capacities = np.random.randint(cap_low, cap_high, num_bins)

    model = CQM()

    obj = BQM(vartype='BINARY')
    x = {(i, j): obj.add_variable(f'x_{i}_{j}') for i in items for j in bins}

    for i in items:
        for j in bins:
            obj.set_linear(x[(i, j)], -values[i])

    model.set_objective(obj)

    # Each item at most goes to one bin.
    for i in items:
        model.add_constraint([(x[(i, j)], 1) for j in bins] + [(-1,)], sense="<=")

    # Build knapsack capacity constraints
    for j in bins:
        model.add_constraint([(x[(i, j)], weights[i]) for i in items] + [(-capacities[j],)], sense="<=")

    return model
