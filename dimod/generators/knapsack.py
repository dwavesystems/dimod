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


__all__ = ['knapsack']


def knapsack(num_items, seed=32, value_range=(10, 30), weight_range=(10, 30), tightness_ratio=0.5):
    """
    Given the number of items, the code generates a random knapsack problem,
    formulated as a Constrained Quadratic model. The capacity of the bin is set to be
    half of the sum of the weights.

    Args:
        num_items (int):
            Number of items to choose from

        seed (int, optional, default=32):
            Seed for numpy random number generator

        value_range (tuple, optional, default=(10, 30)):
            The range of the randomly generated values for each item

        weight_range (tuple, optional, default=(10, 30)):
            The range of the randomly generated weights for each item

        tightness_ratio (float, optional, default=0.5):
            ratio of capacity over sum of weights

    Returns:
        :obj:`.ConstrainedQuadraticModel`.

    """

    np.random.seed(seed)
    items = list(range(num_items))
    value = {i: np.random.randint(*value_range) for i in items}
    weight = {i: np.random.randint(*weight_range) for i in items}
    capacity = int(sum(weight.values()) * tightness_ratio)

    model = CQM()
    obj = BQM(vartype='BINARY')
    x = {i: obj.add_variable(f'x_{i}') for i in items}

    for i in items:
        obj.set_linear(x[i], -value[i])

    model.set_objective(obj)
    constraint = [(x[i], weight[i]) for i in items] + [(-capacity, )]
    model.add_constraint(constraint, sense="<=", label='capacity')

    return model
