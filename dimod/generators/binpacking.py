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


__all__ = ['bin_packing']


def bin_packing(num_items, seed=32, weight_range=(10, 30)):
    """
    Given the number of items, the code generates a random bin packing problem
    formulated as a Constrained Quadratic model. The weights for each item are
    integers uniformly drawn from in the weight_range. The bin capacity is set
    to num_items * mean(weights) / 5.

    Args:
        num_items (int):
            Number of items to choose from

        seed (int, optional, default=32):
            Seed for numpy random number generator

        weight_range (tuple, optional, default=(10, 30)):
            The range of the randomly generated weights for each item

    Returns:
        :obj:`.ConstrainedQuadraticModel`.

    """
    max_num_bins = num_items
    np.random.seed(seed)
    items = list(range(num_items))
    weights = list(np.random.randint(*weight_range, num_items))
    bins = list(range(max_num_bins))
    bin_capacity = num_items * np.mean(weights) / 5
    model = CQM()

    obj = BQM(vartype='BINARY')
    y = {j: obj.add_variable(f'y_{j}') for j in bins}

    for j in bins:
        obj.set_linear(y[j], 1)

    model.set_objective(obj)

    x = {(i, j): model.add_variable(f'x_{i}_{j}', vartype='BINARY') for i in items for j in bins}

    # Each item goes to one bin
    for i in items:
        model.add_constraint([(x[(i, j)], 1) for j in bins] + [(-1,)],
                             sense="==")

    # Bin capacity constraint
    for j in bins:
        model.add_constraint([(x[(i, j)], weights[i]) for i in items] +
                             [(y[j], -bin_capacity)], sense="<=")

    return model


if __name__ == "__main__":

    cqm = bin_packing(10)
    print(cqm.variables)
