# Copyright 2024 D-Wave Inc.
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

from itertools import product

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel
from dimod.typing import ArrayLike

__all__ = ['quadratic_assignment']

def quadratic_assignment(distance_matrix: ArrayLike,
                         flow_matrix: ArrayLike,
                         ) -> ConstrainedQuadraticModel:
    """Generates a constrained quadratic model encoding a quadratic-assignment problem.

    Given distance and flow matrices, generates a :class:`~dimod.ConstrainedQuadraticModel` 
    for the corresponding quadratic-assignment problem. 

    Args:
        distances: Distances between locations :math:`i` and :math:`j` as a NumPy array.

        flows: Flows between facilities :math:`i` and :math:`j` as a NumPy array.

    Returns:
        The constrained quadratic model encoding the quadratic-assignment problem. Variables are
        denoted as ``x_{i}_{j}`` where ``x_{i}_{j} == 1`` means that facility ``i`` is
        placed in location ``j``.

    """

    distance_matrix = np.atleast_2d(np.asarray(distance_matrix))
    flow_matrix = np.atleast_2d(np.asarray(flow_matrix))

    if distance_matrix.shape != flow_matrix.shape:
        raise ValueError("'distance_matrix' and 'flow_matrix' must have the same shape")

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("'distance_matrix' must be square")

    if distance_matrix.ndim != 2:
        raise ValueError("'distance_matrix' must be 2-dimensional")

    num_locations = distance_matrix.shape[0]

    model = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    x = {(i, j): obj.add_variable(f'x_{i}_{j}') for i in range(num_locations)
         for j in range(num_locations)}

    for i, j, k, l in product(range(num_locations), repeat=4):
        if (i,j) != (k,l):
            obj.set_quadratic(x[(i,j)], x[(k,l)], flow_matrix[i][k]*distance_matrix[j][l]
                              + flow_matrix[k][i]*distance_matrix[j][l])

    model.set_objective(obj)

    for i in range(num_locations):
        constraint_vars = [x[(i,j)] for j in range(num_locations)]
        model.add_discrete(constraint_vars, label=f'discrete_constraint_{i}')

    for j in range(num_locations):
        constraint = [(x[(i,j)], 1) for i in range(num_locations)] + [(-1,)]
        model.add_constraint(constraint, sense='==', label=f'facility_constraint_{j}')

    return model
