# Copyright 2024 D-Wave Systems Inc.
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
from itertools import product 

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.constrained import ConstrainedQuadraticModel

from dimod import Binary 

def quadratic_assignment(distance_matrix: typing.Optional[np.typing.ArrayLike] = None,
                    flow_matrix: typing.Optional[np.typing.ArrayLike] = None,
                    ) -> ConstrainedQuadraticModel:
    """Generates a constrained quadratic model encoding a quadratic assignment problem.

    Given distance and flow matrices, generates a :class:`~dimod.ConstrainedQuadraticModel` 
    for the corresponding quadratic assignment problem. 

    Args:
        distances: numpy array containing the distance between location i and location j

        flows: numpy array containing the flow between facility i and facility j

    Returns:
        The constrained quadratic model encoding the quadratic assignment problem. Variables are
        denoted as ``x_{i}_{j}`` where ``x_{i}_{j} == 1`` means that facility ``i`` is
        placed in location ``j``.

    """

    if distance_matrix.shape != flow_matrix.shape:
        raise ValueError("'distance_matrix' and 'flow_matrix' must have the same shape")

    num_locations = distance_matrix.shape[0]

    model = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    x = {(i,j): obj.add_variable(f'x_{i}_{j}') for i in range(num_locations) for j in range(num_locations)}

    for i, j, k, l in product(range(num_locations), repeat=4):
        if (i,j) != (k,l): 
            obj.set_quadratic(x[(i,j)], x[(k,l)], flow_matrix[i][k]*distance_matrix[j][l] + flow_matrix[k][i]*distance_matrix[j][l])
    
    model.set_objective(obj)

    for i in range(num_locations):
        constraint = [(x[(i,j)], 1) for j in range(num_locations)] + [(-1,)]
        model.add_constraint(constraint, sense="==")

    for j in range(num_locations):
        constraint = [(x[(i,j)], 1) for i in range(num_locations)] + [(-1,)]
        model.add_constraint(constraint, sense='==')

    return model
