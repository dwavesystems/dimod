# Copyright 2023 D-Wave Systems Inc.
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

from dimod.constrained import ConstrainedQuadraticModel
from dimod.binary.binary_quadratic_model import quicksum
from dimod.quadratic import Integer

__all__ = ["magic_square"]


def magic_square(size: int, power: int = 1) -> ConstrainedQuadraticModel:
    """Generate a magic square of a particular size.
    
    Args:
        size (int):
            The sidelength of the magic square, the magic square will
            have size * size integer variables.
        power (int, optional):
            If set to 1 the problem is a normal magic square, if set
            to 2 it is a magic square of squares. Must be 1 or 2.
            Defaults to 1.

    Returns:
        dimod.ConstrainedQuadraticModel:
            A coonstrained quadratic model which represents a magic square.
            The variables are labeled by "var_i_j" where i and j are two
            non-negative integers which represent their column and row
            respectively. If power is greater than 1 then the value of the
            variable represents the base, so the magic square is satisfied
            by var_i_j**power .
    """
    if power not in [1, 2]:
        raise ValueError(f"power must be either 1 or 2, recieved {power}")
    
    variables = {}
    
    for i, j in product(range(size), range(size)):
        variables[i, j] = Integer(f"var_{i}_{j}", lower_bound=1)
    
    # the sum across every row, column, diagonal
    constraint_sum = Integer("sum", lower_bound=1)
    
    cqm = ConstrainedQuadraticModel()
    
    for i in range(size):
        
        if power == 1:
            # Every row must equal the same value
            cqm.add_constraint_from_comparison(
                quicksum(variables[i, j] for j in range(size)) -
                constraint_sum == 0, label=f"row_{i}")

            # Every column must equal the same value
            cqm.add_constraint_from_comparison(
                    quicksum(variables[j, i] for j in range(size)) -
                    constraint_sum == 0, label=f"col_{i}")
        else:
            # Every row must equal the same value
            cqm.add_constraint_from_comparison(
                quicksum(variables[i, j]**power for j in range(size)) -
                constraint_sum == 0, label=f"row_{i}")

            # Every column must equal the same value
            cqm.add_constraint_from_comparison(
                    quicksum(variables[j, i]**power for j in range(size)) -
                    constraint_sum == 0, label=f"col_{i}")

    if power == 1:
        # the diagonals must equal the same value
        cqm.add_constraint_from_comparison(
            quicksum(variables[i, i] for i in range(size)) -
            constraint_sum == 0, label="diagonal")
        
        cqm.add_constraint_from_comparison(
            quicksum(variables[i, size-1-i] for i in range(size)) -
            constraint_sum == 0, label="antidiagonal")
    else:
        # the diagonals must equal the same value
        cqm.add_constraint_from_comparison(
            quicksum(variables[i, i]**power for i in range(size)) -
            constraint_sum == 0, label="diagonal")
        
        cqm.add_constraint_from_comparison(
            quicksum(variables[i, size-1-i]**power for i in range(size)) -
            constraint_sum == 0, label="antidiagonal")
    
    # every integer must be unique
    cqm.add_constraint_from_comparison(
        quicksum(
            variables[i, j]**2 +
            variables[k, l]**2 -
            2*variables[i, j]*variables[k, l]
            for i, j, k, l in product(range(size), repeat=4)
            if ((k > i and l == j) or (l > j)))
        >= (size**4 - size**2)/2, label="uniqueness")
    
    return cqm
