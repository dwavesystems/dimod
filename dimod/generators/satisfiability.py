# Copyright 2022 D-Wave Systems Inc.
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

from __future__ import annotations

import collections.abc
import itertools
import typing

import numpy as np

import dimod  # for typing

from dimod.binary import BinaryQuadraticModel
from dimod.vartypes import Vartype


__all__ = ["random_nae3sat", "random_2in4sat"]


def _kmcsat_interactions(num_variables: int, k: int, num_clauses: int,
                         *,
                         plant_solution: bool = False,
                         seed: typing.Union[None, int, np.random.Generator] = None,
                         ) -> typing.Iterator[typing.Tuple[int, int, int]]:
    rng = np.random.default_rng(seed)

    # Use of for and while loops is for clarity, optimizations are possible.
    for _ in range(num_clauses):
        # randomly select the variables
        variables = rng.choice(num_variables, k, replace=False)

        # randomly assign the negations
        signs = 2 * rng.integers(0, 1, endpoint=True, size=k) - 1
        while plant_solution and abs(sum(signs))>1:
            # Rejection sample until signs are compatible with an all 1 ground
            # state:
            signs = 2 * rng.integers(0, 1, endpoint=True, size=k) - 1
            
            
        # get the interactions for each clause
        for (u, usign), (v, vsign) in itertools.combinations(zip(variables, signs), 2):
            yield u, v, usign*vsign


def random_kmcsat(variables: typing.Union[int, typing.Sequence[dimod.typing.Variable]],
                  k: int,
                  num_clauses: int,
                  *,
                  plant_solution: bool = False,
                  seed: typing.Union[None, int, np.random.Generator] = None,
                  ) -> BinaryQuadraticModel:
    """Generate a random k Max-Cut satisfiability problem as a binary quadratic model.

    kMC-SAT [#ZK]_ is an NP-complete problem class
    that consists in satisfying a number of
    clauses of ``k`` literals (variables, or their negations).
    Each clause should encode a max-cut problem over the clause literals.
    
    Each clause contributes -:code:`k//2` to the energy when the clause is 
    satisfied, and at least 0 when unsatisfied. The energy :math:`H(s)` for a 
    spin assignment :math:`s` is thus lower bounded by :math:`E_{SAT}=-`:code:`k//2 * num_clauses`, 
    this lower bound matches the ground state energy in satisfiable instances. 
    For k>3, energy penalties per clause violation are non-uniform.


    Args:
        num_variables: The number of variables in the problem.
        k: number of variables participating in the clause.
        num_clauses: The number of clauses. Each clause contains three literals.
        plant_solution: Create literals uniformly subject to the constraint that the
            all 1 (and all -1) are ground states (satisfy all clauses).
        seed: Passed to :func:`numpy.random.default_rng()`, which is used
            to generate the clauses and the variable negations.
    Returns:
        A binary quadratic model with spin variables.

    .. note:: The clauses are randomly sampled from the space of k-variable
        clauses *with replacement* which can result in collisions. However,
        collisions are allowed in standard problem definitions, are absent with
        high probability in interesting cases, and are almost always harmless
        when they do occur. 

        For large problems planting of an all 1 solution can
        be achieved (in some special cases) without modification of the hardness
        qualities of the instance class. Planting of a not all 1 ground state
        can be achieved with a spin-reversal transform without loss of generality. [#DKR]_
    .. [#ZK] Lenka Zdeborová and Florent Krzakala,
       "Quiet Planting in the Locked Constraint Satisfaction Problems",
       https://epubs.siam.org/doi/10.1137/090750755
    .. [#DKR] Adam Douglass, Andrew D. King & Jack Raymond,
       "Constructing SAT Filters with a Quantum Annealer",
       https://link.springer.com/chapter/10.1007/978-3-319-24318-4_9
    """
    if isinstance(variables, collections.abc.Sequence):
        num_variables = len(variables)
        labels = variables
    else:
        num_variables = variables
        labels = None
        
    if num_variables < 1:
        raise ValueError("number of variables must be non-negative")
    elif k < 1:
        raise ValueError("number of variables must be non-negative")
    elif num_clauses < 0:
        raise ValueError("{num_clauses} must be non-negative")
    elif num_variables < k:
        raise ValueError(f"must use at least {k}<= number of variables")

    bqm = BinaryQuadraticModel(num_variables, Vartype.SPIN)
    bqm.add_quadratic_from(_kmcsat_interactions(num_variables, k, num_clauses,
                                                plant_solution=plant_solution, seed=seed))

    if labels:
        bqm.relabel_variables(dict(enumerate(labels)))

    return bqm


def random_nae3sat(variables: typing.Union[int, typing.Sequence[dimod.typing.Variable]],
                   num_clauses: int,
                   *,
                   plant_solution: bool = False,
                   seed: typing.Union[None, int, np.random.Generator] = None,
                   ) -> BinaryQuadraticModel:
    """Generate a random not-all-equal 3-satisfiability problem as a binary quadratic model.

    Not-all-equal 3-satisfiability (NAE3SAT_) is an NP-complete problem class
    that consists in satisfying a number of conjunctive
    clauses of three literals (variables, or their negations).
    For valid solutions, the literals in each clause should be not-all-equal;
    i.e. any assignment of values except ``(+1, +1, +1)`` or ``(-1, -1, -1)``
    are valid for each clause.

    Each clause contributes -1 to the energy when the clause is satisfied,
    and +3 when unsatisfied. The energy :math:`H(s)` for a spin assignment :math:`s`
    is thus lower bounded by :math:`E_{SAT}=-`:code:`num_clauses`, this lower 
    bound matches the ground state energy in satisfiable instances. The number 
    of violated clauses is :math:`(H(s) - E_{SAT})/4`.

    NAE3SAT problems have been studied with the D-Wave quantum annealer [#DKR]_.

    .. _NAE3SAT: https://en.wikipedia.org/wiki/Not-all-equal_3-satisfiability

    
    Args:
        num_variables: The number of variables in the problem.
        num_clauses: The number of clauses. Each clause contains three literals.
        plant_solution: Create literals uniformly subject to the constraint that the
            all 1 (and all -1) are ground states satisfying all clauses.
        seed: Passed to :func:`numpy.random.default_rng()`, which is used
            to generate the clauses and the variable negations.

    Returns:
        A binary quadratic model with spin variables.

    Example:

        Generate a NAE3SAT problem with a given clause-to-variable ratio (rho).

        >>> num_variables = 75
        >>> rho = 2.1
        >>> bqm = dimod.generators.random_nae3sat(num_variables, round(num_variables*rho))

    .. note:: The clauses are randomly sampled from the space of 3-variable
        clauses *with replacement* which can result in collisions. However,
        collisions are allowed in standard problem definitions, are absent with
        high probability in interesting cases, and are almost always harmless
        when they do occur. 

        Planting of a not all 1 solution state
        can be achieved with a spin-reversal transform without loss of 
        generality. Planting can significantly modify the hardness of 
        optimization problems.

    .. [#DKR] Adam Douglass, Andrew D. King & Jack Raymond,
       "Constructing SAT Filters with a Quantum Annealer",
       https://link.springer.com/chapter/10.1007/978-3-319-24318-4_9

    """
    return random_kmcsat(variables, 3, num_clauses, plant_solution=plant_solution, seed=seed)


def random_2in4sat(variables: typing.Union[int, typing.Sequence[dimod.typing.Variable]],
                   num_clauses: int,
                   *,
                   plant_solution: bool = False,
                   seed: typing.Union[None, int, np.random.Generator] = None,
                   ) -> BinaryQuadraticModel:
    """Generate a random 2-in-4 satisfiability problem as a binary quadratic model.

    2-in-4 satisfiability [#DKR]_ is an NP-complete problem class
    that consists in satisfying a number of conjunctive
    clauses of four literals (variables, or their negations).
    For valid solutions, two of the literals in each clause should ``+1`` and
    the other two should be ``-1``.

    Each clause contributes -2 to the energy when the clause is satisfied,
    and at least 0 when unsatisfied. The energy :math:`H(s)` for a spin 
    assignment :math:`s` is thus lower bounded by :math:`E_{SAT}=-2`:code:`num_clauses`, 
    this lower bound matches the ground state energy in satisfiable instances. 
    The number of violated clauses is at most :math:`(H(s) - E_{SAT})/2`.

    
    Args:
        num_variables: The number of variables in the problem.
        num_clauses: The number of clauses. Each clause contains three literals.
        plant_solution: Create literals uniformly subject to the constraint that the
            all 1 (and all -1) are ground states satisfying all clauses.
        seed: Passed to :func:`numpy.random.default_rng()`, which is used
            to generate the clauses and the variable negations.
    Returns:
        A binary quadratic model with spin variables.

    .. note:: The clauses are randomly sampled from the space of 4-variable
        clauses *with replacement* which can result in collisions. However,
        collisions are allowed in standard problem definitions, are absent with
        high probability in interesting cases, and are almost always harmless
        when they do occur. 

        Planting of a not all 1 ground state
        can be achieved with a spin-reversal transform without loss of 
        generality. Planting can significantly modify the hardness of 
        optimization problems. However, for large problems planting of a 
        solution can be achieved (in the SAT phase) without modification of the
        hardness qualities of the instance class. [#ZK]_ 
    .. [#DKR] Adam Douglass, Andrew D. King & Jack Raymond,
       "Constructing SAT Filters with a Quantum Annealer",
       https://link.springer.com/chapter/10.1007/978-3-319-24318-4_9
    .. [#ZK] Lenka Zdeborová and Florent Krzakala,
       "Quiet Planting in the Locked Constraint Satisfaction Problems",
       https://epubs.siam.org/doi/10.1137/090750755

    """
    return random_kmcsat(variables, 4, num_clauses, plant_solution=plant_solution, seed=seed)
