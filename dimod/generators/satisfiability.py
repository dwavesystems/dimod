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
import sys

import numpy as np

import dimod  # for typing
import warnings
import networkx as nx # for configuration_model

from dimod.binary import BinaryQuadraticModel
from dimod.vartypes import Vartype

__all__ = ["random_nae3sat", "random_2in4sat","random_kmcsat","random_kmcsat_cqm","kmcsat_clauses"]

def _cut_poissonian_degree_distribution(num_variables,num_stubs,cut=2,seed=None):
    ''' Sampling of the cutPoisson distribution by rejection sampling method.
    
    Select degree distribution uniformly at random from Poissonian
    distribution subject to constraint that sockets are not exhausted
    and degree is equal to or greater than cut.
    '''
    if num_stubs < num_variables*cut:
        raise ValueError('Mean connectivity must be at least as large as the cut value')
    rng = np.random.default_rng(seed)
    
    degrees = []
    while num_variables > 1 and num_stubs > cut*num_variables:
        lam = (num_stubs/num_variables/2)
        degree = rng.poisson(lam=lam)
        if degree >= cut and num_stubs-degree>=cut*(num_variables-1): 
            degrees.append(degree)
            num_variables = num_variables-1
            num_stubs = num_stubs - degree
    if num_variables == 1:
        degrees.append(num_stubs)
    else:
        degrees = degrees + [cut]*num_variables
    return degrees

def kmcsat_clauses(num_variables: int, k: int, num_clauses: int,
                    *,
                    variables_list: list = None,
                    signs_list: list = None,
                    plant_solution: bool = False,
                    graph_ensemble: str = 'Poissonian',
                    max_config_model_rejections = 1024,
                    seed: typing.Union[None, int, np.random.Generator] = None,
) -> (list, list):
    
    rng = np.random.default_rng(seed)
    if variables_list is None:
        rngNX = np.random.RandomState(rng.integers(32767)) # networkx requires legacy method
        # Use of for and while loops, and rejection sampling, is for clarity, optimizations are possible.

        # Establish connectivity pattern amongst variables (the graph):
        variables_list = []
        if graph_ensemble == 'cutPoissonian':
            # Sample sequentially connectivity and reject unviable cases:
            clause_degree_sequence = [k]*num_clauses

            degrees = _cut_poissonian_degree_distribution(num_variables, num_clauses*k, cut=2, seed=rng)
            G = nx.bipartite.configuration_model(degrees, clause_degree_sequence, create_using=nx.Graph(), seed=rngNX)
            if max_config_model_rejections > 0:
                # A single-shot of the configuration model does not guarantee that all clauses contain k
                # variables. A small subset may contain fewer than k variables. By default we enforce
                # via rejection sampling a requirement that all clauses contain exactly k variables:
                while max_config_model_rejections > 0 and G.number_of_edges() != num_clauses*k:
                    # An overflow is possible, but only for pathological cases
                    degrees = _cut_poissonian_degree_distribution(num_variables, num_clauses*k, cut=2, seed=rng)
                    G = nx.bipartite.configuration_model(degrees, clause_degree_sequence, create_using=nx.Graph(), seed=rngNX)
                    max_config_model_rejections = max_config_model_rejections - 1
                    
                if max_config_model_rejections == 0:
                    warn_message = ('configuration model consistently rejected sampled cutPoissonian '
                                    'degree sequences, the model returned contains clauses with < k literals. '
                                    'Likely cause is a pathological parameterization of the graph ensemble. '
                                    'If you intended sampling to fail set max_config_model_rejections=0 to '
                                    'suppress this warning. Expected ' + str(num_clauses*k) +
                                    ' stubs, last attempt ' + str(G.number_of_edges())
                    )
                    warnings.warn(warn_message,
                                  UserWarning, stacklevel=3
                    )
            # Extract a list of variables for each clause from the graphical representation
            for i in range(num_variables, num_variables+num_clauses):
                variables_list.append(list(G.neighbors(i)))
        else:
            if graph_ensemble is None or graph_ensemble == 'Poissonian':
                pass
            else:
                raise ValueError('Unsupported graph ensemble, supported types are'
                             '"Poissonian" (by default) and "cutPoissonian".')
            for _ in range(num_clauses):
                # randomly select the variables
                variables_list.append(rng.choice(num_variables, k, replace=False))
    
    if signs_list is None:
        signs_list = []
        # Convert variables to literals:
        for variables in variables_list:
            # randomly assign the negations
            k = len(variables)
            signs = 2 * rng.integers(0, 1, endpoint=True, size=k) - 1
            while plant_solution and abs(sum(signs))>1:
                # Rejection sample until signs are compatible with an all 1 ground
                # state:
                signs = 2 * rng.integers(0, 1, endpoint=True, size=k) - 1
            signs_list.append(signs)
    return variables_list,signs_list

def _kmcsat_interactions(num_variables: int, k: int, num_clauses: int,
                         *,
                         variables_list: list = None,
                         signs_list: list = None,
                         plant_solution: bool = False,
                         graph_ensemble: str = 'Poissonian',
                         max_config_model_rejections = 1024,
                         seed: typing.Union[None, int, np.random.Generator] = None,
) -> typing.Iterator[typing.Tuple[int, int, int]]:
    variables_list, signs_list = kmcsat_clauses(num_variables, k, num_clauses,
                                                variables_list = variables_list,
                                                signs_list = signs_list,
                                                plant_solution=plant_solution,
                                                graph_ensemble=graph_ensemble,
                                                max_config_model_rejections=max_config_model_rejections,
                                                seed=seed)
    # get the interactions for each clause
    for variables,signs in zip(variables_list,signs_list):
        for (u, usign), (v, vsign) in itertools.combinations(zip(variables, signs), 2):
            yield u, v, usign*vsign


def random_kmcsat(variables: typing.Union[int, typing.Sequence[dimod.typing.Variable]],
                  k: int,
                  num_clauses: int,
                  *,
                  variables_list: list = None,
                  signs_list: list = None,
                  plant_solution: bool = False,
                  graph_ensemble: str = 'Poissonian',
                  max_config_model_rejections = 1024,
                  seed: typing.Union[None, int, np.random.Generator] = None
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
        graph_ensemble: By default, variables are assigned uniformly at random
            to clauses yielding a 'Poissonian' ensemble. An alternative choice
            is CutPoissonian that guarantees all variables participate in a
            at least two interactions - with high probability a single giant
            problem component containing all variables is produced. 
        max_config_model_rejections: This is relevant only when selecting
            ``graph_ensemble``='cutPoissonian'. The creation of this ensemble
            requires sampling of graphs with fixed degree sequences via the 
            configuration model, which is not guaranteed to succeed. When 
            sampling fails some max-cut SAT clauses are assigned fewer than 
            k literals. A failure mode can be avoided wih high probability, 
            except at pathological parameterization, by setting a large value 
            (the default). 
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

        A 1RSB analysis indicates the following critical behaviour [#MM] (page 443) in
        canonical random graphs as a function of the clause to variable ratio
        alpha = num_clauses /num_var.
        graph_class k alpha_dynamical  alpha_sat
        Poisson     3 1.50             2.11 (alpha_rigidity=1.72)
                    4 0.58             0.64
                    5 1.02             1.39
                    6 0.48             0.57
        cutPoisson  3 1.61             2.16
                    4 0.62             0.7067L
                    5 1.08             1.41
                    6 0.47             0.5959L
        In a Poisson graph, each clause connects at random to variables, the
        marginal connectivity distribution of variables converges to a Poisson 
        distribution.
        In a cutPoisson graph, each clause connects at random to variables, 
        subject to the constraint each variable has connectivity atleast 2.
        For locked problems (marked L) the threshold is exact, and planting 
        is quiet (for alpha<alpha_S, planting=true/false are statistically
        similar in free energy [extensive properties, those that determine
        computational hardness]).

    .. [#MM] Marc Mézard and Andrea Montanari
       "Information, Physics and Computation"
       DOI: DOI:10.1093/acprof:oso/9780198570837.001.0001
       "https://web.stanford.edu/~montanar/RESEARCH/book.html"
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
                                                variables_list = variables_list,
                                                signs_list = signs_list,
                                                plant_solution=plant_solution, seed=seed))

    if labels:
        bqm.relabel_variables(dict(enumerate(labels)))

    return bqm

def random_kmcsat_cqm(num_variables = None,
                      variables_list_obj: list = [],
                      signs_list_obj: list = [],
                      *,
                      variables_list_cons: list = [],
                      signs_list_cons: list = [],
                      constraint_form = 'quadratic',
                      binarize = True):
    if num_variables == None:
        num_variables=0
        if len(variables_list_obj)>0:
            num_variables = max(num_variables,np.max(np.array(variables_list_obj)))
        if len(variables_list_cons)>0:
            num_variables=max(num_variables,np.max(np.array(variables_list_cons)))
        num_variables=num_variables + 1
    
    cqm = dimod.CQM()
    #Add the binary variables we need up front
    for i in range(num_variables):
        if binarize:
            cqm.add_variable('BINARY')
        else:
            cqm.add_variable('SPIN')
    if len(variables_list_obj)>0:
        num_clauses=len(variables_list_obj)
        k=len(variables_list_obj[0])
        bqm = random_kmcsat(num_variables,k,num_clauses,variables_list=variables_list_obj,signs_list=signs_list_obj)
        if binarize:
            bqm.change_vartype('BINARY')
        cqm.set_objective(bqm)
    
    for variables,signs in zip(variables_list_cons, signs_list_cons):
        num_clauses = 1
        k = len(variables)
        if constraint_form == 'quadratic':
            val = -(k//2)*(k-k//2) + (k//2)*(k//2-1)/2 + (k-k//2)*(k-k//2-1)/2 
            bqm = random_kmcsat(num_variables,k,num_clauses,variables_list=[variables],signs_list=[signs])
            if binarize:
                bqm.change_vartype('BINARY')
            label = cqm.add_constraint_from_model(bqm, '==', val)
            #print(cqm.constraints[label].to_polystring())
        else:
            equation_form = [(v,s) for v,s in zip(variables,signs)]
            if k&1:
                #Slack variable equality:
                aux_label = cqm.add_variable('SPIN')
                equation_form.append((aux_label,1))
            if binarize:
                rhs = (k+1)//2 
            else: 
                rhs = 0
            label = cqm.add_constraint_from_iterable(equation_form, '==', rhs=rhs)
    return cqm
                  

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
