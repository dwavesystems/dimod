import numpy as np
import dimod


__all__ = ["nae3sat"]


def kmc_sat(num_variables, num_clauses=1, k=3, seed=None):
    """k Max-Cut SAT problem generator.

    This generator makesinstances of the kMC-SAT problem. This problem class is a conjunction of
    one or more clauses, where in each clause a max-sat problem is formulated. The value `k` corresponds
    to the number of clauses per variable.

    When k=3, this probem is known as not-all-equal 3 SAT (NAE3SAT), since the max-sat solution of each
    clause correspond to the three variables not taking the same value. When k=4, this problem is known
    as 2in4 SAT, since the max-sat solution corresponds to having two True and two False variables on
    each clause.

    Input:
        * num_variables (int): The number of variables in the problem
        * num_clauses (int): The number of clauses in the problem
        * k (int): The number of variables per clause
        * seed (int): A seed to be passed to the random number generator

    Returns:
        * clause_vars (2D np.array): Each row is a clause, each column is a variable
        * clause_signs (2D np.array): Each row is a clause, each column indicates if the variable is negated
    """

    rnd = np.random.RandomState(seed)
    clause_vars = np.zeros(shape=(num_clauses, k), dtype=int)
    for c in range(num_clauses):
        clause_vars[c, :] = rnd.choice(num_variables, size=k, replace=False)
    clause_signs = rnd.choice([-1, 1], size=(num_clauses, k))
    return clause_vars, clause_signs


def kmc_sat_to_bqm(clause_vars, clause_signs):
    """Translator from k Max-Cut SAT instance to Binary Quadratic Model

    Input:
        * clause_vars (2D np.array): Each row is a clause, each column is a variable
        * clause_signs (2D np.array): Each row is a clause, each column indicates if the variable is negated

    Returns:
        * bqm (dimod.BinaryQuadraticModel): Problem bqm
    """

    bqm = dimod.BinaryQuadraticModel(vartype="SPIN")
    for vars, signs in zip(clause_vars, clause_signs):
        bqm.add_interactions_from(
            {
                (vars[i], vars[j]): signs[i] * signs[j]
                for i in range(len(vars))
                for j in range(i)
            }
        )
    return bqm


def nae3sat(num_variables, rho=2.1, seed=None):

    """Generator for Not-All-Equal 3-SAT (NAE3SAT) Binary Quadratic Models.

    NAE3SAT is an NP-complete problem class that consists in satistying a number of conjunctive
    clauses that involve three variables (or variable negations). The variables on each clause
    should be not-all-equal. Ie. all solutions except 111 or 000 are valid for each class.

    Input:
        * num_variables (int): The number of variables in the problem
        * rho (float): The clause-to-variable ratio

    Returns:
        * bqm (dimod.BinaryQuadraticModel): Problem bqm
    """

    rnd = np.random.RandomState(seed)
    num_clauses = rnd.choice(
        range(int(rho * num_variables) - 1, int(rho * num_variables) + 2)
    )
    clause_vars, clause_signs = kmc_sat(
        num_variables=num_variables, num_clauses=num_clauses, k=3, seed=seed
    )
    bqm = kmc_sat_to_bqm(clause_vars=clause_vars, clause_signs=clause_signs)
    return bqm
