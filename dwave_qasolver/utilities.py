def ising_energy(h, J, solution):
    """Calculate the Ising energy of the given solution.

    H(s) = sum_i h_i * s_i + sum_(i, j) J_(i,j) * s_i * s_j

    Args:
        h: The linear biases in a dict of the form {var: bias, ...}.
        J: The quadratic biases in a dict of the form
        {(var0, var1): bias, ...}.
        solution: A dict of spins of the form {var: spin, ...} where
        each spin is either -1 or 1.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    https://en.wikipedia.org/wiki/Ising_model

    """
    energy = 0

    # add the contribution from the linear biases
    for v in h:
        energy += h[v] * solution[v]

    # add the contribution from the quadratic biases
    for v0, v1 in J:
        energy += J[(v0, v1)] * solution[v0] * solution[v1]

    return energy


def qubo_energy(Q, solution):
    """Calculate the quadratic polynomial value of the given solution
    to a quadratic unconstrained binary optimization (QUBO) problem.

    E(x) = sum_(i, j) Q_(i, j) * x_i * x_j

    Args:
        Q: A dict of the qubo coefficients of the form
        {(var0, var1): coeff, ...}
        solution: A dict of binary variables of the form
        {var: bin, ...} where each bin is either 0 or 1.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    """

    energy = 0
    for v0, v1 in Q:
        energy += solution[v0] * solution[v1] * Q[(v0, v1)]

    return energy
