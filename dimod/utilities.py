"""
A collection of utility functions useful for Samplers.
"""

from dimod import _PY2

__all__ = ['ising_energy', 'qubo_energy', 'ising_to_qubo', 'qubo_to_ising']

if _PY2:
    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()
else:
    def iteritems(d):
        return d.items()

    def itervalues(d):
        return d.values()


def ising_energy(sample, h, J, offset=0.0):
    """Calculate the Ising energy of the given sample.

    H(s) = sum_i h_i * s_i + sum_(i, j) J_(i,j) * s_i * s_j

    https://en.wikipedia.org/wiki/Ising_model

    Args:
        sample: A dict of spins of the form {v: spin, ...} where
            each spin is either -1 or 1.
        h: The linear biases in a dict of the form {v: bias, ...}.
        J: The quadratic biases in a dict of the form {(u, v): bias, ...}.
        offset (numeric, optional): The constant offset to be applied
            to the energy. Default 0.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    """
    # add the contribution from the linear biases
    for v in h:
        offset += h[v] * sample[v]

    # add the contribution from the quadratic biases
    for v0, v1 in J:
        offset += J[(v0, v1)] * sample[v0] * sample[v1]

    return offset


def qubo_energy(sample, Q, offset=0.0):
    """Calculate the quadratic polynomial value of the given sample
    to a quadratic unconstrained binary optimization (QUBO) problem.

    E(x) = sum_(i, j) Q_(i, j) * x_i * x_j

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    Args:
        sample: A dict of binary variables of the form
            {v: bin, ...} where each bin is either 0 or 1.
        Q: A dict of the QUBO coefficients of the form
            {(u, v): coeff, ...}
        offset (numeric, optional): The constant offset to be applied
            to the energy. Default 0.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    """
    for v0, v1 in Q:
        offset += sample[v0] * sample[v1] * Q[(v0, v1)]

    return offset


def ising_to_qubo(h, J, offset=0.0):
    """Converts an Ising problem to a QUBO problem.

    Map an Ising model defined over -1/+1 variables to a binary quadratic
    program x' * Q * x defined over 0/1 variables. We return the Q defining
    the BQP model as well as the offset in energy between the two problem
    formulations, i.e. s' * J * s + h' * s = offset + x' * Q * x. The linear term
    of the BQP is contained along the diagonal of Q.

    See qubo_to_ising(Q) for the inverse function.

    Args:
        h (dict): A dict of the linear coefficients of the Ising problem.
        J (dict): A dict of the quadratic coefficients of the Ising problem.
        offset (numeric, optional): The constant offset to be applied
            to the energy. Default 0.

    Returns:
        (dict, float): A 2-tuple containing:

            dict: the QUBO coefficients.

            float: The new energy offset.

    """
    # the linear biases are the easiest
    q = {(v, v): 2. * bias for v, bias in iteritems(h)}

    # next the quadratic biases
    for (u, v), bias in iteritems(J):
        if bias == 0.0:
            continue
        q[(u, v)] = 4. * bias
        q[(u, u)] -= 2. * bias
        q[(v, v)] -= 2. * bias

    # finally calculate the offset
    offset += sum(itervalues(J)) - sum(itervalues(h))

    return q, offset


def qubo_to_ising(Q, offset=0.0):
    """Converts a QUBO problem to an Ising problem.

    Map a binary quadratic program x' * Q * x defined over 0/1 variables to
    an Ising model defined over -1/+1 variables. We return the h and J
    defining the Ising model as well as the offset in energy between the
    two problem formulations, i.e. x' * Q * x = offset + s' * J * s + h' * s. The
    linear term of the QUBO is contained along the diagonal of Q.

    See ising_to_qubo(h, J) for the inverse function.

    Args:
        Q: A dict of the QUBO coefficients.
        offset (numeric, optional): The constant offset to be applied
            to the energy. Default 0.

    Returns:
        (dict, dict, float): A 3-tuple containing:

            dict: the linear coefficients of the Ising problem.

            dict: the quadratic coefficients of the Ising problem.

            float: The new energy offset.

    """
    h = {}
    J = {}
    linear_offset = 0.0
    quadratic_offset = 0.0

    for (u, v), bias in iteritems(Q):
        if u == v:
            if u in h:
                h[u] += .5 * bias
            else:
                h[u] = .5 * bias
            linear_offset += bias

        else:
            if bias != 0.0:
                J[(u, v)] = .25 * bias

            if u in h:
                h[u] += .25 * bias
            else:
                h[u] = .25 * bias

            if v in h:
                h[v] += .25 * bias
            else:
                h[v] = .25 * bias

            quadratic_offset += bias

    offset += .5 * linear_offset + .25 * quadratic_offset

    return h, J, offset
