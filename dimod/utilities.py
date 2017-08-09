"""
A collection of utility functions useful for Samplers.
"""

import sys

from collections import defaultdict

__all__ = ['ising_energy', 'qubo_energy', 'ising_to_qubo', 'qubo_to_ising']

PY2 = sys.version_info[0] == 2
if PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


def ising_energy(h, J, sample):
    """Calculate the Ising energy of the given sample.

    H(s) = sum_i h_i * s_i + sum_(i, j) J_(i,j) * s_i * s_j

    https://en.wikipedia.org/wiki/Ising_model

    Args:
        h: The linear biases in a dict of the form {var: bias, ...}.
        J: The quadratic biases in a dict of the form
        {(var0, var1): bias, ...}.
        sample: A dict of spins of the form {var: spin, ...} where
        each spin is either -1 or 1.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    """
    energy = 0

    # add the contribution from the linear biases
    for v in h:
        energy += h[v] * sample[v]

    # add the contribution from the quadratic biases
    for v0, v1 in J:
        energy += J[(v0, v1)] * sample[v0] * sample[v1]

    return energy


def qubo_energy(Q, sample):
    """Calculate the quadratic polynomial value of the given sample
    to a quadratic unconstrained binary optimization (QUBO) problem.

    E(x) = sum_(i, j) Q_(i, j) * x_i * x_j

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    Args:
        Q: A dict of the QUBO coefficients of the form
        {(var0, var1): coeff, ...}
        sample: A dict of binary variables of the form
        {var: bin, ...} where each bin is either 0 or 1.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    """
    energy = 0

    for v0, v1 in Q:
        energy += sample[v0] * sample[v1] * Q[(v0, v1)]

    return energy


def ising_to_qubo(h, J):
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

    Returns:
        (dict, float): A dict of the QUBO coefficients. The energy offset.

    """

    q = defaultdict(float)
    offset = 0

    # the linear biases are the easiest
    for v, bias in iteritems(h):
        q[(v, v)] = 2 * bias
        offset -= bias

    # next the quadratic biases
    for (u, v), bias in iteritems(J):
        q[(u, v)] += 4 * bias
        q[(u, u)] -= 2 * bias
        q[(v, v)] -= 2 * bias
        offset += bias

    # finally convert q to a dict, rather than default dict
    q = dict((k, v) for k, v in iteritems(q) if v != 0)
    return q, offset


def qubo_to_ising(Q):
    """Converts a QUBO problem to an Ising problem.

    Map a binary quadratic program x' * Q * x defined over 0/1 variables to
    an Ising model defined over -1/+1 variables. We return the h and J
    defining the Ising model as well as the offset in energy between the
    two problem formulations, i.e. x' * Q * x = offset + s' * J * s + h' * s. The
    linear term of the QUBO is contained along the diagonal of Q.

    See ising_to_qubo(h, J) for the inverse function.

    Args:
        Q: A dict of the QUBO coefficients.

    Returns:
        (dict, dict, float):
        A dict of the linear coefficients of the Ising problem.
        A dict of the quadratic coefficients of the Ising problem.
        The energy offset.

    """
    h = defaultdict(float)
    j = {}
    offset = 0

    for (i, k), e in iteritems(Q):
        if i == k:
            # linear biases
            h[i] += 0.5 * e
            offset += 0.5 * e
        else:
            # quadratic biases
            j[(i, k)] = 0.25 * e
            h[i] += 0.25 * e
            h[k] += 0.25 * e
            offset += 0.25 * e

    # remove the 0 entries of J
    h = dict(h)
    j = dict((k, v) for k, v in iteritems(j) if v != 0)

    return h, j, offset
