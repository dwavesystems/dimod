import numpy as np

from ezdw import chimera


def generate_random_problem(edges, k, random):
    """
    A random problem has zero fields, and couplers chosen uniformly at random in {-k, -k+1, ..., -1, 1, ..., k}
    if `k` is finite, else [-1, 1].
    :param edges: the (Chimera) edges for which coupler values are provided.
    :param k: a non-negative integer, possibly infinite precision for the coupler values.
    :param random: the random state used for sampling.
    :return: a random problem.
    """
    assert k > 0
    if np.isinf(k):

        def value_factory(): return random.uniform(-1, 1)
    else:
        assert np.equal(np.mod(k, 1), 0), 'Finite precision must be integral.'
        values = np.arange(-k, k + 1)
        values = np.delete(values, k)  # Remove the 0 value.

        def value_factory(): return random.choice(values)
    J = {coupler: value_factory() for coupler in edges}
    return J


def generate_anticluster_problem(edges, k, random, dims=None):
    """
    An anticluster problem has zero fields, inter-cell couplers chosen uniformly at random from {-k, k}, and
    intra-cell couplers chosen uniformly at random from {-1, 1}.
    :param edges: the (Chimera) edges for which coupler values are provided.
    :param k: a non-negative, integer precision for the inter-cell coupler values.
    :param random: the random state used for sampling.
    :param dims: Chimera dimensions tuple (see arguments to `ezdw.chimera.get_chimera_dimensions`). If
        `None`, smallest square MxMx4 Chimera graph is inferred from edges.
    :return: an anticluster problem.
    """
    assert k > 0
    assert not np.isinf(k)
    assert np.equal(np.mod(k, 1), 0), 'Precision must be integral.'
    if dims is None:
        n = max(i for e in edges for i in e)+1
        dims = (chimera.get_min_square_chimera_dimensions(n),)
    J = {}
    for lu, lv in edges:
        cu, cv = chimera.ltoc(lu, *dims), chimera.ltoc(lv, *dims)
        assert chimera.is_edge(cu, cv)
        jj = k if chimera.is_intercell_edge(cu, cv) else 1
        J[lu, lv] = random.choice([-jj, jj])
    return J
