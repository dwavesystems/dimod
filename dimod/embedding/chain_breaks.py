from collections import Counter, defaultdict, Callable

from six import iteritems, itervalues

import numpy as np

__all__ = ['majority_vote', 'discard', 'weighted_random', 'MinimizeEnergy']


def discard_matrix(samples_matrix, chain_list):
    """Discard broken chains."""
    if not isinstance(samples_matrix, np.matrix):
        samples_matrix = np.matrix(samples_matrix, dtype='int8')
    num_samples, __ = samples_matrix.shape

    variables = []
    for chain in chain_list:
        chain = list(chain)

        v = chain[0]
        variables.append(v)

        if len(chain) == 1:
            continue

        # there must be a better way
        unbroken = np.array((samples_matrix[:, chain] == samples_matrix[:, v]).all(axis=1)).flatten()

        samples_matrix = samples_matrix[unbroken, :]

    return samples_matrix[:, variables]


def discard(sample, embedding):
    """Discards the sample if broken.

    Args:
        sample (Mapping): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.

    Yields:
        dict: The unembedded sample is no chains were broken.

    """
    unembeded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembeded[v] = vals.pop()
        else:
            return

    yield unembeded


def majority_vote(sample, embedding):
    """Determines the sample values by majority vote.

    Args:
        sample (Mapping): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen to match the most common value in the chain.

    """
    unembeded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembeded[v] = vals.pop()
        else:
            unembeded[v] = _most_common(vals)

    yield unembeded


def weighted_random(sample, embedding):
    """Determines the sample values by weighed random choice.

    Args:
        sample (Mapping): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen randomly, weighted by the frequency of the values
        within the chain.

    """
    unembeded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        # pick a random element uniformly from all vals, this weights them by
        # the proportion of each
        unembeded[v] = random.choice(vals)

    yield unembeded


class MinimizeEnergy(Callable):
    """Determine the sample values by minimizing the local energy.

    Args:
        linear (dict): The linear biases of the source model. Should be a dict of
            the form {v: bias, ...} where v is a variable in the source model
            and bias is the linear bias associated with v.
        quadratic (dict): The quadratic biases of the source model. Should be a dict
            of the form {(u, v): bias, ...} where u, v are variables in the
            source model and bias is the quadratic bias associated with (u, v).

    Examples:
        This is a callable object

        .. code-block:: python

            # Get an Ising problem, source graph in this case is a triangle and use it to define
            # the chain resolution method.
            h = {'a': 0, 'b': 0, 'c': 0}
            J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
            method = dimod.embedding.MinimizeEnergy(h, J)

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # Now say we have a set of target samples
            samples = [{0: +1, 1: -1, 2: +1, 3: +1},
                       {0: -1, 1: -1, 2: -1, 3: -1},
                       {0: +1, 1: +1, 2: +1, 3: +1}]

            for source_sample in dimod.iter_unembed(samples, embedding, chain_break_method=method):
                pass

    """

    def __init__(self, linear=None, quadratic=None):
        if linear is None and quadratic is None:
            raise TypeError("the minimize_energy method requires `linear` and `quadratic` keyword arguments")
        self._linear = linear if linear is not None else defaultdict(float)
        self._quadratic = quadratic if quadratic is not None else dict()

    def __call__(self, sample, embedding):
        """
        Args:
            sample (dict): A sample of the form {v: val, ...} where v is
                a variable in the target graph and val is the associated value as
                determined by a binary quadratic model sampler.
            embedding (dict): The mapping from the source graph to the target graph.
                Should be of the form {v: {s, ...}, ...} where v is a node in the
                source graph and s is a node in the target graph.

        Yields:
            dict: The unembedded sample. When there is a chain break, the value
            is chosen to minimize the energy relative to its neighbors.
        """
        unembeded = {}
        broken = {}  # keys are the broken source variables, values are the energy contributions

        vartype = set(itervalues(sample))
        if len(vartype) > 2:
            raise ValueError("sample has more than two different values")

        # first establish the values of all of the unbroken chains
        for v, chain in iteritems(embedding):
            vals = [sample[u] for u in chain]

            if _all_equal(vals):
                unembeded[v] = vals.pop()
            else:
                broken[v] = self._linear[v]  # broken tracks the linear energy

        # now, we want to determine the energy for each of the broken variable
        # as much as we can
        for (u, v), bias in iteritems(self._quadratic):
            if u in unembeded and v in broken:
                broken[v] += unembeded[u] * bias
            elif v in unembeded and u in broken:
                broken[u] += unembeded[v] * bias

        # in order of energy contribution, pick spins for the broken variables
        while broken:
            v = max(broken, key=lambda u: abs(broken[u]))  # biggest energy contribution

            # get the value from vartypes that minimizes the energy
            val = min(vartype, key=lambda b: broken[v] * b)

            # set that value and remove it from broken
            unembeded[v] = val
            del broken[v]

            # add v's energy contribution to all of the nodes it is connected to
            for u in broken:
                if (u, v) in self._quadratic:
                    broken[u] += val * self._quadratic[(u, v)]
                if (v, u) in self._quadratic:
                    broken[u] += val * self._quadratic[(v, u)]

        yield unembeded


def _all_equal(iterable):
    """True if all values in `iterable` are equal, else False."""
    iterator = iter(iterable)
    first = next(iterator)
    return all(first == rest for rest in iterator)


def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)
