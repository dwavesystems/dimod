"""Chain-break-resolution generators available to :func:`.iter_unembed`.

Chain-break-resolution generators enable :func:`.iter_unembed` to use different techniques
of resolving chain breaks without keeping large numbers of samples in memory. Each generator
yields zero or more unembedded samples.

"""

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

        try:
            v = chain[0]
        except IndexError:
            raise ValueError("each chain in chain_list must contain at least one variable")
        variables.append(v)

        if len(chain) == 1:
            continue

        # there must be a better way
        unbroken = np.array((samples_matrix[:, chain] == samples_matrix[:, v]).all(axis=1)).flatten()

        samples_matrix = samples_matrix[unbroken, :]

    return samples_matrix[:, variables]


def discard(sample, embedding):
    """Discard the sample if a chain is broken.

    Args:
        sample (Mapping):
            Sample as a dict of form {t: val, ...}, where t is
            a variable in the target graph and val its associated value as
            determined by a binary quadratic model sampler.
        embedding (dict):
            Mapping from source graph to target graph as a dict
            of form {s: {t, ...}, ...}, where s is a source-model variable and t is
            a target-model variable.

    Yields:
        dict: The unembedded sample if no chains were broken.

    Examples:
        This example unembeds a sample from a target graph that chains nodes 0 and 1 to
        represent source node a. The first sample has an unbroken chain, the second a broken
        chain.

        >>> import dimod
        >>> embedding = {'a': {0, 1}, 'b': {2}}
        >>> samples = {0: 1, 1: 1, 2: 0}
        >>> next(dimod.embedding.discard(samples, embedding), 'No sample')  # doctest: +SKIP
        {'a': 1, 'b': 0}
        >>> samples = {0: 1, 1: 0, 2: 0}
        >>> next(dimod.embedding.discard(samples, embedding), 'No sample')
        'No sample'

    """
    unembedded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembedded[v] = vals.pop()
        else:
            return

    yield unembedded


def majority_vote(sample, embedding):
    """Determine the sample values for chains by majority vote.

    Args:
        sample (Mapping):
            Sample as a dict of form {t: val, ...}, where t is
            a variable in the target graph and val its associated value as
            determined by a binary quadratic model sampler.
        embedding (dict):
            Mapping from source graph to target graph as a dict
            of form {s: {t, ...}, ...}, where s is a source-model variable and t is
            a target-model variable.

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen to match the most common value in the chain. For broken chains
        without a majority, one of the two values is chosen arbitrarily.

    Examples:
        This example unembeds a sample from a target graph that chains nodes 0 and 1 to
        represent source node a and nodes 2, 3, and 4 to represent source node b.
        Both samples have broken chains for source node b, with different majority values.

        >>> import dimod
        >>> embedding = {'a': {0, 1}, 'b': {2, 3, 4}}
        >>> samples = {0: 1, 1: 1, 2: 0, 3: 0, 4: 1}
        >>> next(dimod.embedding.majority_vote(samples, embedding), 'No sample')  # doctest: +SKIP
        {'a': 1, 'b': 0}
        >>> samples = {0: 1, 1: 1, 2: 1, 3: 0, 4: 1}
        >>> next(dimod.embedding.majority_vote(samples, embedding), 'No sample')  # doctest: +SKIP
        {'a': 1, 'b': 1}

    """
    unembedded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembedded[v] = vals.pop()
        else:
            unembedded[v] = _most_common(vals)

    yield unembedded


def weighted_random(sample, embedding):
    """Determine the sample values of chains by weighed random choice.

    Args:
        sample (Mapping):
            Sample as a dict of form {t: val, ...}, where t is
            a variable in the target graph and val its associated value as
            determined by a binary quadratic model sampler.
        embedding (dict):
            Mapping from source graph to target graph as a dict
            of form {s: {t, ...}, ...}, where s is a source-model variable and t is
            a target-model variable.

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen randomly, weighted by frequency of the chain's values.

    Examples:
        This example unembeds a sample from a target graph that chains nodes 0 and 1 to
        represent source node a and nodes 2, 3, and 4 to represent source node b.
        The sample has broken chains for both source nodes.

        >>> import dimod
        >>> embedding = {'a': {0, 1}, 'b': {2, 3, 4}}
        >>> samples = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
        >>> next(dimod.embedding.weighted_random(samples, embedding), 'No sample')  # doctest: +SKIP
        {'a': 0, 'b': 1}

    """
    unembedded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        # pick a random element uniformly from all vals, this weights them by
        # the proportion of each
        unembedded[v] = np.random.choice(vals)

    yield unembedded


class MinimizeEnergy(Callable):
    """Determine the sample values of broken chains by minimizing local energy.

    Args:
        linear (dict): Linear biases of the source model as a dict of
            form {s: bias, ...}, where s is a source-model variable
            and bias its associated linear bias.
        quadratic (dict): Quadratic biases of the source model as a dict
            of form {(u, v): bias, ...}, where u, v are source-model variables
            and bias the associated quadratic bias.

    Examples:
        This example embeds from a triangular graph to a square graph,
        chaining target-nodes 2 and 3 to represent source-node c, and unembeds
        using the `MinimizeEnergy` method four synthetic samples. The first two
        sample have unbroken chains, the second two have broken chains.

        >>> import dimod
        >>> h = {'a': 0, 'b': 0, 'c': 0}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> method = dimod.embedding.MinimizeEnergy(h, J)
        >>> samples = [{0: +1, 1: -1, 2: +1, 3: +1},
        ...            {0: -1, 1: -1, 2: -1, 3: -1},
        ...            {0: -1, 1: -1, 2: +1, 3: -1},
        ...            {0: +1, 1: +1, 2: -1, 3: +1}]
        ...
        >>> for source_sample in dimod.iter_unembed(samples, embedding, chain_break_method=method):  # doctest: +SKIP
        ...     print(source_sample)
        ...
        {'a': 1, 'c': 1, 'b': -1}
        {'a': -1, 'c': -1, 'b': -1}
        {'a': -1, 'c': 1, 'b': -1}
        {'a': 1, 'c': -1, 'b': 1}

    """

    def __init__(self, linear=None, quadratic=None):
        if linear is None and quadratic is None:
            raise TypeError("the minimize_energy method requires either `linear` or `quadratic` keyword arguments")
        self._linear = linear if linear is not None else defaultdict(float)
        self._quadratic = quadratic if quadratic is not None else dict()

    def __call__(self, sample, embedding):
        """
        Args:
            sample (dict): Sample as a dict of form {t: val, ...}, where t is
                a target-graph variable and val its associated value as
                determined by a binary quadratic model sampler.
            embedding (dict): Mapping from source graph to target graph as a
                dict of form {s: {t, ...}, ...} where s is a source-graph node
                and t is a target-graph node.

        Yields:
            dict: The unembedded sample. When there is a chain break, the value
            is chosen to minimize the energy relative to its neighbors.
        """
        unembedded = {}
        broken = {}  # keys are the broken source variables, values are the energy contributions

        vartype = set(itervalues(sample))
        if len(vartype) > 2:
            raise ValueError("sample has more than two different values")

        # first establish the values of all of the unbroken chains
        for v, chain in iteritems(embedding):
            vals = [sample[u] for u in chain]

            if _all_equal(vals):
                unembedded[v] = vals.pop()
            else:
                broken[v] = self._linear[v]  # broken tracks the linear energy

        # now, we want to determine the energy for each of the broken variable
        # as much as we can
        for (u, v), bias in iteritems(self._quadratic):
            if u in unembedded and v in broken:
                broken[v] += unembedded[u] * bias
            elif v in unembedded and u in broken:
                broken[u] += unembedded[v] * bias

        # in order of energy contribution, pick spins for the broken variables
        while broken:
            v = max(broken, key=lambda u: abs(broken[u]))  # biggest energy contribution

            # get the value from vartypes that minimizes the energy
            val = min(vartype, key=lambda b: broken[v] * b)

            # set that value and remove it from broken
            unembedded[v] = val
            del broken[v]

            # add v's energy contribution to all of the nodes it is connected to
            for u in broken:
                if (u, v) in self._quadratic:
                    broken[u] += val * self._quadratic[(u, v)]
                if (v, u) in self._quadratic:
                    broken[u] += val * self._quadratic[(v, u)]

        yield unembedded


def _all_equal(iterable):
    """True if all values in `iterable` are equal, else False."""
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        # empty iterable is all equal
        return True
    return all(first == rest for rest in iterator)


def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    counts = Counter(iterable)
    if counts:
        (val, __), = counts.most_common(1)
        return val
    else:
        raise ValueError("iterable must contain at least one value")
