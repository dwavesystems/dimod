from __future__ import division

import itertools

from six import iteritems, itervalues

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.embedding.chain_breaks import majority_vote
from dimod.embedding.utils import chain_to_quadratic
from dimod.response import Response
from dimod.vartypes import Vartype

__all__ = ['embed_bqm', 'embed_ising', 'embed_qubo', 'iter_unembed']


def embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=1.0, embed_singleton_variables=True):
    """Embeds a binary quadratic model onto another graph via an embedding.

    Args:
        source_bqm (:obj:`.BinaryQuadraticModel`):
            The binary quadratic model to be embedded.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float, optional):
            Magnitude quadratic bias (in SPIN-space) that should be used to create chains. Note
            that the energy penalty of chain breaks will be 2 * chain_strength.

    Returns:
        :obj:`.BinaryQuadraticModel`: The target binary quadratic model.

    """
    if embed_singleton_variables:
        unused = set(target_adjacency) - set().union(*embedding.values())
        if unused:
            # if there are unused variables then we may be alterning the embedding so we need to
            # make a copy, we'll be adding new key/value pairs so we only need a shallow copy
            embedding = embedding.copy()
    else:
        unused = False

    # create a new empty binary quadratic model with the same class as source_bqm
    target_bqm = source_bqm.__class__.empty(source_bqm.vartype)

    # go ahead and add the offset
    target_bqm.add_offset(source_bqm.offset)

    # start with the linear biases, spreading the source bias equally over the target variables in
    # the chain
    for v, bias in iteritems(source_bqm.linear):

        if v in embedding:
            chain = embedding[v]
        elif embed_singleton_variables and unused:
            # if a variable is in the source_bqm but is not mentioned in embedding, then we try to map
            # it to a source node not already mentioned in embedding
            chain = embedding[v] = {unused.pop()}
        else:
            raise ValueError('no embedding provided for source variable {}'.format(v))

        if any(u not in target_adjacency for u in chain):
            raise ValueError('chain variable {} not in target_adjacency'.format(v))

        b = bias / len(chain)

        target_bqm.add_variables_from({u: b for u in chain})

    # next up the quadratic biases, spread the quadratic biases evenly over the available
    # interactions
    for (u, v), bias in iteritems(source_bqm.quadratic):
        available_interactions = {(s, t) for s in embedding[u] for t in embedding[v] if s in target_adjacency[t]}

        if not available_interactions:
            raise ValueError("no edges in target graph between source variables {}, {}".format(u, v))

        b = bias / len(available_interactions)

        target_bqm.add_interactions_from((u, v, b) for u, v in available_interactions)

    for chain in itervalues(embedding):
        quadratic_chain_biases = chain_to_quadratic(chain, target_adjacency, chain_strength)
        target_bqm.add_interactions_from(quadratic_chain_biases, vartype=Vartype.SPIN)  # these are spin

        # add the energy for satisfied chains to the offset
        energy_diff = -sum(itervalues(quadratic_chain_biases))
        target_bqm.add_offset(energy_diff)

    return target_bqm


def embed_ising(souce_h, source_J, embedding, target_adjacency, chain_strength=1.0):
    """Embeds an Ising problem onto another graph via an embedding.

    Args:
        source_h (dict[variable, bias]/list[bias]):
            Linear biases of the Ising problem. If a list, the list's indices are used as
            variable labels.

        source_J (dict[(variable, variable), bias]):
            Quadratic biases of the Ising problem.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float, optional):
            Magnitude quadratic bias (in SPIN-space) that should be used to create chains. Note
            that the energy penalty of chain breaks will be 2 * chain_strength.

    Returns:
        tuple: A 2-tuple:

            dict[variable, bias]: Linear biases of the target Ising problem.

            dict[(variable, variable), bias]: Quadratic biases of the target Ising problem.

    """
    source_bqm = BinaryQuadraticModel.from_ising(souce_h, source_J)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength)
    target_h, target_J, __ = target_bqm.to_ising()
    return target_h, target_J


def embed_qubo(source_Q, embedding, target_adjacency,  chain_strength=1.0):
    """Embed an Ising problem onto another graph via an embedding.

    Args:
        source_Q (dict[(variable, variable), bias]):
            Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float, optional):
            Magnitude quadratic bias (in SPIN-space) that should be used to create chains. Note
            that the energy penalty of chain breaks will be 2 * chain_strength.

    Returns:
        dict[(variable, variable), bias]: Quadratic biases of the target QUBO.

    """
    source_bqm = BinaryQuadraticModel.from_qubo(source_Q)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength)
    target_Q, __ = target_bqm.to_qubo()
    return target_Q


def iter_unembed(target_samples, embedding, chain_break_method=None):
    """Yield unembedded samples.

    Args:
        target_samples (iterable[mapping[variable, value]]):
            Iterable of samples. Each sample of the form {v: val, ...} where v is a variable in the
            target graph and val is the associated value.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        chain_break_method (function, optional, default=:func:`.majority_vote`):
            The method used to resolve chain breaks.

    Yields:
        dict[variable, value]: An unembedded sample of the form {v: val, ...} where v is a variable
            in the source graph and val is the associated value.

    Notes:
        This is implemented as an iterator to keep down memory footprint. It is intended to be used
        with :meth:`.Response.from_dicts`.

    Examples:
        todo

    """
    if chain_break_method is None:
        chain_break_method = majority_vote

    for target_sample in target_samples:
        for source_sample in chain_break_method(target_sample, embedding):
            yield source_sample
