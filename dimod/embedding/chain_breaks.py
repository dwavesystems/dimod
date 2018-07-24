# Copyright 2018 D-Wave Systems Inc.
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
#
# ================================================================================================
from __future__ import division

from collections import Callable
from heapq import heapify, heappop

import numpy as np

from dimod.vartypes import SPIN

__all__ = ['broken_chains',
           'discard',
           'majority_vote',
           'weighted_random',
           'MinimizeEnergy',
           ]


def broken_chains(samples, chains):
    """Find the broken chains.

    Args:
        samples (array_like):
            Samples as a nS x nV array_like object where nS is the number of samples and nV is the
            number of variables.

        chains (list):
            List of chains of length nC where nC is the number of chains.
            Each chain should be an array_like collection of indices in samples.

    Returns:
        np.ndarray: A nS x nC boolean array. If i, j is True, then chain j in sample i is broken.

    Examples:
        >>> samples = np.array([[-1, +1, -1, +1], [-1, -1, +1, +1]], dtype=np.int8)
        >>> chains = [[0, 1], [2, 3]]
        >>> dimod.broken_chains(samples, chains)
        array([[True, True],
               [ False,  False]])

        >>> samples = np.array([[-1, +1, -1, +1], [-1, -1, +1, +1]], dtype=np.int8)
        >>> chains = [[0, 2], [1, 3]]
        >>> dimod.broken_chains(samples, chains)
        array([[False, False],
               [ True,  True]])

    Notes:
        Assumes that samples are binary and that one of the values is 1.

    """
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("expected samples to be a numpy 2D array")

    num_samples, num_variables = samples.shape
    num_chains = len(chains)

    broken = np.zeros((num_samples, num_chains), dtype=bool, order='F')

    for cidx, chain in enumerate(chains):
        if isinstance(chain, set):
            chain = list(chain)
        chain = np.asarray(chain)

        if chain.ndim > 1:
            raise ValueError("chains should be 1D array_like objects")

        # chains of length 1, or 0 cannot be broken
        if len(chain) <= 1:
            continue

        all_ = (samples[:, chain] == 1).all(axis=1)
        any_ = (samples[:, chain] == 1).any(axis=1)
        broken[:, cidx] = np.bitwise_xor(all_, any_)

    return broken


def discard(samples, chains):
    """Discard broken chains."""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("expected samples to be a numpy 2D array")

    num_samples, num_variables = samples.shape
    num_chains = len(chains)

    broken = broken_chains(samples, chains)

    unbroken_idxs, = np.where(~broken.any(axis=1))

    chain_variables = np.fromiter((np.asarray(chain)[0] for chain in chains),
                                  count=num_chains, dtype=int)

    return samples[np.ix_(unbroken_idxs, chain_variables)], unbroken_idxs


def majority_vote(samples, chains):
    """Use the most common element in broken chains"""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("expected samples to be a numpy 2D array")

    num_samples, num_variables = samples.shape
    num_chains = len(chains)

    unembedded = np.empty((num_samples, num_chains), dtype='int8', order='F')

    # determine if spin or binary. If samples are all 1, then either method works, so we use spin
    # because it is faster
    if samples.all():  # spin-valued
        for cidx, chain in enumerate(chains):
            # we just need the sign for spin. We don't use np.sign because in that can return 0
            # and fixing the 0s is slow.
            unembedded[:, cidx] = 2*(samples[:, chain].sum(axis=1) >= 0) - 1
    else:  # binary-valued
        for cidx, chain in enumerate(chains):
            mid = len(chain) / 2
            unembedded[:, cidx] = (samples[:, chain].sum(axis=1) >= mid)

    return unembedded, np.arange(num_samples)  # we keep all of the samples in this case


def weighted_random(samples, chains):
    """Determine the sample values of chains by weighed random choice."""
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("expected samples to be a numpy 2D array")

    # it sufficies to choose a random index from each chain and use that to construct the matrix
    idx = [np.random.choice(chain) for chain in chains]

    num_samples, num_variables = samples.shape
    return samples[:, idx], np.arange(num_samples)  # we keep all of the samples in this case


class MinimizeEnergy(Callable):
    """"""
    def __init__(self, bqm, embedding):
        # this is an awkward construction but we need it to maintain consistency with the other
        # chain break methods
        self.chain_to_var = {frozenset(chain): v for v, chain in embedding.items()}

        self.bqm = bqm

    def __call__(self, samples, chains):
        samples = np.asarray(samples)
        if samples.ndim != 2:
            raise ValueError("expected samples to be a numpy 2D array")

        chain_to_var = self.chain_to_var
        variables = [chain_to_var[frozenset(chain)] for chain in chains]
        chains = [np.asarray(list(chain)) if isinstance(chain, set) else np.array(chain) for chain in chains]

        # we want the bqm by index
        bqm = self.bqm.relabel_variables({v: idx for idx, v in enumerate(variables)}, inplace=False)

        num_chains = len(chains)

        if bqm.vartype is SPIN:
            ZERO = -1
        else:
            ZERO = 0

        def _minenergy(arr):

            # energies = []

            unbroken_arr = np.zeros((num_chains,), dtype=np.int8)
            broken = []

            for cidx, chain in enumerate(chains):

                eq1 = (arr[chain] == 1)
                if not np.bitwise_xor(eq1.all(), eq1.any()):
                    # not broken
                    unbroken_arr[cidx] = arr[chain][0]
                else:
                    broken.append(cidx)

            energies = []
            for cidx in broken:
                en = bqm.linear[cidx] + sum(unbroken_arr[idx] * bqm.adj[cidx][idx] for idx in bqm.adj[cidx])
                energies.append([-abs(en), en, cidx])
            heapify(energies)

            while energies:
                _, e, i = heappop(energies)

                unbroken_arr[i] = val = ZERO if e > 0 else 1

                for energy_triple in energies:
                    k = energy_triple[2]
                    energy_triple[1] += val * bqm.adj[i][k]
                    energy_triple[0] = -abs(energy_triple[1])

                heapify(energies)

            return unbroken_arr

        num_samples, num_variables = samples.shape
        return np.apply_along_axis(_minenergy, 1, samples), np.arange(num_samples)
