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

import numpy as np

__all__ = ['broken_chains',
           'discard',
           'majority_vote',
           'weighted_random',
           # 'MinimizeEnergy',
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
        chain = np.asarray(chain)

        if chain.ndim > 1:
            raise ValueError("chains should be 1D array_like objects")

        # # chains of length 1, or 0 cannot be broken
        if len(chain) <= 1:
            continue

        all_ = (samples[:, chain] == 1).all(axis=1)
        any_ = (samples[:, chain] == 1).any(axis=1)
        broken[:, cidx] = np.bitwise_xor(all_, any_)

    return np.ascontiguousarray(broken)


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

    # it sufficies to choose a random index in each chain and use that to construct the matrix
    idx = [np.random.choice(chain) for chain in chains]

    return samples[:, idx], np.arange(num_samples)  # we keep all of the samples in this case


# class MinimizeEnergy(Callable):
#     def __init__(self, bqm, variable_order):
#         self.linear, self.quadratic, __ = bqm.to_numpy_vectors(variable_order)

#     def __call__(self, samples, chains):
#         samples = np.asarray(samples)
#         if samples.ndim != 2:
#             raise ValueError("expected samples to be a numpy 2D array")

#         raise NotImplementedError
