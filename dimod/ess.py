# Copyright 2026 D-Wave
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

from __future__ import annotations

from math import floor

import numpy as np

__all__ = ['estimate_effective_sample_size']


def estimate_effective_sample_size(x: np.ndarray, b: int | None = None) -> float:
    """Estimates the effective sample size of ``x``.

    The effective sample size (ESS) is the number of effectively independent samples drawn from
    Markov chains' stationary distribution. The univariate estimator implemented here is the
    (multivariate) estimator defined in the first equation at the top of page 14 in
    `<Revisiting the Gelman-Rubin Diagnostic https://arxiv.org/abs/1812.09384>`_.

    Args:
        x: An (m, n) matrix where rows index independent Markov chains and columns index
            time steps.
        b: Batch size of the estimator. If ``None``, then ``b`` is set to the floor of the
            square root of ``n``. Defaults to None.

    Returns:
        float: An estimate of the effective sample size of ``x``.
    """
    if x.ndim != 2:
        raise ValueError("The input matrix ``x`` should have shape (m, n) where m indexes "
                         f"independent Markov chains and n indexes time. ``x`` has shape {x.shape}")
    m, n = x.shape
    if b is None:
        b = int(floor(n**0.5))
    if b > n or b < 3:
        raise ValueError(
            f"Batch size should be at least three but no more than ``n``. Batch size is {b} and "
            f"``n`` is {n}. If size was not given, it defaults to the floor of square-root of ``n`` "
            f"(n={n})."
        )

    s_squared = x.var(1, ddof=1).mean()
    # = second equation at the top of page 7
    # = average of "sample variance within series"

    # This estimator $\hat{\tau}^2_L$ is defined in equation (5) of
    # Revisiting the Gelman-Rubin Diagnostic (https://arxiv.org/abs/1812.09384)
    tau_squared = (2 * _estimate_replicated_batch_means(x, b)
                   - _estimate_replicated_batch_means(x, b // 3))
    # = equation (5)
    # = nVar(xbar_i.) = total variance of the mean-within-series

    sigma_squared = ((n - 1) / n) * s_squared + tau_squared / n
    # = first equation at the top of page 10
    # = estimate of the distribution's variance

    ess = m * n * sigma_squared / tau_squared
    # = estimate of the effective sample size
    # = first equation at the top of page 14
    return ess.item()


def _estimate_replicated_batch_means(x: np.ndarray, b: int) -> float:
    """Computes the replicated batch means estimate.

    This estimator (:math:`\\hat\\tau^2_b`) is defined in the equation above equation (5) of
    `<Revisiting the Gelman-Rubin Diagnostic https://arxiv.org/abs/1812.09384>`_.

    The estimator batches each Markov chain into batches of size ``b``, estimates the mean of each
    batch, and computes the sample variance of these batched means.

    The first few columns of ``x`` may be dropped in the estimation process in order to satisfy the
    requirement that the length of the Markov chain is divisible by the batch size.

    Args:
        x: An (m, n) matrix where rows index independent Markov chains and columns index
            time steps.
        b: Batch size of the estimator.

    Returns:
        float: Replicated batch means estimate.
    """
    n = x.shape[1]

    n_batches = n // b
    trimmed_length = b * n_batches

    x = x[:, (n - trimmed_length):]
    ybar = np.mean(np.split(x, n_batches, axis=1), axis=2).mT

    res = b*np.var(ybar, ddof=1)
    # NOTE: this is equivalent to
    # res = ((ybar - muhat) ** 2).sum() * b / (n_batches * m - 1)
    # where
    # muhat = x.mean()
    return res
