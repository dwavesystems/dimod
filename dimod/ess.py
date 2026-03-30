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

from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from dimod import SampleSet

from math import floor, isnan
from warnings import warn

import numpy as np

__all__ = ['compute_ess', 'compute_ess_sampleset']


def compute_ess(x: np.ndarray, batch_size: int | None = None) -> float:
    """Estimates the effective sample size of ``x``.

    NOTE: The estimate can be nan or negative for extreme cases (e.g., constants). This can occur
    by definition of the estimator and is not a bug.

    Examples:
        These two examples demonstrate typical use cases of the estimator based on an energy statistic.
        The first example measures the QPU's ESS. The second example measures the ESS of two
        Metropolis-Hastings samplers (one with one sweep, another with ten sweeps).

        Example (1): QPU
        .. code-block:: python

            import numpy as np
            from dimod.ess import estimate_effective_sample_size
            from dimod.generators import power_r
            from dwave.system import DWaveCliqueSampler

            markov_chain_length = 100
            num_vars = 100
            num_chains = 10
            bqm = power_r(512, num_vars)
            bqm.normalize()
            qpu = DWaveCliqueSampler()
            qpu_energy = np.array(
                [qpu.sample(bqm, num_reads=markov_chain_length, answer_mode="raw").record.energy
                for _ in range(num_chains)]
            )
            print("Effective sample size per chain (QPU):",
                  estimate_effective_sample_size(qpu_energy)/num_chains)
            # Effective sample size per chain (QPU): 71.87414368659094


        Example (2): Metropolis-Hastings
        .. code-block::python
            import numpy as np
            from dwave.samplers import SimulatedAnnealingSampler
            from dimod.ess import estimate_effective_sample_size
            from dimod.generators import power_r
            neal = SimulatedAnnealingSampler()
            markov_chains = []
            markov_chain_length = 100
            num_vars = 100
            num_chains = 10
            num_sweeps = 1  # Increase number of sweeps to increase ESS
            bqm = power_r(512, num_vars)
            bqm.normalize()
            initial_state = None
            for chain_idx in range(num_chains):
                chain = []
                for time_idx in range(markov_chain_length):
                    sample_set = neal.sample(
                        bqm, beta_schedule=[1.0]*num_sweeps,
                        beta_schedule_type="custom", initial_states=initial_state)
                    chain.append(sample_set.record.energy.item())
                    initial_state = (sample_set.record.sample, sample_set.variables)
                markov_chains.append(chain)
            mc_energy = np.array(markov_chains)
            print("Effective sample size per chain (MH):",
                  estimate_effective_sample_size(mc_energy)/num_chains)
            # Effective sample size per chain (MH): 12.496482970401358

        Use a larger number of sweeps to achieve larger ESS
        >>> num_sweeps = 10
        Effective sample size per chain (MH): 64.44999653861495


    Args:
        x: An (m, n) matrix where rows index independent Markov chains and columns index
            time steps.
        batch_size: Batch size of the estimator. If ``None``, then ``batch_size`` is set to the floor
            of the square root of ``n``. Defaults to None.

    Returns:
        float: An estimate of the effective sample size of ``x``. The estimate can be NaN or
        even negative when the input chain is nearly constant. This can occur by definition of the
        estimator and is not a bug.
    """
    if x.ndim != 2:
        raise ValueError("The input matrix ``x`` should have shape (m, n) where m indexes "
                         f"independent Markov chains and n indexes time. ``x`` has shape {x.shape}")
    m, n = x.shape
    if batch_size is None:
        batch_size = floor(n**0.5)
    if batch_size > n or batch_size < 3:
        raise ValueError(
            f"Batch size should be at least three but no more than the chain length of the Markov "
            f"chain. Batch size is {batch_size} and chain length is {n}. If size was not given, it defaults"
            f"to the floor of square-root of the chain length."
        )

    s_squared = x.var(1, ddof=1).mean()
    # = second equation at the top of page 7
    # = average of "sample variance within series"

    # This estimator $\hat{\tau}^2_L$ is defined in equation (5) of
    # Revisiting the Gelman-Rubin Diagnostic (https://arxiv.org/abs/1812.09384)
    tau_squared = (2 * _estimate_replicated_batch_means(x, batch_size)
                   - _estimate_replicated_batch_means(x, batch_size // 3))
    # = equation (5)
    # = nVar(xbar_i.) = total variance of the mean-within-series

    sigma_squared = ((n - 1) / n) * s_squared + tau_squared / n
    # = first equation at the top of page 10
    # = estimate of the distribution's variance

    ess = (m * n * sigma_squared / tau_squared).item()
    # = estimate of the effective sample size
    # = first equation at the top of page 14
    if isnan(ess) or ess < 0:
        warn("The estimated effective sample size is NaN or negative. This can occur by definition "
             "of the estimator and is not a bug.")
    return ess


def compute_ess_sampleset(
    sample_set: SampleSet,
    test_function: Callable[[SampleSet], Iterable[float]] | None = None,
    batch_size: int | None = None
):
    """Estimates the effective sample size of ``sample_set``.

    NOTE: The estimate can be nan or negative for extreme cases (e.g., constants). This can occur
    by definition of the estimator and is not a bug.

    This function differs from ``compute_ess`` in that it assumes its input, ``sample_set``, consists
    of states from a single Markov chain. By default, when no test
    function is supplied, the effective sample size is estimated based on the sample set's energies.


    Examples:
        This example demonstrates a typical use case of the estimator, measuring a QPU sample's
        ESS based on the magnetization of the system.


        Example (1): QPU
        .. code-block::python
            import numpy as np
            from dwave.system import DWaveCliqueSampler
            from dimod.ess import estimate_effective_sample_size
            from dimod.generators import power_r
            markov_chain_length = 100
            num_vars = 33
            bqm = power_r(512, num_vars)
            bqm.normalize()
            qpu = DWaveCliqueSampler()

            def test_fn(ss):
                return ss.record.sample.mean(1)
            sample_set = qpu.sample(bqm, num_reads=markov_chain_length, answer_mode="raw")
            print("Effective sample size (QPU):",
                  estimate_effective_sample_size_sampleset(sample_set, test_fn))
            # Effective sample size per chain (QPU): 69.69037448554732


    Args:
        sample_set: A sample set with m ordered reads and n variables. In a typical use case with
            QPU samples, the sample set should have been attained with the sampling parameter
            ``answer_mode`` set to "raw".
        test_function: A function mapping ``sample_set`` to an iterable of floats of length m (the
            sample size of the sample set). If ``None``, then the default test function is the energy
            of the model as reported by the sample set. Defaults to None.
        batch_size: Batch size of the estimator. If ``None``, then ``batch_size`` is set to the floor
            of the square root of ``n``. Defaults to None.

    Returns:
        float: An estimate of the effective sample size of ``sample_set``. The estimate can be NaN or
        even negative when the input chain is nearly constant. This can occur by definition of the
        estimator and is not a bug.
    """
    if test_function is None:
        def test_function(ss: SampleSet):
            return ss.record.energy

    try:
        stat = test_function(sample_set)
    except Exception as e:
        raise RuntimeError(
            "Test function raised an error. Does the test function have the correct signature? "
            "The test function should consume a ``dimod.SampleSet`` and return an iterable of"
            "floats with length equal to the sample size."
        ) from e

    if not isinstance(stat, Iterable):
        raise ValueError(
            "The test function should return an *iterable* of floats with length equal to the "
            f"sample size. The test function returned {stat}."
        )

    stat = list(stat)
    X = np.array(stat)

    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError(
            "The test function should return an iterable of *floats* with length equal to the "
            f"sample size. The test function returned {stat}."
        )

    if X.ndim != 1:
        raise ValueError(
            "The test function should consume a sample set and return a *flat* iterable of "
            f"floats. The output of `list(test_function(sample_set))` is {stat}."
        )

    if X.shape[0] != len(sample_set):
        raise ValueError("The test function should consume a sample set and return an iterable of "
                         "floats of *length* equal to the sample size of the sample set. The sample "
                         f"size of the sample set is {len(sample_set)} and the length of the test "
                         f"function output has shape {X.shape}.")

    return compute_ess(X.reshape(1, -1), batch_size)


def _estimate_replicated_batch_means(x: np.ndarray, batch_size: int) -> float:
    """Computes the replicated batch means estimate.

    This estimator (:math:`\\hat\\tau^2_b`) is defined in the equation above equation (5) of
    `<Revisiting the Gelman-Rubin Diagnostic https://arxiv.org/abs/1812.09384>`_.

    The estimator batches each Markov chain into batches of size ``batch_size``, estimates the mean
    of each batch, and computes the sample variance of these batched means.

    The first few columns of ``x`` may be dropped in the estimation process in order to satisfy the
    requirement that the length of the Markov chain is divisible by the batch size.

    Args:
        x: An (m, n) matrix where rows index independent Markov chains and columns index
            time steps.
        batch_size: Batch size of the estimator.

    Returns:
        float: Replicated batch means estimate.
    """
    n = x.shape[1]

    n_batches = n // batch_size
    trimmed_length = batch_size * n_batches

    x = x[:, (n - trimmed_length):]
    ybar = np.mean(np.split(x, n_batches, axis=1), axis=2).T

    res = batch_size * np.var(ybar, ddof=1)
    # NOTE: this is equivalent to
    # res = ((ybar - muhat) ** 2).sum() * b / (n_batches * m - 1)
    # where
    # muhat = x.mean()
    return res
