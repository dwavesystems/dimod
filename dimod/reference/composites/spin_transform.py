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
"""
On the D-Wave system, coupling :math:`J_{i,j}` adds a small bias to qubits :math:`i` and
:math:`j` due to leakage. This can become significant for chained qubits. Additionally,
qubits are biased to some small degree in one direction or another.
Applying a spin-reversal transform can improve results by reducing the impact of possible
analog and systematic errors. A spin-reversal transform does not alter the Ising problem;
the transform simply amounts to reinterpreting spin up as spin down, and visa-versa, for
a particular spin.
"""
from random import random
import itertools

import numpy as np

from dimod.core.composite import Composite
from dimod.core.sampler import Sampler
from dimod.core.structured import Structured
from dimod.sampleset import SampleSet
from dimod.vartypes import Vartype

__all__ = ['SpinReversalTransformComposite']


class SpinReversalTransformComposite(Sampler, Composite):
    """Composite for applying spin reversal transform preprocessing.

    Spin reversal transforms (or "gauge transformations") are applied
    by flipping the spin of variables in the Ising problem. After
    sampling the transformed Ising problem, the same bits are flipped in the
    resulting sample.

    Args:
        sampler: A `dimod` sampler object.

    Attributes:
        children (list):
            [`sampler`] where `sampler` is the input sampler.
        structure:
            Inherited from input `sampler`.

    Examples:
        This example composes a dimod ExactSolver sampler with spin transforms then
        uses it to sample an Ising problem.

        >>> import dimod
        ...
        >>> # Compose the sampler
        >>> base_sampler = dimod.ExactSolver()
        >>> composed_sampler = dimod.SpinReversalTransformComposite(base_sampler)
        >>> base_sampler in composed_sampler.children
        True
        >>> # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> print(next(response.data()))           # doctest: +SKIP
        Sample(sample={'a': 1, 'b': 1}, energy=-1.5)

    References
    ----------
    .. [KM] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    """
    children = None
    parameters = None
    properties = None

    def __init__(self, child):
        self.children = [child]

        if isinstance(child, Structured):
            # todo something like Structured.__init__(self)
            raise NotImplementedError

        self.parameters = parameters = {'num_spin_reversal_transforms': [],
                                        'spin_reversal_variables': []}
        parameters.update(child.parameters)

        self.properties = {'child_properties': child.properties}

    def sample(self, bqm, num_spin_reversal_transforms=2, spin_reversal_variables=None, **kwargs):
        """Sample from the binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.
            num_spin_reversal_transforms (integer, optional, default=2):
                Number of spin reversal transform runs.
            spin_reversal_variables (list/dict, optional, default=None):
                Variables to which to apply the spin reversal. If None, every variable
                has a 50% probability of being selected.

        Returns:
            :obj:`.SampleSet`

        Examples:
            This example runs 100 spin reversals applied to one variable of a QUBO problem.

            >>> import dimod
            ...
            >>> base_sampler = dimod.ExactSolver()
            >>> composed_sampler = dimod.SpinReversalTransformComposite(base_sampler)
            >>> Q = {('a', 'a'): -1, ('b', 'b'): -1, ('a', 'b'): 2}
            >>> response = composed_sampler.sample_qubo(Q,
            ...               num_spin_reversal_transforms=100,
            ...               spin_reversal_variables={'a'})
            >>> len(response)
            400
            >>> print(next(response.data()))           # doctest: +SKIP
            Sample(sample={'a': 0, 'b': 1}, energy=-1.0)

        """
        # make a main response
        responses = []

        for ii in range(num_spin_reversal_transforms):
            if spin_reversal_variables is None:
                # apply spin transform to each variable with 50% chance
                transform = list(v for v in bqm.linear if random() > .5)
            else:
                transform = list(spin_reversal_variables)

            flipped_bqm = bqm.copy()

            for v in transform:
                flipped_bqm.flip_variable(v)

            flipped_response = self.child.sample(bqm, **kwargs)

            tf_idxs = [flipped_response.variables.index[v] for v in flipped_response.variables]

            if bqm.vartype is Vartype.SPIN:
                flipped_response.record.sample[:, tf_idxs] = -1 * flipped_response.record.sample[:, tf_idxs]
            else:
                flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]

            responses.append(flipped_response)

        # # stack the records
        record = np.rec.array(np.hstack((resp.record for resp in responses)))

        vartypes = set(resp.vartype for resp in responses)
        if len(vartypes) > 1:
            raise RuntimeError("inconsistent vartypes returned")
        vartype = vartypes.pop()

        info = {}
        for resp in responses:
            info.update(resp.info)

        labels = responses[0].variables

        return SampleSet(record, labels, info, vartype)
