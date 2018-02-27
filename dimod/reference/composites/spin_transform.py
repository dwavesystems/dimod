from random import random
import time
import itertools

from dimod.core.composite import Composite
from dimod.core.sampler import Sampler
from dimod.core.structured import Structured
from dimod.response import Response
from dimod.vartypes import Vartype

__all__ = ['SpinReversalTransformComposite']


class SpinReversalTransformComposite(Sampler, Composite):
    """Composite for applying spin reversal transform preprocessing.

    Spin reversal transforms (or "gauge transformations") are applied
    by flipping the spin of variables in the Ising problem. We can
    then sample using the transformed Ising problem and flip the same
    bits in the resulting sample.

    Args:
        sampler: A dimod sampler object.

    Examples:
        Composing a sampler.

        >>> base_sampler = dimod.ExactSolver()
        >>> composed_sampler = dimod.SpinReversalTransformComposite(base_sampler)

        The composed sampler can now be used as a dimod sampler.

        >>> h = {0: -1, 1: 1}
        >>> response = composed_sampler.sample_ising(h, {})
        >>> list(response.samples())
        [{0: 1, 1: -1}, {0: -1, 1: -1}, {0: 1, 1: 1}, {0: -1, 1: 1}]

        The base sampler is also in `children` attribute of the composed
        sampler.

        >>> base_sampler in composed_sampler.children
        True

    References
    ----------
    .. [KM] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    Attributes:
        children (list): [`sampler`] where `sampler` is the input sampler.
        structure: Inherited from input `sampler`.

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
        """todo"""
        # make a main response
        response = None

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

            tf_idxs = [flipped_response.label_to_idx[v] for v in flipped_response.variable_labels]

            if bqm.vartype is Vartype.SPIN:
                flipped_response.samples_matrix[:, tf_idxs] = -1 * flipped_response.samples_matrix[:, tf_idxs]
            else:
                flipped_response.samples_matrix[:, tf_idxs] = 1 - flipped_response.samples_matrix[:, tf_idxs]

            if response is None:
                response = flipped_response
            else:
                response.update(flipped_response)

        return response
