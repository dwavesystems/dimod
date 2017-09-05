from __future__ import division

from random import random
import sys
import time
import itertools

from dimod.sampler_template import TemplateSampler
from dimod.composite_template import TemplateComposite
from dimod.responses import SpinResponse, BinaryResponse
from dimod.decorators import ising, qubo
from dimod.utilities import ising_to_qubo, qubo_to_ising

__all__ = ['SpinTransform']

_PY2 = sys.version_info[0] == 2
if _PY2:
    iteritems = lambda d: d.iteritems()
    range = xrange
    zip = itertools.izip
else:
    iteritems = lambda d: d.items()


class SpinTransform(TemplateComposite, TemplateSampler):
    """TODO
    """
    def __init__(self, sampler):
        TemplateComposite.__init__(self)
        TemplateSampler.__init__(self)

        self.children.append(sampler)  # TemplateComposite creates children attribute
        self._child = sampler  # faster access than self.children[0]

    @ising(1, 2)
    def sample_ising(self, h, J,
                     num_spin_reversal_transforms=1, spin_reversal_variables=None,
                     **kwargs):
        """TODO

        NB: num runs sampler each time

        """
        sampler = self._child

        # dispatch all of the jobs, in case the samples are resolved upon response read.
        # keep track of which variables were transformed
        dispatched = []
        for __ in range(num_spin_reversal_transforms):
            h_spin, J_spin, transform = \
                apply_spin_reversal_transform(h, J, spin_reversal_variables)

            response = sampler.sample_ising(h_spin, J_spin, **kwargs)

            dispatched.append((response, transform))

        # put all of the responses into one
        st_response = SpinResponse()

        for response, transform in dispatched:
            samples, energies, sample_data = zip(*response.items(data=True))

            # flip the bits in the samples
            st_samples = (_apply_srt_sample_spin(sample, transform) for sample in samples)

            # keep track of which bits were flipped in data
            st_sample_data = (_apply_srt_sample_data(dat, transform) for dat in sample_data)

            st_response.add_samples_from(st_samples, energies, st_sample_data)

            st_response.data.update(response.data)

        return st_response

    @ising(1, 2)
    def sample_structured_ising(self, h, J,
                                num_spin_reversal_transforms=1, spin_reversal_variables=None,
                                **kwargs):
        """TODO

        NB: num runs sampler each time

        """
        sampler = self._child

        # dispatch all of the jobs, in case the samples are resolved upon response read.
        # keep track of which variables were transformed
        dispatched = []
        for __ in range(num_spin_reversal_transforms):
            h_spin, J_spin, transform = \
                apply_spin_reversal_transform(h, J, spin_reversal_variables)

            response = sampler.sample_structured_ising(h_spin, J_spin, **kwargs)

            dispatched.append((response, transform))

        # put all of the responses into one
        st_response = SpinResponse()

        for response, transform in dispatched:
            samples, energies, sample_data = zip(*response.items(data=True))

            # flip the bits in the samples
            st_samples = (_apply_srt_sample_spin(sample, transform) for sample in samples)

            # keep track of which bits were flipped in data
            st_sample_data = (_apply_srt_sample_data(dat, transform) for dat in sample_data)

            st_response.add_samples_from(st_samples, energies, st_sample_data)

            st_response.data.update(response.data)

        return st_response


def _apply_srt_sample_spin(sample, transform):
    # flips the bits in a spin sample
    return {v: -s if v in transform else s for v, s in iteritems(sample)}


def _apply_srt_sample_data(data, transform):
    # stores information about the transform in the sample's data field
    if 'spin_reversal_variables' in data:
        data['spin_reversal_variables_{}'.format(time.time())] = transform
    else:
        data['spin_reversal_variables'] = transform
    return data


def apply_spin_reversal_transform(h, J, spin_reversal_variables=None):
    """Applies spin reversal transforms to an Ising problem.

    Spin reversal transforms (or "gauge transformations") are applied
    by flipping the spin of variables in the Ising problem. We can
    then sample using the transformed Ising problem and flip the same
    bits in the resulting sample.

    Args:
        h (dict): The linear terms in the Ising problem. Should be of
            the form {v: bias, ...} where v is a variable in the Ising
            problem, and bias is the linear bias associated with v.
        J (dict): A dictionary of the quadratic terms in the Ising
            problem. Should be of the form {(u, v): bias} where u,
            v are variables in the Ising problem and bias is the
            quadratic bias associated with u, v.
        spin_reversal_variables (iterable, optional): An iterable of
            variables in the Ising problem. These are the variables
            that have their spins flipped. If set to None, each variable
            has a 50% chance of having its bit flipped. Default None.

    Returns:
        h_spin (dict): the transformed linear biases, in the same
            form as `h`.
        J_spin (dict): the transformed quadratic biases, in the same
            form as `J`.
        spin_reversal_variables (set): The variables which had their
            spins flipped. If `spin_reversal_variables` were provided,
            then this will be the same.

    References:
    .. _KM:
        Andrew D. King and Catherine C. McGeoch. Algorithm engineering
            for a quantum annealing platform.
            https://arxiv.org/abs/1410.2628, 2014.

    """

    if spin_reversal_variables is None:
        # apply spin transform to each variable with 50% chance
        transform = set(v for v in h if random() > .5)
    else:
        transform = set(spin_reversal_variables)

    # apply spins transform to the linear biases
    h_spin = {v: -bias if v in transform else bias for v, bias in iteritems(h)}

    # apply spins transform to the quadratic biases
    def quad_bias(edge):
        u, v = edge
        bias = J[edge]
        if u in transform:
            bias = -bias
        if v in transform:
            bias = -bias
        return bias
    J_spin = {edge: quad_bias(edge) for edge in J}

    return h_spin, J_spin, transform


def apply_spin_reversal_transform_qubo(Q, spin_reversal_variables=None):
    """Applies spin reversal transforms to the Ising formulation of the
    given QUBO.

    Spin reversal transforms (or "gauge transformations") are applied
    by flipping the spin of variables in an Ising problem. We can
    then sample using the transformed Ising problem and flip the same
    bits in the resulting sample.

    Args:
        Q (dict): A dictionary defining the QUBO. Should be of the form
            {(u, v): bias} where u, v are variables and bias is numeric.
        spin_reversal_variables (iterable, optional): An iterable of
            variables in the Ising problem. These are the variables
            that have their spins flipped. If set to None, each variable
            has a 50% chance of having its bit flipped. Default None.

    Returns:
        Q_spin (dict): the transformed QUBO, in the same form as `Q`.
        spin_reversal_variables (set): The variables which had their
            spins flipped. If `spin_reversal_variables` were provided,
            then this will be the same.
        energy_offset (float): The energy offset between the energy
            defined by `Q` and `Q_spin`.

    Examples:
        >>> Q = {(0, 0): -1, (0, 1): 1, (1, 1): 1}
        >>> Q_spin, __, offset = dimod.apply_spin_reversal_transform_qubo(Q, {1})
        >>> dimod.qubo_energy(Q, {0: 1, 1: 0})
        -1.0
        >>> dimod.qubo_energy(Q_spin, {0: 1, 1: 1})
        -2.0
        >>> dimod.qubo_energy(Q_spin, {0: 1, 1: 1}) + offset
        -1.0

    References:
    .. _KM:
        Andrew D. King and Catherine C. McGeoch. Algorithm engineering
            for a quantum annealing platform.
            https://arxiv.org/abs/1410.2628, 2014.

    """

    # transform to Ising problem
    h, J, off_ising = qubo_to_ising(Q)

    # apply spin transforms to the Ising
    h_spin, J_spin, transform = apply_spin_reversal_transform(h, J, spin_reversal_variables)

    # convert back
    Q_spin, off_qubo = ising_to_qubo(h_spin, J_spin)

    # there may have been a net energy change
    return Q_spin, transform, off_ising + off_qubo
