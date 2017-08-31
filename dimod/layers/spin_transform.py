from __future__ import division

from random import random
import sys
import time

from dimod.decorators import ising, qubo
from dimod.utilities import ising_to_qubo, qubo_to_ising

__all__ = ['SpinReversal']

PY2 = sys.version_info[0] == 2
if PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


def SpinTransformation(Sampler):
    class _STSampler(Sampler):
        """TODO
        """
        @ising(1, 2)
        def sample_ising(self, h, J,
                         num_spin_reversal_transforms=1, spin_reversal_variables=None,
                         **kwargs):
            """TODO

            """

            # first up we want to apply the spins to input h, J
            h_spin, J_spin, transform = apply_spin_transform(h, J, spin_variables)

            # sample from the transformed h, J
            response = Sampler.sample_ising(self, h_spin, J_spin, **kwargs)

            # unapply the transform, the samples are dicts so we can apply
            # in place, the energy is not affected
            for sample in response:
                for v in transform:
                    sample[v] = -sample[v]

            # store the transformed variables in data for posterity
            if 'spin_transform_variables' not in response.data:
                response.data['spin_transform_variables'] = transform
            else:
                # there is already a spin_variables field, so let's
                # make a (hopefully) unique one.
                response.data['spin_transform_variables_{}'.format(time.time())] = transform

            return response

    #     def sample_structured_ising(self, h, J, spin_variables=None, **kwargs):
    #         """TODO

    #         """

    #         # first up we want to apply the spins to input h, J
    #         h_spin, J_spin, transform = apply_spin_transform(h, J, spin_variables)

    #         # sample from the transformed h, J
    #         response = Sampler.sample_structured_ising(self, h_spin, J_spin, **kwargs)

    #         # unapply the transform, the samples are dicts so we can apply
    #         # in place, the energy is not affected
    #         for sample in response:
    #             for v in transform:
    #                 sample[v] = -sample[v]

    #         # store the transformed variables in data for posterity
    #         if 'spin_transform_variables' not in response.data:
    #             response.data['spin_transform_variables'] = transform
    #         else:
    #             # there is already a spin_variables field, so let's
    #             # make a (hopefully) unique one.
    #             response.data['spin_transform_variables_{}'.format(time.time())] = transform

    #         return response

    #     def sample_qubo(self, Q, spin_variables=None, **kwargs):
    #         """TODO

    #         """

    #         # first up we want to apply the spins to input h, J
    #         Q_spin, transform, offset = apply_spin_transform_qubo(Q, spin_variables)

    #         # sample from the transformed h, J
    #         response = Sampler.sample_qubo(self, Q_spin, **kwargs)

    #         # unapply the transform, the samples are dicts so we can apply
    #         # in place, the energy is not affected
    #         for sample in response:
    #             for v in transform:
    #                 sample[v] = 1 - sample[v]

    #         # also need to update the energy
    #         response._energies = [en + offset for en in response._energies]

    #         # store the transformed variables in data for posterity
    #         if 'spin_transform_variables' not in response.data:
    #             response.data['spin_transform_variables'] = transform
    #         else:
    #             # there is already a spin_variables field, so let's
    #             # make a (hopefully) unique one.
    #             response.data['spin_transform_variables_{}'.format(time.time())] = transform

    #         return response

    #     def sample_structured_qubo(self, Q, spin_variables=None, **kwargs):
    #         """TODO

    #         """

    #         # first up we want to apply the spins to input h, J
    #         Q_spin, transform, offset = apply_spin_transform_qubo(Q, spin_variables)

    #         # sample from the transformed h, J
    #         response = Sampler.sample_structured_qubo(self, Q_spin, **kwargs)

    #         # unapply the transform, the samples are dicts so we can apply
    #         # in place, the energy is not affected
    #         for sample in response:
    #             for v in transform:
    #                 sample[v] = 1 - sample[v]

    #         # also need to update the energy
    #         response._energies = [en + offset for en in response._energies]

    #         # store the transformed variables in data for posterity
    #         if 'spin_transform_variables' not in response.data:
    #             response.data['spin_transform_variables'] = transform
    #         else:
    #             # there is already a spin_variables field, so let's
    #             # make a (hopefully) unique one.
    #             response.data['spin_transform_variables_{}'.format(time.time())] = transform

    #         return response

    # return _STSampler


def apply_spin_reversal_transform(h, J, spin_reversal_variables=None):
    """Applies spin reveral transforms to an Ising problem.

    Spin reversal transforms (or "gauge transformations") are applied
    by flipping the spin of random variables in the Ising problem.
    We can then sample using the transformed Ising problem and flip
    the same bits in the resulting sample.

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
        h_spin (dict): the transformed h, in the same form as `h`.
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


# def apply_spin_transform_qubo(Q, spin_variables=None):
#     """TODO"""
#     h, J, off_ising = qubo_to_ising(Q)
#     h_spin, J_spin, transform = apply_spin_transform(h, J)
#     Q_spin, off_qubo = ising_to_qubo(h_spin, J_spin)
#     return Q_spin, transform, off_ising + off_qubo
