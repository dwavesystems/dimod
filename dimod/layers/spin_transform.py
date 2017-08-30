from __future__ import division

import random
import sys
import time

from dimod.decorators import ising, qubo
from dimod.utilities import ising_to_qubo, qubo_to_ising


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
        def sample_ising(self, h, J, spin_variables=None, **kwargs):
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

        def sample_structured_ising(self, h, J, spin_variables=None, **kwargs):
            """TODO

            """

            # first up we want to apply the spins to input h, J
            h_spin, J_spin, transform = apply_spin_transform(h, J, spin_variables)

            # sample from the transformed h, J
            response = Sampler.sample_structured_ising(self, h_spin, J_spin, **kwargs)

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

        def sample_qubo(self, Q, spin_variables=None, **kwargs):
            """TODO

            """

            # first up we want to apply the spins to input h, J
            Q_spin, transform, offset = apply_spin_transform_qubo(Q, spin_variables)

            # sample from the transformed h, J
            response = Sampler.sample_qubo(self, Q_spin, **kwargs)

            # unapply the transform, the samples are dicts so we can apply
            # in place, the energy is not affected
            for sample in response:
                for v in transform:
                    sample[v] = 1 - sample[v]

            # also need to update the energy
            response._energies = [en + offset for en in response._energies]

            # store the transformed variables in data for posterity
            if 'spin_transform_variables' not in response.data:
                response.data['spin_transform_variables'] = transform
            else:
                # there is already a spin_variables field, so let's
                # make a (hopefully) unique one.
                response.data['spin_transform_variables_{}'.format(time.time())] = transform

            return response

        def sample_structured_qubo(self, Q, spin_variables=None, **kwargs):
            """TODO

            """

            # first up we want to apply the spins to input h, J
            Q_spin, transform, offset = apply_spin_transform_qubo(Q, spin_variables)

            # sample from the transformed h, J
            response = Sampler.sample_structured_qubo(self, Q_spin, **kwargs)

            # unapply the transform, the samples are dicts so we can apply
            # in place, the energy is not affected
            for sample in response:
                for v in transform:
                    sample[v] = 1 - sample[v]

            # also need to update the energy
            response._energies = [en + offset for en in response._energies]

            # store the transformed variables in data for posterity
            if 'spin_transform_variables' not in response.data:
                response.data['spin_transform_variables'] = transform
            else:
                # there is already a spin_variables field, so let's
                # make a (hopefully) unique one.
                response.data['spin_transform_variables_{}'.format(time.time())] = transform

            return response

    return _STSampler


def apply_spin_transform(h, J, spin_variables=None):
    """
    TODO
    """

    if spin_variables is None:
        transform = set(random.sample(h, len(h) // 4))
    else:
        transform = set(spin_variables)

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


def apply_spin_transform_qubo(Q, spin_variables=None):
    """TODO"""
    h, J, off_ising = qubo_to_ising(Q)
    h_spin, J_spin, transform = apply_spin_transform(h, J)
    Q_spin, off_qubo = ising_to_qubo(h_spin, J_spin)
    return Q_spin, transform, off_ising + off_qubo
