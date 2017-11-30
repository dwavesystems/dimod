from random import random
import time
import itertools

from dimod import _PY2
from dimod.template_composite import TemplateComposite
from dimod.responses import SpinResponse, BinaryResponse
from dimod.decorators import ising, qubo
from dimod.utilities import ising_to_qubo, qubo_to_ising

__all__ = ['SpinReversalTransform']

if _PY2:
    range = xrange
    zip = itertools.izip

    def iteritems(d):
        return d.iteritems()
else:
    def iteritems(d):
        return d.items()


class SpinReversalTransform(TemplateComposite):
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
        >>> composed_sampler = dimod.SpinReversalTransform(base_sampler)

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
    def __init__(self, sampler):
        # puts sampler into self.children
        TemplateComposite.__init__(self, sampler)

        self._child = sampler  # faster access than self.children[0]

        # copy over the structure
        self.structure = sampler.structure

    @ising(1, 2)
    def sample_ising(self, h, J,
                     num_spin_reversal_transforms=1, spin_reversal_variables=None,
                     **kwargs):
        """Applies spin reversal transforms to an Ising problem, then samples
        using the child sampler's `sample_ising` method.

        Args:
            h (dict/list): The linear terms in the Ising problem. If a
                dict, should be of the form {v: bias, ...} where v is
                a variable in the Ising problem, and bias is the linear
                bias associated with v. If a list, should be of the form
                [bias, ...] where the indices of the biases are the
                variables in the Ising problem.
            J (dict): A dictionary of the quadratic terms in the Ising
                problem. Should be of the form {(u, v): bias} where u,
                v are variables in the Ising problem and bias is the
                quadratic bias associated with u, v.
            num_spin_reversal_transforms (int, optional): Default 1. The
                number of different spin reversal transforms to apply to
                the given Ising problem. Note that the child sampler will
                be invoked for each spin reversal transform.
            spin_reversal_variables (iterable, optional): An iterable of
                variables in the Ising problem. These are the variables
                that have their spins flipped. If set to None, each variable
                has a 50% chance of having its bit flipped. Note that if a
                variable is in spin_reversal_variables but not in h or J
                then it will be ignored. Default None.
            **kwargs: Any other keyword arguments are passed unchanged to
                the child sampler's `sample_ising` method.


        Notes:
            As noted in the section defining the `num_spin_reversal_transforms`
            parameter, the child sampler will be invoked for each different
            spin reversal transform. So if the child sampler accepts a
            `num_reads` keyword parameter, the total number of reads
            performed will be `num_reads` * `num_spin_reversal_transforms`.

        """
        if not isinstance(num_spin_reversal_transforms, int):
            raise TypeError("input `num_spin_reversal_transforms` must be an 'int'")

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
            has a 50% chance of having its bit flipped. Note that if a
            variable is in spin_reversal_variables but not in h or J
            then it will be ignored. Default None.

    Returns:
        h_spin (dict): the transformed linear biases, in the same
            form as `h`.
        J_spin (dict): the transformed quadratic biases, in the same
            form as `J`.
        spin_reversal_variables (set): The variables which had their
            spins flipped. If `spin_reversal_variables` were provided,
            then this will be the same.

    References
    ----------
    .. [KM] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

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
