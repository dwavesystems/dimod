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
# =============================================================================
"""
Composites that convert binary quadratic model samplers into polynomial samplers
or that work with binary polynomials.

Higher-order composites implement three sampling methods (similar to
:class:`.Sampler`):

* :meth:`.PolySampler.sample_poly`
* :meth:`.PolySampler.sample_hising`
* :meth:`.PolySampler.sample_hubo`

"""
from __future__ import division

import numpy as np

from dimod.core.polysampler import ComposedPolySampler, PolySampler
from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.higherorder.utils import make_quadratic, poly_energies
from dimod.response import SampleSet

__all__ = 'HigherOrderComposite', 'PolyScaleComposite', 'PolyTruncateComposite'


class HigherOrderComposite(ComposedPolySampler):
    """Convert a binary quadratic model sampler to a binary polynomial sampler.

    Energies of the returned samples do not include the penalties.

    Args:
        sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Example:
        This example uses :class:`.HigherOrderComposite` to instantiate a
        composed sampler that submits a simple Ising problem to a sampler.
        The composed sampler creates a bqm from a higher order problem.

        >>> sampler = dimod.HigherOrderComposite(dimod.ExactSolver())
        >>> h = {0: -0.5, 1: -0.3, 2: -0.8}
        >>> J = {(0, 1, 2): -1.7}
        >>> sampleset = sampler.sample_hising(h, J, discard_unsatisfied=True)
        >>> sampleset.first # doctest: +SKIP
        Sample(sample={0: 1, 1: 1, 2: 1},
               energy=-3.3,
               num_occurrences=1,
               penalty_satisfaction=True)

        """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        """A list containing the wrapped sampler."""
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param['penalty_strength'] = []
        param['discard_unsatisfied'] = []
        param['keep_penalty_variables'] = []
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample_ising(self, h, J, offset=0, *args, **kwargs):
        # need to handle offset input for backwards compatibility
        if offset:
            J[()] = offset
        return ComposedPolySampler.sample_ising(self, h, J, *args, **kwargs)

    def sample_poly(self, poly, penalty_strength=1.0,
                    keep_penalty_variables=False,
                    discard_unsatisfied=False, **parameters):
        """Sample from the given binary polynomial.

        Takes the given binary polynomial, introduces penalties, reduces the
        higher-order problem into a quadratic problem and sends it to its child
        sampler.

        Args:
            poly (:obj:`.BinaryPolynomial`):
                A binary polynomial.

            penalty_strength (float, optional): Strength of the reduction constraint.
                Insufficient strength can result in the binary quadratic model
                not having the same minimization as the polynomial.

            keep_penalty_variables (bool, optional): default is True. if False
                will remove the variables used for penalty from the samples

            discard_unsatisfied (bool, optional): default is False. If True
                will discard samples that do not satisfy the penalty conditions.

            **parameters: Parameters for the sampling method, specified by
            the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        bqm = make_quadratic(poly, penalty_strength, vartype=poly.vartype)
        response = self.child.sample(bqm, **parameters)

        return polymorph_response(response, poly, bqm,
                                  penalty_strength=penalty_strength,
                                  keep_penalty_variables=keep_penalty_variables,
                                  discard_unsatisfied=discard_unsatisfied)


def penalty_satisfaction(response, bqm):
    """ Creates a penalty satisfaction list

    Given a sampleSet and a bqm object, will create a binary list informing
    whether the penalties introduced during degree reduction are satisfied for
    each sample in sampleSet

    Args:
        response (:obj:`.SampleSet`): Samples corresponding to provided bqm

        bqm (:obj:`.BinaryQuadraticModel`): a bqm object that contains
            its reduction info.

    Returns:
        :obj:`numpy.ndarray`: a binary array of penalty satisfaction information

    """
    record = response.record
    label_dict = response.variables.index

    if len(bqm.info['reduction']) == 0:
        return np.array([1] * len(record.sample))

    penalty_vector = np.prod([record.sample[:, label_dict[qi]] *
                              record.sample[:, label_dict[qj]]
                              == record.sample[:,
                                 label_dict[valdict['product']]]
                              for (qi, qj), valdict in
                              bqm.info['reduction'].items()], axis=0)
    return penalty_vector


def polymorph_response(response, poly, bqm,
                       penalty_strength=None,
                       keep_penalty_variables=True,
                       discard_unsatisfied=False):
    """ Transforms the sampleset for the higher order problem.

    Given a response of a penalized HUBO, this function creates a new sampleset
    object, taking into account penalty information and calculates the
    energies of samples for the higherorder problem.

    Args:
        response (:obj:`.SampleSet`): response for a penalized hubo.

        poly (:obj:`.BinaryPolynomial`):
            A binary polynomial.

        bqm (:obj:`dimod.BinaryQuadraticModel`): Binary quadratic model of the
            reduced problem.

        penalty_strength (float, optional): default is None, if provided,
            will be added to the info field of the returned sampleSet object.

        keep_penalty_variables (bool, optional): default is True. if False
            will remove the variables used for penalty from the samples

        discard_unsatisfied (bool, optional): default is False. If True
            will discard samples that do not satisfy the penalty conditions.

    Returns:
        (:obj:`.SampleSet'): A sampleSet object that has additional penalty
            information. The energies of samples are calculated for the HUBO
            ignoring the penalty variables.

    """
    record = response.record
    penalty_vector = penalty_satisfaction(response, bqm)
    original_variables = bqm.variables

    if discard_unsatisfied:
        samples_to_keep = list(map(bool, list(penalty_vector)))
        penalty_vector = np.array([True] * np.sum(samples_to_keep))
    else:
        samples_to_keep = list(map(bool, [1] * len(record.sample)))

    samples = record.sample[samples_to_keep]
    energy_vector = poly.energies((samples, response.variables))

    if not keep_penalty_variables:
        original_variables = poly.variables
        idxs = [response.variables.index[v] for v in original_variables]
        samples = np.asarray(samples[:, idxs])

    num_samples, num_variables = np.shape(samples)

    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy_vector.dtype),
                 ('penalty_satisfaction',
                  penalty_vector.dtype)]
    datatypes.extend((name, record[name].dtype, record[name].shape[1:])
                     for name in record.dtype.names if
                     name not in {'sample',
                                  'energy'})

    data = np.rec.array(np.empty(num_samples, dtype=datatypes))
    data.sample = samples
    data.energy = energy_vector
    for name in record.dtype.names:
        if name not in {'sample', 'energy'}:
            data[name] = record[name][samples_to_keep]

    data['penalty_satisfaction'] = penalty_vector
    response.info['reduction'] = bqm.info['reduction']
    if penalty_strength is not None:
        response.info['penalty_strength'] = penalty_strength
    return SampleSet(data, original_variables, response.info,
                     response.vartype)


class PolyScaleComposite(ComposedPolySampler):
    """Composite to scale biases of a binary polynomial.

    Args:
        child (:obj:`.PolySampler`):
            A binary polynomial sampler.

    Examples:

       >>> linear = {'a': -4.0, 'b': -4.0}
       >>> quadratic = {('a', 'b'): 3.2, ('a', 'b', 'c'): 1}
       >>> sampler = dimod.PolyScaleComposite(dimod.HigherOrderComposite(dimod.ExactSolver()))
       >>> response = sampler.sample_hising(linear, quadratic, scalar=0.5,
       ...                ignored_terms=[('a','b')])

    """

    def __init__(self, child):
        if not isinstance(child, PolySampler):
            raise TypeError("Child sampler must be a PolySampler")
        self._children = [child]

    @property
    def children(self):
        """The child sampler in a list"""
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param.update({'scalar': [],
                      'bias_range': [],
                      'poly_range': [],
                      'ignored_terms': [],
                      })
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample_poly(self, poly, scalar=None, bias_range=1, poly_range=None,
                    ignored_terms=None, **parameters):
        """Scale and sample from the given binary polynomial.

        If scalar is not given, problem is scaled based on bias and polynomial
        ranges. See :meth:`.BinaryPolynomial.scale` and
        :meth:`.BinaryPolynomial.normalize`

        Args:
            poly (obj:`.BinaryPolynomial`): A binary polynomial.

            scalar (number, optional):
                Value by which to scale the energy range of the binary polynomial.

            bias_range (number/pair, optional, default=1):
                Value/range by which to normalize the all the biases, or if
                `poly_range` is provided, just the linear biases.

            poly_range (number/pair, optional):
                Value/range by which to normalize the higher order biases.

            ignored_terms (iterable, optional):
                Biases associated with these terms are not scaled.

            **parameters:
                Other parameters for the sampling method, specified by
                the child sampler.

        """

        if ignored_terms is None:
            ignored_terms = set()
        else:
            ignored_terms = {frozenset(term) for term in ignored_terms}

        # scale and normalize happen in-place so we need to make a copy
        original, poly = poly, poly.copy()

        if scalar is not None:
            poly.scale(scalar, ignored_terms=ignored_terms)
        else:
            poly.normalize(bias_range=bias_range, poly_range=poly_range,
                           ignored_terms=ignored_terms)

            # we need to know how much we scaled by, which we can do by looking
            # at the biases
            try:
                v = next(v for v, bias in original.items()
                         if bias and v not in ignored_terms)
            except StopIteration:
                # nothing to scale
                scalar = 1
            else:
                scalar = poly[v] / original[v]

        sampleset = self.child.sample_poly(poly, **parameters)

        if ignored_terms:
            # we need to recalculate the energy
            sampleset.record.energy = original.energies((sampleset.record.sample,
                                                         sampleset.variables))
        else:
            sampleset.record.energy /= scalar

        return sampleset


class PolyTruncateComposite(ComposedPolySampler):
    """Composite to truncate the returned samples

    Post-processing is expensive and sometimes one might want to only
    treat the lowest energy samples. This composite layer allows one to
    pre-select the samples within a multi-composite pipeline

    Args:
        child_sampler (:obj:`dimod.PolySampler`):
            A dimod binary polynomial sampler.

        n (int):
            Maximum number of rows in the returned sample set.

        sorted_by (str/None, optional, default='energy'):
            Selects the record field used to sort the samples before
            truncating. Note that sample order is maintained in the
            underlying array.

        aggregate (bool, optional, default=False):
            If True, aggregate the samples before truncating.

    Note:
        If aggregate is True :attr:`.SampleSet.record.num_occurrences` are
        accumulated but no other fields are.

    """
    def __init__(self, child_sampler, n, sorted_by='energy', aggregate=False):

        if n < 1:
            raise ValueError('n should be a positive integer, recived {}'.format(n))

        self._children = [child_sampler]
        self._truncate_kwargs = dict(n=n, sorted_by=sorted_by)
        self._aggregate = aggregate

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self.child.parameters.copy()

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample_poly(self, poly, **kwargs):
        """Sample from the binary polynomial and truncate output.

        Args:
            poly (obj:`.BinaryPolynomial`): A binary polynomial.

            **kwargs:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        tkw = self._truncate_kwargs
        if self._aggregate:
            return self.child.sample_poly(poly, **kwargs).aggregate().truncate(**tkw)
        else:
            return self.child.sample_poly(poly, **kwargs).truncate(**tkw)
