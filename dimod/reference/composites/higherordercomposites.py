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

"""
Composites that convert binary quadratic model samplers into polynomial samplers
or that work with binary polynomials.

Higher-order composites implement three sampling methods (similar to
:class:`.Sampler`):

* :meth:`.PolySampler.sample_poly`
* :meth:`.PolySampler.sample_hising`
* :meth:`.PolySampler.sample_hubo`

"""
import warnings

from collections import defaultdict

import numpy as np

from dimod.core.polysampler import ComposedPolySampler, PolySampler
from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.higherorder.utils import make_quadratic, poly_energies
from dimod.sampleset import SampleSet, append_variables



__all__ = ['HigherOrderComposite',
           'PolyScaleComposite',
           'PolyTruncateComposite',
           'PolyFixedVariableComposite',
           ]


class HigherOrderComposite(ComposedPolySampler):
    """Convert a binary quadratic model sampler to a binary polynomial sampler.

    Energies of the returned samples do not include the penalties.

    Args:
        sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Example:
        This example uses :class:`.HigherOrderComposite` to instantiate a
        composed sampler that submits a simple Ising problem to a sampler.
        The composed sampler creates a binary quadratic model (BQM) from a
        higher order problem.

        >>> sampler = dimod.HigherOrderComposite(dimod.ExactSolver())
        >>> h = {0: -0.5, 1: -0.3, 2: -0.8}
        >>> J = {(0, 1, 2): -1.7}
        >>> sampleset = sampler.sample_hising(h, J, discard_unsatisfied=True)
        >>> set(sampleset.first.sample.values()) == {1}
        True

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

    def sample_poly(self, poly, penalty_strength=1.0,
                    keep_penalty_variables=False,
                    discard_unsatisfied=False, **parameters):
        """Sample from the given binary polynomial.

        Introduces penalties to reduce the given higher-order binary polynomial
        to a quadratic problem and sends it to its child sampler.

        Args:
            poly (:obj:`.BinaryPolynomial`):
                A binary polynomial.

            penalty_strength (float, optional):
                Strength of the reduction constraint. Insufficient strength can
                result in the binary quadratic model not having the same
                minimization as the polynomial.

            keep_penalty_variables (bool, optional, default=True):
                Setting to False removes the variables used for penalty from
                the samples.

            discard_unsatisfied (bool, optional, default=False):
                Setting to True discards samples that do not satisfy the penalty
                conditions.

            initial_state (dict, optional):
                Only accepted when the child sampler accepts an initial state.
                The initial state is given in terms of the variables in
                the binary polynomial. The corresponding initial values are
                populated for use by the child sampler.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        bqm = make_quadratic(poly, penalty_strength, vartype=poly.vartype)

        if 'initial_state' in parameters:
            initial_state = expand_initial_state(bqm,
                                                 parameters['initial_state'])
            parameters['initial_state'] = initial_state

        response = self.child.sample(bqm, **parameters)

        return polymorph_response(response, poly, bqm,
                                  penalty_strength=penalty_strength,
                                  keep_penalty_variables=keep_penalty_variables,
                                  discard_unsatisfied=discard_unsatisfied)


def expand_initial_state(bqm, initial_state):
    """Determine values for the initial state.

    Used on a binary quadratic model generated from a higher order polynomial.

    Args:
        bqm (:obj:`.BinaryQuadraticModel`): a binary quadratic model (BQM)
        object that contains its reduction info.

        initial_state (dict):
            An initial state for the higher order polynomial that generated the
            BQM.

    Returns:
        dict: A fully specified initial state.

    """
    # Developer note: this function relies heavily on assumptions about the
    # existance and structure of bqm.info['reduction']. We should consider
    # changing the way that the reduction information is passed.
    if not bqm.info['reduction']:
        return initial_state  # saves making a copy

    initial_state = dict(initial_state)  # so we can edit it in-place

    for (u, v), changes in bqm.info['reduction'].items():

        uv = changes['product']
        initial_state[uv] = initial_state[u] * initial_state[v]

        if 'auxiliary' in changes:
            # need to figure out the minimization from the initial_state
            aux = changes['auxiliary']

            en = (initial_state[u] * bqm.adj[aux].get(u, 0) +
                  initial_state[v] * bqm.adj[aux].get(v, 0) +
                  initial_state[uv] * bqm.adj[aux].get(uv, 0))

            initial_state[aux] = min(bqm.vartype.value, key=lambda val: en*val)

    return initial_state


def penalty_satisfaction(response, bqm):
    """Create a penalty satisfaction list.

    Given a sample set and a binary quadratic model (BQM) object, creates a
    binary list informing whether the penalties introduced during degree
    reduction are satisfied for each sample in the sample set.

    Args:
        response (:obj:`.SampleSet`): Samples corresponding to provided BQM

        bqm (:obj:`.BinaryQuadraticModel`): a BQM object that contains
            its reduction info.

    Returns:
        :obj:`numpy.ndarray`: a binary array of penalty satisfaction information

    """
    record = response.record
    label_to_idx = response.variables.index

    if len(bqm.info['reduction']) == 0:
        return np.array([1] * len(record.sample))

    penalty_vector = np.prod([record.sample[:, label_to_idx(qi)] *
                              record.sample[:, label_to_idx(qj)]
                              == record.sample[:,
                                 label_to_idx(valdict['product'])]
                              for (qi, qj), valdict in
                              bqm.info['reduction'].items()], axis=0)
    return penalty_vector


def polymorph_response(response, poly, bqm,
                       penalty_strength=None,
                       keep_penalty_variables=True,
                       discard_unsatisfied=False):
    """Transform the sample set for the higher order problem.

    Given a response of a penalized HUBO, this function creates a new sample set
    object, taking into account penalty information and calculates the
    energies of samples for the higherorder problem.

    Args:
        response (:obj:`.SampleSet`): response for a penalized hubo.

        poly (:obj:`.BinaryPolynomial`):
            A binary polynomial.

        bqm (:obj:`dimod.BinaryQuadraticModel`): Binary quadratic model of the
            reduced problem.

        penalty_strength (float, optional, default=None):
            If provided, added to the info field of the
            returned :obj:`dimod.SampleSet`.

        keep_penalty_variables (bool, optional, default=True):
            Setting to False removes variables used for penalty from the samples.

        discard_unsatisfied (bool, optional, default=False):
            Setting to True discards samples that do not satisfy the penalty
            conditions.

    Returns:
        (:obj:`.SampleSet'): A SampleSet object that has additional penalty
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
        idxs = [response.variables.index(v) for v in original_variables]
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

        If `scalar` is not given, problem is scaled based on bias and polynomial
        ranges. See :meth:`.BinaryPolynomial.scale` and
        :meth:`.BinaryPolynomial.normalize`

        Args:
            poly (obj:`.BinaryPolynomial`): A binary polynomial.

            scalar (number, optional):
                Value by which to scale the energy range of the binary polynomial.

            bias_range (number/pair, optional, default=1):
                Value/range by which to normalize all the biases, or if
                `poly_range` is provided, just the linear biases.

            poly_range (number/pair, optional):
                Value/range by which to normalize higher-order biases.

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
    """Composite that truncates returned samples.

    Post-processing is expensive and sometimes one might want to only
    treat the lowest-energy samples. This composite layer allows one to
    pre-select the samples within a multi-composite pipeline.

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
        If `aggregate` is True, :attr:`.SampleSet.record.num_occurrences` are
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


class PolyFixedVariableComposite(ComposedPolySampler):
    """Composite that fixes variables of a problem.

    Fixes variables of a binary polynomial and modifies linear and k-local terms
    accordingly. Returned samples include the fixed variable.

    Args:
       sampler (:obj:`dimod.PolySampler`):
            A dimod polynomial sampler.

    Examples:
       This example uses :class:`.PolyFixedVariableComposite` to instantiate a
       composed sampler that submits a simple high-order Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and k-local terms
       biases.

       >>> h = {1: -1.3, 2: 1.2, 3: -3.4, 4: -0.5}
       >>> J = {(1, 4): -0.6, (1, 2, 3): 0.2, (1, 2, 3, 4): -0.1}
       >>> poly = dimod.BinaryPolynomial.from_hising(h, J, offset=0)
       >>> sampler = dimod.PolyFixedVariableComposite(dimod.ExactPolySolver())
       >>> sampleset = sampler.sample_poly(poly, fixed_variables={3: -1, 4: 1})

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        params = self.child.parameters.copy()
        params['fixed_variables'] = []
        return params

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample_poly(self, poly, fixed_variables=None, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            poly (:obj:`dimod.BinaryPolynomial`):
                Binary polynomial model to be sampled from.

            fixed_variables (dict):
                A dictionary of variable assignments.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        child = self.child
        if fixed_variables is None:
            sampleset = child.sample_poly(poly, **parameters)
            return sampleset
        else:
            poly_copy = fix_variables(poly, fixed_variables)
            sampleset = child.sample_poly(poly_copy, **parameters)
            if len(sampleset):
                return append_variables(sampleset, fixed_variables)
            elif fixed_variables:
                return type(sampleset).from_samples_bqm(fixed_variables, bqm=poly)
            else:
                return sampleset


def fix_variables(poly, fixed_variables):
    if () in poly.keys():
        offset = poly[()]
    else:
        offset = 0.0
    poly_copy = defaultdict(float)
    for k, v in poly.items():
        k = set(k)
        for var, value in fixed_variables.items():
            if var in k:
                k -= {var}
                v *= value
        k = frozenset(k)
        if len(k) > 0:
            poly_copy[k] += v
        else:
            offset += v
    poly_copy[()] = offset
    return BinaryPolynomial(poly_copy, poly.vartype)
