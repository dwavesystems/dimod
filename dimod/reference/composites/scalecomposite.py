# Copyright 2019 D-Wave Systems Inc.
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
A composite that scales problem variables as directed. if scalar is not given
calculates it based on quadratic and bias ranges.

"""
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from numbers import Number

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.core.composite import ComposedSampler

__all__ = 'ScaleComposite',


class ScaleComposite(ComposedSampler):
    """Composite to scale variables of a problem

    Scales the variables of a bqm and modifies linear and quadratic terms
    accordingly.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.ScaleComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler scales linear, quadratic biases and offset as
       indicated by options.

       >>> h = {'a': -4.0, 'b': -4.0}
       >>> J = {('a', 'b'): 3.2}
       >>> sampler = dimod.ScaleComposite(dimod.ExactSolver())
       >>> response = sampler.sample_ising(h, J, scalar=0.5,
       ...                ignored_interactions=[('a','b')])

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param.update({'scalar': [],
                      'bias_range': [],
                      'quadratic_range': [],
                      'ignored_variables': [],
                      'ignored_interactions': [],
                      'ignore_offset': []})
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, scalar=None, bias_range=1, quadratic_range=None,
               ignored_variables=None, ignored_interactions=None,
               ignore_offset=False, **parameters):
        """ Scale and sample from the provided binary quadratic model.

        if scalar is not given, problem is scaled based on bias and quadratic
        ranges. See :meth:`.BinaryQuadraticModel.scale` and
        :meth:`.BinaryQuadraticModel.normalize`

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            scalar (number):
                Value by which to scale the energy range of the binary quadratic model.

            bias_range (number/pair):
                Value/range by which to normalize the all the biases, or if
                `quadratic_range` is provided, just the linear biases.

            quadratic_range (number/pair):
                Value/range by which to normalize the quadratic biases.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        ignored_variables, ignored_interactions = _check_params(
            ignored_variables, ignored_interactions)

        child = self.child
        bqm_copy = _scaled_bqm(bqm, scalar, bias_range, quadratic_range,
                               ignored_variables, ignored_interactions,
                               ignore_offset)
        response = child.sample(bqm_copy, **parameters)

        return _scale_back_response(bqm, response, bqm_copy.info['scalar'],
                                    ignored_variables, ignored_interactions,
                                    ignore_offset)

    def sample_ising(self, h, J, offset=0, scalar=None,
                     bias_range=1, quadratic_range=None,
                     ignored_variables=None, ignored_interactions=None,
                     ignore_offset=False, **parameters):
        """ Scale and sample from the problem provided by h, J, offset

        if scalar is not given, problem is scaled based on bias and quadratic
        ranges.

        Args:
            h (dict): linear biases

            J (dict): quadratic or higher order biases

            offset (float, optional): constant energy offset

            scalar (number):
                Value by which to scale the energy range of the binary quadratic model.

            bias_range (number/pair):
                Value/range by which to normalize the all the biases, or if
                `quadratic_range` is provided, just the linear biases.

            quadratic_range (number/pair):
                Value/range by which to normalize the quadratic biases.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        if any(len(inter) > 2 for inter in J):
            # handle HUBO
            import warnings
            msg = ("Support for higher order Ising models in ScaleComposite is "
                   "deprecated and will be removed in dimod 0.9.0. Please use "
                   "PolyScaleComposite.sample_hising instead.")
            warnings.warn(msg, DeprecationWarning)

            from dimod.reference.composites.higherordercomposites import PolyScaleComposite
            from dimod.higherorder.polynomial import BinaryPolynomial

            poly = BinaryPolynomial.from_hising(h, J, offset=offset)

            ignored_terms = set()
            if ignored_variables is not None:
                ignored_terms.update(frozenset(v) for v in ignored_variables)
            if ignored_interactions is not None:
                ignored_terms.update(frozenset(inter) for inter in ignored_interactions)
            if ignore_offset:
                ignored_terms.add(frozenset())

            return PolyScaleComposite(self.child).sample_poly(poly, scalar=scalar,
                                                              bias_range=bias_range,
                                                              poly_range=quadratic_range,
                                                              ignored_terms=ignored_terms,
                                                              **parameters)

        bqm = BinaryQuadraticModel.from_ising(h, J, offset=offset)
        return self.sample(bqm, scalar=scalar,
                           bias_range=bias_range,
                           quadratic_range=quadratic_range,
                           ignored_variables=ignored_variables,
                           ignored_interactions=ignored_interactions,
                           ignore_offset=ignore_offset, **parameters)


def _scale_back_response(bqm, response, scalar, ignored_interactions,
                         ignored_variables, ignore_offset):
    """Helper function to scale back the response of sample method"""
    if len(ignored_interactions) + len(
            ignored_variables) + ignore_offset == 0:
        response.record.energy = np.divide(response.record.energy, scalar)
    else:
        response.record.energy = bqm.energies((response.record.sample,
                                               response.variables))
    return response


def _check_params(ignored_variables, ignored_interactions):
    """Helper for sample methods"""

    if ignored_variables is None:
        ignored_variables = set()
    elif not isinstance(ignored_variables, abc.Container):
        ignored_variables = set(ignored_variables)

    if ignored_interactions is None:
        ignored_interactions = set()
    elif not isinstance(ignored_interactions, abc.Container):
        ignored_interactions = set(ignored_interactions)

    return ignored_variables, ignored_interactions


def _calc_norm_coeff(h, J, bias_range, quadratic_range, ignored_variables,
                     ignored_interactions):
    """Helper function to calculate normalization coefficient"""

    if ignored_variables is None or ignored_interactions is None:
        raise ValueError('ignored interactions or variables cannot be None')

    def parse_range(r):
        if isinstance(r, Number):
            return -abs(r), abs(r)
        return r

    def min_and_max(iterable):
        if not iterable:
            return 0, 0
        return min(iterable), max(iterable)

    if quadratic_range is None:
        linear_range, quadratic_range = bias_range, bias_range
    else:
        linear_range = bias_range

    lin_range, quad_range = map(parse_range, (linear_range,
                                              quadratic_range))

    lin_min, lin_max = min_and_max([v for k, v in h.items()
                                    if k not in ignored_variables])
    quad_min, quad_max = min_and_max([v for k, v in J.items()
                                      if not check_isin(k,
                                                        ignored_interactions)])

    inv_scalar = max(lin_min / lin_range[0], lin_max / lin_range[1],
                     quad_min / quad_range[0], quad_max / quad_range[1])

    if inv_scalar != 0:
        return 1. / inv_scalar
    else:
        return 1.


def _scaled_bqm(bqm, scalar, bias_range, quadratic_range,
                ignored_variables, ignored_interactions,
                ignore_offset):
    """Helper function of sample for scaling"""

    bqm_copy = bqm.copy()
    if scalar is None:
        scalar = _calc_norm_coeff(bqm_copy.linear, bqm_copy.quadratic,
                                  bias_range, quadratic_range,
                                  ignored_variables, ignored_interactions)

    bqm_copy.scale(scalar, ignored_variables=ignored_variables,
                   ignored_interactions=ignored_interactions,
                   ignore_offset=ignore_offset)
    bqm_copy.info.update({'scalar': scalar})
    return bqm_copy


def check_isin(key, key_list):
    return sum(set(key) == set(key_tmp) for key_tmp in key_list)
