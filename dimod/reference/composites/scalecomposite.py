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
A composite that fixes the variables provided and removes them from the
bqm object before sending to its child sampler.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
of technical terms in descriptions of Ocean tools.
"""
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from numbers import Number

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.core.composite import ComposedSampler
from dimod.higherorder import poly_energies, _relabeled_poly

__all__ = ['ScaleComposite']


class ScaleComposite(ComposedSampler):
    """Composite to scale variables of a problem

    Inherits from :class:`dimod.ComposedSampler`.

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

       >>> linear = {'a': -4.0, 'b': -4.0}
       >>> quadratic = {('a', 'b'): 3.2}
       >>> sampler = dimod.ScaleComposite(dimod.SimulatedAnnealingSampler())
       >>> response = sampler.sample_ising(linear, quadratic, scalar=0.5,
       ...                ignored_interactions=[('a','b')],num_reads = 100)
       >>> print(response.aggregate().record) # doctest: +SKIP
       >>> response = sampler.sample_ising(linear, quadratic, scalar=0.5,
       ...                                              num_reads = 100)
       >>> print(response.aggregate().record) # doctest: +SKIP
       [([-1,  1], -3.2, 58) ([ 1, -1], -3.2, 42)]
       [([1, 1], -4.8, 100)]

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self.child.parameters.copy()

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

        child = self.child

        bqm_copy = bqm.copy()
        if scalar is None:
            bqm_copy.normalize(bias_range, quadratic_range,
                               ignored_variables=ignored_variables,
                               ignored_interactions=ignored_interactions,
                               ignore_offset=ignore_offset)
        else:
            bqm_copy.scale(scalar, ignored_variables=ignored_variables,
                           ignored_interactions=ignored_interactions,
                           ignore_offset=ignore_offset)
        response = child.sample(bqm_copy, **parameters)

        response.record.energy = bqm.energies((response.record.sample,
                                               response.variables))
        return response

    def sample_ising(self, h, J, offset=0, scalar=None,
                     bias_range=1, quadratic_range=None,
                     ignored_variables=None, ignored_interactions=None,
                     ignore_offset=False, **parameters):
        """ Scale and sample from the problem provided by h, J, offset

        if scalar is not given, problem is scaled based on bias and quadratic
        ranges.

        Args:
            h (dict): linear biases corresponding to the HUBO form

            J (dict): higher order biases corresponding to the HUBO form

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

        if max(map(len, J.keys())) == 2:
            bqm = BinaryQuadraticModel.from_ising(h, J, offset=offset)
            return self.sample(bqm, scalar=scalar,
                               bias_range=bias_range,
                               quadratic_range=quadratic_range,
                               ignored_variables=ignored_variables,
                               ignored_interactions=ignored_interactions,
                               ignore_offset=ignore_offset, **parameters)

        # handle HUBO
        child = self.child
        if scalar is None:
            scalar = _calc_norm_coeff(h, J, bias_range, quadratic_range)
        h_sc = dict(h)
        J_sc = dict(J)
        if scalar != 1:
            h_sc, J_sc, offset_sc = _scale(h, J, offset, scalar,
                                           ignored_variables=ignored_variables,
                                           ignored_interactions=
                                           ignored_interactions,
                                           ignore_offset=ignore_offset)
        response = child.sample_ising(h_sc, J_sc, offset=offset_sc,
                                      **parameters)

        poly = _relabeled_poly(h, J, response.variables.index)
        response.record.energy = np.add(poly_energies(response.record.sample,
                                                      poly), offset)
        return response


def _calc_norm_coeff(h, J, bias_range, quadratic_range):
    """Helper function to calculate normalization coefficient"""

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

    lin_min, lin_max = min_and_max(h.values())
    quad_min, quad_max = min_and_max(J.values())

    inv_scalar = max(lin_min / lin_range[0], lin_max / lin_range[1],
                     quad_min / quad_range[0], quad_max / quad_range[1])
    if inv_scalar != 0:
        return 1. / inv_scalar
    else:
        return 1.


def _scale(h, j, offset, scalar,
           ignored_variables=None,
           ignored_interactions=None,
           ignore_offset=False):
    """Helper function to calculate scale of a problem"""
    if not isinstance(scalar, Number):
        raise TypeError("expected scalar to be a Number")

    if ignored_variables is None:
        ignored_variables = set()
    elif not isinstance(ignored_variables, abc.Container):
        ignored_variables = set(ignored_variables)

    if ignored_interactions is None:
        ignored_interactions = set()
    elif not isinstance(ignored_interactions, abc.Container):
        ignored_interactions = set(ignored_interactions)

    j_sc = {}
    for u, v in j.items():
        if u in ignored_interactions:
            j_sc[u] = v
        else:
            j_sc[u] = v * scalar

    if not ignore_offset:
        offset = offset * scalar

    h_sc = {}
    for k, v in h.items():
        if k not in ignored_variables:
            h_sc[k] = v * scalar
        else:
            h_sc[k] = v

    return h_sc, j_sc, offset
