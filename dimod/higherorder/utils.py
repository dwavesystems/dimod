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

import itertools
import warnings

from collections import Counter

import numpy as np

from six import iteritems

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.higherorder.polynomial import BinaryPolynomial
from dimod.sampleset import as_samples
from dimod.vartypes import as_vartype, Vartype

__all__ = ['make_quadratic']


def _spin_product(variables):
    """A bqm with a gap of 1 that represents the product of two spin variables.

    Note that spin-product requires an auxiliary variable.

    Args:
        variables (list):
            multiplier, multiplicand, product, aux

    Returns:
        :obj:`.BinaryQuadraticModel`

    """
    multiplier, multiplicand, product, aux = variables

    return BinaryQuadraticModel({multiplier: -.5,
                                 multiplicand: -.5,
                                 product: -.5,
                                 aux: -1.},
                                {(multiplier, multiplicand): .5,
                                 (multiplier, product): .5,
                                 (multiplier, aux): 1.,
                                 (multiplicand, product): .5,
                                 (multiplicand, aux): 1.,
                                 (product, aux): 1.},
                                2.,
                                Vartype.SPIN)


def _binary_product(variables):
    """A bqm with a gap of 1 that represents the product of two binary variables.

    Args:
        variables (list):
            multiplier, multiplicand, product

    Returns:
        :obj:`.BinaryQuadraticModel`

    """
    multiplier, multiplicand, product = variables

    return BinaryQuadraticModel({multiplier: 0.0,
                                 multiplicand: 0.0,
                                 product: 3.0},
                                {(multiplier, multiplicand): 1.0,
                                 (multiplier, product): -2.0,
                                 (multiplicand, product): -2.0},
                                0.0,
                                Vartype.BINARY)


def _new_product(variables, u, v):
    # make a new product variable not in variables, then add it
    p = '{}*{}'.format(u, v)
    while p in variables:
        p = '_' + p
    variables.add(p)
    return p


def _new_aux(variables, u, v):
    # make a new auxiliary variable not in variables, then add it
    aux = 'aux{},{}'.format(u, v)
    while aux in variables:
        aux = '_' + aux
    variables.add(aux)
    return aux


def make_quadratic(poly, strength, vartype=None, bqm=None):
    """Create a binary quadratic model from a higher order polynomial.

    Args:
        poly (dict):
            Polynomial as a dict of form {term: bias, ...}, where `term` is a tuple of
            variables and `bias` the associated bias.

        strength (float):
            The energy penalty for violating the prodcut constraint.
            Insufficient strength can result in the binary quadratic model not
            having the same minimizations as the polynomial.

        vartype (:class:`.Vartype`/str/set, optional):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            If `bqm` is provided, `vartype` is not required.

        bqm (:class:`.BinaryQuadraticModel`, optional):
            The terms of the reduced polynomial are added to this binary quadratic model.
            If not provided, a new binary quadratic model is created.

    Returns:
        :class:`.BinaryQuadraticModel`

    Examples:

        >>> poly = {(0,): -1, (1,): 1, (2,): 1.5, (0, 1): -1, (0, 1, 2): -2}
        >>> bqm = dimod.make_quadratic(poly, 5.0, dimod.SPIN)

    """
    if vartype is None:
        if bqm is None:
            raise ValueError("one of vartype or bqm must be provided")
        else:
            vartype = bqm.vartype
    else:
        vartype = as_vartype(vartype)  # handle other vartype inputs
        if bqm is None:
            bqm = BinaryQuadraticModel.empty(vartype)
        else:
            bqm = bqm.change_vartype(vartype, inplace=False)

    bqm.info['reduction'] = {}

    # we want to be able to mutate the polynomial so copy. We treat this as a
    # dict but by using BinaryPolynomial we also get automatic handling of
    # square terms
    poly = BinaryPolynomial(poly, vartype=bqm.vartype)
    variables = set().union(*poly)

    while any(len(term) > 2 for term in poly):
        # determine which pair of variables appear most often
        paircounter = Counter()
        for term in poly:
            if len(term) <= 2:
                # we could leave these in but it can lead to cases like
                # {'ab': -1, 'cdef': 1} where ab keeps being chosen for
                # elimination. So we just ignore all the pairs
                continue
            for u, v in itertools.combinations(term, 2):
                pair = frozenset((u, v))  # so order invarient
                paircounter[pair] += 1
        pair, __ = paircounter.most_common(1)[0]
        u, v = pair

        # make a new product variable p == u*v and replace all (u, v) with p
        p = _new_product(variables, u, v)
        terms = [term for term in poly if u in term and v in term]
        for term in terms:
            new = tuple(w for w in term if w != u and w != v) + (p,)
            poly[new] = poly.pop(term)

        # add a constraint enforcing the relationship between p == u*v
        if vartype is Vartype.BINARY:
            constraint = _binary_product([u, v, p])

            bqm.info['reduction'][(u, v)] = {'product': p}
        elif vartype is Vartype.SPIN:
            aux = _new_aux(variables, u, v)  # need an aux in SPIN-space

            constraint = _spin_product([u, v, p, aux])

            bqm.info['reduction'][(u, v)] = {'product': p, 'auxiliary': aux}
        else:
            raise RuntimeError("unknown vartype: {!r}".format(vartype))

        # scale constraint and update the polynomial with it
        constraint.scale(strength)
        for v, bias in constraint.linear.items():
            try:
                poly[v, ] += bias
            except KeyError:
                poly[v, ] = bias
        for uv, bias in constraint.quadratic.items():
            try:
                poly[uv] += bias
            except KeyError:
                poly[uv] = bias
        try:
            poly[()] += constraint.offset
        except KeyError:
            poly[()] = constraint.offset

    # convert poly to a bqm (it already is one)
    for term, bias in poly.items():
        if len(term) == 2:
            u, v = term
            bqm.add_interaction(u, v, bias)
        elif len(term) == 1:
            v, = term
            bqm.add_variable(v, bias)
        elif len(term) == 0:
            bqm.add_offset(bias)
        else:
            # still has higher order terms, this shouldn't happen
            msg = ('Internal error: not all higher-order terms were reduced. '
                   'Please file a bug report.')
            raise RuntimeError(msg)

    return bqm


def poly_energy(sample_like, poly):
    """Calculates energy of a sample from a higher order polynomial.

    Args:
         sample (samples_like):
            A raw sample. `samples_like` is an extension of NumPy's
            array_like structure. See :func:`.as_samples`.

        poly (dict):
            Polynomial as a dict of form {term: bias, ...}, where `term` is a
            tuple of variables and `bias` the associated bias.

    Returns:
        float: The energy of the sample.

    """

    msg = ("poly_energy is deprecated and will be removed in dimod 0.9.0."
           "In the future, use BinaryPolynomial.energy")
    warnings.warn(msg, DeprecationWarning)
    # dev note the vartype is not used in the energy calculation and this will
    # be deprecated in the future
    return BinaryPolynomial(poly, 'SPIN').energy(sample_like)


def poly_energies(samples_like, poly):
    """Calculates energy of samples from a higher order polynomial.

    Args:
        sample (samples_like):
            A collection of raw samples. `samples_like` is an extension of
            NumPy's array_like structure. See :func:`.as_samples`.

        poly (dict):
            Polynomial as a dict of form {term: bias, ...}, where `term` is a
            tuple of variables and `bias` the associated bias. Variable
            labeling/indexing of terms in poly dict must match that of the
            sample(s).

    Returns:
        list/:obj:`numpy.ndarray`: The energy of the sample(s).

    """
    msg = ("poly_energies is deprecated and will be removed in dimod 0.9.0."
           "In the future, use BinaryPolynomial.energies")
    warnings.warn(msg, DeprecationWarning)
    # dev note the vartype is not used in the energy calculation and this will
    # be deprecated in the future
    return BinaryPolynomial(poly, 'SPIN').energies(samples_like)
