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

import itertools

from collections import Counter

from six import iteritems

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import vartype_argument
from dimod.vartypes import Vartype

__all__ = ['make_quadratic']


def _spin_product(variables):
    """Create a bqm with a gap of 2 that represents the product of two variables.

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
    """Create a bqm with a gap of 2 that represents the product of two variables.

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


def make_quadratic(poly, strength, vartype=None, bqm=None):
    """Create a binary quadratic model from a higher order polynomial.

    Args:
        poly (dict):
            A polynomial. Should be a dict of the form {term: bias, ...} where term is a tuple of
            variables and bias is their associated bias.

        strength (float):
            The strength of the reduction constraint. Insufficient strength can result in the
            binary quadratic model not having the same minimizations as the polynomial.

        vartype (:class:`.Vartype`, optional):
            The vartype of the polynomial. If a bqm is provided, then vartype is not required.

        bqm (:class:`.BinaryQuadraticModel`, optional):
            The terms of the reduced polynomial are added to this bqm. If not provided a new
            empty binary quadratic model is created.

    Returns:
        :class:`.BinaryQuadraticModel`

    Examples:

        >>> import dimod
        ...
        >>> poly = {(0,): -1, (1,): 1, (2,): 1.5, (0, 1): -1, (0, 1, 2): -2}
        >>> bqm = make_quadratic(poly, 5.0, dimod.SPIN)

    """

    if bqm is None:
        if vartype is None:
            raise ValueError("one of vartype and create_using must be provided")
        bqm = BinaryQuadraticModel.empty(vartype)
    else:
        if not isinstance(bqm, BinaryQuadraticModel):
            raise TypeError('create_using must be a BinaryQuadraticModel')
        if vartype is not None and vartype is not bqm.vartype:
            raise ValueError("one of vartype and create_using must be provided")
    bqm.info['reduction'] = {}

    new_poly = {}
    for term, bias in iteritems(poly):
        if len(term) == 0:
            bqm.add_offset(bias)
        elif len(term) == 1:
            v, = term
            bqm.add_variable(v, bias)
        else:
            new_poly[term] = bias

    return _reduce_degree(bqm, new_poly, vartype, strength)


def _reduce_degree(bqm, poly, vartype, scale):

    if all(len(term) <= 2 for term in poly):
        # termination criteria, we are already quadratic
        bqm.add_interactions_from(poly)
        return bqm

    # determine which pair of variables appear most often
    paircounter = Counter()
    for term in poly:
        if len(term) > 2:
            for u, v in itertools.combinations(term, 2):
                pair = frozenset((u, v))
                paircounter[pair] += 1

    pair, __ = paircounter.most_common(1)[0]
    u, v = pair

    # make a new product variable and aux variable and add constraint that u*v == p
    p = '{}*{}'.format(u, v)

    while p in bqm.linear:
        p = '_' + p

    if vartype is Vartype.BINARY:
        constraint = _binary_product([u, v, p])

        bqm.info['reduction'][(u, v)] = {'product': p}
    else:
        aux = 'aux{},{}'.format(u, v)
        while aux in bqm.linear:
            aux = '_' + aux
        constraint = _spin_product([u, v, p, aux])

        bqm.info['reduction'][(u, v)] = {'product': p, 'auxiliary': aux}

    constraint.scale(scale)
    bqm.update(constraint)

    new_poly = {}
    for interaction, bias in poly.items():
        if u in interaction and v in interaction:

            if len(interaction) == 2:
                # in this case we are reducing a quadratic bias, so it becomes linear and can
                # be removed
                assert len(interaction) >= 2
                bqm.add_variable(p, bias)
                continue

            interaction = tuple(s for s in interaction if s not in pair)
            interaction += (p,)

        if interaction in new_poly:
            new_poly[interaction] += bias
        else:
            new_poly[interaction] = bias

    return _reduce_degree(bqm, new_poly, vartype, scale)
