# Copyright 2022 D-Wave Systems Inc.
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

from __future__ import annotations

import collections.abc
import itertools
import math
import typing

import numpy as np

import dimod  # for typing

from dimod.binary import BinaryQuadraticModel
from dimod.vartypes import Vartype


__all__ = ["nae3sat"]


def _iter_interactions(num_variables: int, num_clauses: int,
                       seed: typing.Union[None, int, np.random.Generator] = None,
                       ) -> typing.Iterator[typing.Tuple[int, int, int]]:

    rng = np.random.default_rng(seed)

    for _ in range(num_clauses):
        x, y, z = rng.choice(num_variables, 3, replace=False)
        xs, ys, zs = 2 * rng.integers(0, 1, endpoint=True, size=3) - 1
        yield (x, y, xs*ys)
        yield (x, z, xs*zs)
        yield (y, z, ys*zs)


def _iter_interactions_without_replacement(
        num_variables: int, num_clauses: int,
        seed: typing.Union[None, int, np.random.Generator] = None,
        ) -> typing.Iterator[typing.Tuple[int, int, int]]:

    try:
        max_num_clauses = math.comb(num_variables, 3)
    except AttributeError:
        # Python < 3.8
        # This is a bit less picky and has overflow issues but
        # probably fine for any cases users are likely to actually
        # submit
        max_num_clauses = math.factorial(num_variables)
        max_num_clauses //= (6 * math.factorial(num_variables - 3))

    if num_clauses > max_num_clauses:
        raise ValueError("rho results in too many clauses")

    rng = np.random.default_rng(seed)

    seen: typing.Set[typing.FrozenSet[int]] = set()
    while len(seen) < num_clauses:
        x, y, z = rng.choice(num_variables, 3, replace=False)

        fz = frozenset((x, y, z))  # so we're order-invarient
        if fz in seen:
            continue
        seen.add(fz)

        xs, ys, zs = 2 * rng.integers(0, 1, endpoint=True, size=3) - 1

        yield (x, y, xs*ys)
        yield (x, z, xs*zs)
        yield (y, z, ys*zs)


def nae3sat(variables: typing.Union[int, typing.Sequence[dimod.typing.Variable]],
            rho: float = 2.1,
            *,
            seed: typing.Union[None, int, np.random.Generator] = None,
            replace: bool = True,
            ) -> BinaryQuadraticModel:
    """Generator for Not-All-Equal 3-SAT (NAE3SAT) Binary Quadratic Models.

    NAE3SAT_ is an NP-complete problem class that consists in satistying a number of conjunctive
    clauses that involve three variables (or variable negations). The variables on each clause
    should be not-all-equal. Ie. all solutions except ``(+1, +1, +1)`` or
    ``(-1, -1, -1)`` are valid for each class.

    .. _NAE3SAT: https://en.wikipedia.org/wiki/Not-all-equal_3-satisfiability

    Args:
        num_variables: The number of variables in the problem.
        rho: The clause-to-variable ratio.
        seed: Passed to :func:`numpy.random.default_rng()`, which is used
            to generate the clauses and the negations.
        replace: If true, then clauses are randomly sampled from the space
            of all possible clauses. This can result in the same three variables
            being present in multiple clauses.
            As the number of variables grows the probability of this happening
            shrinks rapidly and therefore it is often better to allow sampling
            with replacement for performance.

    Returns:
        A binary quadratic models with spin variables.

    """
    if isinstance(variables, collections.abc.Sequence):
        num_variables = len(variables)
        labels = variables
    elif variables < 0:
        raise ValueError("variables must be a sequence or a positive integer")
    else:
        num_variables = variables
        labels = None

    if rho < 0:
        raise ValueError("rho must be positive")

    num_clauses = round(rho * num_variables)

    bqm = BinaryQuadraticModel(num_variables, Vartype.SPIN)
    if replace:
        bqm.add_quadratic_from(_iter_interactions(num_variables, num_clauses, seed))
    else:
        bqm.add_quadratic_from(_iter_interactions_without_replacement(num_variables, num_clauses, seed))

    if labels:
        bqm.relabel_variables(dict(enumerate(labels)))

    return bqm
