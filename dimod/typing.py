# Copyright 2021 D-Wave Systems Inc.
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
Type hints for common dimod inputs.
"""
import collections.abc
import typing

import numpy as np

from dimod.vartypes import VartypeLike

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    # support numpy < 1.20
    ArrayLike = collections.abc.Sequence
    DTypeLike = typing.Any

try:
    from numpy.typing import NDArray
except ImportError:
    # support numpy < 1.21
    NDArray = collections.abc.Sequence


__all__ = ['Bias',
           'GraphLike',
           'Polynomial',
           'SampleLike',
           'SamplesLike',
           'Variable',
           'VartypeLike',
           ]

# use float for python types, https://www.python.org/dev/peps/pep-0484/#the-numeric-tower
# exclude np.complexfloating from numpy types
Bias = typing.Union[float, np.floating, np.integer]
"""A :obj:`~typing.Union` representing objects that can be used as biases.

This includes:

* Python's :class:`int` and :class:`float`.
* NumPy's :class:`~numpy.floating` and :class:`~numpy.integer`.

"""

# it would be nice to be able to exclude None from Variable. We can define
# a special abstract base class that excludes None, but making it behave like
# a typing class is a pain accross our supported Python versions. In the future
# we should handle it

Variable = collections.abc.Hashable
"""Objects that can be used as variable labels.

.. note::

    In `dimod` variables can be labelled using any hashable object except for
    :obj:`None`. However, for simplicity we alias :class:`~collections.abc.Hashable` which
    does permit :obj:`None`.

"""


GraphLike = typing.Union[
    int,  # number of nodes
    tuple[
        collections.abc.Collection[Variable],
        collections.abc.Collection[tuple[Variable, Variable]]
    ],
    collections.abc.Collection[tuple[Variable, Variable]],  # edges
    ]
"""Objects that can be interpreted as a graph.

This includes:

* An :class:`int`, interpreted as a complete graph with nodes labelled ``range(n)``.
* A list of edges
* A 2-tuple containing a list of nodes and a list of edges
* A :class:`networkx.Graph`.

"""

try:
    import networkx as nx
except ImportError:
    pass
else:
    GraphLike = typing.Union[GraphLike, nx.Graph]

Polynomial = collections.abc.Mapping[collections.abc.Sequence[Variable], Bias]
"""A polynomial represented by a mapping."""


class QuadraticVectors(typing.NamedTuple):
    row_indices: NDArray[np.integer]
    col_indices: NDArray[np.integer]
    biases: NDArray[np.floating]


class BQMVectors(typing.NamedTuple):
    linear_biases: NDArray[np.floating]
    quadratic: QuadraticVectors
    offset: Bias


class LabelledBQMVectors(typing.NamedTuple):
    linear_biases: NDArray[np.floating]
    quadratic: QuadraticVectors
    offset: Bias
    labels: collections.abc.Sequence[Variable]


class DQMVectors(typing.NamedTuple):
    case_starts: NDArray[np.integer]
    linear_biases: NDArray[np.floating]
    quadratic: QuadraticVectors
    labels: collections.abc.Sequence[Variable]
    offset: Bias


SampleLike = typing.Union[
    collections.abc.Sequence[Bias],
    collections.abc.Mapping[Variable, Bias],
    tuple[collections.abc.Sequence[Bias], collections.abc.Sequence[Variable]],
    tuple[np.ndarray, collections.abc.Sequence[Variable]],
    np.ndarray,  # this is overgenerous, but there is no way to specify 1-dimensional
    ]
"""Objects that can be interpreted as a single sample.

This includes:

* A one-dimensional NumPy array_like_.
* A 2-:class:`tuple` containing a one-dimensional NumPy array_like_ and a
  list of variable labels.
* A :class:`dict` where the keys are variable labels and the values are the
  assignments.

NumPy array_like_ is a very flexible definition.

.. _array_like: https://numpy.org/devdocs/glossary.html#term-array_like

"""


SamplesLike = typing.Union[
    SampleLike,
    collections.abc.Sequence[collections.abc.Sequence[Bias]],  # 2d array
    tuple[collections.abc.Sequence[collections.abc.Sequence[Bias]], collections.abc.Sequence[Variable]],
    collections.abc.Sequence[SampleLike],
    collections.abc.Iterator[SampleLike],
    ]
"""Objects that can be interpreted as a collection of samples.

This includes:

* Any :obj:`SampleLike`.
* A two-dimensional NumPy array_like_.
* A 2-:class:`tuple` containing a two-dimensional array_like_ and a
  list of variable labels.
* A :class:`list` of :class:`dict` where each dict has the same keys.

.. _array_like: https://numpy.org/devdocs/glossary.html#term-array_like

"""
