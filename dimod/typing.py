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
import typing

from typing import Collection, Hashable, Mapping, NamedTuple, Sequence, Tuple, Union

import numpy as np

from dimod.vartypes import VartypeLike

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    # support numpy < 1.20
    ArrayLike = typing.Sequence
    DTypeLike = typing.Any

try:
    from numpy.typing import NDArray
except ImportError:
    # support numpy < 1.21
    NDArray = typing.Sequence


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
Bias = Union[float, np.floating, np.integer]

Variable = Hashable  # todo: exclude None

try:
    import networkx as nx
except ImportError:
    GraphLike = Union[
        int,  # number of nodes
        Tuple[Collection[Variable], Collection[Tuple[Variable, Variable]]],  # nodes/edges
        Collection[Tuple[Variable, Variable]],  # edges
        ]
else:
    GraphLike = Union[
        int,  # number of nodes
        Tuple[Collection[Variable], Collection[Tuple[Variable, Variable]]],  # nodes/edges
        Collection[Tuple[Variable, Variable]],  # edges
        nx.Graph,
        ]

Polynomial = Mapping[Sequence[Variable], Bias]

class QuadraticVectors(NamedTuple):
    row_indices: NDArray[np.integer]
    col_indices: NDArray[np.integer]
    biases: NDArray[np.floating]


class BQMVectors(NamedTuple):
    linear_biases: NDArray[np.floating]
    quadratic: QuadraticVectors
    offset: Bias


class LabelledBQMVectors(NamedTuple):
    linear_biases: NDArray[np.floating]
    quadratic: QuadraticVectors
    offset: Bias
    labels: Sequence[Variable]


class DQMVectors(NamedTuple):
    case_starts: NDArray[np.integer]
    linear_biases: NDArray[np.floating]
    quadratic: QuadraticVectors
    labels: Sequence[Variable]
    offset: Bias


SampleLike = Union[
    typing.Sequence[float],
    typing.Mapping[Variable, Bias],
    ArrayLike,  # this is overgenerous, but we cannot easily specify it better
    ]


SamplesLike = Union[
    SampleLike,
    typing.Sequence[typing.Sequence[float]],  # 2d array
    typing.Tuple[typing.Sequence[float], typing.List[Variable]],
    typing.Tuple[typing.Sequence[typing.Sequence[float]], typing.List[Variable]],
    typing.Sequence[SampleLike],
    typing.Iterator[SampleLike],
    ]
