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

import typing

import numpy as np

from dimod.typing import SampleLike, SamplesLike


valid_sample_like: typing.List[SampleLike] = [
    [0],
    [0, 1],
    {'a': 1, 'b': 1},
    ([0, 1], 'ab'),
    (np.ones(5), 'abcde'),
    np.ones(5),
    ]

invalid_sample_like: typing.List[SampleLike] = [
    [complex(1)],  # E: has incompatible type
    [object()],  # E: has incompatible type
    object(),  # E: has incompatible type
    [[0, 1], [1, 2]],  # E: has incompatible type
]


valid_samples_like: typing.List[SamplesLike] = [
    [0],
    [0, 1],
    {'a': 1, 'b': 1},
    ([0, 1], 'ab'),
    (np.ones(5), 'abcde'),
    np.ones(5),
    [[0, 1], [1, 0]],
    ([[0, 1], [1, 0]], 'ab'),
    [([0, 1], 'ab'), ([1, 1], 'ab')],
    iter([([0, 1], 'ab'), ([1, 1], 'ab')]),
]
