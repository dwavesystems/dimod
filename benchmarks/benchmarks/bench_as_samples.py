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
import itertools

import numpy as np
import dimod


from benchmarks.common import Benchmark


class NumpyArray(Benchmark):
    def setup(self):
        self.empty = np.asarray([], dtype=np.int8)
        self.arr100x100 = np.ones((100, 100), dtype=np.int8)
        self.labels100 = list(range(100))

    def time_empty(self):
        dimod.as_samples(self.empty)

    def time_100x100(self):
        dimod.as_samples(self.arr100x100)

    def time_empty_labelled(self):
        dimod.as_samples((self.empty, self.labels100))

    def time_100x100_labelled(self):
        dimod.as_samples((self.arr100x100, self.labels100))


class Dicts(Benchmark):
    def setup(self):
        # 1000 variables
        variables = list(''.join(letters) for letters in itertools.product('abcdefghij', repeat=3))

        self.dict1000 = {v: 1 for v in variables}
        self.dict1x1000 = [{v: 1 for v in variables}]
        self.dict1000x1000 = [{v: 1 for v in variables} for _ in range(1000)]

    def time_empty(self):
        dimod.as_samples({})

    def time_dict1000(self):
        dimod.as_samples(self.dict1000)

    def time_dict1x1000(self):
        dimod.as_samples(self.dict1x1000)

    def time_dict1000x1000(self):
        dimod.as_samples(self.dict1000x1000)


class Lists(Benchmark):
    def setup(self):
        self.list1000 = [1]*1000
        self.list1x1000 = [[1]*1000]
        self.list1000x1000 = [[1]*1000 for _ in range(1000)]

    def time_empty(self):
        dimod.as_samples([])

    def time_list1000(self):
        dimod.as_samples(self.list1000)

    def time_list1x1000(self):
        dimod.as_samples(self.list1x1000)

    def time_list1000x1000(self):
        dimod.as_samples(self.list1000x1000)
