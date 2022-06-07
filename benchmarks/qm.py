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

import dimod

import itertools


class TimeAddLinearFrom:
    params = ([str, int], [dict, list])

    def setUp(self, type_, container):
        self.qm = qm = dimod.QuadraticModel()
        qm.add_variables_from('BINARY', map(type_, range(5000)))

        self.iterator = container((type_(v), 1) for v in range(5000))

    def time_add_linear_from(self, *args):
        self.qm.add_linear_from(self.iterator)


class TimeAddVariablesFrom:
    params = [str, int]

    def setUp(self, type_):
        self.qm = dimod.QuadraticModel()
        self.variables = list(map(type_, range(5000)))

    def time_add_variables_from(self, *args):
        self.qm.add_variables_from('BINARY', self.variables)


class TimeFromFile:

    dense = dimod.QM()
    dense.add_variables_from('BINARY', range(2500))
    dense.add_quadratic_from((u, v, 1) for u, v in itertools.combinations(dense.variables, 2))

    dense_fp = dense.to_file()

    # ideally this would be more determistic than just specifying the seed, but
    # this is a lot easier.
    sparse = dimod.QM.from_bqm(dimod.generators.gnm_random_bqm(5000, 5000, 'SPIN', random_state=42))

    sparse_fp = sparse.to_file()

    def setup(self):
        self.dense_fp.seek(0)
        self.sparse_fp.seek(0)

    def time_from_file_dense(self):
        dimod.QM.from_file(self.dense_fp)

    def time_from_file_sparse(self):
        dimod.QM.from_file(self.sparse_fp)
