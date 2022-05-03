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
