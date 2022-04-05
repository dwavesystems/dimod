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


class TimeAddVariablesFrom:
    variables_str = list(map(str, range(5000)))
    variables_int = range(5000)

    def setUp(self):
        self.qm = dimod.QuadraticModel()

    def time_add_variables_from_str(self):
        self.qm.add_variables_from('BINARY', self.variables_str)

    def time_add_variables_from_int(self):
        self.qm.add_variables_from('BINARY', self.variables_int)
