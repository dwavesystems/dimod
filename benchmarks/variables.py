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

from dimod.variables import Variables


class TimeConstuction:
    num_variables = 1000

    iterables = dict(range=range(num_variables),
                     strings=list(map(str, range(num_variables))),
                     integers=list(range(1000)),
                     empty=[],
                     none=None,
                     variables=Variables(range(1000)),
                     )

    params = iterables.keys()
    param_names = ['iterable']

    def time_construction(self, key):
        Variables(self.iterables[key])


class TimeIteration:
    num_variables = 1000

    variables = dict(string=Variables(map(str, range(num_variables))),
                     index=Variables(range(num_variables)),
                     integer=Variables(range(num_variables, 0, -1))
                     )

    params = variables.keys()
    param_names = ['labels']

    def time_iteration(self, key):
        for v in self.variables[key]:
            pass
