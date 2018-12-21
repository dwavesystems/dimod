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

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.higherorder import polymorph_response, make_quadratic
import dimod

__all__ = ['HigherOrderComposite']

class HigherOrderComposite(dimod.ComposedSampler):

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param['penalty_strength'] = []
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, h, J, offset=0, penalty_strength=1.0, **parameters):
        bqm = BinaryQuadraticModel(linear=h, quadratic={}, offset=0,
                                   vartype=dimod.SPIN)
        bqm = make_quadratic(J, penalty_strength, bqm=bqm)
        response = self.child.sample(bqm, **parameters)

        return polymorph_response(response, h, J, offset, bqm,
                                  penalty_strength=penalty_strength)
