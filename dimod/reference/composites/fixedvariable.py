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

"""
A composite that fixes the variables provided and removes them from the binary
quadratic model before sending to its child sampler.
"""
import warnings

try:
    from dwave.preprocessing import FixVariablesComposite
except ImportError:
    from dimod.reference.composites._preprocessing import NotFound as FixVariablesComposite

from dimod.reference.composites._preprocessing import NotFound

__all__ = ['FixedVariableComposite']


class FixedVariableComposite(FixVariablesComposite):
    """Composite to fix variables of a problem to provided.


    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.FixedVariableComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and quadratic
       biases according.

       >>> h = {1: -1.3, 4: -0.5}
       >>> J = {(1, 4): -0.6}
       >>> sampler = dimod.FixedVariableComposite(dimod.ExactSolver())
       >>> sampleset = sampler.sample_ising(h, J, fixed_variables={1: -1})

    """
    def __init__(self, child):
        if isinstance(self, NotFound):
            # we recommend --no-deps because its dependencies are the same as
            # dimods and it would be a circular install otherwise
            raise TypeError(
                f"{type(self).__name__!r} has been moved to dwave-preprocessing. "
                "You must install dwave-preprocessing in order to use it. "
                "You can do so with "
                "'pip install \"dwave-preprocessing<0.4\" --no-deps'.",
                )

        # otherwise warn about it's new location but let it proceed
        warnings.warn(
            f"{type(self).__name__!s} has been deprecated and will be removed from dimod 0.11.0. "
            "You can get similar functionality in dwave-preprocessing "
            " To avoid this warning, import 'from dwave.preprocessing import FixVariablesComposite'.",
            DeprecationWarning, stacklevel=2
            )

        super().__init__(child, algorithm='explicit')

    def sample(self, bqm, fixed_variables=None, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            fixed_variables (dict):
                A dictionary of variable assignments.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        return super().sample(bqm, fixed_variables=fixed_variables, **parameters)
