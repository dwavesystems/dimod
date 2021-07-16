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

"""A composite that uses the roof duality algorithm [#bht]_ [#bh]_ to fix some
variables in the binary quadratic model before passing it on to its child
sampler.

.. [#bht] Boros, E., P.L. Hammer, G. Tavares. Preprocessing of Unconstrained
    Quadratic Binary Optimization. Rutcor Research Report 10-2006, April, 2006.

.. [#bh] Boros, E., P.L. Hammer. Pseudo-Boolean optimization. Discrete Applied
    Mathematics 123, (2002), pp. 155-225

"""
import warnings

try:
    from dwave.preprocessing import FixVariablesComposite
except ImportError:
    from dimod.reference.composites._preprocessing import NotFound as FixVariablesComposite

from dimod.reference.composites._preprocessing import NotFound

__all__ = ['RoofDualityComposite']


class RoofDualityComposite(FixVariablesComposite):
    """Uses roof duality to assign some variables before invoking child sampler.

    Uses the :func:`~dimod.roof_duality.fix_variables` function to determine
    variable assignments, then fixes them before calling the child sampler.
    Returned samples include the fixed variables.

    Args:
       child (:obj:`dimod.Sampler`):
            A dimod sampler. Used to sample the binary quadratic model after
            variables have been fixed.

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

        super().__init__(child, algorithm='roof_duality')

    def sample(self, bqm, sampling_mode=True, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            sampling_mode (bool, optional, default=True):
                In sampling mode, only roof-duality is used. When
                `sampling_mode` is false, strongly connected components are used
                to fix more variables, but in some optimal solutions these
                variables may take different values.

            **parameters:
                Parameters for the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        return super().sample(bqm, strict=sampling_mode, **parameters)
