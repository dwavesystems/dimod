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
On the D-Wave system, coupling :math:`J_{i,j}` adds a small bias to qubits :math:`i` and
:math:`j` due to leakage. This can become significant for chained qubits. Additionally,
qubits are biased to some small degree in one direction or another.
Applying a spin-reversal transform can improve results by reducing the impact of possible
analog and systematic errors. A spin-reversal transform does not alter the Ising problem;
the transform simply amounts to reinterpreting spin up as spin down, and visa-versa, for
a particular spin.
"""

try:
    from dwave.preprocessing import SpinReversalTransformComposite as _SpinReversalTransformComposite
except ImportError:
    from dimod.reference.composites._preprocessing import NotFound as _SpinReversalTransformComposite

from dimod.reference.composites._preprocessing import DeprecatedToPreprocessing


__all__ = ['SpinReversalTransformComposite']


class SpinReversalTransformComposite(DeprecatedToPreprocessing, _SpinReversalTransformComposite):
    pass


SpinReversalTransformComposite.__doc__ = _SpinReversalTransformComposite.__doc__
