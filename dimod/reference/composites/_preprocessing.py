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

"""Unified warnings/exceptions for stuff moved to dwave-preprocessing."""

__all__ = ['DeprecatedToPreprocessing']


class DeprecatedToPreprocessing:
    """Removed.

    .. deprecated:: 0.10.0

        The following composites were migrated to
        `dwave-preprocessing <https://github.com/dwavesystems/dwave-preprocessing>`_

        * ``ClipComposite``
        * ``ConnectedComponentsComposite``
        * ``FixedVariableComposite``
        * ``RoofDualityComposite``
        * ``ScaleComposite``
        * ``SpinReversalTransformComposite``

        You must install ``dwave-preprocessing`` in order to use them.

    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
                f"{type(self).__name__!r} was moved to dwave-preprocessing in dimod 0.10.0. "
                "You must install dwave-preprocessing in order to use it. "
                "You can do so with 'pip install dwave-preprocessing'. "
                "This stub class will be removed in dimod 0.13.0.",
            )
