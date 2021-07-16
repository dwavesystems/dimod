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

import warnings

__all__ = ['NotFound', 'DeprecatedToPreprocessing']


class NotFound:
    pass


class DeprecatedToPreprocessing:
    def __init__(self, *args, **kwargs):
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
            f"{type(self).__name__!s} has been moved to dwave-preprocessing "
            "and will be removed from dimod 0.11.0. To avoid this warning, "
            "import it with 'from dwave.preprocessing import "
            f"{type(self).__name__!s}'.",
            DeprecationWarning, stacklevel=2
            )

        super().__init__(*args, **kwargs)
