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

"""
A composite that clips problem variables below and above threshold. if lower
and upper bounds is not given it does nothing.

"""

try:
    from dwave.preprocessing import ClipComposite as _ClipComposite
except ImportError:
    from dimod.reference.composites._preprocessing import NotFound as _ClipComposite

from dimod.reference.composites._preprocessing import DeprecatedToPreprocessing


__all__ = ['ClipComposite']


class ClipComposite(DeprecatedToPreprocessing, _ClipComposite):
    pass


ClipComposite.__doc__ = _ClipComposite.__doc__
