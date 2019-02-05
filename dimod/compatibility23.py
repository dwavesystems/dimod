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

import sys
import inspect

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from collections import namedtuple

_PY2 = sys.version_info.major == 2

if _PY2:

    def getargspec(f):
        return inspect.getargspec(f)

    def SortKey(obj):
        return obj

else:

    _ArgSpec = namedtuple('ArgSpec', ('args', 'varargs', 'keywords', 'defaults'))

    def getargspec(f):
        argspec = inspect.getfullargspec(f)
        # FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)

        return _ArgSpec(argspec.args, argspec.varargs, argspec.varkw, argspec.defaults)

    # Based on an answer https://stackoverflow.com/a/34757114/8766655
    # by kindall https://stackoverflow.com/users/416467/kindall
    class SortKey(object):
        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            try:
                return self.obj < other.obj
            except TypeError:
                pass

            if isinstance(self.obj, type(other.obj)):
                if not isinstance(self.obj, abc.Sequence):
                    raise TypeError("cannot compare types")
                for v0, v1 in zip(self.obj, other.obj):
                    if SortKey(v0) < SortKey(v1):
                        return True
                return len(self.obj) < len(other.obj)

            return type(self.obj).__name__ < type(other.obj).__name__
