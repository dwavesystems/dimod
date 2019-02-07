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
# =============================================================================

import sys
import inspect

from functools import total_ordering

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
    @total_ordering
    class SortKey(object):
        def __init__(self, obj):
            self.obj = obj
            self.name = type(self.obj).__name__

        def __eq__(self, other):
            return self.obj == other.obj

        def __lt__(self, other):
            try:
                return self.obj < other.obj
            except TypeError:
                pass

            if self.name == other.name:

                if isinstance(self.obj, abc.Sequence):
                    # this case happens when there are two sequences of the same type
                    # that have nested objects that python3 cannot compare
                    for u, v in zip(self.obj, other.obj):
                        su = SortKey(u)
                        sv = SortKey(v)
                        if su < sv:
                            return True
                        if sv < su:
                            return False
                    # the prefix case should be caught by the try-catch loop at
                    # the top but just in case
                    return len(self.obj) < len(other.obj)

                # if they are of the same type but they failed the original
                # try-catch block and they are not a sequence then we can't
                # resolve the order (this happens for instance with dicts which
                # python2 can sort but not in python3)
                msg = "cannot sort types {!r} and {!r}"
                raise TypeError(msg.format(self.obj.name, other.obj.name))

            return self.name < other.name
