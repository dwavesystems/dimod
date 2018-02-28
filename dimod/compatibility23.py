import sys
import inspect

from collections import namedtuple

_PY2 = sys.version_info.major == 2

if _PY2:

    def getargspec(f):
        return inspect.getargspec(f)

else:

    _ArgSpec = namedtuple('ArgSpec', ('args', 'varargs', 'keywords', 'defaults'))

    def getargspec(f):
        argspec = inspect.getfullargspec(f)
        # FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)

        return _ArgSpec(argspec.args, argspec.varargs, argspec.varkw, argspec.defaults)
