import sys
import itertools
import inspect

from collections import namedtuple

_PY2 = sys.version_info.major == 2

if _PY2:

    range_ = xrange

    zip_ = itertools.izip

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    def iterkeys(d):
        return d.iterkeys()

    zip_longest = itertools.izip_longest

    def getargspec(f):
        return inspect.getargspec(f)

else:

    range_ = range

    zip_ = zip

    def iteritems(d):
        return iter(d.items())

    def itervalues(d):
        return iter(d.values())

    def iterkeys(d):
        return iter(d.keys())

    zip_longest = itertools.zip_longest

    def getargspec(f):
        ArgSpec = namedtuple('ArgSpec', ('args', 'varargs', 'keywords', 'defaults'))

        # FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)
        argspec = inspect.getfullargspec(f)

        return ArgSpec(argspec.args, argspec.varargs, argspec.varkw, argspec.defaults)
