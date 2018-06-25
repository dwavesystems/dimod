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

from __future__ import absolute_import

import abc


class SamplerABCMeta(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = abc.ABCMeta.__new__(mcls, name, bases, namespace, **kwargs)

        samplermixins = {name
                         for name, value in namespace.items()
                         if getattr(value, "__issamplemixin__", False)}
        if len(samplermixins) == 3:
            abstracts = samplermixins
        else:
            abstracts = set()

        for base in bases:
            samplermixins = {name
                             for name in getattr(base, "__abstractmethods__", set())
                             if getattr(getattr(cls, name, None), "__issamplemixin__", False)}
            if len(samplermixins) == 3:
                abstracts.update(samplermixins)

        # if we found any, update abstract methods
        if abstracts:
            cls.__abstractmethods__ = frozenset(abstracts.union(cls.__abstractmethods__))

        return cls


def samplemixinmethod(method):
    """Marks a method as being a mixin.

    Adds the '__issamplemixin__' attribute with value True to the decorated function.

    Examples:
        >>> @samplemixinmethod
        >>> def f():
        ...     pass
        >>> f.__issamplemixin__
        True

    """
    # NB: decorator name was chosen to be consistent with @classmethod and @staticmethod
    method.__issamplemixin__ = True
    return method
