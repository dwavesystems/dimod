"""todo"""
from __future__ import absolute_import

from abc import ABCMeta, abstractproperty


from _weakrefset import WeakSet


class SamplerABCMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super(ABCMeta, mcls).__new__(mcls, name, bases, namespace, **kwargs)
        # Compute set of abstract method names
        abstracts = {name
                     for name, value in namespace.items()
                     if getattr(value, "__isabstractmethod__", False)}
        samplermixins = {name
                         for name, value in namespace.items()
                         if getattr(value, "__issamplemixin__", False)}
        if len(samplermixins) == 3:
            abstracts.update(samplermixins)
        for base in bases:
            for name in getattr(base, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)
            samplermixins = {name
                             for name in getattr(base, "__abstractmethods__", set())
                             if getattr(getattr(cls, name, None), "__issamplemixin__", False)}
            if len(samplermixins) == 3:
                abstracts.update(samplermixins)
        cls.__abstractmethods__ = frozenset(abstracts)
        # Set up inheritance registry
        cls._abc_registry = WeakSet()
        cls._abc_cache = WeakSet()
        cls._abc_negative_cache = WeakSet()
        cls._abc_negative_cache_version = ABCMeta._abc_invalidation_counter
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
