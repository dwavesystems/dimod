# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

__all__ = ['Scoped']


class Scoped(abc.ABC):
    """Abstract base class for components that allocate scope-bound resources.

    :class:`.Scoped` requires the concrete implementation to provide resources
    clean-up via :meth:`~.Scoped.close` method. With that, context manager
    protocol is automatically supported (resources are released on context exit).

    The scoped resource should either be used from a `runtime context`_, e.g.:

    >>> with SomeScopedSampler() as sampler:       # doctest: +SKIP
    ...     sampler.sample(...)

    or explicitly disposed with a call to the :meth:`.close` method.

    .. _runtime context: https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers

    .. versionadded:: 0.12.19
        :class:`.Scoped` abstract base class.

    Both :class:`~dimod.code.sampler.Sampler` and :class:`~dimod.core.composite.Composite`
    implement :class:`.Scoped` interface.
    """

    @abc.abstractmethod
    def close(self):
        """Release allocated system resources that are hard or impossible to
        garbage-collect.
        """
        pass

    def __enter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release system resources allocated and raise any exception triggered
        within the runtime context.
        """
        self.close()
        # raise exceptions from the runtime context as they're not handled here
        return None
