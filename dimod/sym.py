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

import contextlib

__all__ = ['symbolic']


class symbolic(contextlib.ContextDecorator):
    """Context manager for creating quadratic models symbolically.

    This context manager is reentrant but not thread safe.
    """
    _count: int = 0

    def __init__(self):
        pass

    def __enter__(self):
        symbolic._count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        symbolic._count -= 1

    @classmethod
    def active(cls) -> bool:
        return cls._count > 0
