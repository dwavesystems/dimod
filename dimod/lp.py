# Copyright 2022 D-Wave Systems Inc.
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

from __future__ import annotations

import collections.abc
import functools
import io
import os
import shutil
import tempfile
import typing

import dimod  # for typing

from dimod.cylp import cyread_lp_file


__all__ = ['loads', 'load']


def load(file_like: typing.Union[str, bytes, io.IOBase]) -> dimod.ConstrainedQuadraticModel:
    """Construct a constrained quadratic model from a LP file.

    LP files are a common format for encoding optimization models. See
    documentation from Gurobi_ and CPLEX_.

    Args:
        file_like: Either a :class:`str` or :class:`bytes` object representing
            a path, or a file-like_ object.

    Returns:

        An example of reading from a LP file.

        .. code-block:: python

            with open('example.lp', 'rb') as f:
                cqm = dimod.lp.load(f)

        An example of loading from a path

        .. code-block:: python

            cqm = dimod.lp.load('example.lp')

    See also:

        :func:`~dimod.serialization.lp.loads`

    .. versionadded:: 0.11.0

    .. _Gurobi: https://www.gurobi.com/documentation/9.5/refman/lp_format.html

    .. _CPLEX: https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-lp-file-format-algebraic-representation

    .. _file-like: https://docs.python.org/3/glossary.html#term-file-object

    """

    if isinstance(file_like, (str, bytes)):
        return cyread_lp_file(file_like)

    # ok, we got a file-like

    if not file_like.readable():
        raise ValueError("file_like must be readable")

    if file_like.seekable() and file_like.tell():
        # this file is current pointing somewhere other than the beginning,
        # so we make a copy so our reader can start at the beginning
        filename = ''
    else:
        try:
            filename = file_like.name
        except AttributeError:
            filename = ''

    if os.path.isfile(filename):
        return cyread_lp_file(filename)        

    # copy it into somewhere that our reader can get it
    with tempfile.NamedTemporaryFile('wb', delete=False) as tf:
        shutil.copyfileobj(file_like, tf)

    try:
        return cyread_lp_file(tf.name)
    finally:
        # remove the file so we're not accumulating memory/disk space
        os.unlink(tf.name)


def loads(obj: typing.Union[str, typing.ByteString]) -> dimod.ConstrainedQuadraticModel:
    """Construct a constrained quadratic model from a string formatted as a LP file.

    LP files are a common format for encoding optimization models. See
    documentation from Gurobi_ and CPLEX_.

    Args:
        obj: A :class:`str` or :class:`bytes` formatted like an LP file.

    Returns:

        An example of reading a string formatted as an LP file.

        .. testcode::

            lp = '''
            Minimize
                x0 - 2 x1
            Subject To
                x0 + x1 = 1
            Binary
                x0 x1
            End
            '''

            cqm = dimod.lp.loads(lp)


    See also:

        :func:`~dimod.serialization.lp.load`

    .. versionadded:: 0.11.0

    .. _Gurobi: https://www.gurobi.com/documentation/9.5/refman/lp_format.html

    .. _CPLEX: https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-lp-file-format-algebraic-representation

    .. _file-like: https://docs.python.org/3/glossary.html#term-file-object

    """

    if isinstance(obj, str):
        obj = obj.encode()

    return load(io.BytesIO(obj))
