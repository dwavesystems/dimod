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
import tempfile
import typing

import dimod  # for typing

from dimod.serialization.cylp import read_lp_file as cyread_lp_file


__all__ = ['read_lp', 'read_lp_file']


@functools.singledispatch
def read_lp(lp, *args, **kwargs) -> dimod.ConstrainedQuadraticModel:
    raise TypeError("unsupported lp type:", type(lp))


@read_lp.register(collections.abc.ByteString)
def _read_lp_bytestring(bytestring: bytes) -> dimod.ConstrainedQuadraticModel:
    # create a named temporary file that we can pass
    with tempfile.NamedTemporaryFile('wb', delete=False) as f:
        f.write(bytestring)

    try:
        cqm = read_lp_file(f.name)
    finally:
        # remove the file so we're not accumulating memory/disk space
        os.unlink(f.name)

    return cqm


@read_lp.register(io.IOBase)
def _read_lp_filelike(fp: io.IOBase) -> dimod.ConstrainedQuadraticModel:
    try:
        filename = fp.name
    except AttributeError:
        # todo: when does this happen?
        raise NotImplementedError

    if fp.tell():
        # todo: figure out what to do in this case
        raise NotImplementedError

    # ok, we have a named file and the user wants us to start at the beginning
    # so let's just use that
    return read_lp_file(filename)


@read_lp.register(str)
def _read_lp_string(string: str) -> dimod.ConstrainedQuadraticModel:
    # create a named temporary file that we can pass
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(string)

    try:
        cqm = read_lp_file(f.name)
    finally:
        # remove the file so we're not accumulating memory/disk space
        os.unlink(f.name)

    return cqm


def read_lp_file(filename: str) -> dimod.ConstrainedQuadraticModel:
    return cyread_lp_file(filename)
