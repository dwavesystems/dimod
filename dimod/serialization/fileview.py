# Copyright 2019 D-Wave Systems Inc.
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

import abc
import io
import json
import struct
import tempfile
import warnings

from collections import namedtuple
from typing import BinaryIO, ByteString, Callable, Mapping, Tuple

import numpy as np

from dimod.variables import iter_deserialize_variables, iter_serialize_variables


__all__ = ['FileView', 'load']


# we want to use SpooledTemporaryFile but have it also include the methods
# from io.IOBase. This is (probably) forthcoming in future python, see
# https://bugs.python.org/issue35112
if issubclass(tempfile.SpooledTemporaryFile, io.IOBase):
    warnings.warn("Using deprecated SpooledTemporaryFile wrapper, "
                  "functionality is now included in SpooledTemporaryFile",
                  DeprecationWarning)


class SpooledTemporaryFile(tempfile.SpooledTemporaryFile):
    # This is not part of io.IOBase, but it is implemented in io.BytesIO
    # and io.TextIOWrapper

    def readinto(self, *args, **kwargs):
        return self._file.readinto(*args, **kwargs)

    def readable(self):
        return self._file.readable()

    def seekable(self):
        return self._file.seekable()

    def writable(self):
        return self._file.writable()


class Section(abc.ABC):
    @property
    @abc.abstractmethod
    def magic(self):
        """A 4-byte section identifier. Must be a class variable."""
        pass

    @classmethod
    @abc.abstractmethod
    def loads_data(cls, buff):
        """Accepts a bytes-like object and returns the saved data."""
        pass

    @abc.abstractmethod
    def dump_data(self):
        """Returns a bytes-like object encoding the relevant data."""
        pass

    def dumps(self):
        """Wraps .dump_data to include the identifier and section length."""
        magic = self.magic

        if not isinstance(magic, bytes):
            raise TypeError("magic string should by bytes object")
        if len(magic) != 4:
            raise ValueError("magic string should be 4 bytes in length")

        length = bytes(4)  # placeholder 4 bytes for length

        data = self.dump_data()

        data_length = len(data)

        parts = [magic, length, data]

        if (data_length + len(magic) + len(length)) % 64:
            pad_length = 64 - (data_length + len(magic) + len(length)) % 64
            parts.append(b' '*pad_length)
            data_length += pad_length

        parts[1] = np.dtype('<u4').type(data_length).tobytes()

        assert sum(map(len, parts)) % 64 == 0

        return b''.join(parts)

    @classmethod
    def load(cls, fp):
        """Wraps .loads_data and checks the identifier and length."""
        magic = fp.read(len(cls.magic))
        if magic != cls.magic:
            raise ValueError("unknown subheader, expected {} but recieved "
                             "{}".format(cls.magic, magic))
        length = np.frombuffer(fp.read(4), '<u4')[0]
        return cls.loads_data(fp.read(int(length)))


class VariablesSection(Section):
    magic = b'VARS'

    def __init__(self, variables):
        self.variables = variables

    def dump_data(self):
        serializable = list(iter_serialize_variables(self.variables))
        return json.dumps(serializable).encode('ascii')

    @classmethod
    def loads_data(self, data):
        return iter_deserialize_variables(json.loads(data.decode('ascii')))


def FileView(bqm, version=(1, 0), ignore_labels=False):
    warnings.warn("FileView is deprecated, please use `bqm.to_file` instead",
                  DeprecationWarning, stacklevel=2)
    return bqm.to_file(version=version, ignore_labels=ignore_labels)


class _BytesIO(io.RawIOBase):
    # A stub implementation that mimics io.BytesIO but does not make a copy
    # in the case of a memoryview or bytearray. This is necessary because,
    # although io.BytesIO avoids a copy of bytes objects in python 3.5+, it
    # still copies the mutable versions.
    #
    # This is based on the version in the _pyio library
    # https://github.com/python/cpython/blob/3.5/Lib/_pyio.py#L831
    #
    # Copyright 2001-2019 Python Software Foundation; All Rights Reserved
    #
    # 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
    #    the Individual or Organization ("Licensee") accessing and otherwise using Python
    #    3.5.9 software in source or binary form and its associated documentation.
    #
    # 2. Subject to the terms and conditions of this License Agreement, PSF hereby
    #    grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
    #    analyze, test, perform and/or display publicly, prepare derivative works,
    #    distribute, and otherwise use Python 3.5.9 alone or in any derivative
    #    version, provided, however, that PSF's License Agreement and PSF's notice of
    #    copyright, i.e., "Copyright 2001-2019 Python Software Foundation; All Rights
    #    Reserved" are retained in Python 3.5.9 alone or in any derivative version
    #    prepared by Licensee.
    #
    # 3. In the event Licensee prepares a derivative work that is based on or
    #    incorporates Python 3.5.9 or any part thereof, and wants to make the
    #    derivative work available to others as provided herein, then Licensee hereby
    #    agrees to include in any such work a brief summary of the changes made to Python
    #    3.5.9.
    #
    # 4. PSF is making Python 3.5.9 available to Licensee on an "AS IS" basis.
    #    PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  BY WAY OF
    #    EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
    #    WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE
    #    USE OF PYTHON 3.5.9 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
    #
    # 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.5.9
    #    FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
    #    MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.5.9, OR ANY DERIVATIVE
    #    THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.
    #
    # 6. This License Agreement will automatically terminate upon a material breach of
    #    its terms and conditions.
    #
    # 7. Nothing in this License Agreement shall be deemed to create any relationship
    #    of agency, partnership, or joint venture between PSF and Licensee.  This License
    #    Agreement does not grant permission to use PSF trademarks or trade name in a
    #    trademark sense to endorse or promote products or services of Licensee, or any
    #    third party.
    #
    # 8. By copying, installing or otherwise using Python 3.5.9, Licensee agrees
    #    to be bound by the terms and conditions of this License Agreement.

    def __init__(self, buff):
        self._buffer = memoryview(buff)
        self._pos = 0

    def read(self, size=None):
        if size is None:
            size = -1
        if size < 0:
            size = len(self._buffer)

        if len(self._buffer) <= self._pos:
            return b''
        newpos = min(len(self._buffer), self._pos + size)
        b = self._buffer[self._pos: newpos]
        self._pos = newpos
        return bytes(b)

    def readable(self):
        return True

    def seek(self, pos, whence=0):
        if whence == 0:
            if pos < 0:
                raise ValueError("negative seek position %r" % (pos,))
            self._pos = pos
        elif whence == 1:
            self._pos = max(0, self._pos + pos)
        elif whence == 2:
            self._pos = max(0, len(self._buffer) + pos)
        else:
            raise ValueError("unsupported whence value")
        return self._pos

    def seekable(self):
        return True


_loaders: Mapping[bytes, Callable] = dict()


def register(prefix: bytes, loader: Callable):
    """Register a new loader."""
    _loaders[prefix] = loader


def load(fp, cls=None):
    """Load a model from a file.

    Args:
        fp (bytes-like/file-like):
            If file-like, should be readable, seekable file-like object. If
            bytes-like it will be wrapped with `io.BytesIO`.

        cls (class, optional):
            Deprecated keyword argument. Is ignored.

    Returns:
        The loaded model.

    """
    if cls is not None:
        warnings.warn("'cls' keyword argument is deprecated and ignored",
                      DeprecationWarning, stacklevel=2)

    if isinstance(fp, ByteString):
        file_like: BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
    else:
        file_like = fp

    if not file_like.seekable:
        raise ValueError("expected file-like to be seekable")

    pos = file_like.tell()

    lengths = sorted(set(map(len, _loaders)))
    for num_bytes in lengths:
        prefix = file_like.read(num_bytes)
        file_like.seek(pos)

        try:
            return _loaders[prefix](file_like)
        except KeyError:
            pass

    raise ValueError("cannot load the given file-like")


# for slightly more explicit naming
load.register = register


def make_header(prefix: bytes, data: Mapping, version: Tuple[int, int]) -> bytearray:
    """Construct a header for serializing dimod models.

    The first `len(prefix)` bytes will be exactly the contents of `prefix`.

    The next 1 byte is an unsigned byte: the major version of the file
    format.

    The next 1 byte is an unsigned byte: the minor version of the file
    format.

    The next 4 bytes form a little-endian unsigned int, the length of
    the header data `HEADER_LEN`.

    The next `HEADER_LEN` bytes form the header data. This will be `data`
    json-serialized and encoded with 'ascii'.

    The header is padded with spaces to make the entire length divisible by 64.

    """
    header = bytearray()

    header += prefix
    header += bytes(version)

    version_start = len(header)

    header += bytes(4)  # will hold the data length

    data_start = len(header)

    header += json.dumps(data, sort_keys=True).encode('ascii')
    header += b'\n'

    # want the entire thing to be divisible by 64 for alignment
    if len(header) % 64:
        header += b' '*(64 - len(header) % 64)

    assert not len(header) % 64

    header[version_start:version_start+4] = struct.pack('<I', len(header) - data_start)

    return header


def write_header(file_like: BinaryIO, prefix: bytes, data: Mapping, version: Tuple[int, int]):
    """Write a header constructed by :func:`.make_header` to a file-like."""
    file_like.write(make_header(prefix, data, version))


HeaderInfo = namedtuple('HeaderInfo', ['data', 'version'])


def read_header(file_like: BinaryIO, prefix: bytes) -> HeaderInfo:
    """Read the information from a header constructed by :func:`.make_header`.

    The return value should be accessed by attribute for easy future expansion.
    """
    read_prefix = file_like.read(len(prefix))
    if read_prefix != prefix:
        raise ValueError("unknown file type, expected magic string "
                         f"{prefix!r} but got {read_prefix!r} "
                         "instead")

    version = tuple(file_like.read(2))

    header_len = struct.unpack('<I', file_like.read(4))[0]
    data = json.loads(file_like.read(header_len).decode('ascii'))

    return HeaderInfo(data, version)
