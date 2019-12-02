# distutils: language = c++
# cython: language_level=3
#
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
# =============================================================================
import io
import json

import numpy as np

try:
    from dimod.bqm import AdjArrayBQM, AdjMapBQM, AdjVectorBQM
except ImportError:
    pass  # not available in python < 3.5
from dimod.bqm import AdjDictBQM  # should always be available


class FileView(io.RawIOBase):
    """A seekable, readable view into a binary quadratic model.

    Format specification:

    The first 8 bytes are a magic string: exactly "DIMODBQM".

    The next 1 byte is an unsigned byte: the major version of the file format.

    The next 1 byte is an unsigned byte: the minor version of the file format.

    The next 4 bytes form a little-endian unsigned int, the length of the header
    data HEADER_LEN.

    The next HEADER_LEN bytes form the header data. This is a json-serialized
    dictionary. The dictionary is exactly:

    .. code-block:: python

        dict(shape=bqm.shape,
             dtype=bqm.dtype.name,
             itype=bqm.itype.name,
             ntype=bqm.ntype.name,
             vartype=bqm.vartype.name,
             type=type(bqm).__name__,
             variables=list(bqm.variables),
             )

    it is terminated by a newline `\n` and padded with spaces to make the entire
    length of the entire header divisible by 16.

    Args:
        bqm:
            Only currently support AdjArrayBQM. Also the BQM is NOT locked,
            though in the future it should be and this view should be used
            in a context-manager.

    See also
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html

    """

    MAGIC_PREFIX = b'DIMODBQM'
    VERSION = bytes([1, 0])  # version 1.0

    def __init__(self, bqm):
        super(FileView, self).__init__()

        self.bqm = bqm  # todo: increment viewcount
        self.pos = 0

        # construct the header

        prefix = self.MAGIC_PREFIX
        version = bytes([1, 0])  # version 1.0

        data = dict(shape=bqm.shape,
                    dtype=bqm.dtype.name,
                    itype=bqm.itype.name,
                    ntype=bqm.ntype.name,
                    vartype=bqm.vartype.name,
                    type=type(bqm).__name__,
                    variables=list(bqm.variables),  # this is ordered
                    )

        header_data = json.dumps(data, sort_keys=True).encode('ascii')
        header_data += b'\n'

        # whole header length should be divisible by 16
        header_data += b' '*(16 - (len(prefix) +
                                   len(version) +
                                   4 +
                                   len(header_data)) % 16)

        header_len = np.dtype('<u4').type(len(header_data)).tobytes()

        self.header = header = prefix + version + header_len + header_data

        assert len(header) % 16 == 0  # sanity check

        # the lengths of the various components
        num_var, num_int = bqm.shape

        self.header_length = len(self.header)  # the ENTIRE header
        self.offset_length = bqm.dtype.itemsize  # one bias
        self.linear_length = num_var*(bqm.ntype.itemsize + bqm.dtype.itemsize)
        self.quadratic_length = 2*num_int*(bqm.itype.itemsize + bqm.dtype.itemsize)

        # this is useful information to have cached later
        self.accumulated_degees = np.add.accumulate(bqm.degrees(array=True),
                                                    dtype=np.intc)

    def close(self):
        # todo: decrement viewcount
        super(FileView, self).close()

    def _readinto_header(self, buff, pos):
        header = self.header
        num_bytes = min(len(buff), len(header) - pos)
        if num_bytes < 0 or pos < 0:
            return 0
        buff[:num_bytes] = header[pos:pos+num_bytes]
        return num_bytes

    def readinto(self, buff):
        pos = self.pos
        start = pos

        header_length = self.header_length
        offset_length = self.offset_length
        linear_length = self.linear_length
        quadratic_length = self.quadratic_length

        buff = memoryview(buff)  # we're going to be slicing

        bqm = self.bqm
        num_variables, num_interactions = bqm.shape

        written = 0  # tracks the sections we've been through

        if pos >= written and pos < header_length:
            pos += self._readinto_header(buff, pos)
        written += header_length

        if pos >= written and pos < written + offset_length and pos - start < len(buff):
            pos += bqm.readinto_offset(buff[pos-start:], pos - written)
        written += offset_length

        if pos >= written and pos < written + linear_length and pos - start < len(buff):
            pos += bqm.readinto_linear(buff[pos-start:], pos - written,
                                       self.accumulated_degees)
        written += linear_length

        if pos >= written and pos < written + quadratic_length and pos - start < len(buff):
            pos += bqm.readinto_quadratic(buff[pos-start:], pos-written,
                                          self.accumulated_degees)

        self.pos = pos
        return pos - start

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = pos = offset
        elif whence == io.SEEK_CUR:
            self.pos = pos = self.pos + offset
        elif whence == io.SEEK_END:
            self.pos = pos = (self.header_length +
                              self.offset_length +
                              self.linear_length +
                              self.quadratic_length +
                              offset)
        else:
            raise ValueError("unknown value for 'whence'")
        return pos

    def seekable(self):
        return True


def load(fp, cls=None):
    """

    Args:
        fp (bytes-like/file-like):
            If file-like, should be readable, seekable file-like object. If
            bytes-like it will be wrapped with `io.BytesIO`.

        cls (class, optional):
            The class of binary quadratic model. If not provided, the bqm will
            be of the same class that was saved. Note: currently only works
            for AdjArrayBQM.

    Returns:
        The loaded bqm.

    """

    if isinstance(fp, (bytes, bytearray)):
        fp = io.BytesIO(fp)

    magic = fp.read(len(FileView.MAGIC_PREFIX))

    if magic != FileView.MAGIC_PREFIX:
        raise NotImplementedError

    version = fp.read(len(FileView.VERSION))
    if version != FileView.VERSION:
        raise NotImplementedError

    # next get the header
    header_len = np.frombuffer(fp.read(4), '<u4')[0]
    header_data = fp.read(header_len)

    data = json.loads(header_data.decode('ascii'))

    if cls is None:
        cls = globals().get(data['type'])

    offset = len(FileView.MAGIC_PREFIX) + len(FileView.VERSION) + 4 + header_len

    return cls._load(fp, data, offset=offset)
