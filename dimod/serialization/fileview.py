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

from dimod.bqm.utils import ilinear_biases, ineighborhood
from dimod.variables import iter_deserialize_variables


# we try to pick values much higher than io's
SEEK_OFFSET = 100
SEEK_LINEAR = 101
SEEK_QUADRATIC = 102


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
        bqm (:class:`~dimod.core.bqm.BQM`):
            The binary quadratic model.

    Note:
        Currently the BQM is not locked while the file view is open, in the
        future this will change.

    See also
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html

    """

    MAGIC_PREFIX = b'DIMODBQM'
    VERSION = bytes([1, 0])  # version 1.0

    SEEK_OFFSET = SEEK_OFFSET
    SEEK_LINEAR = SEEK_LINEAR
    SEEK_QUADRATIC = SEEK_QUADRATIC

    def __init__(self, bqm):
        super(FileView, self).__init__()

        self.bqm = bqm  # todo: increment viewcount
        self.pos = 0

        # the lengths of the various components
        num_var, num_int = bqm.shape

        self.offset_length = bqm.dtype.itemsize  # one bias
        self.linear_length = num_var*(bqm.ntype.itemsize + bqm.dtype.itemsize)
        self.quadratic_length = 2*num_int*(bqm.itype.itemsize + bqm.dtype.itemsize)

    @property
    def neighborhood_starts(self):
        """The indices of the neighborhood starts."""
        # lazy construction
        try:
            return self._neighborhood_starts
        except AttributeError:
            pass

        bqm = self.bqm

        starts = np.zeros(bqm.num_variables, dtype=bqm.ntype)

        if bqm.num_variables:
            starts[1:] = np.add.accumulate(bqm.degrees(array=True),
                                           dtype=bqm.ntype)[:-1]

        self._neighborhood_starts = starts

        return self.neighborhood_starts

    @property
    def header(self):
        """The header associated with the BQM."""
        # lazy construction

        try:
            return self._header
        except AttributeError:
            pass

        bqm = self.bqm
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

        self._header = header = prefix + version + header_len + header_data

        assert len(header) % 16 == 0  # sanity check

        return self.header

    @property
    def header_end(self):
        """The location (in bytes) that the header ends."""
        return len(self.header)

    @property
    def offset_start(self):
        """The location (in bytes) that the offset starts."""
        return self.header_end

    @property
    def offset_end(self):
        return self.offset_start + self.offset_length

    @property
    def linear_start(self):
        """The location (in bytes) that the linear data starts."""
        return self.offset_end

    @property
    def linear_end(self):
        """The location (in bytes) that the linear data end."""
        return self.linear_start + self.linear_length

    @property
    def quadratic_start(self):
        """The location (in bytes) that the quadratic data starts."""
        return self.linear_end

    @property
    def quadratic_end(self):
        """The location (in bytes) that the quadratic data end."""
        return self.quadratic_start + self.quadratic_length

    def close(self):
        """Close the file view. The BQM will no longer be viewable."""
        # todo: decrement viewcount
        super(FileView, self).close()
        del self.bqm

    def readinto(self, buff):
        """Read bytes into a pre-allocated, writable bytes-like object.

        Args:
            buff (bytes-like):
                A pre-allocated writeable bytes-like object.

        Returns:
            int: The number of bytes read. If 0 bytes are read this indicated
            the end of the file view.

        """
        buff = memoryview(buff)  # we're going to be slicing

        num_read = 0
        while num_read < len(buff):
            n = self.readinto1(buff[num_read:])
            if n == 0:
                break
            num_read += n

        return num_read

    # developer note: we use RawIOBase with an "extra" implemented readinto1
    # because the BufferedIOBase's stub methods are `read1` and `read`, whereas
    # for performance reasons, we want to implement `readinto1` and `readinto`.
    def readinto1(self, buff):
        """Read bytes into a pre-allocated, writable bytes-like object.

        `readinto1` differs from :meth:`.readinto` by only reading a single
        c++ object at a time.

        Args:
            buff (bytes-like):
                A pre-allocated writeable bytes-like object.

        Returns:
            int: The number of bytes read. If 0 bytes are read this indicated
            the end of the file view.

        """
        pos = self.pos
        bqm = self.bqm

        if pos < 0:
            raise RuntimeError("invalid position")

        elif pos < self.header_end:
            # header
            data = memoryview(self.header)[pos:]

        elif pos < self.offset_end:
            # offset
            data = memoryview(bqm.offset.tobytes())[pos - self.offset_start:]

        elif pos < self.linear_end:
            # linear biases
            ldata = ilinear_biases(bqm)
            data = memoryview(ldata).cast('B')[pos - self.linear_start:]

        elif pos < self.quadratic_end:
            # quadratic biases

            quadratic_itemsize = bqm.itype.itemsize + bqm.dtype.itemsize

            # position relative to the start of the quadratic biases
            qpos = pos - self.quadratic_start

            # which pair (in the concatenated neighborhoods) we're on
            pair_idx = qpos // quadratic_itemsize

            # use the pair to figure out which variable we're on. In practice
            # pair_idx is often 0 which makes this fast.
            vi = np.searchsorted(self.neighborhood_starts, pair_idx, side='right') - 1

            # use the variable and the pair_idx to determine which pair within
            # the neighborhood we're on
            ni = pair_idx - int(self.neighborhood_starts[vi])

            start = ni * quadratic_itemsize + qpos % quadratic_itemsize

            qdata = ineighborhood(bqm, vi)

            data = memoryview(qdata).cast('B')[start:]

        else:
            data = bytes()

        num_bytes = min(len(buff), len(data))
        buff[:num_bytes] = data[:num_bytes]

        self.pos += num_bytes

        return num_bytes

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        """Change the stream position to the given `offset`.

        Args:
            offset (int):
                The offset relative to `whence`.

            whence (int):
                In addition to values for whence provided in the :mod:`io`
                module, additional values for whence are:

                    * SEEK_OFFSET or 100 - the start of the offset data
                    * SEEK_LINEAR or 101 - the start of the linear data
                    * SEEK_QUADRATIC or 102 - the start of the quadratic data

        Returns:
            The new stream position.

        """
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.quadratic_end + offset
        elif whence == SEEK_OFFSET:
            self.pos = self.offset_start + offset
        elif whence == SEEK_LINEAR:
            self.pos = self.linear_start + offset
        elif whence == SEEK_QUADRATIC:
            self.pos = self.quadratic_start + offset
        else:
            raise ValueError("unknown value for 'whence'")
        return self.pos

    def seekable(self):
        return True


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

    def readable():
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

    def seekable():
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

    if isinstance(fp, (bytes, bytearray, memoryview)):
        fp = _BytesIO(fp)

    magic = fp.read(len(FileView.MAGIC_PREFIX))

    if magic != FileView.MAGIC_PREFIX:
        # todo: expand on error message (print actual magic prefix)
        raise ValueError("unknown file type")

    version = fp.read(len(FileView.VERSION))
    if version != FileView.VERSION:
        raise ValueError("Given serialization does not have a matching version")

    # next get the header
    header_len = np.frombuffer(fp.read(4), '<u4')[0]
    header_data = fp.read(header_len)

    data = json.loads(header_data.decode('ascii'))

    # convert list-variable names to tuples (including nested)
    data['variables'] = list(iter_deserialize_variables(data['variables']))

    from dimod.bqm import AdjArrayBQM, AdjMapBQM, AdjVectorBQM

    if cls is None:
        cls = locals().get(data['type'])

    offset = len(FileView.MAGIC_PREFIX) + len(FileView.VERSION) + 4 + header_len

    return cls._load(fp, data, offset=offset)
