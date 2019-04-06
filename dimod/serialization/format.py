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

# developer note: It is hard to provide code examples for these because doing
# so will overwrite the settings in sphinx's setup.

import numbers
import sys

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from collections import deque

import numpy as np

from six import StringIO
from six.moves import map

import dimod

__all__ = 'set_printoptions', 'Formatter'

_format_options = {
    'width': 79,
    'depth': None,
    'sorted_by': 'energy',
}


def set_printoptions(**kwargs):
    """Set print options globally.

    Args:
        width (int, optional, default=79):
            The maximum number of characters to a single line.

        depth (int, optional, default=None):
            The maximum number of rows printed, summation is used if
            exceeded. Default is unlimited.

        sorted_by (str/None, optional, default='energy'):
            Selects the field used to sort the samples when printing samplesets.
            If None, samples are printed in record order.

    Note:
        All arguments must be provided as keyword arguments.

    """
    _format_options.update(kwargs)


def _spinstr(v, rjust=0):
    s = '-1' if v <= 0 else '+1'
    return s.rjust(rjust)


def _binarystr(v, rjust=0):
    s = '0' if v <= 0 else '1'
    return s.rjust(rjust)


class _SampleTable(object):
    """Creates the table for printing samples. Acts like a deque in that
    it can be rotated and appended on either side.
    """
    def __init__(self, space_length=1):
        self.deque = deque()
        self.items_length = 0
        self.space_length = space_length  # space between columns

    @property
    def ncol(self):
        return len(self.deque)

    @property
    def width(self):
        """the width of the table if made into a string"""
        return self.items_length + (self.ncol - 1)*self.space_length

    def append(self, header, f, _left=False):
        """Add a column to the table.

        Args:
            header (str):
                Column header

            f (function(datum)->str):
                Makes the row string from the datum. Str returned by f should
                have the same width as header.

        """
        self.items_length += len(header)
        if _left:
            self.deque.appendleft((header, f))
        else:
            self.deque.append((header, f))

    def appendleft(self, header, f):
        self.append(header, f, _left=True)

    def append_index(self, num_rows):
        """Add an index column.

        Left justified, width is determined by the space needed to print the
        largest index.
        """
        width = len(str(num_rows - 1))

        def f(datum):
            return str(datum.idx).ljust(width)
        header = ' '*width

        self.append(header, f)

    def append_sample(self, v, vartype, _left=False):
        """Add a sample column"""
        vstr = str(v).rjust(2)  # the variable will be len 0, or 1
        length = len(vstr)

        if vartype is dimod.SPIN:
            def f(datum):
                return _spinstr(datum.sample[v], rjust=length)
        else:
            def f(datum):
                return _binarystr(datum.sample[v], rjust=length)

        self.append(vstr, f, _left=_left)

    def appendleft_sample(self, v, vartype):
        self.append_sample(v, vartype, _left=True)

    def append_vector(self, name, vector, _left=False):
        """Add a data vectors column."""
        if np.issubdtype(vector.dtype, np.integer):
            # determine the length we need
            largest = str(max(vector.max(), vector.min(), key=abs))
            length = max(len(largest), min(7, len(name)))  # how many spaces we need to represent

            if len(name) > length:
                header = name[:length-1] + '.'
            else:
                header = name.rjust(length)

            def f(datum):
                return str(getattr(datum, name)).rjust(length)
        elif np.issubdtype(vector.dtype, np.floating):
            largest = np.format_float_positional(max(vector.max(), vector.min(), key=abs),
                                                 precision=6, trim='0')
            length = max(len(largest), min(7, len(name)))  # how many spaces we need to represent
            if len(name) > length:
                header = name[:length-1] + '.'
            else:
                header = name.rjust(length)

            def f(datum):
                return np.format_float_positional(getattr(datum, name),
                                                  precision=6, trim='0',
                                                  ).rjust(length)
        else:
            length = 7
            if len(name) > length:
                header = name[:length-1] + '.'
            else:
                header = name.rjust(length)

            def f(datum):
                r = repr(getattr(datum, name))
                if len(r) > length:
                    r = r[:length-3] + '...'
                return r.rjust(length)

        self.append(header, f, _left=_left)

    def appendleft_vector(self, name, vector):
        self.append_vector(name, vector, _left=True)

    def dump_to_list(self):
        """deconstructs self into a list"""
        return [self.deque.popleft() for _ in range(self.ncol)]

    def pop(self):
        header, _ = self.deque.pop()
        self.items_length -= len(header)

    def popleft(self):
        header, _ = self.deque.popleft()
        self.items_length -= len(header)

    def rotate(self, r):
        self.deque.rotate(r)


class Formatter(object):
    """Used to create nice string formats for dimod objects.

    Args:
        width (int, optional, default=79):
            The maximum number of characters to a single line.

        depth (int, optional, default=None):
            The maximum number of rows printed, summation is used if
            exceeded. Default is unlimited.

        sorted_by (str/None, optional, default='energy'):
            Selects the field used to sort the samples when printing samplesets.
            If None, samples are printed in record order.

    Examples:
        >>> from dimod.serialization.format import Formatter
        >>> sampleset = dimod.SampleSet.from_samples(([-1, 1], ['a', 'b']), dimod.SPIN, energy=1)
        >>> Formatter(width=45).print(sampleset)
           a  b energy num_oc.
        0 -1 +1      1       1
        ['SPIN', 1 rows, 1 samples, 2 variables]
        >>> Formatter(width=30).print(sampleset)
           a  b energy num_oc.
        0 -1 +1      1       1
        ['SPIN',
         1 rows,
         1 samples,
         2 variables]

    """
    def __init__(self, **kwargs):
        self.options = options = _format_options.copy()
        options.update(kwargs)

    def format(self, obj, **kwargs):
        """Return the formatted representation of the object as a string."""
        sio = StringIO()
        self.fprint(obj, stream=sio, **kwargs)
        return sio.getvalue()

    def fprint(self, obj, stream=None, **kwargs):
        """Prints the formatted representation of the object on stream"""
        if stream is None:
            stream = sys.stdout

        options = self.options
        options.update(kwargs)

        if isinstance(obj, dimod.SampleSet):
            self._print_sampleset(obj, stream, **options)
            return

        raise TypeError("cannot format type {}".format(type(obj)))

    def _print_sampleset(self, sampleset, stream,
                         width, depth, sorted_by,
                         **other):

        if len(sampleset) > 0:
            self._print_samples(sampleset, stream, width, depth, sorted_by)
        else:
            stream.write('Empty SampleSet\n')

            # write the data vectors
            stream.write('Record Fields: [')
            self._print_items(sampleset.record.dtype.names, stream, width - len('Data Vectors: [') - 1)
            stream.write(']\n')

            # write the variables
            stream.write('Variables: [')
            self._print_items(sampleset.variables, stream, width - len('Variables: [') - 1)
            stream.write(']\n')

        # add the footer
        stream.write('[')
        footer = [repr(sampleset.vartype.name),
                  '{} rows'.format(len(sampleset)),
                  '{} samples'.format(sampleset.record.num_occurrences.sum()),
                  '{} variables'.format(len(sampleset.variables))
                  ]
        if sum(map(len, footer)) + (len(footer) - 1)*2 > width - 2:
            # if the footer won't fit in width
            stream.write(',\n '.join(footer))
        else:
            # if width the minimum footer object then we don't respect it
            stream.write(', '.join(footer))
        stream.write(']')

    def _print_samples(self, sampleset, stream, width, depth, sorted_by):
        if len(sampleset) == 0:
            raise ValueError("Cannot print empty samplesets")

        # we need to know what goes into each row. We know we will use
        # datum as returned by sampleset.data() to populate the values,
        # so let's store our row formatters in the following form:
        #   row[(header, f(datum): str)]

        table = _SampleTable()

        # there are a minimum set of headers:
        #     idx energy num_oc.
        table.append_index(len(sampleset))
        table.append_vector('energy', sampleset.record.energy)
        table.append_vector('num_occurrences', sampleset.record.num_occurrences)

        # if there are more vectors, let's just put a placeholder in for now
        # we might replace it later if we still have space
        if len(sampleset.record.dtype.names) > len(sampleset._REQUIRED_FIELDS):
            table.append('...', lambda _: '...')

        # next we want to add variables until we run out of width
        table.rotate(-1)  # move the index to the end
        num_added = 0
        for v in sampleset.variables:
            table.append_sample(v, sampleset.vartype)
            num_added += 1

            if table.width > width:
                # we've run out of space, need to make room for the last
                # variable and a spacer
                last = sampleset.variables[-1]

                table.appendleft_sample(last, sampleset.vartype)
                table.appendleft('...', lambda _: '...')

                while table.width > width:
                    # remove variables until we have space for the last one
                    table.pop()
                    num_added -= 1

                break
        table.rotate(num_added + 1)  # move the index back to the front

        # finally any remaining space should be used for other fields. We assume
        # at this point that deque looks like [idx variables energy num_occ. ...]
        other_fields = set(sampleset.record.dtype.names).difference(sampleset._REQUIRED_FIELDS)
        if other_fields:
            num_added = 0
            while len(other_fields):
                name = min(other_fields, key=len)

                table.appendleft_vector(name, sampleset.record[name])
                other_fields.remove(name)
                num_added += 1

                if table.width > width:
                    table.popleft()
                    num_added -= 1
                    break
            else:
                # we have no other fields to add
                assert len(other_fields) == 0
                table.pop()  # remove the summary
            table.rotate(-num_added)  # put index back at the front

        # turn rows into a list because we're done rotating etc
        rows = table.dump_to_list()

        # ok, now let's print.
        stream.write(' '.join(header for header, _ in rows))
        stream.write('\n')

        if depth is None:
            depth = float('inf')

        for idx, datum in enumerate(sampleset.data(index=True)):
            stream.write(' '.join(f(datum) for _, f in rows))
            stream.write('\n')

            if idx + 3 >= depth and len(sampleset) > depth:
                stream.write('...\n')
                datum = next(sampleset.data(reverse=True, index=True))  # get the last one
                stream.write(' '.join(f(datum) for _, f in rows))
                stream.write('\n')
                break

    def _print_items(self, iterable, stream, width):

        iterator = map(repr, iterable)

        try:
            first = next(iterator)
        except StopIteration:
            # nothing to represent
            return
        else:
            # we could check width here but honestly that seems like
            # an edge case not worth considering
            stream.write(first)
            width -= len(first)

        for item in iterator:
            # we're not the first object
            stream.write(', ')
            width -= 2

            if len(item) > width - 3:
                stream.write('...')
                break

            stream.write(item)
            width -= len(item)
