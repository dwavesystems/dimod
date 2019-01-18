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

from itertools import islice

import numpy as np

import dimod

_format_options = {
    'linewidth': 79,
    'precision': 6,
    'max_samples': 50,
    'sorted_by': 'energy',
    'column_fill': '  ',  # between columns, private
}
# NB: Some of these are 'private' in that they are not documented in
# set_printoptions


def set_printoptions(**kwargs):
    """Set print options globally.

    Args:
        linewidth (int, optional):
            The maximum number of characters to a single line. Default is 75.

        precision (int, optional):
            Number of digits of precision for floating point output. Default is
            6.

        max_samples (int, optional):
            The maximum number of samples printed, summation is used if
            exceeded. Default is 50.

        sorted_by (str/None, optional):
            Selects the field used to sort the samples when printing samplesets.
            If None, samples are printed in record order. Default is 'energy'.

    Note:
        All arguments must be provided as keyword arguments.

    """
    _format_options.update(kwargs)


def _spinstr(v):
    return '-1' if v <= 0 else '+1'


def _binarystr(v):
    return '0' if v <= 0 else '1'


def _numstr(val, precision=6):
    if isinstance(val, numbers.Integral):
        return str(int(val))
    return np.format_float_positional(val, unique=True, precision=precision,
                                      trim='0')


def _total_width(widths, fill_width):
    return widths.sum() + fill_width*(len(widths) - 1)


def _column_formatting(charr, energy_idx, options):
    """Make columns all the same width and limit the total width"""

    column_fill = options['column_fill']
    linewidth = options['linewidth']

    lenfunc = np.frompyfunc(len, 1, 1)

    column_widths = lenfunc(charr).max(axis=0)

    if _total_width(column_widths, len(column_fill)) > linewidth:
        idxs = np.ones(len(column_widths), dtype=bool)
        idxs[energy_idx - 2] = False

        new_charr = charr[:, idxs]
        new_energy_idx = energy_idx - 1

        new_charr[:, new_energy_idx - 2] = '..'

        return _column_formatting(new_charr, energy_idx-1, options)

    for ci in range(1, charr.shape[1]):  # skip index
        width = lenfunc(charr[:, ci]).max()

        def fmt(s):
            return s.rjust(width)

        # apply to column
        charr[:, ci] = np.frompyfunc(fmt, 1, 1)(charr[:, ci])

    width = lenfunc(charr[:, 0]).max()

    def fmt(s):
        return s.ljust(width)

    # apply to column
    charr[:, 0] = np.frompyfunc(fmt, 1, 1)(charr[:, 0])

    return '\n'.join(column_fill.join(row) for row in charr)


def sampleset_to_string(sampleset, **kwargs):
    """Get the string representation of a sampleset.

    Args:
        sampleset (:obj:`.SampleSet`):
            A sample set.

        **kwargs:
            Any keyword arguments will override the printing defaults. See
            :func:`.set_printoptions` for argument descriptions.

    Returns:
        str

    """

    # todo: _format overrides
    options = _format_options.copy()
    options.update(kwargs)

    max_samples = options['max_samples']
    sorted_by = options['sorted_by']
    precision = options['precision']

    fields = [field for field in sampleset.record.dtype.names
              if field != 'sample']
    variables = sampleset.variables

    #
    # let's assume it fits our width and just worry about number of rows for now
    #

    # variables + index + (fields - 'sample') so the index/'sample' cancel
    ncols = len(sampleset.variables) + len(sampleset.record.dtype.names)

    nrows = min(max_samples, len(sampleset)) + 1  # +1 for the header

    charr = np.empty((nrows, ncols), dtype=object)

    # set up the header

    var_headers = [str(v) for v in variables]
    field_headers = [field if len(field) <= 7 else field[:7]+'.'  # limit to 7 characters
                     for field in fields]
    charr[0, :] = [''] + var_headers + field_headers

    #
    # now populate the samples/dtypes
    #

    if sampleset.vartype is dimod.SPIN:
        splfmt = _spinstr
    else:
        splfmt = _binarystr

    def _row_from_datum(index, datum):
        row = [str(index)]
        row.extend(splfmt(datum.sample[v]) for v in variables)
        row.extend(_numstr(getattr(datum, field)) for field in fields)
        return row

    if len(sampleset) > max_samples:
        for ci, datum in enumerate(sampleset.data(sorted_by=sorted_by), start=1):
            if ci > max_samples - 2:  # 3 = header + skiprow + lastrow
                break

            charr[ci, :] = _row_from_datum(ci-1, datum)

        # skip row
        charr[-2, :] = '..'

        # last row
        datum = next(iter(sampleset.data(reverse=True)))  # get the last sample
        charr[-1, :] = _row_from_datum(len(sampleset) - 1, datum)

    else:
        for ci, datum in enumerate(sampleset.data(sorted_by=sorted_by), start=1):
            charr[ci, :] = _row_from_datum(ci-1, datum)

    #
    # format and limit the number of columns
    #

    charr_str = _column_formatting(charr, len(variables)+1, options)

    footer = '\n\n[ {} rows, {} variables ]'.format(len(sampleset), len(sampleset.variables))

    return charr_str + footer
