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
# ================================================================================================
from itertools import islice

import numpy as np

import dimod


class PassVar(object):
    def __str__(self):
        return '..'


class SampleSetFormatter(object):

    min_column_width = 2
    column_space = 2

    max_rows = 40
    max_columns = 12  # 17
    header = True
    footer = True
    index = True

    def __init__(self, sampleset):

        self.sampleset = sampleset

        # sampleset's size/shape/fields won't change, so we can build the (empty)
        # array

        nrows = min(self.max_rows,
                    (len(sampleset) +  # one row for header
                        self.header +  # two for footer
                        2*self.footer))
        nrows -= 2*self.footer  # two rows for footer

        ncols = min(self.max_columns,
                    (len(sampleset.variables) +
                        len(sampleset.record.dtype.names) - 1 +
                        self.index))

        self.char = char = np.full((nrows, ncols), '', dtype=object)

        # get columns
        self.fields = fields = [field for field in sampleset.record.dtype.names
                                if field != 'sample']

        max_variables = self.max_columns - len(fields) - self.index
        if len(sampleset.variables) > max_variables:
            variables = sampleset.variables[:max_variables-2]
            variables.append(PassVar())
            variables.append(sampleset.variables[-1])
        else:
            # we can do all the variables
            variables = sampleset.variables
        variable_labels = list(map(str, variables))
        self.variables = variables

        if self.header:
            char[0, :] = ['']*self.index + variable_labels + fields

    def to_string(self):
        sampleset = self.sampleset
        num_samples = len(sampleset)
        char = self.char
        nrows, ncols = char.shape

        variables = self.variables
        fields = self.fields

        if sampleset.vartype is dimod.SPIN:
            def _str(datum, v):
                if isinstance(v, PassVar):
                    return str(v)
                return '+1' if datum.sample[v] > 0 else '-1'
        else:
            def _str(datum, v):
                if isinstance(v, PassVar):
                    return str(v)
                return '1' if datum.sample[v] > 0 else '0'

        ci = 1*self.header
        for i, datum in enumerate(sampleset.data()):
            row = [str(i)]*self.header
            row.extend(_str(datum, v) for v in variables)
            row.extend(str(getattr(datum, field)) for field in fields)

            char[ci, :] = row
            ci += 1

            if ci >= nrows:
                break

        if num_samples > nrows - self.header:
            char[-2, :] = '..'

            # last line
            datum = next(iter(sampleset.data(reverse=True)))  # get the last sample
            row = [str(num_samples-1)]*self.header
            row.extend(_str(datum, v) for v in variables)
            row.extend(str(getattr(datum, field)) for field in fields)
            char[-1, :] = row

        # formatting

        lenfunc = np.frompyfunc(len, 1, 1)

        for ci in range(self.index, ncols):
            width = max(lenfunc(char[:, ci]).max(), self.min_column_width) + self.column_space

            def fmt(s):
                return s.rjust(width)

            # apply to column
            char[:, ci] = np.frompyfunc(fmt, 1, 1)(char[:, ci])

        if self.index:
            width = max(lenfunc(char[:, 0]).max(), self.min_column_width)

            def fmt(s):
                return s.ljust(width)

            # apply to column
            char[:, 0] = np.frompyfunc(fmt, 1, 1)(char[:, 0])

        footer = '\n\n[ {} rows, {} variables ]'.format(len(sampleset), len(sampleset.variables))

        return '\n'.join(''.join(row) for row in char) + self.footer*footer
