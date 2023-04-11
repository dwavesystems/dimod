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
import string
import tempfile
import typing

import dimod  # for typing

from dimod.cylp import cyread_lp_file
from dimod.sym import Sense
from dimod.vartypes import Vartype


__all__ = ['dump', 'dumps', 'loads', 'load']


LABEL_VALID_CHARS = set(string.ascii_letters + string.digits + "'!\"#$%&(),.;?@_‘’{}~")
LABEL_INVALID_FIRST_CHARS = set('eE.' + string.digits)


def _sign(bias: float) -> str:
    return '-' if bias < 0 else '+'


def _abs(bias: float) -> str:
    # todo: use bias.is_integer() instead of int(bias) == bias once we can require
    # numpy>=1.22.0.  this is needed to support Float32BQM on numpy<1.22.0.
    return repr(abs(int(bias))) if int(bias) == bias else repr(abs(bias))


def _sense(s: Sense) -> str:
    return '=' if s.value == "==" else s.value


def _label_error(label: typing.Hashable, rule: str) -> str:
    raise ValueError(f'Label {label!r} cannot be output to an LP file; {rule}. '
                     'Use CQM.relabel_variables() or CQM.relabel_constraints() to fix')


def _validate_label(label: typing.Hashable):
    if not isinstance(label, str):
        _label_error(label, 'labels must be strings')

    if not len(label):
        _label_error(label, 'labels must be nonempty')

    if len(label) > 255:
        _label_error(label, 'labels must be <= 255 characters in length')

    if not all(c in LABEL_VALID_CHARS for c in label):
        _label_error(label, f'labels must be composed of only these characters: '
                           f'{"".join(sorted(LABEL_VALID_CHARS))}')

    if any(label.startswith(c) for c in LABEL_INVALID_FIRST_CHARS):
        _label_error(label, f'labels must not start with any of these characters: '
                           f'{"".join(sorted(LABEL_INVALID_FIRST_CHARS))}')


class _WidthLimitedFile:
    TARGET_LINE_LEN = 80

    def __init__(self, fp: typing.TextIO):
        """Wrap a file-like object to provide a custom `write` method that
        inserts line breaks into the output stream.  This is done to generate attractive
        output (e.g. lines that are not extremely short) for most inputs, while satisfying
        the line length requirement of LP files for all inputs.

        This design has the advantage of a simple and readable implementation as compared
        to the approach of tracking line length and emitting newlines conditionally,
        through additional statements interspersed with the f.write() calls below.  And it
        has the advantage of producing more attractive output as compared to the dead
        simple approach of emitting newlines unconditionally in the f.write() calls below.

        This design does incur a run-time performance penalty of about 50% versus those
        alternatives.  In the balance of our goals, this seems to be the best option.
        """
        self.fp = fp

        self._line_len = 0

        self.tell = fp.tell

    def write(self, s: str):
        """Check if writing ``s`` would result in our line length target being
        exceeded, and if so, emit a newline.  Then write ``s``.

        Note that this function does not gaurantee a maximum line length.  Lines longer
        than the target length can be emitted.  Callers, i.e. the f.write() calls in the
        code below, must pass ``s`` arguments that are sufficiently short, to avoid
        exceeding the maximum line length for LP files.  The longest lines emitted by the
        current implementation are those encoding quadratic terms for an objective or
        constraints, when two variables have maximum length labels.  We have a unit test
        (test_long_line_breaking) to check this case.
        """
        try:
            pos = s.index('\n')
            rpos = s.rindex('\n')
        except ValueError:
            pos = len(s)
            rpos = None

        if self._line_len + pos > self.TARGET_LINE_LEN - 1:
            self.fp.write('\n ')
            self._line_len = 1

        self.fp.write(s)

        if rpos is not None:
            self._line_len = len(s) - 1 - rpos
        else:
            self._line_len += len(s)


def dump(cqm: dimod.ConstrainedQuadraticModel, file_like: typing.TextIO):
    """Serialize a constrained quadratic model as an LP file.

    LP files are a common format for encoding optimization models. See
    documentation from Gurobi_ and CPLEX_.

    Args:
        cqm: A constrained quadratic model.
        file_like: A ``.write()`` supporting file-like_ object.

    .. _file-like: https://docs.python.org/3/glossary.html#term-file-object

    .. _Gurobi: https://www.gurobi.com/documentation/9.5/refman/lp_format.html

    .. _CPLEX: https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-lp-file-format-algebraic-representation

    """
    # check that there are no soft constraints, LP format does not support them
    if len(cqm._soft) > 0:
        raise ValueError(f"LP file does not support soft constraints, {len(cqm._soft)} were given")

    # check that constraint labels are serializable
    for c in cqm.constraints:
        _validate_label(c)

    vartypes = {var: cqm.vartype(var) for var in cqm.variables}

    # check that variable labels and types are serializable
    for v, vartype in vartypes.items():
        _validate_label(v)

        if vartype == Vartype.SPIN:
            raise ValueError(
                'SPIN variables not supported in LP files, convert them to BINARY beforehand.')

    f = _WidthLimitedFile(file_like)

    # write the objective
    for var, bias in cqm.objective.iter_linear():
        if bias:
            if not f.tell():
                # LP files allow to omit this part if the objective is empty, so
                # we add these lines if we are sure that there are nonzero terms.
                f.write("Minimize\n")
                f.write(' obj: ')
            f.write(f"{_sign(bias)} {_abs(bias)} {var} ")

    if not cqm.objective.is_linear():
        if not f.tell():
            # if the objective is only quadratic then the header is written now
            f.write("Minimize\n")
            f.write(' obj: ')

        f.write('+ [ ')
        for u, v, bias in cqm.objective.iter_quadratic():
            # multiply bias by two because all quadratic terms are eventually
            # divided by two outside the squared parenthesis
            f.write(f"{_sign(bias)} {_abs(2 * bias)} {u} * {v} ")
        f.write(']/2 ')

    if cqm.objective.offset:
        offset = cqm.objective.offset

        if not f.tell():
            # if the objective has only an offset then the header is written now
            f.write("Minimize\n")
            f.write(' obj: ')
        f.write(f"{_sign(offset)} {_abs(offset)} ")

    # write the constraints
    f.write("\n\n")
    f.write("Subject To \n")

    for label, constraint in cqm.constraints.items():
        f.write(f' {label}: ')

        for var, bias in constraint.lhs.iter_linear():
            if bias:
                f.write(f"{_sign(bias)} {_abs(bias)} {var} ")

        if constraint.lhs.quadratic:
            f.write('+ [ ')
            for u, v, bias in constraint.lhs.iter_quadratic():
                f.write(f"{_sign(bias)} {_abs(bias)} {u} * {v} ")
            f.write('] ')

        rhs = constraint.rhs - constraint.lhs.offset
        f.write(f" {_sense(constraint.sense)} {rhs}\n")

    # write variable bounds
    f.write('\n')
    f.write('Bounds\n')

    for v, vartype in vartypes.items():
        if vartype in (Vartype.INTEGER, Vartype.REAL):
            f.write(f' {cqm.lower_bound(v)} <= {v} <= {cqm.upper_bound(v)}\n')
        elif vartype is not Vartype.BINARY:
            raise RuntimeError(f"unexpected vartype {vartype}")

    # write variable names
    for section, vartype_ in (('Binary', Vartype.BINARY), ('General', Vartype.INTEGER)):
        f.write('\n')
        f.write(f'{section}\n')

        for v, vartype in vartypes.items():
            if vartype is vartype_:
                f.write(f' {v}')

    # conclude
    f.write('\n')
    f.write('End')
    return f


def dumps(cqm: dimod.ConstrainedQuadraticModel) -> str:
    """Serialize a constrained quadratic model as an LP file.

    LP files are a common format for encoding optimization models. See
    documentation from Gurobi_ and CPLEX_.

    Args:
        cqm: A constrained quadratic model.

    Returns:
        A string encoding the constrained quadratic model as an LP file.

    .. _Gurobi: https://www.gurobi.com/documentation/9.5/refman/lp_format.html

    .. _CPLEX: https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-lp-file-format-algebraic-representation

    """
    with io.StringIO() as f:
        dump(cqm, f)
        f.seek(0)
        return f.read()


def load(file_like: typing.Union[str, bytes, io.IOBase]) -> dimod.ConstrainedQuadraticModel:
    """Construct a constrained quadratic model from a LP file.

    LP files are a common format for encoding optimization models. See
    documentation from Gurobi_ and CPLEX_.

    Note that if the objective function is specified as a maximization function
    then it will be converted to a minimization function by flipping the sign
    of all of the biases.

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

    if (filename is not None) and os.path.isfile(filename):
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

    Note that if the objective function is specified as a maximization function
    then it will be converted to a minimization function by flipping the sign
    of all of the biases.

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
