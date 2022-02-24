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

from __future__ import division

import itertools
import collections.abc as abc

from numbers import Number

import numpy as np

from dimod.decorators import vartype_argument
from dimod.sampleset import as_samples
from dimod.utilities import iter_safe_relabels
from dimod.vartypes import Vartype

__all__ = ['BinaryPolynomial']


def asfrozenset(term):
    """Convert to frozenset if it is not already"""
    return term if isinstance(term, frozenset) else frozenset(term)


class BinaryPolynomial(abc.MutableMapping):
    """A polynomial with binary variables and real-valued coefficients.

    Args:
        poly (mapping/iterable):
            Polynomial as a mapping of form {term: bias, ...}, where `term` is
            a collection of variables and `bias` the associated bias. It can also
            be an iterable of 2-tuples (term, bias).

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Attributes:
        degree (int):
            Degree of the polynomial.

        variables (set):
            Variables of the polynomial.

        vartype (:class:`.Vartype`):
            One of :class:`~dimod.Vartype.SPIN` or :class:`~dimod.Vartype.BINARY`.

    Examples:

        Binary polynomials can be constructed in many different ways. The
        following are all equivalent

        >>> poly = dimod.BinaryPolynomial({'a': -1, 'ab': 1}, dimod.SPIN)
        >>> poly = dimod.BinaryPolynomial({('a',): -1, ('a', 'b'): 1}, dimod.SPIN)
        >>> poly = dimod.BinaryPolynomial([('a', -1), (('a', 'b'), 1)], dimod.SPIN)
        >>> poly = dimod.BinaryPolynomial({'a': -1, 'ab': .5, 'ba': .5}, dimod.SPIN)

        Binary polynomials act a mutable mappings but the terms can be accessed with
        any sequence.

        >>> poly = dimod.BinaryPolynomial({'a': -1, 'ab': 1}, dimod.BINARY)
        >>> poly['ab']
        1
        >>> poly['ba']
        1
        >>> poly[{'a', 'b'}]
        1
        >>> poly[('a', 'b')]
        1
        >>> poly['cd'] = 4
        >>> poly['dc']
        4

    """
    @vartype_argument('vartype')
    def __init__(self, poly, vartype):
        if isinstance(poly, abc.Mapping):
            poly = poly.items()

        # we need to aggregate the repeated terms
        self._terms = terms = {}
        for term, bias in poly:

            fsterm = asfrozenset(term)

            # when SPIN-valued, s^2 == 1, so we need to handle that case
            # in BINARY, x^2 == x
            if len(fsterm) < len(term) and vartype is Vartype.SPIN:
                new = set()
                term = tuple(term)  # make sure it has .count
                for v in fsterm:
                    if term.count(v) % 2:
                        new.add(v)
                fsterm = frozenset(new)

            if fsterm in terms:
                terms[fsterm] += bias
            else:
                terms[fsterm] = bias

        self.vartype = vartype

    def __contains__(self, term):
        return asfrozenset(term) in self._terms

    def __delitem__(self, term):
        del self._terms[asfrozenset(term)]

    def __eq__(self, other):
        if not isinstance(other, BinaryPolynomial):
            try:
                other = type(self)(other, self.vartype)
            except Exception:
                # not a polynomial
                return False

        self_terms = self._terms
        other_terms = other._terms

        return (
            self.vartype == other.vartype
            and all(
                (not bias or other_terms.get(term, 0.) == bias)
                for term, bias in self.items()
            )
            and all(
                (not bias or self_terms.get(term, 0.) == bias)
                for term, bias in other.items()
            )
        )

    def __ne__(self, other):
        return not (self == other)

    def __getitem__(self, term):
        return self._terms[asfrozenset(term)]

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    def __setitem__(self, term, bias):
        self._terms[asfrozenset(term)] = bias

    def __repr__(self):
        return '{!s}({!r}, {!r})'.format(self.__class__.__name__,
                                         self._terms, self.vartype.name)

    @property
    def variables(self):
        """Variables of the polynomial."""
        return set().union(*self._terms)

    @property
    def degree(self):
        """Degree of the polynomial."""
        if len(self) == 0:
            return 0
        return max(map(len, self._terms))

    def copy(self):
        """Create a shallow copy."""
        return type(self)(self, self.vartype)

    def energy(self, sample_like, dtype=float):
        """Energy of the given sample.

        Args:
            sample_like (samples_like):
                A raw sample. `sample-like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

            dtype (:class:`numpy.dtype`, optional):
                Data type of the returned energy. Defaults to float.

        Returns:
            The energy.

        Examples:
            >>> poly = dimod.BinaryPolynomial({('x', 'y'): -1, ('x', 'y', 'z'): 0.5},
            ...                               'BINARY')
            >>> poly.energy({'x': 1, 'y': 1, 'z': 0})
            -1.0
        """
        energy, = self.energies(sample_like, dtype=dtype)
        return energy

    def energies(self, samples_like, dtype=float):
        """Energies of the given samples.

        Args:
            samples_like (samples_like):
                A collection of raw samples. `samples-like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

            dtype (:class:`numpy.dtype`, optional):
                Data type of the returned energies. Defaults to float.

        Returns:
            :obj:`numpy.ndarray`: The energies.

        Examples:
            >>> poly = dimod.BinaryPolynomial({('x', 'y'): -1, ('x', 'y', 'z'): 0.5},
            ...                               'BINARY')
            >>> samples = [{'x': 1, 'y': 1, 'z': 0},
            ...            {'x': 1, 'y': 1, 'z': 1}]
            >>> poly.energies(samples)
            array([-1. , -0.5])
        """
        samples, labels = as_samples(samples_like)
        if labels:
            idx, label = zip(*enumerate(labels))
            labeldict = dict(zip(label, idx))
        else:
            labeldict = {}

        num_samples = samples.shape[0]

        energies = np.zeros(num_samples, dtype=dtype)
        for term, bias in self.items():
            if len(term) == 0:
                energies += bias
            else:
                energies += np.prod([samples[:, labeldict[v]] for v in term], axis=0) * bias

        return energies

    def relabel_variables(self, mapping, inplace=True):
        """Relabel variables of a binary polynomial as specified by mapping.

        Args:
            mapping (dict):
                Dict mapping current variable labels to new ones. If an
                incomplete mapping is provided, unmapped variables retain their
                current labels.

            inplace (bool, optional, default=True):
                If True, the binary polynomial is updated in-place; otherwise, a
                new binary polynomial is returned.

        Returns:
            :class:`.BinaryPolynomial`: A binary polynomial with the variables
            relabeled. If `inplace` is set to True, returns itself.

        Examples:
            >>> poly = dimod.BinaryPolynomial({('x', 'y'): -1, ('x', 'y', 'z'): 0.5},
            ...                               'BINARY')
            >>> poly2 = poly.relabel_variables({'x': 'k', 'y': 'l'}, inplace=False)
            >>> 'k' in poly2.variables and 'z' in poly2.variables
            True
        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        for submap in iter_safe_relabels(mapping, self.variables):

            for oldterm, bias in list(self.items()):
                newterm = frozenset((submap.get(v, v) for v in oldterm))

                if newterm != oldterm:
                    self[newterm] = bias
                    del self[oldterm]

        return self

    def normalize(self, bias_range=1, poly_range=None, ignored_terms=None):
        """Normalize biases of the binary polynomial to fit the provided range(s).

        If ``poly_range`` is provided, ``bias_range`` is used as the range for
        linear biases only while ``poly_range`` is used for other biases.

        Args:
            bias_range (number/pair):
                Value/range by which to normalize all biases, or, if ``poly_range``
                is provided, just the linear biases.

            poly_range (number/pair, optional):
                Value/range by which to normalize higher-order biases.

            ignored_terms (iterable, optional):
                Biases associated with these terms are not scaled.

        """

        def parse_range(r):
            if isinstance(r, Number):
                return -abs(r), abs(r)
            return r

        if ignored_terms is None:
            ignored_terms = set()
        else:
            ignored_terms = {asfrozenset(term) for term in ignored_terms}

        if poly_range is None:
            linear_range, poly_range = bias_range, bias_range
        else:
            linear_range = bias_range

        lin_range, poly_range = map(parse_range, (linear_range, poly_range))

        # determine the current ranges for linear, higherorder
        lmin = lmax = 0
        pmin = pmax = 0
        for term, bias in self.items():

            if term in ignored_terms:
                # we don't use the ignored terms to calculate the scaling
                continue

            if len(term) == 1:
                lmin = min(bias, lmin)
                lmax = max(bias, lmax)
            elif len(term) > 1:
                pmin = min(bias, pmin)
                pmax = max(bias, pmax)

        inv_scalar = max(lmin / lin_range[0], lmax / lin_range[1],
                         pmin / poly_range[0], pmax / poly_range[1])

        if inv_scalar != 0:
            self.scale(1 / inv_scalar, ignored_terms=ignored_terms)

    def scale(self, scalar, ignored_terms=None):
        """Multiply the polynomial by the given scalar.

        Args:
            scalar (number):
                Value to multiply the polynomial by.

            ignored_terms (iterable, optional):
                Biases associated with these terms are not scaled.

        """

        if ignored_terms is None:
            ignored_terms = set()
        else:
            ignored_terms = {asfrozenset(term) for term in ignored_terms}

        for term in self:
            if term not in ignored_terms:
                self[term] *= scalar

    @classmethod
    def from_hising(cls, h, J, offset=None):
        """Construct a binary polynomial from a higher-order Ising problem.

        Args:
            h (dict):
                The linear biases.

            J (dict):
                The higher-order biases.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            :obj:`.BinaryPolynomial`

        Examples:
            >>> poly = dimod.BinaryPolynomial.from_hising({'a': 2}, {('a', 'b'): -1}, 0)
            >>> poly.degree
            2

        """
        poly = {(k,): v for k, v in h.items()}
        poly.update(J)
        if offset is not None:
            poly[frozenset([])] = offset
        return cls(poly, Vartype.SPIN)

    def to_hising(self):
        """Construct a higher-order Ising problem from a binary polynomial.

        Returns:
            tuple: A 3-tuple of the form (`h`, `J`, `offset`) where `h` includes
            the linear biases, `J` has the higher-order biases and `offset` is
            the linear offset.

        Examples:
            >>> poly = dimod.BinaryPolynomial({'a': -1, ('a', 'b'): 1, ('a', 'b', 'c'): -1}, dimod.SPIN)
            >>> h, J, off = poly.to_hising()
            >>> h
            {'a': -1}

        """
        if self.vartype is Vartype.BINARY:
            return self.to_spin().to_hising()

        h = {}
        J = {}
        offset = 0
        for term, bias in self.items():
            if len(term) == 0:
                offset += bias
            elif len(term) == 1:
                v, = term
                h[v] = bias
            else:
                J[tuple(term)] = bias

        return h, J, offset

    @classmethod
    def from_hubo(cls, H, offset=None):
        """Construct a binary polynomial from a higher-order unconstrained
        binary optimization (HUBO) problem.

        Args:
            H (dict):
                Coefficients of a higher-order unconstrained binary optimization
                (HUBO) model.

        Returns:
            :obj:`.BinaryPolynomial`

        Examples:
            >>> poly = dimod.BinaryPolynomial.from_hubo({('a', 'b', 'c'): -1})
            >>> poly.degree
            3

        """
        poly = cls(H, Vartype.BINARY)
        if offset is not None:
            poly[()] = poly.get((), 0) + offset
        return poly

    def to_hubo(self):
        """Construct a higher-order unconstrained binary optimization (HUBO)
        problem from a binary polynomial.

        Returns:
            tuple: A 2-tuple of the form (`H`, `offset`) where `H` is the HUBO
            and `offset` is the linear offset.

        Examples:
            >>> poly = dimod.BinaryPolynomial({'a': -1, ('a', 'b', 'c'): -1}, dimod.BINARY)
            >>> H, off = poly.to_hubo()

        """
        if self.vartype is Vartype.SPIN:
            return self.to_binary().to_hubo()

        H = {tuple(term): bias for term, bias in self.items() if term}
        offset = self[tuple()] if tuple() in self else 0
        return H, offset

    def to_binary(self, copy=False):
        """Return a binary polynomial over `{0, 1}` variables.

        Args:
            copy (optional, default=False):
                If True, the returned polynomial is always a copy. Otherwise,
                if the polynomial is binary-valued already it returns itself.

        Returns:
            :obj:`.BinaryPolynomial`

        """
        if self.vartype is Vartype.BINARY:
            if copy:
                return self.copy()
            else:
                return self

        new = BinaryPolynomial({}, Vartype.BINARY)

        # s = 2x - 1
        for term, bias in self.items():
            for t in map(frozenset, powerset(term)):
                newbias = bias * 2**len(t) * (-1)**(len(term) - len(t))

                if t in new:
                    new[t] += newbias
                else:
                    new[t] = newbias

        return new

    def to_spin(self, copy=False):
        """Return a binary polynomial over `{-1, +1}` variables.

        Args:
            copy (optional, default=False):
                If True, the returned polynomial is always a copy. Otherwise,
                if the polynomial is spin-valued already it returns itself.

        Returns:
            :obj:`.BinaryPolynomial`

        """
        if self.vartype is Vartype.SPIN:
            if copy:
                return self.copy()
            else:
                return self

        new = BinaryPolynomial({}, Vartype.SPIN)

        # x = (s + 1) / 2
        for term, bias in self.items():
            newbias = bias / (2**len(term))

            for t in map(frozenset, powerset(term)):
                if t in new:
                    new[t] += newbias
                else:
                    new[t] = newbias

        return new


def powerset(iterable):
    return itertools.chain.from_iterable(itertools.combinations(iterable, r)
                                         for r in range(len(iterable)+1))
