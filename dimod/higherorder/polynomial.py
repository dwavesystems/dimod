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
# ============================================================================
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.decorators import vartype_argument
from dimod.sampleset import as_samples
from dimod.utilities import resolve_label_conflict
from dimod.vartypes import Vartype

def asfrozenset(term):
    """Convert to frozenset if it is not already"""
    return term if isinstance(term, frozenset) else frozenset(term)


class Polynomial(abc.MutableMapping):
    def __init__(self, poly):
        if isinstance(poly, abc.Mapping):
            poly = poly.items()
         
        # we need to aggregate the repeated terms
        self._terms = terms = {}        
        for term, bias in ((asfrozenset(term), bias) for term, bias in poly):
            if term in terms:
                terms[term] += bias
            else:
                terms[term] = bias

    def __contains__(self, term):
        return asfrozenset(term) in self._terms

    def __delitem__(self, term):
        del self._terms[asfrozenset(term)]

    def __getitem__(self, term):
        return self._terms[asfrozenset(term)]

    def __iter__(self):
        return iter(self._terms)

    def __len__(self):
        return len(self._terms)

    def __setitem__(self, term, bias):
        self._terms[asfrozenset(term)] = bias

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._terms)

    @property
    def variables(self):
        return set().union(*self._terms)

    @property
    def degree(self):
        if len(self) == 0:
            return 0
        return max(map(len, self._terms.values()))

    def copy(self):
        """Make a shallow copy"""
        return self.__class__(self)

    def energy(self, sample_like, dtype=np.float):
        energy, = self.energies(sample_like, dtype=dtype)
        return energy

    def energies(self, samples_like, dtype=np.float):
        samples, labels = as_samples(samples_like)
        idx, label = zip(*enumerate(labels))
        labeldict = dict(zip(label, idx))

        num_samples = samples.shape[0]

        energies = np.zeros(num_samples, dtype=dtype)
        for term, bias in self.items():
            if len(term) == 0:
                energies += bias
            else:
                energies += np.prod([samples[:, labeldict[v]] for v in term], axis=0) * bias

        return energies


    def relabel_variables(self, mapping, inplace=True):

        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        try:
            old_labels = set(mapping)
            new_labels = set(mapping.values())
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        variables = self.variables
        for v in new_labels:
            if v in variables and v not in old_labels:
                raise ValueError(('A variable cannot be relabeled "{}" without also relabeling '
                                  "the existing variable of the same name").format(v))

        shared = old_labels & new_labels
        if shared:
            old_to_intermediate, intermediate_to_new = resolve_label_conflict(mapping, old_labels, new_labels)

            self.relabel_variables(old_to_intermediate, inplace=True)
            self.relabel_variables(intermediate_to_new, inplace=True)
            return self

        for oldterm, bias in list(self.items()):
            newterm = frozenset((mapping.get(v, v) for v in oldterm))

            if newterm != oldterm:
                self[newterm] = bias
                del self[oldterm]

        return self

class BinaryPolynomial(Polynomial):
    @vartype_argument('vartype')
    def __init__(self, poly, vartype):
        Polynomial.__init__(self, poly)
        self.vartype = vartype

    @classmethod
    def from_ising(cls, h, J, offset=None):  
        poly = {(k,): v for k, v in h.items()}
        poly.update(J)
        if offset is not None:
            poly[frozenset([])] = offset
        return cls(poly, Vartype.SPIN)

    def copy(self):
        return self.__class__(self, self.vartype)
