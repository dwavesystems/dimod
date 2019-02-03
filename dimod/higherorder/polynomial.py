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

from dimod.decorators import vartype_argument

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

    @property
    def variables(self):
        return set().union(*self._terms)


class BinaryPolynomial(Polynomial):
    @vartype_argument('vartype')
    def __init__(self, poly, vartype):
        super(BinaryPolynomial, self).__init__(self, poly)
        self.vartype = vartype
