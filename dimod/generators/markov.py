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

import dimod

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ['sample_markov_network', 'markov_network_bqm']


###############################################################################
# The following code is partially based on https://github.com/tbabej/gibbs
#
# MIT License
# ===========
#
# Copyright 2017 Tomas Babej
# https://github.com/tbabej/gibbs
#
# This software is released under MIT licence.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


@binary_quadratic_model_sampler(1)
def sample_markov_network(MN, sampler=None, fixed_variables=None,
                          return_sampleset=False,
                          **sampler_args):
    """Samples from a markov network using the provided sampler.

    Parameters
    ----------
    G : NetworkX graph
        A Markov Network as returned by :func:`.markov_network`

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    fixed_variables : dict
        A dictionary of variable assignments to be fixed in the markov network.

    return_sampleset : bool (optional, default=False)
        If True, returns a :obj:`dimod.SampleSet` rather than a list of samples.

    **sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    samples : list[dict]/:obj:`dimod.SampleSet`
        A list of samples ordered from low-to-high energy or a sample set.

    Examples
    --------

    >>> import dimod
    ...
    >>> potentials = {('a', 'b'): {(0, 0): -1,
    ...                            (0, 1): .5,
    ...                            (1, 0): .5,
    ...                            (1, 1): 2}}
    >>> MN = dnx.markov_network(potentials)
    >>> sampler = dimod.ExactSolver()
    >>> samples = dnx.sample_markov_network(MN, sampler)
    >>> samples[0]     # doctest: +SKIP
    {'a': 0, 'b': 0}

    >>> import dimod
    ...
    >>> potentials = {('a', 'b'): {(0, 0): -1,
    ...                            (0, 1): .5,
    ...                            (1, 0): .5,
    ...                            (1, 1): 2}}
    >>> MN = dnx.markov_network(potentials)
    >>> sampler = dimod.ExactSolver()
    >>> samples = dnx.sample_markov_network(MN, sampler, return_sampleset=True)
    >>> samples.first       # doctest: +SKIP
    Sample(sample={'a': 0, 'b': 0}, energy=-1.0, num_occurrences=1)

    >>> import dimod
    ...
    >>> potentials = {('a', 'b'): {(0, 0): -1,
    ...                            (0, 1): .5,
    ...                            (1, 0): .5,
    ...                            (1, 1): 2},
    ...               ('b', 'c'): {(0, 0): -9,
    ...                            (0, 1): 1.2,
    ...                            (1, 0): 7.2,
    ...                            (1, 1): 5}}
    >>> MN = dnx.markov_network(potentials)
    >>> sampler = dimod.ExactSolver()
    >>> samples = dnx.sample_markov_network(MN, sampler, fixed_variables={'b': 0})
    >>> samples[0]           # doctest: +SKIP
    {'a': 0, 'c': 0, 'b': 0}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """

    bqm = markov_network_bqm(MN)

    if fixed_variables:
        # we can modify in-place since we just made it
        bqm.fix_variables(fixed_variables)

    sampleset = sampler.sample(bqm, **sampler_args)

    if fixed_variables:
        # add the variables back in
        sampleset = dimod.append_variables(sampleset, fixed_variables)

    if return_sampleset:
        return sampleset
    else:
        return list(map(dict, sampleset.samples()))


def markov_network_bqm(MN):
    """Construct a binary quadratic model for a markov network.


    Parameters
    ----------
    G : NetworkX graph
        A Markov Network as returned by :func:`.markov_network`

    Returns
    -------
    bqm : :obj:`dimod.BinaryQuadraticModel`
        A binary quadratic model.

    """

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # the variable potentials
    for v, ddict in MN.nodes(data=True, default=None):
        potential = ddict.get('potential', None)

        if potential is None:
            continue

        # for single nodes we don't need to worry about order

        phi0 = potential[(0,)]
        phi1 = potential[(1,)]

        bqm.add_variable(v, phi1 - phi0)
        bqm.offset += phi0

    # the interaction potentials
    for u, v, ddict in MN.edges(data=True, default=None):
        potential = ddict.get('potential', None)

        if potential is None:
            continue

        # in python<=3.5 the edge order might not be consistent so we use the
        # one that was stored
        order = ddict['order']
        u, v = order

        phi00 = potential[(0, 0)]
        phi01 = potential[(0, 1)]
        phi10 = potential[(1, 0)]
        phi11 = potential[(1, 1)]

        bqm.add_variable(u, phi10 - phi00)
        bqm.add_variable(v, phi01 - phi00)
        bqm.add_interaction(u, v, phi11 - phi10 - phi01 + phi00)
        bqm.offset += phi00

    return bqm
