# Copyright 2020 D-Wave Systems Inc.
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

import collections.abc
import functools
import warnings

import dimod

__all__ = ['load_sampler_bqm_tests']

# in the future we may want a way for folks to load new BQM classes into this
# framework
BQM_SUBCLASSES = [dimod.Float32BQM,
                  dimod.Float64BQM,
                  dimod.DictBQM,
                  ]


def test_sample(self, sampler, bqm):
    sampleset = sampler.sample(bqm)

    self.assertEqual(set(sampleset.variables), set(bqm.variables))
    self.assertIs(sampleset.vartype, bqm.vartype)
    dimod.testing.assert_sampleset_energies(sampleset, bqm, precision=4)


def test_sample_ising(self, sampler, h, J):
    sampleset = sampler.sample_ising(h, J)

    self.assertEqual(set(sampleset.variables), set(h).union(*J))
    self.assertIs(sampleset.vartype, dimod.SPIN)

    for sample, en in sampleset.data(['sample', 'energy']):
        self.assertAlmostEqual(dimod.ising_energy(sample, h, J), en)


def test_sample_qubo(self, sampler, Q):
    sampleset = sampler.sample_qubo(Q)

    self.assertEqual(set(sampleset.variables), set().union(*Q))
    self.assertIs(sampleset.vartype, dimod.BINARY)

    for sample, en in sampleset.data(['sample', 'energy']):
        self.assertAlmostEqual(dimod.qubo_energy(sample, Q), en)


class TestCaseOverloadWarning(UserWarning):
    pass


# this is functionally quite similar to functools.partialmethod but it avoids
# the confusing error message on a failed test that partial creates.
def parameterized_method(f, *args, specifiers=None):
    @functools.wraps(f)
    def method(self):
        return f(self, *args)

    if specifiers:
        method.__name__ += '_' + '_'.join(specifiers)

    return method


def create_bqm_tests(sampler, max_num_variables=None):
    """Yields test methods that can be added to a TestCase."""

    if max_num_variables is None:
        max_num_variables = float('inf')
    elif max_num_variables < 0:
        raise ValueError('max_num_variables should be a positive int')

    # empty

    specifiers = ['empty', sampler.__name__]

    h = {}
    J = {}

    yield parameterized_method(test_sample_ising, sampler(), h, J,
                               specifiers=specifiers)

    Q = {}

    yield parameterized_method(test_sample_qubo, sampler(), Q,
                               specifiers=specifiers)

    for BQM in BQM_SUBCLASSES:
        yield parameterized_method(test_sample, sampler(), BQM(h, J, 1.5, 'SPIN'),
                                   specifiers=['spin', BQM.__name__] + specifiers)
        yield parameterized_method(test_sample, sampler(), BQM.from_qubo(Q, 1.5),
                                   specifiers=['binary', BQM.__name__] + specifiers)

    # 1 variable

    if max_num_variables < 1:
        return

    u = (('a',),)  # complicated label

    specifiers = ['1var', sampler.__name__]

    h = {u: 6.0}
    J = {}

    yield parameterized_method(test_sample_ising, sampler(), h, J,
                               specifiers=specifiers)

    Q = {(u, u): 6.0}

    yield parameterized_method(test_sample_qubo, sampler(), Q,
                               specifiers=specifiers)

    for BQM in BQM_SUBCLASSES:
        yield parameterized_method(test_sample, sampler(), BQM(h, J, 1.5, 'SPIN'),
                                   specifiers=['spin', BQM.__name__] + specifiers)
        yield parameterized_method(test_sample, sampler(), BQM.from_qubo(Q, 1.5),
                                   specifiers=['binary', BQM.__name__] + specifiers)

    if max_num_variables < 2:
        return

    v = 0  # new variable

    specifiers = ['1path', sampler.__name__]

    h = {u: 6.0}
    J = {(u, v): -3}

    yield parameterized_method(test_sample_ising, sampler(), h, J,
                               specifiers=specifiers)

    Q = {(u, u): 6.0, (u, v): -3}

    yield parameterized_method(test_sample_qubo, sampler(), Q,
                               specifiers=specifiers)

    for BQM in BQM_SUBCLASSES:
        yield parameterized_method(test_sample, sampler(), BQM(h, J, 3, 'SPIN'),
                                   specifiers=['spin', BQM.__name__] + specifiers)
        yield parameterized_method(test_sample, sampler(), BQM.from_qubo(Q, 8),
                                   specifiers=['binary', BQM.__name__] + specifiers)

    if max_num_variables < 3:
        return

    w = 'c'  # new variable

    specifiers = ['2path', sampler.__name__]

    h = {u: 6.0}
    J = {(u, v): -3, (v, w): 105}

    yield parameterized_method(test_sample_ising, sampler(), h, J,
                               specifiers=specifiers)

    Q = {(u, u): 6.0, (u, v): -3, (v, w): 105}

    yield parameterized_method(test_sample_qubo, sampler(), Q,
                               specifiers=specifiers)

    for BQM in BQM_SUBCLASSES:
        yield parameterized_method(test_sample, sampler(), BQM(h, J, -4., 'SPIN'),
                                   specifiers=['spin', BQM.__name__] + specifiers)
        yield parameterized_method(test_sample, sampler(), BQM.from_qubo(Q, 16),
                                   specifiers=['binary', BQM.__name__] + specifiers)


def load_sampler_bqm_tests(sampler, max_num_variables=None,
                           suppress_overload_warning=False):
    """Populate the decorated TestCase with sampler tests using small BQMs.

    Args:
        sampler (obj/callable):
            A :class:`.Sampler`. If given as an object, the same sampler will
            be used for all tests, if given as a callable a new sampler will
            be instantiated for each test.

        max_num_variables (int, optional):
            The maximum BQM size to be tested. Defaults to unlimited.

        suppress_overload_warning (bool, optional, default=False):
            If True and the decorated class has a method matching one of the
            ones generated by `load_sampler_bqm_tests`, a warning will be
            emitted.

    Example:

        For example, if one wanted to test the :class:`.ExactSolver`,

        .. code-block:: python

            import unittest

            import dimod.testing

            @dimod.testing.load_sampler_bqm_tests(dimod.ExactSolver)
            class TestExactSolver(unittest.TestCase):
                pass

            unittest.main()

    """

    if isinstance(sampler, collections.abc.Callable):
        sampler_factory = sampler
    else:
        def sampler_factory():
            return sampler
        sampler_factory.__name__ = str(sampler)

    def decorator(cls):
        for test_method in create_bqm_tests(sampler_factory,
                                            max_num_variables=max_num_variables):

            if not hasattr(cls, test_method.__name__):
                setattr(cls, test_method.__name__, test_method)
            elif not suppress_overload_warning:
                msg = ('{} has been overloaded. If this is the desired '
                       'behaviour, this warning can be suppressed with the '
                       'suppress_overload_warning kwarg.')
                warnings.warn(msg, TestCaseOverloadWarning)

        return cls

    return decorator
