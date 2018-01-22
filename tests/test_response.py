import unittest

import itertools
import random
import itertools

import dimod


class ResponseGenericTests(object):
    """Provides tests that should pass for any dimod Response.
    unittests for a response object should inherit from this class.
    """
    def test_empty_response(self):
        response = self.response_factory()

        # there are three methods that return an iterator, when dumped into a list
        # they should all be empty
        self.assertEqual(list(response.samples()), [])
        self.assertEqual(list(response.energies()), [])
        self.assertEqual(list(response.data()), [])

        # response itself should also be able to be dumped to a list, iterating over
        # the samples
        self.assertEqual(list(response), [])

        # finally an empty response should have length 0
        self.assertEqual(len(response), 0)

    def test_samples(self):
        response = self.response_factory()
        response.add_sample({0: self.zero}, 1, n=5)
        response.add_sample({0: self.one}, -1, n=1)

        self.assertEqual(list(response.samples()),
                         [{0: self.one}, {0: self.zero}])
        for sample, data in response.samples(data=True):
            self.assertIn('n', data)
        self.assertEqual([spl for spl, __ in response.samples(data=True)],
                         [{0: self.one}, {0: self.zero}])

    def test_energies(self):
        response = self.response_factory()
        response.add_sample({0: self.zero}, 1, n=5)
        response.add_sample({0: self.one}, -1, n=1)
        self.assertEqual(list(response.energies()),
                         [-1, 1])
        for energy, data in response.energies(data=True):
            self.assertIn('n', data)
        self.assertEqual([en for en, __ in response.energies(data=True)],
                         [-1, 1])

    # items is being depreciated
    def test_items(self):
        response = self.response_factory()
        response.add_sample({0: self.zero}, 1, n=5)
        response.add_sample({0: self.one}, -1, n=5)
        self.assertEqual(list(response.data(keys=['sample', 'energy'])), [({0: self.one}, -1), ({0: self.zero}, 1)])

    def test_add_samples_from(self):
        """There are several different ways that responses can be added."""
        response = self.response_factory()

        sample = {0: self.zero}
        energy = 1.
        data = {'n': 107}

        samples = itertools.repeat(sample, 10)
        energies = itertools.repeat(energy, 10)

        response.add_samples_from(samples, energies, **data)

        samples = itertools.repeat(sample, 10)
        energies = itertools.repeat(energy, 10)

        response.add_samples_from(samples, energies)

        items = itertools.repeat((sample, energy), 10)
        response.add_samples_from(*zip(*items), **data)

        items = itertools.repeat((sample, energy), 10)
        response.add_samples_from(*zip(*items))

        self.assertEqual(len(response), 40)

    def test_relabel_variables(self):
        response = self.response_factory()

        response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
        response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})

        mapping = {'a': self.one, 'b': 0}
        rl_response = response.relabel_samples(mapping)
        response.relabel_samples(mapping)

        mapping = {'a': self.one, 'b': self.one}
        response = self.response_factory()

        response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
        response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})
        with self.assertRaises(dimod.MappingError):
            # mapping without unique variables
            response.relabel_samples(mapping)

        # check when we relabel only a subset
        response = self.response_factory()
        response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
        response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})
        with self.assertRaises(KeyError):
            rl_response = response.relabel_samples({'a': 'c'})

    def test_setting_info_on_construction(self):
        response = self.response_factory({'name': 'hello'})

        self.assertEqual(response.info['name'], 'hello')

        with self.assertRaises(TypeError):
            response = self.response_factory(5)

    def test_adding_single_sample_with_no_data(self):
        """There was a bug where every sample referenced the same data dict"""
        response = self.response_factory()

        response.add_sample({0: self.one, 1: self.zero}, 0.0)
        response.add_sample({0: self.one, 1: self.zero}, -1.)

        data_ids = set()
        for sample, data in response.samples(data=True):
            data_ids.add(id(data))

        self.assertEqual(len(data_ids), 2)

    def test_add_sample_type_checking(self):

        response = self.response_factory()

        with self.assertRaises(TypeError):
            response.add_sample({}, {})

        with self.assertRaises(TypeError):
            response.add_sample(1, 1)

        # add_sample should raise a TypeError when num_occurences set to a float
        with self.assertRaises(TypeError):
            response.add_sample({}, 0., num_occurences=1.)

    def test__str__(self):
        # just make sure it doesn't fail

        response = self.response_factory()

        for __ in range(100):
            response.add_sample({0: self.zero, 1: self.one}, 1, data={'n': 5})
            response.add_sample({0: self.one, 1: self.zero}, -1, data={'n': 1})

        s = response.__str__()

    def test_add_samples_type_checking(self):

        response = self.response_factory()

        with self.assertRaises(TypeError):
            response.add_samples_from([{}], [{}])

        with self.assertRaises(TypeError):
            response.add_samples_from([1], [1])

        with self.assertRaises(TypeError):
            response.add_samples_from([{}], [1], [7])

        # add_samples_from should raise a ValueError when given samples and energies are not the same length
        with self.assertRaises(ValueError):
            response.add_samples_from([{}, {}], [])

    def test__iter__(self):
        """iterating over response should be the same as iterating over samples"""
        response = self.response_factory()

        response.add_sample({0: self.zero, 1: self.one}, 1, data={'n': 5})
        response.add_sample({0: self.zero, 1: self.one}, 1, data={'n': 1})

        for sample in response:
            self.assertEqual(sample, {0: self.zero, 1: self.one})

    def test_cast(self):
        # cast to TemplateResponse
        response = self.response_factory()

        # cast an empty response
        new_response = response.cast(dimod.TemplateResponse)
        self.assertEqual(len(new_response), 0)

        # populate the response
        variables = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
        h = {v: random.uniform(-2, 2) for v in variables}
        J = {(u, v): random.uniform(-1, 1) for u, v in itertools.combinations(variables, 2)}
        for __ in range(10):
            sample = {v: random.choice((self.zero, self.one)) for v in variables}
            energy = sum(h[v] * sample[v] for v in sample)
            energy += sum(sample[v] * sample[u] * bias for (u, v), bias in J.items())
            response.add_sample(sample, energy)

        # cast to template without any transforms
        new_response = response.cast(dimod.TemplateResponse)
        for sample, new_sample in zip(response, new_response):
            self.assertEqual(sample, new_sample)

        # cast with offset
        new_response = response.cast(dimod.TemplateResponse, offset=2)
        for energy, new_energy in zip(response.energies(), new_response.energies()):
            self.assertAlmostEqual(energy, new_energy - 2)

        # cast with varmap (to a totally new system)
        varmap = {self.zero: -2, self.one: 2}
        new_response = response.cast(dimod.TemplateResponse, varmap=varmap)
        for sample, new_sample in zip(response, new_response):
            self.assertEqual({v: varmap[val] for v, val in sample.items()}, new_sample)

    def test_data(self):
        # cast to TemplateResponse
        response = self.response_factory()

        # populate the response with the add_sample method, setting some keywords
        variables = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 0, 1, 2, 3}
        h = {v: random.uniform(-2, 2) for v in variables}
        J = {(u, v): random.uniform(-1, 1) for u, v in itertools.combinations(variables, 2)}
        for i in range(10):
            sample = {v: random.choice((self.zero, self.one)) for v in variables}
            energy = sum(h[v] * sample[v] for v in sample)
            energy += sum(sample[v] * sample[u] * bias for (u, v), bias in J.items())

            if not i % 2:
                response.add_sample(sample, energy, sample_idx=i, neg_sample_idx=-i)
            else:
                response.add_sample(sample, energy, sample_idx=i, neg_sample_idx=-i, num_occurences=i)

        # iterate over response.data()
        last_energy = -float('inf')
        for datum in response.data():

            # all keyword parameters present
            self.assertIsInstance(datum, dict)
            self.assertEqual(set(datum), {'energy', 'sample', 'num_occurences', 'sample_idx', 'neg_sample_idx'})

            # in order of energy, low-to-high
            self.assertGreaterEqual(datum['energy'], last_energy)
            last_energy = datum['energy']

        for idx, datum in enumerate(response.data(ordered_by_energy=False)):

            # same requirements as before
            self.assertIsInstance(datum, dict)
            self.assertEqual(set(datum), {'energy', 'sample', 'num_occurences', 'sample_idx', 'neg_sample_idx'})

            # idx should be equal
            self.assertEqual(datum['sample_idx'], idx)
            self.assertEqual(datum['neg_sample_idx'], -idx)

        # iterating over specific keys
        last_energy = -float('inf')
        for energy in response.data(keys=['energy']):
            self.assertGreaterEqual(energy, last_energy)
            last_energy = energy

        # iterating over double keys
        i = 0
        for idx, nidx in response.data(keys=['sample_idx', 'neg_sample_idx'], ordered_by_energy=False):
            self.assertEqual(idx, -nidx)
            self.assertEqual(i, idx)
            i += 1


class TestTemplateResponse(unittest.TestCase, ResponseGenericTests):
    """Tests on the TemplateResponse"""
    def setUp(self):
        self.response_factory = dimod.TemplateResponse
        self.zero = 0
        self.one = 1

    def test_setting_vartype(self):
        # should be set to undefined
        response = dimod.TemplateResponse()
        self.assertIs(response.vartype, dimod.Vartype.UNDEFINED)

        # set to a specific one
        response = dimod.TemplateResponse(vartype=dimod.Vartype.SPIN)
        self.assertIs(response.vartype, dimod.Vartype.SPIN)

        # set to a specific one by accepted variable types
        response = dimod.TemplateResponse(vartype={-1, 1})
        self.assertIs(response.vartype, dimod.Vartype.SPIN)

        # set to a specific one by name
        response = dimod.TemplateResponse(vartype='BINARY')
        self.assertIs(response.vartype, dimod.Vartype.BINARY)


class TestBinaryResponse(unittest.TestCase, ResponseGenericTests):
    def setUp(self):
        self.response_factory = dimod.BinaryResponse
        self.zero = 0
        self.one = 1

    def test_as_spin(self):
        response = self.response_factory()

        # set up a BQM and some samples
        Q = {(0, 0): -1, (0, 1): 1, (1, 1): -1}
        h, J, offset = dimod.qubo_to_ising(Q)

        sample0 = {0: 0, 1: 1}
        sample1 = {0: 1, 1: 1}
        sample2 = {0: 0, 1: 0}

        # load up the samples
        response.add_samples_from([sample0, sample1, sample2], Q=Q)
        response.add_sample(sample0, Q=Q)

        # cast to spin
        spin_response = response.as_spin(-offset)

        # check that the energies are correcct
        for sample, energy in spin_response.data(keys=['sample', 'energy']):
            self.assertEqual(dimod.ising_energy(sample, h, J), energy)

        # make a new spin response
        spin_response = response.as_spin(offset)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in spin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)

    def test_input_checking(self):
        response = self.response_factory()

        with self.assertRaises(ValueError):
            response.add_sample({0: -1}, 0.0)

        with self.assertRaises(TypeError):
            response.add_sample({0: 0})  # neither energy nor Q

        with self.assertRaises(ValueError):
            response.add_samples_from([{0: -1}], [0.0])  # spin-valued sample

        with self.assertRaises(TypeError):
            response.add_samples_from([{0: 0}])  # neither energy nor Q


class TestSpinResponse(unittest.TestCase, ResponseGenericTests):
    def setUp(self):
        self.response_factory = dimod.SpinResponse
        self.zero = -1
        self.one = 1

    def test_add_sample_hJ(self):
        # add_sample with no energy specified, but h, J given

        h = {0: 0, 1: 0}
        J = {(0, 1): 1}

        sample0 = {0: -1, 1: 1}
        sample1 = {0: 1, 1: -1}
        sample2 = {0: 1, 1: 1}

        en0 = -1  # the resulting energy
        en1 = -1
        en2 = 1

        response = self.response_factory()
        response.add_sample(sample0, h=h, J=J)
        self.assertEqual(list(response.energies()), [en0])

        response.add_samples_from([sample0, sample1, sample2], h=h, J=J)
        self.assertEqual(list(response.energies()), [-1, -1, -1, 1])

    def test_as_binary(self):

        h = {0: 0, 1: 0}
        J = {(0, 1): 1}

        sample0 = {0: -1, 1: 1}
        sample1 = {0: 1, 1: -1}
        sample2 = {0: 1, 1: 1}

        Q, offset = dimod.ising_to_qubo(h, J)

        response = self.response_factory()
        response.add_samples_from([sample0, sample1, sample2], h=h, J=J)

        bin_response = response.as_binary(-1 * offset)
        for sample, energy in bin_response.data(keys=['sample', 'energy']):
            self.assertEqual(dimod.qubo_energy(sample, Q), energy)

        bin_response = response.as_binary(-1 * offset)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in bin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)

    def test_input_checking(self):
        response = self.response_factory()

        # binary value for sample
        with self.assertRaises(ValueError):
            response.add_sample({0: 0}, 0.0)

        with self.assertRaises(TypeError):
            response.add_sample({0: 1})  # neither energy nor h, J

        with self.assertRaises(ValueError):
            response.add_samples_from([{0: 0}], [0.0])

        with self.assertRaises(TypeError):
            response.add_samples_from([{0: 1}])  # neither energy nor h, J
