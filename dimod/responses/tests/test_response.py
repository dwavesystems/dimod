import unittest

import itertools
import random
import itertools

from dimod import SpinResponse, BinaryResponse
from dimod import TemplateResponse
from dimod import ising_to_qubo, qubo_to_ising, qubo_energy, ising_energy


class ResponseGenericTests(object):
    one = 1
    zero = 0
    relabel_allowed = True

    def test_empty_object(self):
        response = self.response_factory()

        # should all be empty and we should be able to iterate over them
        self.assertEqual(list(response.samples()), [])
        self.assertEqual(list(response.energies()), [])
        self.assertEqual(list(response.items()), [])
        self.assertEqual(list(response.samples(data=True)), [])
        self.assertEqual(list(response.energies(data=True)), [])
        self.assertEqual(list(response.items(data=True)), [])
        self.assertEqual(len(response), 0)

    def test_samples(self):
        response = self.response_factory()
        response.add_sample({0: self.zero}, 1, data={'n': 5})
        response.add_sample({0: self.one}, -1, data={'n': 1})
        self.assertEqual(list(response.samples()),
                         [{0: self.one}, {0: self.zero}])
        self.assertEqual(list(response.samples(data=True)),
                         [({0: self.one}, {'n': 1}), ({0: self.zero}, {'n': 5})])

    def test_energies(self):
        response = self.response_factory()
        response.add_sample({0: self.zero}, 1, data={'n': 5})
        response.add_sample({0: self.one}, -1, data={'n': 1})
        self.assertEqual(list(response.energies()),
                         [-1, 1])
        self.assertEqual(list(response.energies(data=True)),
                         [(-1, {'n': 1}), (1, {'n': 5})])

    def test_items(self):
        response = self.response_factory()
        response.add_sample({0: self.zero}, 1, data={'n': 5})
        response.add_sample({0: self.one}, -1, data={'n': 1})
        self.assertEqual(list(response.items()), [({0: self.one}, -1), ({0: self.zero}, 1)])
        self.assertEqual(list(response.items(data=True)),
                         [({0: self.one}, -1, {'n': 1}), ({0: self.zero}, 1, {'n': 5})])

    def test_add_samples_from(self):
        """There are several different ways that responses can be added."""
        response = self.response_factory()

        sample0 = {0: self.zero}
        energy0 = 1
        data0 = {'n': 107}

        samples = itertools.repeat(sample0, 10)
        energies = itertools.repeat(energy0, 10)
        sample_data = itertools.repeat(data0, 10)

        response.add_samples_from(samples, energies, sample_data)

        samples = itertools.repeat(sample0, 10)
        energies = itertools.repeat(energy0, 10)

        response.add_samples_from(samples, energies)

        items = itertools.repeat((sample0, energy0, data0), 10)
        response.add_samples_from(*zip(*items))

        items = itertools.repeat((sample0, energy0), 10)
        response.add_samples_from(*zip(*items))

        self.assertEqual(len(response), 40)

    def test_relabel_variables(self):
        response = self.response_factory()

        if not self.relabel_allowed:
            with self.assertRaises(ValueError):
                response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})

            with self.assertRaises(NotImplementedError):
                response.relabel_samples({})

            with self.assertRaises(ValueError):
                response.add_samples_from([{'a': self.zero, 'b': self.one}], [1],
                                          sample_data=[{'n': 5}])

        else:
            response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
            response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})

            mapping = {'a': self.one, 'b': 0}
            rl_response = response.relabel_samples(mapping)
            response.relabel_samples(mapping)

            mapping = {'a': self.one, 'b': self.one}
            response = self.response_factory()

            response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
            response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})
            with self.assertRaises(ValueError):
                # mapping without unique variables
                response.relabel_samples(mapping)

            # check when we relabel only a subset
            response = self.response_factory()
            response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
            response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})
            rl_response = response.relabel_samples({'a': 'c'})

            for sample in rl_response.samples():
                self.assertIn('c', sample)
                self.assertIn('b', sample)

    def test_setting_data_on_construction(self):
        response = self.response_factory({'name': 'hello'})

        self.assertEqual(response.data['name'], 'hello')

        with self.assertRaises(TypeError):
            response = self.response_factory(5)

    def test_adding_single_sample_with_no_data(self):
        response = self.response_factory()

        response.add_sample({0: self.one, 1: self.zero}, 0.0)
        response.add_sample({0: self.one, 1: self.zero}, -1.)

        data_ids = set()
        for sample, data in response.samples(data=True):
            data_ids.add(id(data))

            self.assertEqual(data, {})

        self.assertEqual(len(data_ids), 2)

    def test_add_sample_type_checking(self):

        response = self.response_factory()

        with self.assertRaises(TypeError):
            response.add_sample({}, {})

        with self.assertRaises(TypeError):
            response.add_sample(1, 1)

        with self.assertRaises(TypeError):
            response.add_sample({}, 1, 7)

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


class TestTemplateResponse(unittest.TestCase, ResponseGenericTests):
    """Tests on the TemplateResponse"""
    response_factory = TemplateResponse


class TestBinaryResponse(unittest.TestCase, ResponseGenericTests):
    response_factory = BinaryResponse

    def test_as_spin(self):
        # add_sample with no energy specified, but Q given

        Q = {(0, 0): -1, (0, 1): 1, (1, 1): -1}

        sample0 = {0: 0, 1: 1}
        sample1 = {0: 1, 1: 1}
        sample2 = {0: 0, 1: 0}

        h, J, offset = qubo_to_ising(Q)

        response = self.response_factory()
        response.add_samples_from([sample0, sample1, sample2], Q=Q)

        response.add_sample(sample0, Q=Q)

        spin_response = response.as_spin(-1 * offset)

        for sample, energy in spin_response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)

        spin_response = response.as_spin(-1 * offset, data_copy=True)
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
            response.add_samples_from([{0: -1}], [0.0])

        with self.assertRaises(TypeError):
            response.add_samples_from([{0: 0}])  # neither energy nor Q


class TestSpinResponse(unittest.TestCase, ResponseGenericTests):
    one = 1
    zero = -1  # spin-valued
    response_factory = SpinResponse

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

        Q, offset = ising_to_qubo(h, J)

        response = self.response_factory()
        response.add_samples_from([sample0, sample1, sample2], h=h, J=J)

        bin_response = response.as_binary(-1 * offset)
        for sample, energy in bin_response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)

        bin_response = response.as_binary(-1 * offset, data_copy=True)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in bin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)

    def test_input_checking(self):
        response = self.response_factory()

        with self.assertRaises(ValueError):
            response.add_sample({0: 0}, 0.0)

        with self.assertRaises(TypeError):
            response.add_sample({0: 1})  # neither energy nor h, J

        with self.assertRaises(ValueError):
            response.add_samples_from([{0: 0}], [0.0])

        with self.assertRaises(TypeError):
            response.add_samples_from([{0: 1}])  # neither energy nor h, J
