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

    def test_empty_object(self):
        response = self.response_factory()

        # should be empty
        self.assertEqual(list(response.samples()), [])
        self.assertEqual(list(response.energies()), [])
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

        response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
        response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})

        mapping = {'a': self.one, 'b': 0}
        rl_response = response.relabel_samples(mapping)
        response.relabel_samples(mapping, copy=False)

        mapping = {'a': self.one, 'b': self.one}
        response = self.response_factory()

        response.add_sample({'a': self.zero, 'b': self.one}, 1, data={'n': 5})
        response.add_sample({'a': self.one, 'b': self.zero}, -1, data={'n': 1})
        with self.assertRaises(ValueError):
            response.relabel_samples(mapping, copy=False)


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

        spin_response = response.as_spin(-1 * offset)

        for sample, energy in spin_response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)


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
