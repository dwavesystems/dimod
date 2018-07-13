import unittest

import numpy as np


def data_struct_array(sample, **vectors):  # data_struct_array(sample, *, energy, **vectors):
    """Combine samples and per-sample data into a numpy structured array.

    Args:
        sample (array_like):
            Samples, in any form that can be converted into a numpy array.

        energy (array_like, required):
            Required keyword argument. Energies, in any form that can be converted into a numpy
            1-dimensional array.

        **kwargs (array_like):
            Other per-sample data, in any form that can be converted into a numpy array.

    Returns:
        :obj:`~numpy.ndarray`: A numpy structured array. Has fields ['sample', 'energy', **kwargs]

    """

    if not sample:
        # if samples are empty
        sample = np.zeros((0, 0), dtype=np.int8)
    else:
        sample = np.asanyarray(sample, dtype=np.int8)

        if sample.ndim < 2:
            sample = np.expand_dims(sample, 0)

    num_samples, num_variables = sample.shape

    datavectors = {}
    datatypes = [('sample', np.dtype(np.int8), (num_variables,))]

    for kwarg, vector in vectors.items():
        datavectors[kwarg] = vector = np.asanyarray(vector)

        if vector.shape[0] != num_samples:
            msg = ('{} and sample have a mismatched shape {}, {}. They must have the same size '
                   'in the first axis.').format(kwarg, vector.shape, sample.shape)
            raise ValueError(msg)

        datatypes.append((kwarg, vector.dtype, vector.shape[1:]))

    if 'energy' not in datavectors:
        # consistent error with the one thrown in python3
        raise TypeError('data_struct_array() needs keyword-only argument energy')
    elif datavectors['energy'].shape != (num_samples,):
        raise ValueError('energy should be a vector of length {}'.format(num_samples))

    data = np.rec.array(np.zeros(num_samples, dtype=datatypes))

    data['sample'] = sample

    for kwarg, vector in datavectors.items():
        data[kwarg] = vector

    return data


class TestSamplesStructuredArray(unittest.TestCase):
    def test_empty(self):
        data = data_struct_array([], energy=[])

        self.assertEqual(data.shape, (0,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_single_sample(self):
        data = data_struct_array([-1, 1, -1], energy=[1.5])

        self.assertEqual(data.shape, (1,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_single_sample_nested(self):
        data = data_struct_array([[-1, 1, -1]], energy=[1.5])

        self.assertEqual(data.shape, (1,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_multiple_samples(self):
        data = data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5])

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_extra_data_vector(self):
        data = data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], occurrences=np.asarray([1, 2]))

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)
        self.assertIn('occurrences', data.dtype.fields)

    def test_data_vector_higher_dimension(self):
        data = data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], occurrences=[[0, 1], [1, 2]])

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)
        self.assertIn('occurrences', data.dtype.fields)

    def test_mismatched_vector_samples_rows(self):
        with self.assertRaises(ValueError):
            data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5, 5.6])

    def test_protected_sample_kwarg(self):
        with self.assertRaises(TypeError):
            data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], sample=[5, 6])

    def test_missing_kwarg_energy(self):
        with self.assertRaises(TypeError):
            data_struct_array([[-1, +1, -1], [+1, -1, +1]], occ=[5, 6])


if __name__ == '__main__':
    unittest.main()
