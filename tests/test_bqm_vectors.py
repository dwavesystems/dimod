import unittest

import numpy as np

from dimod.bqm.vectors import vector


class Test_vector(unittest.TestCase):
    def test_with_asarray(self):
        dtypes = [np.float32,
                  np.float64,
                  np.int8,
                  np.int16,
                  np.int32,
                  np.int64,
                  ]

        for dtype in dtypes:
            vec = vector([-1, 1], dtype=dtype)

            arr = np.asarray(vec)

            # when cast to a numpy array it should match dtype
            self.assertEqual(arr.dtype, dtype)

            # modifying the array should also modify the underlying (check that it's a proper view)
            arr[0] = 2
            self.assertEqual(vec[0], 2)

            vec[0] = -2
            self.assertEqual(arr[0], -2)

    def test_vector_construction(self):
        vec = vector([-1, 1], dtype=np.float32)

        newvec = vector(vec, dtype=np.float32)

        # should be a copy
        newvec[0] = 2
        self.assertEqual(vec[0], -1)
        self.assertEqual(newvec[0], 2)

    def test_numpy_construction_float64(self):
        arr = np.array([-1, 1, -1], dtype=np.float64)
        vec = vector(arr)
        self.assertEqual(vec, [-1, 1, -1])

        # should be a copy
        vec[0] = 2
        self.assertEqual(arr[0], -1)
        self.assertEqual(vec[0], 2)

    def test_numpy_construction_float32(self):
        arr = np.array([-1, 1, -1], dtype=np.float32)
        vec = vector(arr)
        self.assertEqual(vec, [-1, 1, -1])

    def test_numpy_construction_all_dtype(self):
        dtypes = [np.float32,
                  np.float64,
                  np.int8,
                  np.int16,
                  np.int32,
                  np.int64,
                  ]

        for dtype in dtypes:
            arr = np.array([-1, 1, -1], dtype=dtype)
            vec = vector(arr)
            self.assertEqual(vec, [-1, 1, -1])

            # should be a copy
            vec[0] = 2
            self.assertEqual(arr[0], -1)
            self.assertEqual(vec[0], 2)

    def test_insert(self):
        v = vector([-1, 1])

        v.insert(1, 0)
        self.assertEqual(v, [-1, 0, 1])

        v.insert(3, 2)
        self.assertEqual(v, [-1, 0, 1, 2])

        v.insert(10000, 3)
        self.assertEqual(v, [-1, 0, 1, 2, 3])
