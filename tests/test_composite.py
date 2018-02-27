import unittest

import dimod


class TestCompositeClass(unittest.TestCase):
    def test_instantiation_base_class(self):
        with self.assertRaises(TypeError):
            dimod.Composite()

    def test_child_property(self):
        class Dummy(dimod.Composite):
            @property
            def children(self):
                return ['a', 'b']

        sampler = Dummy()
        self.assertEqual(sampler.child, 'a')

        class Dummy(dimod.Composite):
            @property
            def children(self):
                return []

        sampler = Dummy()
        with self.assertRaises(RuntimeError):
            sampler.child
