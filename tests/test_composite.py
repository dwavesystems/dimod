# Copyright 2018 D-Wave Systems Inc.
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

import unittest
from unittest import mock

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

    def test_close(self):
        sampler = mock.MagicMock()

        class Dummy(dimod.Composite):
            @property
            def children(self):
                return [sampler]

        composite = Dummy()
        composite.close()

        composite.child.close.assert_called_once()

    def test_context_cleanup(self):
        class Inner(dimod.Sampler):
            parameters = None
            properties = None

            def __init__(self):
                self.resource = mock.Mock()

            def close(self):
                self.resource.close()

            def sample(self, bqm):
                pass

        class Mid(dimod.Composite):
            def __init__(self, *children):
                self._children = children

            @property
            def children(self):
                return self._children

        class Outer(dimod.Composite):
            def __init__(self, child):
                self._child = child
                self.resource = mock.Mock()

            @property
            def children(self):
                return [self._child]

            def close(self):
                super().close()
                self.resource.close()

        inner_1 = Inner()
        inner_2 = Inner()
        mid = Mid(inner_1, inner_2)
        with Outer(mid) as outer:
            pass

        # closed by Composite base close impl
        inner_1.resource.close.assert_called_once()
        inner_2.resource.close.assert_called_once()
        # closed by subclass
        outer.resource.close.assert_called_once()


class TestComposedSamplerInheritance(unittest.TestCase):
    def test_instantiation_base_class(self):
        with self.assertRaises(TypeError):
            dimod.ComposedSampler()

    def test_inheritance(self):
        class Dummy(dimod.ComposedSampler):
            def sample(self, bqm):
                return self.child.sample(bqm)

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

            @property
            def children(self):
                return [self._sampler]

            def __init__(self, sampler):
                self._sampler = sampler

        sampler = mock.MagicMock()
        composite = Dummy(sampler)

        with self.subTest("sampler interface"):
            h = {}
            J = {'ab': 1}
            composite.sample_ising(h, J)
            sampler.sample.assert_called_with(dimod.BQM.from_ising(h, J))

        with self.subTest("composite interface"):
            self.assertEqual(composite.child, sampler)

        with self.subTest("close inheritance"):
            composite.close()
            sampler.close.assert_called_once()
