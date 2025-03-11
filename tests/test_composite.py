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

            @property
            def children(self):
                return [self._child]

            def close(self):
                self._child.close()

        inner_1 = Inner()
        inner_2 = Inner()
        mid = Mid(inner_1, inner_2)
        with Outer(mid) as outer:
            pass

        inner_1.resource.close.assert_called_once()
        inner_2.resource.close.assert_called_once()
