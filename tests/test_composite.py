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


class TestInnermostChild(unittest.TestCase):

    def test_composed_sampler(self):
        innermost_child = dimod.ExactSolver()
        sampler = dimod.ClipComposite(dimod.ScaleComposite(innermost_child))
        self.assertTrue(sampler.innermost_child() is innermost_child)

