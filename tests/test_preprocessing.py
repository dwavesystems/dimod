# Copyright 2022 D-Wave Systems Inc.
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


removed = dict(
    ClipComposite=dimod.ClipComposite,
    ConnectedComponentsComposite=dimod.ConnectedComponentsComposite,
    FixedVariableComposite=dimod.FixedVariableComposite,
    RoofDualityComposite=dimod.RoofDualityComposite,
    ScaleComposite=dimod.ScaleComposite,
    SpinReversalTransformComposite=dimod.SpinReversalTransformComposite
    )


class TestRemovedComposites(unittest.TestCase):
    def test_exception(self):
        for label, composite in removed.items():
            with self.subTest(label):
                with self.assertRaises(TypeError):
                    composite(object())

        with self.subTest('fix_variables'):
            with self.assertRaises(TypeError):
                dimod.fix_variables(object())
