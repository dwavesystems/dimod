# Copyright 2019 D-Wave Systems Inc.
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

import shutil
import tempfile
import unittest

import dimod


class TestFileView(unittest.TestCase):
    def test_empty(self):
        bqm = dimod.AdjVectorBQM('SPIN', _ignore_warning=True)

        with tempfile.TemporaryFile() as tf:
            with bqm.to_file() as bqmf:
                shutil.copyfileobj(bqmf, tf)
            tf.seek(0)
            new = dimod.AdjVectorBQM.from_file(tf)

        self.assertEqual(bqm, new)

    def test_2path(self):
        bqm = dimod.AdjVectorBQM([.1, -.2], [[0, -1], [0, 0]], 'SPIN',
                                 _ignore_warning=True)

        with tempfile.TemporaryFile() as tf:
            with bqm.to_file() as bqmf:
                shutil.copyfileobj(bqmf, tf)
            tf.seek(0)
            new = dimod.AdjVectorBQM.from_file(tf)

        self.assertEqual(bqm, new)
