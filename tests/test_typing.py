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

import itertools
import os
import re
import unittest

import numpy as np

try:
    import mypy
    import mypy.api
except ImportError:
    mypy = None


@unittest.skipUnless(mypy, "mypy not installed")
@unittest.skipUnless(tuple(map(int, np.__version__.split('.')[:2])) >= (1, 20), "need numpy>=1.20")
class TestTyping(unittest.TestCase):
    TEST_DIR = os.path.join(os.path.dirname(__file__), 'typing')
    CACHE_DIR = os.path.join(TEST_DIR, '.mypy_cache')

    def test_mypy(self):
        stdout, stderr, exit_code = mypy.api.run([
            '--cache-dir', self.CACHE_DIR,
            '--follow-imports', 'silent',
            '--show-absolute-path',
            self.TEST_DIR])

        if stderr:
            raise ValueError("mypy raised an error")

        # ok, now need to parse the errors that were raised and check if
        # we expected them or not
        # this relies pretty heavily on their syntax unfortunately
        files = {}  # caching
        for error in stdout.split('\n')[:-2]:

            # strip the drive for windows
            _, error = os.path.splitdrive(error)

            # get the associated line of code
            path, lineno, msg = error.split(':', 2)
            if path not in files:
                with open(path, 'r') as f:
                    files[path] = f.readlines()
            code = files[path][int(lineno) - 1]

            with self.subTest(f"{os.path.basename(path)}:{lineno}"):

                # strip out the comment from the code line
                _, part, comment = code.rstrip().partition('# E: ')

                self.assertEqual(part, '# E: ', "no error comment found")

                # finally check that the comment is present in the message.
                # we could do something more robust but this seems simple
                self.assertIn(comment, msg)
