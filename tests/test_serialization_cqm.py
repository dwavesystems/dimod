# Copyright 2023 D-Wave Systems Inc.
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

"""
This file is intended to be backwards compatible, i.e. it can run with legacy
dimod versions in order to support creating test cases with several serialization
versions.

The last dimod version that supports each serialization version:
dimod 0.10.17: CQM serialization 1.1
dimod 0.11.5:  CQM serialization 1.2
dimod 0.12.3:  CQM serialization 1.3

We don't bother testing serialization version 1.0 because it was only every
supported by dev versions of dimod.

"""

import glob
import numbers
import os
import os.path
import re
import shutil
import typing
import unittest

import dimod

DIMOD_VERSION = tuple(map(int, re.search(r"\d+.\d+.\d+", dimod.__version__).group(0).split(".")))

try:
    # dimod>=0.12.4
    from dimod.constrained.constrained import CQM_SERIALIZATION_VERSION
except ImportError:
    # Legacy versions did not have this value accessible, so let's look it up
    if DIMOD_VERSION < (0, 10, 0):
        CQM_SERIALIZATION_VERSION = None
    elif DIMOD_VERSION < (0, 11, 0):
        CQM_SERIALIZATION_VERSION = (1, 1)
    elif DIMOD_VERSION < (0, 11, 6):
        CQM_SERIALIZATION_VERSION = (1, 2)
    elif DIMOD_VERSION < (0, 12, 4):
        CQM_SERIALIZATION_VERSION = (1, 3)
    else:
        # this shouldn't happen because we should be able to import it
        raise RuntimeError("unable to determine serialization version") from None


@unittest.skipUnless(CQM_SERIALIZATION_VERSION, "no CQM version available")
class TestSerialization(unittest.TestCase):
    dirname: str

    @classmethod
    def setUpClass(cls):
        # determine the data directory and ensure it exists
        cls.dirname = os.path.join(os.path.dirname(__file__), "data", "cqm")
        if not os.path.isdir(cls.dirname):
            os.mkdir(cls.dirname)

    def iter_cqms(self, *, tag: str,
                  ) -> typing.Iterator[typing.Tuple[typing.Tuple[int, int], dimod.CQM]]:
        # Yield all CQMs with the given tag
        for fname in glob.iglob(os.path.join(self.dirname, f"{tag}_v*.cqm")):
            with open(fname, "rb") as f:
                # if the version is greater than current, skip
                data = dimod.serialization.fileview.read_header(f, b'DIMODCQM')
                version = data.version
                if version > CQM_SERIALIZATION_VERSION:
                    continue
                f.seek(0)  # go back to the start

                with self.subTest(version=version):
                    yield version, dimod.ConstrainedQuadraticModel.from_file(f)

    @classmethod
    def save_cqm(cls, cqm: dimod.ConstrainedQuadraticModel, *, tag: str, **kwargs):
        # Add a CQM to the data directory if it doesn't already exist
        fname = os.path.join(cls.dirname,
                             f"{tag}_v{'.'.join(map(str, CQM_SERIALIZATION_VERSION))}.cqm")
        if not os.path.isfile(fname):
            with open(fname, "wb") as f:
                shutil.copyfileobj(cqm.to_file(**kwargs), f)

    @unittest.skipIf(DIMOD_VERSION < (0, 12, 3), "compress kwarg support added in 0.12.3")
    def test_compress(self):
        num_variables = 50
        cqm = dimod.CQM()
        cqm.add_variables('BINARY', range(num_variables))
        cqm.set_objective((v, 1) for v in range(num_variables))
        cqm.add_constraint(((v, 1) for v in range(num_variables)), '==', 0, label="c1")

        self.save_cqm(cqm, tag="compressed", compress=True)
        self.save_cqm(cqm, tag="uncompressed", compress=False)

        for version1, cqm1 in self.iter_cqms(tag="compressed"):
            for version2, cqm2 in self.iter_cqms(tag="uncompressed"):
                with self.subTest(version1=version1, version2=version2):
                    self.assertTrue(cqm1.is_equal(cqm2))

    def test_discrete(self):
        cqm = dimod.CQM()

        bqm = dimod.BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=', label="bqm rhs")
        cqm.add_constraint(bqm, '>=', label="bqm lhs")
        cqm.set_objective(dimod.Integer('c'))
        cqm.add_constraint(dimod.Spin('a')*dimod.Integer('d')*5 <= 3, label="mixed")
        cqm.add_discrete('efg', label="discrete")

        self.save_cqm(cqm, tag="discrete")

        for version, new in self.iter_cqms(tag="discrete"):
            with self.subTest(version=version):
                self.assertTrue(new.objective.variables >= cqm.objective.variables)
                for v, bias in cqm.objective.iter_linear():
                    self.assertEqual(new.objective.get_linear(v), bias)
                for u, v, bias in cqm.objective.iter_quadratic():
                    self.assertEqual(new.objective.get_quadratic(u, v), bias)
                self.assertEqual(new.objective.offset, cqm.objective.offset)

                self.assertEqual(set(cqm.constraints), set(new.constraints))
                for label, constraint in cqm.constraints.items():
                    self.assertTrue(constraint.lhs.is_equal(new.constraints[label].lhs))
                    self.assertEqual(constraint.rhs, new.constraints[label].rhs)
                    self.assertEqual(constraint.sense, new.constraints[label].sense)
                self.assertSetEqual(cqm.discrete, new.discrete)

    def test_empty(self):
        cqm = dimod.ConstrainedQuadraticModel()
        self.save_cqm(cqm, tag="empty")
        for version, new in self.iter_cqms(tag="empty"):
            with self.subTest(version=version):
                self.assertTrue(cqm.is_equal(new))

    def test_no_objective(self):
        cqm = dimod.CQM()
        bqm = dimod.BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=', label='ub')
        cqm.add_constraint(bqm, '>=', label='lb')

        self.save_cqm(cqm, tag="no-objective")

        for version, new in self.iter_cqms(tag="no-objective"):
            with self.subTest(version=version):
                self.assertEqual(set(cqm.constraints), set(new.constraints))
                for label, comp in cqm.constraints.items():
                    self.assertTrue(comp.lhs.is_equal(new.constraints[label].lhs))
                    self.assertEqual(comp.sense, new.constraints[label].sense)
                    self.assertEqual(comp.rhs, new.constraints[label].rhs)

                # as a special case, the objective might contain additional variables
                # this only happens in serialization versions < 2
                self.assertEqual(sum(new.objective.linear.values()), 0)
                self.assertEqual(len(new.objective.quadratic), 0)

    @unittest.skipUnless(CQM_SERIALIZATION_VERSION >= (1, 3), "no soft constraint support")
    def test_soft(self):
        cqm = dimod.CQM()
        bqm = dimod.BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=', weight=2.0, penalty='quadratic', label='soft1')
        cqm.add_constraint(dimod.Spin('a') * dimod.Integer('d') * 5 <= 3, weight=3.0, label='soft2')

        self.save_cqm(cqm, tag="soft")

        for version, new in self.iter_cqms(tag="soft"):
            with self.subTest(version=version):
                self.assertTrue(new.objective.variables >= cqm.objective.variables)
                for v, bias in cqm.objective.iter_linear():
                    self.assertEqual(new.objective.get_linear(v), bias)
                for u, v, bias in cqm.objective.iter_quadratic():
                    self.assertEqual(new.objective.get_quadratic(u, v), bias)
                self.assertEqual(new.objective.offset, cqm.objective.offset)

                self.assertEqual(set(cqm.constraints), set(new.constraints))
                for label, constraint in cqm.constraints.items():
                    self.assertTrue(constraint.lhs.is_equal(new.constraints[label].lhs))
                    self.assertEqual(constraint.rhs, new.constraints[label].rhs)
                    self.assertEqual(constraint.sense, new.constraints[label].sense)

                for label, info in cqm._soft.items():
                    self.assertEqual(info.weight, new._soft[label].weight)
                    self.assertIsInstance(new._soft[label].weight, numbers.Number)
                    self.assertEqual(info.penalty, new._soft[label].penalty)

    def test_unused_variable(self):
        cqm = dimod.CQM()
        cqm.add_variable(vartype='BINARY', v='x')

        self.save_cqm(cqm, tag="unused-variable")

        for version, new in self.iter_cqms(tag="unused-variable"):
            with self.subTest(version=version):
                self.assertEqual(new.variables, cqm.variables)
