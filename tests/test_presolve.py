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

import concurrent.futures
import itertools
import os.path
import unittest

import dimod
import numpy as np

from dwave.preprocessing import Presolver, Feasibility, InvalidModelError, TechniqueFlags

try:
    NUM_CPUS = len(os.sched_getaffinity(0))
except AttributeError:
    # windows
    NUM_CPUS = os.cpu_count()


class TestApply(unittest.TestCase):
    # Developer note: most of the testing for normalization/presolve is done at the C++
    # level. These tests are mostly testing the Cython/Python interface.

    def test_invalid_model(self):
        cqm = dimod.CQM()
        i = cqm.add_variable("INTEGER")
        cqm.objective.set_linear(i, float("nan"))

        presolver = Presolver(cqm)

        with self.assertRaises(InvalidModelError):
            presolver.apply()


class TestNormalization(unittest.TestCase):
    # Developer note: most of the testing for normalization is done at the C++
    # level. These tests are mostly testing the Cython/Python interface.

    @unittest.skipIf(NUM_CPUS < 4, "insufficient CPUs available")
    def test_concurrency(self):
        # obviously this is difficult to test fully, but we can at least run some
        # smoke tests to try to detect deadlocks and corrupted models

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variables("BINARY", 1000)
        for _ in range(1000):
            # these will be flipped, so if we're modifying the same model in
            # several threads at the same time we should see some issues
            cqm.add_constraint(((v, 1) for v in range(1000)), '>=', 1)

        presolver = Presolver(cqm)

        num_threads = 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            concurrent.futures.wait(
                [executor.submit(presolver.normalize) for i in range(num_threads)])

        cqm = presolver.detach_model()

        self.assertEqual(cqm.num_variables(), 1000)
        self.assertEqual(cqm.num_constraints(), 1000)

        linear = {v: -1 for v in range(1000)}
        for comp in cqm.constraints.values():
            self.assertEqual(comp.rhs, -1)
            self.assertEqual(comp.sense, dimod.sym.Sense.Le)
            self.assertEqual(comp.lhs.linear, linear)

    def test_empty(self):
        presolver = Presolver(dimod.ConstrainedQuadraticModel())

        self.assertFalse(presolver.normalize())  # no changes made

    def test_ge(self):
        cqm = dimod.ConstrainedQuadraticModel()
        i, j = dimod.Integers("ij")
        c = cqm.add_constraint(i + j >= 5)

        presolver = Presolver(cqm)

        self.assertTrue(presolver.normalize())  # constraint will be flipped
        self.assertFalse(presolver.normalize())  # no changes, already normalized

    def test_invalid_model(self):
        cqm = dimod.CQM()
        i = cqm.add_variable("INTEGER")
        cqm.objective.set_linear(i, float("nan"))

        presolver = Presolver(cqm)

        with self.assertRaises(InvalidModelError):
            presolver.normalize()


class TestPresolve(unittest.TestCase):
    # Developer note: most of the testing for presolve is done at the C++
    # level. These tests are mostly testing the Cython/Python interface.

    @unittest.skipIf(NUM_CPUS < 4, "insufficient CPUs available")
    def test_concurrency(self):
        # obviously this is difficult to test fully, but we can at least run some
        # smoke tests to try to detect deadlocks and corrupted models

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variables("INTEGER", 100)
        for i in range(100):
            # these will be presolved, so if we're modifying the same model in
            # several threads at the same time we should see some issues
            cqm.add_constraint(((v, 1) for v in range(i, 100)), '==', 1)

        presolver = Presolver(cqm)
        presolver.normalize()

        num_threads = 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = [executor.submit(presolver.presolve) for i in range(num_threads)]

        # presolve should only make any changes once
        self.assertEqual(sum(future.result() for future in jobs), 1)

        cqm = presolver.detach_model()

        self.assertEqual(cqm.num_variables(), 0)
        self.assertEqual(cqm.num_constraints(), 0)

    def test_empty(self):
        presolver = Presolver(dimod.ConstrainedQuadraticModel())

        self.assertFalse(presolver.normalize())  # no changes made
        self.assertFalse(presolver.presolve())  # no changes made

    def test_normalized(self):
        cqm = dimod.ConstrainedQuadraticModel()
        i, j = dimod.Integers("ij")
        c = cqm.add_constraint(i + j >= 100)

        presolver = Presolver(cqm)

        self.assertTrue(presolver.normalize())  # constraint will be flipped
        self.assertFalse(presolver.presolve())  # no changes made

    def test_not_normalized(self):
        cqm = dimod.ConstrainedQuadraticModel()
        i, j = dimod.Integers("ij")
        c = cqm.add_constraint(i + j >= 100)

        presolver = Presolver(cqm)

        with self.assertRaises(TypeError):
            self.assertFalse(presolver.presolve())

    def test_quadratic_expression(self):
        cqm = dimod.CQM()
        cqm.add_variables(dimod.BINARY, "zyx")

        # add them to a constraint in a different order
        x, y, z = dimod.Binaries('xyz')
        cqm.add_constraint(x + 2*y + 3*z + 4*x*y + 5*x*z + 6*y*z <= 7)

        # Fix 1
        cqm.add_constraint(y == 1)
        presolver = Presolver(cqm)
        presolver.set_techniques(TechniqueFlags.DomainPropagation)
        presolver.apply()

        new = presolver.copy_model()
        self.assertEqual(new.constraints[0].lhs.linear, {1: 5, 0: 9})
        self.assertEqual(new.constraints[0].lhs.quadratic, {(1, 0): 5.0})

        # Fix 2
        cqm.add_constraint(y == 1)
        cqm.add_constraint(z == 1)
        presolver = Presolver(cqm)
        presolver.set_techniques(TechniqueFlags.DomainPropagation)
        presolver.apply()

        new = presolver.copy_model()
        self.assertEqual(new.constraints[0].lhs.linear, {0: 10})
        self.assertEqual(new.constraints[0].lhs.quadratic, {})

    def test_quadratic_expression_smoke1(self):
        # This test is just for smoke, previously this problem raised memory errors
        # Other tests test for correctness
        cqm = dimod.lp.loads("""
        Minimize

        Subject To
         c0: - 17.051066840461537 v3  <= -9.106723614321982
         c1: + 11.484217680577922 v4 + 11.484217680577922 v1 + [
         - 22.968435361155844 v1 * v4 ]  <= 6.053502603325116
         c2: + 10.031746008066332 v2 + 10.031746008066332 v1 + [
         - 20.063492016132663 v1 * v2 ]  <= 5.025359547352213
         c3: - 3.0013102085723125 v0 - 12.856230580454906 v1 - 17.97895647564603 v4
         - 8.741753211664253 v2 - 14.227160635984685 v3  <= -29.111621736003023
         c4: + 16.929643545497036 v4 + 17.41326947801976 v2 + 2.1373835985793814 v3
         + 17.6955617528591 v0 + 15.074552221756992 v1 + [ - 29.58451989383531 v2 * v4
         - 4.274767197158763 v3 * v4 - 5.242019062204211 v0 * v2
         - 30.149104443513984 v1 * v0 ]  <= 16.741825427748314
         c5: + 4.272554217017247 v3 + 4.272554217017247 v1 + 7.787383623073452 v4
         + 7.787383623073452 v0 + [ - 8.545108434034494 v1 * v3
         - 15.574767246146903 v0 * v4 ]  <= 5.979476612426666
         c6: + 5.820219874968961 v3 + 12.672662789234511 v0 + 31.221562386236577 v4
         + 11.296889417103412 v2 + 21.305716829157575 v1 + [
         - 11.640439749937922 v0 * v3 - 13.704885828531099 v4 * v0
         - 14.360292059916867 v2 * v4 - 34.37794688402519 v1 * v4
         - 8.233486774289954 v1 * v2 ]  <= 21.530163530436162
         c7: - 7.514948602616437 v1  <= -4.2953168220913325
         c8: - 10.159724167038 v4 - 18.648347945794026 v3 - 5.095291100557395 v1
         - 18.17151250285826 v2 - 15.681874455192489 v0  <= -34.611210435670166
         c9: - 7.7423284407397786 v2  <= -4.796912556777682

        Bounds

        Binary
         v0 v1 v2 v3 v4
        General

        End""")

        presolver = Presolver(cqm)
        presolver.normalize()
        presolver.presolve()

    def test_quadratic_expression_smoke2(self):
        # This test is just for smoke, previously this problem raised memory errors
        # Other tests test for correctness
        cqm = dimod.lp.loads("""
        Subject To
         c0: + 37.796280181141725 v3 + 12.361222016139354 v0 + 9.900498986553224 v4
         + 2.692837991292598 v2 + 18.227397169741746 v1 + [
         - 19.336768049693514 v0 * v3 - 19.800997973106448 v4 * v3
         - 5.385675982585196 v2 * v0 - 36.45479433948349 v1 * v3 ]
          <= 20.947420872251243
         c1: + 24.502538739564844 v1 + 22.76247673463282 v0 + 36.37877362035509 v4
         + 34.63871161542307 v3 + [ - 14.71970552899811 v0 * v1
         - 34.28537195013158 v4 * v1 - 30.80524794026753 v3 * v0
         - 38.47217529057861 v3 * v4 ]  <= 28.951220695396575
         c2: - 7.024165897320047 v1 - 17.222408597919063 v4  <= -12.396213731711883
         c3: + 26.971390424403868 v4 + 13.349524130974341 v1 + 34.673600506593544 v2
         + 4.56864915781716 v0 + 16.483085055346855 v3 + [
         - 26.699048261948683 v1 * v4 - 27.243732586859053 v2 * v4
         - 9.13729831563432 v0 * v2 - 32.96617011069371 v3 * v2 ]
          <= 24.912061857195006
         c4: + 29.80995317158837 v4 + 7.281857767410268 v3 + 9.899507925195692 v0
         + 19.713574313467856 v2 + 42.81070393472186 v1 + [
         - 4.095173392548938 v3 * v4 - 19.799015850391385 v0 * v4
         - 35.725717100236416 v1 * v4 - 10.468542142271598 v1 * v3
         - 39.42714862693571 v1 * v2 ]  <= 26.827564985766625
         c5: - 11.283444468838663 v0 - 6.67991792118767 v4 - 2.9148614631386325 v2
          <= -11.392333040169904
         c6: - 2.8392360194482285 v2 - 12.464455699928497 v1  <= -6.777880150102839
         c7: - 12.077881817597405 v3 - 13.23799843470821 v1  <= -12.382979022647005
         c8: + 12.769456529168128 v3 + 12.769456529168128 v1 + [
         - 25.538913058336256 v1 * v3 ]  <= 5.87279197332948
         c9: - 6.818718217802458 v2 - 4.710431335067147 v1 - 3.788863346856327 v0
         - 17.600626904736487 v3  <= -16.291721005654637

        Bounds

        Binary
         v0 v1 v2 v3 v4
        General

        End""")

        presolver = Presolver(cqm)
        presolver.normalize()
        presolver.presolve()

    def test_time_limit(self):
        # Given a presolveable CQM
        cqm = dimod.CQM()
        i, j = dimod.Integers('ij')
        cqm.add_variables('INTEGER', 'ij')
        cqm.add_constraint(i <= 5)
        cqm.add_constraint(i >= -5)
        cqm.add_constraint(j == 105)

        #  use the default presolvers
        presolver = Presolver(cqm)
        presolver.normalize()

        self.assertFalse(presolver.presolve(time_limit_s=-1))  # negative time_limit does nothing
        self.assertFalse(presolver.presolve(time_limit_s=0))  # 0 time_limit does nothing
        self.assertTrue(presolver.presolve(time_limit_s=100))  # finally it can do work


class TestTechniques(unittest.TestCase):
    def test_add_techniques(self):
        presolver = Presolver(dimod.CQM())
        presolver.set_techniques(TechniqueFlags.None_)
        presolver.add_techniques(TechniqueFlags.DomainPropagation)
        self.assertEqual(presolver.techniques(), TechniqueFlags.DomainPropagation)

    def test_default_techniques(self):
        presolver = Presolver(dimod.CQM())
        self.assertEqual(presolver.techniques(), TechniqueFlags.Default)

        # this is not inclusive, but make sure that we check for specific techniques
        self.assertTrue(presolver.techniques() & TechniqueFlags.RemoveRedundantConstraints)
        self.assertIn(TechniqueFlags.RemoveRedundantConstraints, presolver.techniques())

    def test_functional(self):
        # this is not inclusive - those tests are at the C++ level - but make sure
        # that the techniques actually do something

        cqm = dimod.ConstrainedQuadraticModel()
        i = cqm.add_variable("INTEGER")
        cqm.add_constraint([(i, 1)], "<=", 10, label="bound")

        presolver = Presolver(cqm)
        presolver.set_techniques(TechniqueFlags.None_)

        self.assertFalse(presolver.apply())  # no changes should be made

        presolver.add_techniques(TechniqueFlags.DomainPropagation)
        self.assertTrue(presolver.apply())  # domain prop happens

        # we did domain propagation but not redundant constraint removal
        cqm = presolver.copy_model()
        self.assertEqual(cqm.upper_bound(i), 10)
        self.assertEqual(cqm.num_constraints(), 1)

        presolver.set_techniques(TechniqueFlags.RemoveRedundantConstraints)
        self.assertTrue(presolver.apply())  # removal happens

        # we removed the now redundant constraint
        cqm = presolver.copy_model()
        self.assertEqual(cqm.upper_bound(i), 10)
        self.assertEqual(cqm.num_constraints(), 0)

    def test_load_default_techniques(self):
        presolver = Presolver(dimod.CQM())

        presolver.set_techniques(TechniqueFlags.None_)

        with self.assertWarns(DeprecationWarning):
            presolver.load_default_presolvers()

        self.assertEqual(presolver.techniques(), TechniqueFlags.Default)

    def test_set_techniques(self):
        presolver = Presolver(dimod.CQM())
        presolver.set_techniques(TechniqueFlags.None_)
        self.assertFalse(presolver.techniques())


# Todo: reorganize these tests into individual classes for specific methods
# like the above
class TestPresolver(unittest.TestCase):
    def test_bug0(self):
        random = np.random.RandomState(0)
        cqm = dimod.ConstrainedQuadraticModel()

        dimod_vars = [
            dimod.Binary("a"),
            dimod.Spin("b"),
            dimod.Integer("c", lower_bound=-5000, upper_bound=5000),
            dimod.Real("z", lower_bound=-5000, upper_bound=5000),
        ]

        discrete_labels = "defghi"
        discrete = [dimod.Binary(c) for c in discrete_labels]
        dimod_vars.extend(discrete)

        cqm.add_discrete(discrete_labels)

        obj = dimod.QM()
        obj.add_variable(dimod.INTEGER, "c", lower_bound=-5000, upper_bound=5000)
        obj += sum(random.uniform() * u for u in dimod_vars)
        cqm.set_objective(obj.copy())

        presolver = Presolver(cqm, move=False)
        presolver.apply()
        presolver.clear_model()

        with self.assertRaises(ValueError):
            presolver.restore_samples(np.array([[]]))

    def test_bug1(self):
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.set_objective(dimod.Spin("f"))

        presolver = Presolver(cqm, move=False)
        presolver.apply()
        presolver.clear_model()

        with self.assertRaises(ValueError):
            # wrong number of variables
            presolver.restore_samples(np.array([[]]))

        presolver.restore_samples(np.array([[0]]))

    def test_bug56(self):
        # https://github.com/dwavesystems/dwave-preprocessing/issues/56
        for fname in ['bug56.0.cqm', 'bug56.1.cqm']:
            with self.subTest(fname):
                with open(os.path.join(os.path.dirname(__file__), 'data', fname), 'rb') as f:
                    cqm = dimod.CQM.from_file(f)

                presolver = Presolver(cqm)
                presolver.apply()

                cqm = presolver.detach_model()

                for v in cqm.variables:
                    self.assertNotEqual(cqm.lower_bound(v), cqm.upper_bound(v))

    def test_copy_model(self):
        cqm = dimod.CQM()

        i, j = dimod.Integers('ij')

        cqm.add_variables('INTEGER', 'ij')
        cqm.add_constraint(i <= 5)
        cqm.add_constraint(i >= -5)
        cqm.add_constraint(j == 105)

        presolver = Presolver(cqm)

        presolver.apply()

        model = presolver.copy_model()
        self.assertEqual(len(model.variables), 1)

        # and again, since it's just a copy
        model = presolver.copy_model()
        self.assertEqual(len(model.variables), 1)

    def test_cqm(self):
        cqm = dimod.CQM()

        i, j = dimod.Integers('ij')

        cqm.add_variables('INTEGER', 'ij')
        cqm.add_constraint(i <= 5)
        cqm.add_constraint(i >= -5)
        cqm.add_constraint(j == 105)

        presolver = Presolver(cqm)

        presolver.apply()

        samplearray, labels = presolver.restore_samples([[0], [1]])
        np.testing.assert_array_equal(samplearray, [[0, 105], [1, 105]])
        self.assertEqual(labels, 'ij')

    def test_detach_model(self):
        cqm = dimod.CQM()

        i, j = dimod.Integers('ij')

        cqm.add_variables('INTEGER', 'ij')
        cqm.add_constraint(i <= 5)
        cqm.add_constraint(i >= -5)
        cqm.add_constraint(j == 105)

        presolver = Presolver(cqm)

        presolver.apply()

        model = presolver.detach_model()
        self.assertEqual(len(model.variables), 1)

        with self.assertRaises(InvalidModelError):
            presolver.apply()

    def test_move(self):
        cqm = dimod.CQM()

        i, j = dimod.Integers('ij')

        cqm.add_variables('INTEGER', 'ij')
        cqm.add_constraint(i <= 5)
        cqm.add_constraint(i >= -5)
        cqm.add_constraint(j == 105)

        presolver = Presolver(cqm, move=True)

        self.assertTrue(cqm.is_equal(dimod.CQM()))

        presolver.apply()

        self.assertEqual(presolver.copy_model().num_variables(), 1)

        samplearray, labels = presolver.restore_samples([[0], [1]])
        np.testing.assert_array_equal(samplearray, [[0, 105], [1, 105]])
        self.assertEqual(labels, 'ij')

    def test_no_variable_constraints(self):
        with self.subTest("feasible"):
            cqm = dimod.ConstrainedQuadraticModel()
            i = dimod.Integer('i')

            cqm.add_constraint(i <= 1)
            cqm.add_constraint(i >= -1)
            cqm.add_constraint(i == 0)
            cqm.fix_variable('i', 0)

            presolver = Presolver(cqm)
            presolver.apply()

        with self.subTest("infeas <="):
            cqm = dimod.ConstrainedQuadraticModel()
            i = dimod.Integer('i')
            cqm.add_constraint(i <= 1)
            cqm.fix_variable('i', 2)

            presolver = Presolver(cqm)
            presolver.apply()
            self.assertIs(presolver.feasibility(), Feasibility.Infeasible)

        with self.subTest("infeas =="):
            cqm = dimod.ConstrainedQuadraticModel()
            i = dimod.Integer('i')
            cqm.add_constraint(i == 1)
            cqm.fix_variable('i', 2)

            presolver = Presolver(cqm)
            presolver.apply()
            self.assertIs(presolver.feasibility(), Feasibility.Infeasible)

        with self.subTest("infeas >="):
            cqm = dimod.ConstrainedQuadraticModel()
            i = dimod.Integer('i')
            cqm.add_constraint(i >= 1)
            cqm.fix_variable('i', -1)

            presolver = Presolver(cqm)
            presolver.apply()
            self.assertIs(presolver.feasibility(), Feasibility.Infeasible)

    def test_self_loop(self):
        i = dimod.Integer("i")
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_constraint(i * i <= 0)

        presolver = Presolver(cqm)
        presolver.apply()

        reduced = presolver.detach_model()

        samples = [[reduced.lower_bound(v) for v in reduced.variables],
                   [reduced.upper_bound(v) for v in reduced.variables]]

        samplearray, labels = presolver.restore_samples(samples)

        self.assertEqual(samplearray.shape, (2, 1))
        self.assertEqual(labels, 'i')

    def test_variable_removal(self):
        v0, v1, v2, v3 = dimod.Binaries('wxyz')

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.set_objective(v0 + v1 + v2 + v3)
        cqm.add_constraint(-3 + 7*v0 >= 3)
        cqm.add_constraint(-24 + 5*v0 + 12*v3 + 12*v1 + 19*v2 >= -4)
        cqm.add_constraint(6 - 12*v0 - 12*v2 + 25*v0*v2 >= 3)
        cqm.add_constraint(-5 + 10*v3 >= -1)

        presolver = Presolver(cqm)
        presolver.apply()

    def test_zero_biases(self):
        i, j = dimod.Integers('ij')
        cqm = dimod.ConstrainedQuadraticModel()

        cqm.add_constraint(0*i >= -5)

        cqm.set_lower_bound('i', -5)
        cqm.set_upper_bound('i', +5)

        presolver = Presolver(cqm)
        presolver.apply()

        cqm = presolver.detach_model()

        self.assertEqual(cqm.lower_bound(0), -5)
        self.assertEqual(cqm.upper_bound(0), +5)
        self.assertEqual(cqm.num_constraints(), 0)

    def test_domain_propagation(self):
        i, j, k, l, m, n, o, p = dimod.Reals('ijklmnop')

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.set_objective(i + j + k + l + m + n + o + p)
        cqm.add_constraint(i + 2 * j + k + l <= 100)
        cqm.add_constraint(2 * i + j + 2 * k + 5 * l <= 200)
        cqm.add_constraint(m - 2 * n >= 100)
        cqm.add_constraint(-m + 2 * n >= 100)
        cqm.add_constraint(o - 0.5 * p == 0)
        cqm.add_constraint(0.5 * o - p == 0)

        cqm.set_lower_bound('o', 0)
        cqm.set_upper_bound('o', 1)
        cqm.set_lower_bound('p', 0)
        cqm.set_upper_bound('p', 1)

        presolver = Presolver(cqm)
        presolver.apply()
        updated_cqm = presolver.copy_model()

        self.assertEqual(updated_cqm.lower_bound(0), 0)
        self.assertEqual(updated_cqm.upper_bound(0), 100)
        self.assertEqual(updated_cqm.lower_bound(1), 0)
        self.assertEqual(updated_cqm.upper_bound(1), 50)
        self.assertEqual(updated_cqm.lower_bound(2), 0)
        self.assertEqual(updated_cqm.upper_bound(2), 100)
        self.assertEqual(updated_cqm.lower_bound(3), 0)
        self.assertEqual(updated_cqm.upper_bound(3), 40)
        self.assertEqual(updated_cqm.lower_bound(4), 100 + 99 * 200)
        self.assertEqual(updated_cqm.upper_bound(4), 1e30)
        self.assertEqual(updated_cqm.lower_bound(5), 100 + 99 * 100)
        self.assertEqual(updated_cqm.upper_bound(5), 1e30)
        self.assertEqual(updated_cqm.lower_bound(6), 0)
        self.assertEqual(updated_cqm.upper_bound(6), 0.5**31)
        self.assertEqual(updated_cqm.lower_bound(7), 0)
        self.assertEqual(updated_cqm.upper_bound(7), 0.5**30)
