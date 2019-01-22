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
#
# ================================================================================================

# class ScalingChecker(dimod.Sampler):
#     def __init__(self, originalbqm, scalar):
#         self.bqm = originalbqm.copy()
#         self.bqm.scale(scalar)
#     def sample(self, bqm):
#         assert self.bqm == bqm
#         return ExactSolver().sample(bqm)
#     def sample_ising(self,h, J):

import unittest

import dimod.testing as dtest
from dimod import ExactSolver, ScaleComposite, HigherOrderComposite, \
    BinaryQuadraticModel, Sampler
from dimod.reference.composites.scalecomposite import _scaled_hubo, \
    _check_params, _scaled_bqm


class ScalingChecker(Sampler):
    def __init__(self, child_sampler, bqm=None, h=None, J=None, offset=0,
                 scalar=None, bias_range=1, quadratic_range=None,
                 ignored_variables=None, ignored_interactions=None,
                 ignore_offset=False, **other_params):

        scale_options = dict(scalar=scalar,
                             bias_range=bias_range,
                             quadratic_range=quadratic_range,
                             ignored_variables=ignored_variables,
                             ignored_interactions=ignored_interactions,
                             ignore_offset=ignore_offset)
        self.child = child_sampler

        if bqm is not None:
            self.bqm = _scaled_bqm(bqm, **scale_options)

        elif h is not None and J is not None:
            if max(map(len, J.keys())) == 2:
                bqm = BinaryQuadraticModel.from_ising(h, J, offset=offset)
                self.bqm = _scaled_bqm(bqm, **scale_options)
            else:
                h_sc, J_sc, offset_sc = _scaled_hubo(h, J, offset=offset,
                                                     **scale_options)
                self.h = h_sc
                self.J = J_sc
                self.offset = offset_sc

    def sample(self, bqm, **parameters):
        assert self.bqm == bqm
        return self.child.sample(bqm, **parameters)

    def sample_ising(self, h, J, offset=0, **parameters):
        assert self.h == h
        assert self.J == J
        assert self.offset == offset
        return self.child.sample_ising(h, J, offset=offset, **parameters)

    def parameters(self):
        return self.child.parameters()

    def properties(self):
        return self.child.properties()


class TestScaleComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = ScaleComposite(ExactSolver())
        dtest.assert_sampler_api(sampler)

    def test_scaling_bqm(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}

        ignored_variables, ignored_interactions = _check_params()
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=5.0)
        bqm_new = _scaled_bqm(bqm,
                              bias_range=2,
                              ignored_interactions=ignored_interactions,
                              ignored_variables=ignored_variables)

        sc = 2.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        bqm_scaled = BinaryQuadraticModel.from_ising(hsc, Jsc, offset=5.0 / 2.)
        self.assertEqual(bqm_scaled, bqm_new)

        ignored_variables, ignored_interactions = _check_params()
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=5.0)
        bqm_new = _scaled_bqm(bqm,
                              bias_range=2,
                              ignored_interactions=ignored_interactions,
                              ignored_variables=ignored_variables,
                              ignore_offset=True)

        sc = 2.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        bqm_scaled = BinaryQuadraticModel.from_ising(hsc, Jsc, offset=5.0)
        self.assertEqual(bqm_scaled, bqm_new)

        bqm_new = _scaled_bqm(bqm,
                              quadratic_range=(-1, 0.4),
                              ignored_interactions=ignored_interactions,
                              ignored_variables=ignored_variables)

        sc = 3.2 / 0.4
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}

        bqm_scaled = BinaryQuadraticModel.from_ising(hsc, Jsc, offset=5.0 / sc)
        self.assertEqual(bqm_scaled, bqm_new)

        ignored_variables = ['a', 'b']
        ignored_interactions = None
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=ignored_interactions,
            ignored_variables=ignored_variables)
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic)
        bqm_new = _scaled_bqm(bqm,
                              bias_range=(-2, 2),
                              ignored_interactions=ignored_interactions,
                              ignored_variables=ignored_variables)

        sc = 3.2 / 2.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}

        bqm_scaled = BinaryQuadraticModel.from_ising(hsc, Jsc, offset=0)
        self.assertEqual(bqm_scaled, bqm_new)

        ignored_variables = None
        ignored_interactions = [('a', 'b')]
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=ignored_interactions,
            ignored_variables=ignored_variables)
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic)
        bqm_new = _scaled_bqm(bqm,
                              quadratic_range=0.5,
                              ignored_interactions=ignored_interactions,
                              ignored_variables=ignored_variables)

        sc = 4.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}

        bqm_scaled = BinaryQuadraticModel.from_ising(hsc, Jsc, offset=0)
        self.assertEqual(bqm_scaled, bqm_new)

    def test_scaling_hubo(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        ignored_variables = None
        ignored_interactions = None
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=ignored_interactions,
            ignored_variables=ignored_variables)

        hnew, jnew, offsetnew = _scaled_hubo(linear, quadratic, offset=offset,
                                             bias_range=2,
                                             ignored_interactions=ignored_interactions,
                                             ignored_variables=ignored_variables)

        sc = 2.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        self.assertEqual(hsc, hnew)
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        self.assertEqual(Jsc, jnew)
        self.assertEqual(offsetnew, offset / sc)

        hnew, jnew, offsetnew = _scaled_hubo(linear, quadratic, offset=offset,
                                             bias_range=2,
                                             ignored_interactions=ignored_interactions,
                                             ignored_variables=ignored_variables,
                                             ignore_offset=True)

        sc = 2.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        self.assertEqual(hsc, hnew)
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        self.assertEqual(Jsc, jnew)
        self.assertEqual(offsetnew, offset)

        hnew, jnew, offsetnew = _scaled_hubo(linear, quadratic, offset=offset,
                                             quadratic_range=(-1, 0.4),
                                             ignored_interactions=ignored_interactions,
                                             ignored_variables=ignored_variables)

        sc = 3.2 / 0.4
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        self.assertEqual(hsc, hnew)
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        self.assertEqual(Jsc, jnew)
        self.assertEqual(offsetnew, offset / sc)

        ignored_variables = ['a', 'b']
        ignored_interactions = None
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=ignored_interactions,
            ignored_variables=ignored_variables)

        hnew, jnew, offsetnew = _scaled_hubo(linear, quadratic, offset=offset,
                                             bias_range=(-2, 2),
                                             ignored_interactions=ignored_interactions,
                                             ignored_variables=ignored_variables)
        sc = 3.2 / 2.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        self.assertEqual(hsc, hnew)
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        self.assertEqual(Jsc, jnew)
        self.assertEqual(offsetnew, offset / sc)

        ignored_variables = None
        ignored_interactions = [('a', 'b', 'c')]
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=ignored_interactions,
            ignored_variables=ignored_variables)

        hnew, jnew, offsetnew = _scaled_hubo(linear, quadratic, offset=offset,
                                             quadratic_range=0.5,
                                             ignored_interactions=ignored_interactions,
                                             ignored_variables=ignored_variables)

        sc = 4.
        hsc = {k: v / sc if k not in ignored_variables else v for
               k, v in linear.items()}
        self.assertEqual(hsc, hnew)
        Jsc = {k: v / sc if k not in ignored_interactions else v for
               k, v in quadratic.items()}
        self.assertEqual(Jsc, jnew)
        self.assertEqual(offsetnew, offset / sc)

    def test_sample_hising_nonescale(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5

        ignored_variables, ignored_interactions = _check_params()

        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               penalty_strength=5.
                               )

        sampler = ScaleComposite(
            ScalingChecker(HigherOrderComposite(ExactSolver()),
                           h=linear,
                           J=quadratic, offset=offset,
                           **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)

        self.assertAlmostEqual(response.first.energy, -3.8)

    def test_sample_hising_bias_range(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               penalty_strength=5.,
                               bias_range=2)
        sampler = ScaleComposite(
            ScalingChecker(HigherOrderComposite(ExactSolver()),
                           h=linear,
                           J=quadratic, offset=offset,
                           **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)

        self.assertAlmostEqual(response.first.energy, -3.8)

    def test_sample_hising_quadratic_range(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()

        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               penalty_strength=5.,
                               quadratic_range=(-1, 2)
                               )
        sampler = ScaleComposite(
            ScalingChecker(HigherOrderComposite(ExactSolver()),
                           h=linear,
                           J=quadratic, offset=offset,
                           **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)

        self.assertAlmostEqual(response.first.energy, -3.8)

        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               penalty_strength=5.,
                               quadratic_range=(-1, 0.4)
                               )
        sampler = ScaleComposite(
            ScalingChecker(HigherOrderComposite(ExactSolver()),
                           h=linear,
                           J=quadratic, offset=offset,
                           **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)

        self.assertAlmostEqual(response.first.energy, -3.8)

    def test_sample_hising_ranges(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               penalty_strength=5.,
                               quadratic_range=(-1, 10),
                               bias_range=(-8.0, 5)
                               )
        sampler = ScaleComposite(
            ScalingChecker(HigherOrderComposite(ExactSolver()),
                           h=linear,
                           J=quadratic, offset=offset,
                           **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)

        self.assertAlmostEqual(response.first.energy, -3.8)

    def test_sample_nonescale(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), bqm=bqm,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_bias_range(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}

        ignored_variables, ignored_interactions = _check_params()
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic)
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               bias_range=2.
                               )
        sampler = ScaleComposite(ScalingChecker(ExactSolver(), bqm=bqm,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, -4.8)

    def test_sample_quadratic_range(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               quadratic_range=(-1, 2)
                               )
        sampler = ScaleComposite(ScalingChecker(ExactSolver(), bqm=bqm,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               quadratic_range=(-1, 0.4)
                               )
        sampler = ScaleComposite(ScalingChecker(ExactSolver(), bqm=bqm,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ranges(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               quadratic_range=(-1, 10),
                               bias_range=(-8.0, 5)
                               )
        sampler = ScaleComposite(ScalingChecker(ExactSolver(), bqm=bqm,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ising_quadratic(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params()
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables)
        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample_ising(linear, quadratic,
                                        offset=offset)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ising_ignore_interaction(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=[('a', 'b')])
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               scalar=0.5
                               )

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))

        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ising_ignore_offset(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5

        ignored_variables, ignored_interactions = _check_params()
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ignore_offset(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)

        ignored_variables, ignored_interactions = _check_params()
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_hising_ignore_offset(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5

        ignored_variables, ignored_interactions = _check_params()
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(HigherOrderComposite(
            ExactSolver()), h=linear,
            J=quadratic, offset=offset,
            **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, -3.8)

    def test_sample_ising_ignore_interactions(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=[('a', 'b')])

        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ignore_interactions(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)

        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=[('a', 'b')])
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_hising_ignore_interactions(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params(
            ignored_interactions=[('a', 'b', 'c')])
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(HigherOrderComposite(
            ExactSolver()), h=linear,
            J=quadratic, offset=offset,
            **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, -3.8)

    def test_sample_ising_ignore_variables(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params(
            ignored_variables=['a'])

        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_ignore_variables(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        ignored_variables, ignored_interactions = _check_params(
            ignored_variables=['a'])
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(ExactSolver(), h=linear,
                                                J=quadratic, offset=offset,
                                                **comp_parameters))
        response = sampler.sample(bqm, **comp_parameters)
        self.assertAlmostEqual(response.first.energy, 0.2)

    def test_sample_hising_ignore_variables(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        ignored_variables, ignored_interactions = _check_params(
            ignored_variables=['a'])
        comp_parameters = dict(ignored_interactions=ignored_interactions,
                               ignored_variables=ignored_variables,
                               ignore_offset=True,
                               scalar=0.5)

        sampler = ScaleComposite(ScalingChecker(HigherOrderComposite(
            ExactSolver()), h=linear,
            J=quadratic, offset=offset,
            **comp_parameters))
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        **comp_parameters)
        self.assertAlmostEqual(response.first.energy, -3.8)
