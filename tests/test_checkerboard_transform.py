import unittest
import dimod

import dimod.testing as dit
from dimod import SpinReversalTransformComposite

try:
    import dwave_networkx as dnx
    _dnx = True
except ImportError:
    _dnx = False

try:
    from dwave.system.composites import FixedEmbeddingComposite
    _dwave = True
except ImportError:
    _dwave = False

class TestCheckerboardTransformComposite(unittest.TestCase):
    @unittest.skipUnless(_dnx, "No dwave_networkx package")
    def test_instantiation(self):
        C = dnx.chimera_graph(2, 2, 4)
        for factory in [dimod.ExactSolver, dimod.RandomSampler, dimod.SimulatedAnnealingSampler]:
            structsampler = dimod.StructureComposite(factory(),
                        nodelist=C.nodes(), edgelist=C.edges())

            sampler = dimod.CheckerboardTransformComposite(structsampler, C)

            dit.assert_sampler_api(sampler)
            dit.assert_composite_api(sampler)

    @unittest.skipUnless(_dnx, "No dwave_networkx package")
    def test_transforms_exact(self):

        C = dnx.chimera_graph(2, 2, 2)
        nodelist = list(C.nodes())
        edgelist = list(C.edges())
        structsampler = dimod.StructureComposite(dimod.ExactSolver(),
                    nodelist=nodelist, edgelist=edgelist)

        sampler = dimod.CheckerboardTransformComposite(structsampler, C,
                        aggregate=True)


        h = {v:0.1 for v in nodelist}
        J = {edge:-1.0 for edge in edgelist}
        response = sampler.sample_ising(h,J)

        # All 4 gauges must return same samples
        for datum in response.data():
            self.assertEqual(datum.num_occurrences, 4)

        dit.assert_response_energies(response, dimod.BinaryQuadraticModel.from_ising(h,J))

    @unittest.skipUnless(_dwave, "No dwave-system package")
    def test_transform_embedded(self):
        C = dnx.chimera_graph(1)
        nodelist = list(C.nodes())
        edgelist = list(C.edges())
        structsampler = dimod.StructureComposite(dimod.ExactSolver(),
                    nodelist=nodelist, edgelist=edgelist)
        gauges_sampler = dimod.CheckerboardTransformComposite(structsampler, C,
                        aggregate=True)
        sampler = FixedEmbeddingComposite(gauges_sampler, {'a': [0, 4], 'b': [1, 5], 'c': [2, 6]})
        h = {'a': .5, 'c': 0}
        J = {('a', 'c'): -1}
        response = sampler.sample_ising(h,J)
        # All 4 gauges must return same samples
        for datum in response.data():
            self.assertEqual(datum.num_occurrences, 4)

        dit.assert_response_energies(response, dimod.BinaryQuadraticModel.from_ising(h,J))


    @unittest.skipUnless(_dnx, "No dwave_networkx package")
    def test_chimera(self):
        C = dnx.chimera_graph(4)
        nodelist = list(C.nodes())
        edgelist = list(C.edges())
        structsampler = dimod.StructureComposite(dimod.RandomSampler(),
                            nodelist=nodelist, edgelist=edgelist)

        Q = {(v,v):0.1 for v in nodelist}
        Q.update( {edge:-1.0 for edge in edgelist} )

        sampler = dimod.CheckerboardTransformComposite(structsampler, C)
        response = sampler.sample_qubo(Q, num_reads=1000)

        dit.assert_response_energies(response, dimod.BinaryQuadraticModel.from_qubo(Q))
