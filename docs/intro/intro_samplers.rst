.. _intro_samplers:

=======================
Samplers and Composites
=======================

dimod includes reference :term:`sampler`\ s and :term:`composite`\ s for processing
quadratic (and higher order) models and refining sampling, and for
testing your code during development.

Using Samplers
==============

For an introduction to samplers and composites, see
:std:doc:`Concepts: Samplers and Composites <oceandocs:concepts/samplers>`.
For descriptions of all supported samplers, see
:ref:`Samplers reference documentation <samplers_index>`.

Example: Using a Reference Sampler
----------------------------------

To find solutions to the small four-node
`maximum cut <https://en.wikipedia.org/wiki/Maximum_cut>`_
BQM generated in the :ref:`intro_models` section, shown again in the figure below,
you can use one of dimod's reference samplers: its
:class:`~dimod.reference.samplers.ExactSolver` test sampler, for example,
calculates the energy of all possible samples.

.. figure:: ../_images/four_node_star_graph.png
    :align: center
    :scale: 40 %
    :name: four_node_star_graph2
    :alt: Four-node star graph

    Star graph with four nodes.

>>> qubo = {(0, 0): -3, (1, 1): -1, (0, 1): 2, (2, 2): -1,
...         (0, 2): 2, (3, 3): -1, (0, 3): 2}
>>> dict_bqm = dimod.BQM.from_qubo(qubo)
>>> sampler_exact = dimod.ExactSolver()
>>> sampleset = sampler_exact.sample(dict_bqm)
>>> print(sampleset)
    0  1  2  3 energy num_oc.
1   1  0  0  0   -3.0       1
11  0  1  1  1   -3.0       1
2   1  1  0  0   -2.0       1
...
10  1  1  1  1    0.0       1
['BINARY', 16 rows, 16 samples, 4 variables]

Samplers can be composed. The
`composite pattern <https://en.wikipedia.org/wiki/Composite_pattern>`_ allows
layers of pre- and post-processing to be applied to quadratic programs for a
sampler implementation.

Example: Using a Composed Sampler
---------------------------------

This example uses a composed sampler on the
:std:doc:`Boolean NOT Gate <oceandocs:examples/not>`
example detailed in the :std:doc:`Getting Started <oceandocs:getting_started>`
documentation.

The :class:`~dimod.reference.composites.structure.StructureComposite`
composite enforces the shape of the binary quadratic model. In this case we
only want to accept binary quadratic models with nodes labelled ``'x'``,
``'y'``, and ``'z'``.

>>> from dimod import ExactSolver, StructureComposite
>>> nodelist = ['x', 'y', 'z']
>>> edgelist = [('x', 'y'), ('x', 'z'), ('y', 'z')]
>>> composed_sampler = StructureComposite(ExactSolver(), nodelist, edgelist)
>>> Q = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}
>>> sampleset = composed_sampler.sample_qubo(Q)
>>> print(sampleset)
   x  z energy num_oc.
1  1  0   -1.0       1
3  0  1   -1.0       1
0  0  0    0.0       1
2  1  1    0.0       1
['BINARY', 4 rows, 4 samples, 2 variables]
>>> Q = {('a', 'a'): -1, ('a', 'b'): 2, ('b', 'a'): 0, ('b', 'b'): -1}
>>> try:
...     sampleset = composed_sampler.sample_qubo(Q)
... except ValueError:
...     print("incorrect structure!")
incorrect structure!


Creating Samplers
=================

dimod provides an :ref:`api` you can use to create your own dimod samplers and
composed samplers.

Example: Creating a Sampler
---------------------------

This example creates a dimod sampler by implementing a single method (in this
example the :code:`sample_ising` method).

.. testcode::

    class LinearIsingSampler(dimod.Sampler):

        def sample_ising(self, h, J, **kwargs):
            kwargs = self.remove_unknown_kwargs(**kwargs)
            sample = linear_ising(h, J, **kwargs)  # Defined elsewhere
            energy = dimod.ising_energy(sample, h, J)
            return dimod.SampleSet.from_samples(sample, vartype=dimod.SPIN, energy=energy)

        @property
        def properties(self):
            return dict()

        @property
        def parameters(self):
            return dict()

The :class:`.Sampler` ABC provides the other sample methods "for free"
as mixins.
