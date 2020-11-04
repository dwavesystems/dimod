.. _intro_samplers:

=======================
Samplers and Composites
=======================

dimod includes reference :term:`sampler`\ s and :term:`composite`\ s for processing
binary quadratic models (and higher order modeals) and refining sampling, and for
testing your code during development.

For an introduction to samplers and composites, see
:std:doc:`Samplers and Composites <oceandocs:concepts/samplers>`.

For descriptions of all supported samplers, see :ref:`bqm`.

Using Samplers
==============

To find solutions to the small four-node
`maximum cut <https://en.wikipedia.org/wiki/Maximum_cut>`_
BQM, shown in the figure below, generated in :ref:`intro_bqm`,
you can use one of dimod's reference samplers: its
:class:`~dimod.reference.samplers.ExactSolver` test sampler
calculates the energy of all possible samples.

.. figure:: ../_images/four_node_star_graph.png
    :align: center
    :scale: 40 %
    :name: four_node_star_graph2
    :alt: Four-node star graph

    Star graph with four nodes.

>>> qubo = dict({(0, 0): -3, (1, 1): -1, (0, 1): 2, (2, 2): -1,
...              (0, 2): 2, (3, 3): -1, (0, 3): 2})
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

Using Composed Samplers
=======================

This example uses a composed sampler on the
:std:doc:`Boolean NOT Gate <oceandocs:examples/not>`
example detailed in the :std:doc:`Getting Started <oceandocs:getting_started>`
documentation.

The :class:`~dimod.reference.composites.fixedvariable.FixedVariableComposite`
composite sets the value and removes specified variables from the BQM before
sending it to the sampler. Fixing variable `x`, the input to the NOT gate, to 1
results in valid solution :math:`z=0` having lower energy (-1) than solution
:math:`x=z=1`, which is an invalid state for a NOT gate.

>>> from dimod import FixedVariableComposite, ExactSolver
>>> Q = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}
>>> composed_sampler = FixedVariableComposite(ExactSolver())
>>> sampleset = composed_sampler.sample_qubo(Q, fixed_variables={'x': 1})
>>> print(sampleset)
   x  z energy num_oc.
0  1  0   -1.0       1
1  1  1    0.0       1
['BINARY', 2 rows, 2 samples, 2 variables]

Creating Samplers
=================

dimod provides an :ref:`api` you can use to create your own dimod samplers and
composed samplers.

This example creates a dimod sampler by implementing a single method (in this
example the :meth:`sample_ising` method).

.. testcode::

    class LinearIsingSampler(dimod.Sampler):

        def sample_ising(self, h, J):
            sample = linear_ising(h, J)  # Defined elsewhere
            energy = dimod.ising_energy(sample, h, J)
            return dimod.Response.from_samples([sample], {'energy': [energy]})

        @property
        def properties(self):
            return dict()

        @property
        def parameters(self):
            return dict()

The :class:`.Sampler` ABC provides the other sample methods "for free"
as mixins.
