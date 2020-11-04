.. _intro_dimod:

============
Introduction
============

.. include:: ..\README.rst
  :start-after: index-start-marker1
  :end-before: index-end-marker1

For explanations of the terminology, see the
:std:doc:`Ocean glossary <oceandocs:concepts/glossary>`.

The following sections give an orientation to dimod with usage examples:

.. toctree::
  :maxdepth: 1

  bqm

dimod provides a :term:`binary quadratic model` (BQM) class that encodes
:term:`Ising` and quadratic unconstrained binary optimization (\ :term:`QUBO`\ )
models used by samplers such as the D-Wave system.

It provides useful functionality for working with these models and samplers;
for example :ref:`generators_dimod` to build BQMs and :ref:`utilities_dimod` for calculating the energy of a
sample or serializing dimod objects.

It includes reference :term:`sampler`\ s and :term:`composite`\ s for processing binary quadratic programs
and refining sampling, and useful for testing your code during development.

It also provides an :ref:`api` for constructing new samplers and composed samplers
tailored for your problem.

Additionally, it provides some :ref:`higher_order_composites` and functionality
such as reducing higher-order polynomials to BQMs.

* For an introduction to BQMs, see :std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.
* For an introduction to samplers and composites, see
  :std:doc:`Samplers and Composites <oceandocs:concepts/samplers>`.

Examples
--------

Solving problems with large numbers of variables might necessitate the use of decomposition\ [#]_
methods such as branch-and-bound to reduce the number of variables. The following
illustrative example reduces an Ising model for a small problem (the K4 complete graph),
and converts the reduced-variables model to QUBO formulation.

.. [#] Ocean software's :std:doc:`D-Wave Hybrid <oceandocs:docs_hybrid/sdk_index>` provides tools for
   decomposing large problems.

>>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
>>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
...              (2, 3): 23, (2, 4): 24,
...              (3, 4): 34}
>>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, 0.5, dimod.SPIN)
>>> bqm_k4.vartype
<Vartype.SPIN: frozenset({1, -1})>
>>> len(bqm_k4.linear)
4
>>> bqm_k4.contract_variables(2, 3)
>>> len(bqm_k4.linear)
3
>>> bqm_no3_qubo = bqm_k4.binary
>>> bqm_no3_qubo.vartype
<Vartype.BINARY: frozenset({0, 1})>

The next example uses a composed sampler on the :std:doc:`Boolean NOT Gate <oceandocs:examples/not>`
example detailed in the :std:doc:`Getting Started <oceandocs:getting_started>` documentation.
The :class:`~dimod.reference.samplers.exact_solver.ExactSolver` test sampler calculates the
energy of all possible samples; the :class:`~dimod.reference.composites.fixedvariable.FixedVariableComposite`
composite sets the value and removes specified variables from the BQM before sending it to
the sampler. Fixing variable `x`, the input to the NOT gate, to 1 results in valid solution
:math:`z=0` having lower energy (-1) than solution :math:`x=z=1`, which is an invalid
state for a NOT gate.

>>> from dimod import FixedVariableComposite, ExactSolver
>>> Q = {('x', 'x'): -1, ('x', 'z'): 2, ('z', 'x'): 0, ('z', 'z'): -1}
>>> composed_sampler = FixedVariableComposite(ExactSolver())
>>> sampleset = composed_sampler.sample_qubo(Q, fixed_variables={'x': 1})
>>> print(sampleset)
   x  z energy num_oc.
0  1  0   -1.0       1
1  1  1    0.0       1
['BINARY', 2 rows, 2 samples, 2 variables]

The next example creates a dimod sampler by implementing a single method (in this example
the :meth:`sample_ising` method).

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
