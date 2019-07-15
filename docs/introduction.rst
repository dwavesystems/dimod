============
Introduction
============

dimod provides a binary quadratic model (BQM) class that contains
Ising and quadratic unconstrained binary optimization (QUBO) models used,
as described in :std:doc:`Solving Problems on a D-Wave System <oceandocs:overview/solving_problems>`,
by samplers such as the D-Wave system.

It provides useful functionality for working with these models and samplers;
for example :ref:`generators` to build BQMs and :ref:`utilities` for calculating the energy of a
sample or serializing dimod objects.

It includes reference :ref:`samplers` and :ref:`composites` for processing binary quadratic programs
and refining sampling, and useful for testing your code during development.

It also provides an :ref:`api` for constructing new samplers and composed samplers
tailored for your problem.

Additionally, it provides some :ref:`higher_order_composites` and functionality
such as reducing higher-order polynomials to BQMs.

Example
-------

Solving problems with large numbers of variables might necessitate the use of decomposition\ [#]_
methods such as branch-and-bound to reduce the number of variables. The following
illustrative example reduces an Ising model for a small problem (the K4 complete graph),
and converts the reduced-variables model to QUBO formulation.

.. [#] Ocean software's :std:doc:`D-Wave Hybrid <hybrid:index>` provides tools for
   decomposing large problems.

.. code-block:: python

    >>> import dimod
    >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
    >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
    ...              (2, 3): 23, (2, 4): 24,
    ...              (3, 4): 34}
    >>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, 0.5, dimod.SPIN)
    >>> bqm_k4.vartype
    <Vartype.SPIN: frozenset([1, -1])>
    >>> len(bqm_k4.linear)
    4
    >>> bqm_k4.contract_variables(2, 3)
    >>> len(bqm_k4.linear)
    3
    >>> bqm_no3_qubo = bqm_k4.binary
    >>> bqm_no3_qubo.vartype
    <Vartype.BINARY: frozenset([0, 1])>

Samplers and Composites
-----------------------

:term:`Sampler`\ s are  processes that sample from low-energy states of a problem’s objective function.
A binary quadratic model (BQM) sampler samples from low-energy states in
:term:`model`\ s such as those defined by an :term:`Ising` equation or a :term:`QUBO` problem
and returns an iterable of samples, in order of increasing energy. A dimod
sampler provides ‘sample_qubo’ and ‘sample_ising’ methods as well as the generic
BQM sampler method.

:term:`Composed sampler`\ s apply pre- and/or post-processing to binary quadratic programs without
changing the underlying sampler implementation by layering
`composite patterns <https://en.wikipedia.org/wiki/Composite_pattern>`_ on the
sampler. For example, a composed sampler might add spin transformations when sampling
from the D-Wave system.

:term:`Structured sampler`\ s are restricted to sampling only binary quadratic models defined
on a specific graph.

You can create your own samplers with dimod's :class:`.Sampler` abstract base class (ABC)
providing complementary methods (e.g., ‘sample_qubo’ if only ‘sample_ising’ is implemented),
consistent responses, etc.

Examples
~~~~~~~~

This first example uses a composed sampler on the :std:doc:`Boolean NOT Gate <oceandocs:examples/not>`
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

.. code-block:: python

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

Terminology
-----------

.. glossary::

    chain
        A collection of nodes or variables in the target :term:`graph`\ /\ :term:`model`
        that we want to act as a single node/variable.

    chain strength
        Magnitude of the negative quadratic bias applied
        between variables to form a :term:`chain`.

    composed sampler
        Samplers that apply pre- and/or post-processing to binary quadratic programs without
        changing the underlying :term:`sampler` implementation by layering composite patterns
        on the sampler. For example, a composed sampler might add spin transformations when
        sampling from the D-Wave system.

    graph
        A collection of nodes and edges. A graph can be derived
        from a :term:`model`\ : a node for each variable and an edge for each pair
        of variables with a non-zero quadratic bias.

    model
        A collection of variables with associated linear and
        quadratic biases. Sometimes referred to in other tools as a **problem**.

    sampler
        A process that samples from low energy states of a problem’s :term:`objective function`.
        A binary quadratic model (BQM) sampler samples from low energy states in models such
        as those defined by an :term`Ising` equation or a Quadratic Unconstrained Binary
        Optimization (\ :term:`QUBO`\ ) problem and returns an iterable of samples, in order
        of increasing energy. A dimod sampler provides ‘sample_qubo’ and ‘sample_ising’ methods
        as well as the generic BQM sampler method.

    source
        In the context of :term:`embedding`, the model or induced :term:`graph` that we
        wish to embed. Sometimes referred to in other tools as the **logical** graph/model.

    structured sampler
        Samplers that are restricted to sampling only binary quadratic models defined
        on a specific :term:`graph`.

    target
        :term:`Embedding` attempts to create a target :term:`model` from a target
        :term:`graph`. The process of embedding takes a source model, derives the source
        graph, maps the source graph to the target graph, then derives the target
        model. Sometimes referred to in other tools as the **embedded** graph/model.
