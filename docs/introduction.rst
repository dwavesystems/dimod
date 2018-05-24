============
Introduction
============

The dimod API provides a binary quadratic model (BQM) class that contains Ising and quadratic
unconstrained binary optimization (QUBO) models used by samplers such as the D-Wave system.
It provides utilities for constructing new samplers and composed samplers. It also
provides useful functionality for working with these models and samplers.

Ising and QUBO Formulations
---------------------------

The Ising model is an objective function of :math:`N` variables :math:`\bf s=[s_1,...,s_N]`
corresponding to physical Ising spins, where :math:`h_i` are the biases and
:math:`J_{i,j}` the couplings (interactions) between spins.

.. math::

  \text{Ising:} \qquad
  E(\bf{s}|\bf{h},\bf{J})
  = \left\{ \sum_{i=1}^N h_i s_i +
  \sum_{i<j}^N J_{i,j} s_i s_j  \right\}
  \qquad\qquad s_i\in\{-1,+1\}


The QUBO model is an objective function of :math:`N` binary variables represented as an
upper-diagonal matrix :math:`Q`, where diagonal terms are the linear coefficients and
the nonzero off-diagonal terms the quadratic coefficients.

.. math::

		\text{QUBO:} \qquad E(\bf{x}| \bf{Q})
    =  \sum_{i\le j}^N x_i Q_{i,j} x_j
    \qquad\qquad x_i\in \{0,1\}

The BinaryQuadraticModel class can contain both these models and its methods provide
convenient utilities for working with, and interworking between, the two representations
of a problem.

Example
~~~~~~~

Solving problems with large numbers of variables might necessitate the use of decomposition
methods such as branch-and-bound to reduce the number of variables. The following
example reduces an Ising model for a small problem (the K4 complete graph) for
illustrative purposes, and converts the reduced-variables model to QUBO formulation.

.. code-block:: python

    >>> import dimod
    >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
    >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
    ...              (2, 3): 23, (2, 4): 24,
    ...              (3, 4): 34}
    >>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, 0.5, dimod.SPIN)
    >>> bqm_k4.contract_variables(2, 3)
    >>> bqm_k4
    BinaryQuadraticModel({1: 1, 2: 2, 4: 4}, {(1, 2): 25, (1, 4): 14, (2, 4): 58}, 23.5, Vartype.SPIN)
    >>> bqm_no3_qubo = bqm_k4.binary
    >>> bqm_no3_qubo.linear
    {1: -76.0, 2: -162.0, 4: -136.0}
    >>> bqm_no3_qubo.quadratic
    {(1, 2): 100.0, (1, 4): 56.0, (2, 4): 232.0}
    >>> bqm_no3_qubo.offset
    113.5


Samplers and Composites
-----------------------

*Samplers* are processes that sample from low energy states of a problem’s objective function.
A binary quadratic model (BQM) sampler samples from low energy states in models such
as those defined by an Ising equation or a Quadratic Unconstrained Binary Optimization
(QUBO) problem and returns an iterable of samples, in order of increasing energy. A dimod
sampler provides ‘sample_qubo’ and ‘sample_ising’ methods as well as the generic
BQM sampler method.

*Composed samplers* apply pre- and/or post-processing to binary quadratic programs without
changing the underlying sampler implementation by layering composite patterns on the
sampler. For example, a composed sampler might add spin transformations when sampling
from the D-Wave system.

*Structured samplers* are restricted to sampling only binary quadratic models defined
on a specific graph.

You can create your own samplers with dimod's :class:`.Sampler` abstract base class (ABC)
providing complementary methods (e.g., ‘sample_qubo’ if only ‘sample_ising’ is implemented),
consistent responses, etc.

Example
~~~~~~~

This example creates a dimod sampler by implementing a single method (in this example
the :meth:`sample_ising` method).

.. code-block:: python

    class LinearIsingSampler(dimod.Sampler):

        def sample_ising(self, h, J):
            sample = linear_ising(h, J)  # Defined elsewhere
            energy = dimod.ising_energy(sample, h, J)
            return dimod.Response.from_dicts([sample], {'energy': [energy]})

        @property
        def properties(self):
            return dict()

        @property
        def parameters(self):
            return dict()

The :class:`.Sampler` ABC provides the other sample methods "for free"
as mixins.

Minor-Embedding
---------------

Embedding attempts to create a target :term:`model` from a target :term:`graph`. The process of
embedding takes a source model, derives the source graph, maps the source graph to the target
graph, then derives the target model. Sometimes referred to in other tools as the **embedded** graph/model.

Solving an arbitrarily posed binary quadratic problem on a D-Wave system requires minor-embedding
to a target graph that represents the system’s quantum processing unit.

Terminology
-----------

.. glossary::

    chain
        A collection of nodes or variables in the target graph/model
        that we want to act as a single node/variable.

    chain strength
        Magnitude of the negative quadratic bias applied
        between variables to form a chain.

    composed sampler
        Samplers that apply pre- and/or post-processing to binary quadratic programs without
        changing the underlying sampler implementation by layering composite patterns on the
        sampler. For example, a composed sampler might add spin transformations when sampling
        from the D-Wave system.

    graph
        A collection of nodes and edges. A graph can be derived
        from a model: a node for each variable and an edge for each pair
        of variables with a non-zero quadratic bias.

    model
        A collection of variables with associated linear and
        quadratic biases. Sometimes referred to in other tools as a **problem**.

    sampler
        A process that samples from low energy states of a problem’s objective function.
        A binary quadratic model (BQM) sampler samples from low energy states in models such
        as those defined by an Ising equation or a Quadratic Unconstrained Binary Optimization
        (QUBO) problem and returns an iterable of samples, in order of increasing energy. A dimod
        sampler provides ‘sample_qubo’ and ‘sample_ising’ methods as well as the generic
        BQM sampler method.

    source
        In the context of embedding, the model or induced graph that we wish to embed. Sometimes
        referred to in other tools as the **logical** graph/model.

    structured sampler
        Samplers that are restricted to sampling only binary quadratic models defined
        on a specific graph.

    target
        Embedding attempts to create a target model from a target
        graph. The process of embedding takes a source model, derives the source
        graph, maps the source graph to the target graph, then derives the target
        model. Sometimes referred to in other tools as the **embedded** graph/model.
