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

A sampler is a process that samples from low energy states in models defined by an Ising
equation or a QUBO problem.

Samplers used with Ocean software are expected to have a ‘sample_qubo’ and ‘sample_ising’ method
and return an iterable of samples, in order of increasing energy.

A composite is a transformation applied to a sample. A composed sampler applies the
transformation to a sampler's samples.

The dimod API provides utilities for constructing new samplers and composed samplers
with standard input and response formats.

Example
~~~~~~~
