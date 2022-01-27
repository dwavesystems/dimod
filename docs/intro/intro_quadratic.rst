.. _intro_qm:

===============================
Quadratic Models: Unconstrained
===============================

Unconstrained quadratic models are used to submit problems to :term:`sampler`\ s
such as D-Wave quantum computers\ [#]_ and some hybrid quantum-classical
samplers\ [#]_.

When using such samplers to handle problems with constraints, you typically
formulate the constraints as penalties: see
:std:doc:`sysdocs_gettingstarted:doc_getting_started`.
(:ref:`Constrained models <intro_cqm>`, such as the
:class:`~dimod.ConstrainedQuadraticModel`, can support constraints natively.)

.. [#]
  D-Wave quantum computers accept unconstrained binary quadratic models, such as
  quadratic unconstrained binary optimization (\ :term:`QUBO`\ ) models: binary
  because variables are represented by qubits that return two states and quadratic
  because polynomial terms of two variables can be represented by pairs of coupled
  qubits.

.. [#]
  Some hybrid quantum-classical samplers accept constrained and non-binary models;
  for example, a quadratic model with an integer variable that must be smaller
  than some configured value.

Supported Models
================

* **Quadratic Models**

  The :term:`quadratic model` (QM) class, :class:`~dimod.QuadraticModel`, encodes
  polynomials of binary, integer, and discrete variables, with all terms of degree
  two or less.

  For an introduction to QMs, see
  :std:doc:`Concepts: Quadratic Models <oceandocs:concepts/qm>`. For the QM class,
  its attributes and methods, see the :ref:`QM reference documentation <qm>`.

* **Binary Quadratic Models**

  The :term:`binary quadratic model` (BQM) class,
  :class:`~dimod.binary.binary_quadratic_model.BinaryQuadraticModel`,
  encodes :term:`Ising` and quadratic unconstrained binary optimization
  (\ :term:`QUBO`\ ) models used by samplers such as D-Wave's quantum computers.

  For an introduction to BQMs, see
  :std:doc:`Concepts: Binary Quadratic Models <oceandocs:concepts/bqm>`. For the BQM class,
  its attributes and methods, see the :ref:`BQM reference documentation <bqm>`.

* **Discrete Quadratic Models**

  The :term:`discrete quadratic model` (BQM) class,
  :class:`~dimod.DiscreteQuadraticModel`, encodes polynomials of discrete variables,
  with all terms of degree two or less.

  For an introduction to DQMs, see
  :std:doc:`Concepts: Discrete Quadratic Models <oceandocs:concepts/dqm>`. For the DQM
  class, its attributes and methods, see :ref:`DQM reference documentation <dqm>`.

See examples of using QPU solvers and `Leap <https://cloud.dwavesys.com/leap>`_
hybrid solvers on these models in Ocean documentation's
:std:doc:`Getting Started examples <oceandocs:getting_started>` and the
`dwave-examples GitHub repository <https://github.com/dwave-examples>`_.

Model Construction
==================

dimod provides a variety of model generators. These are especially useful for testing
code and learning.

Example: dimod BQM Generator
----------------------------

This example generates a BQM from a fully-connected graph (a clique) where all
linear biases are zero and quadratic values are uniformly selected -1 or +1 values.

>>> bqm = dimod.generators.random.ran_r(1, 7)

Typically you construct a model when reformulating your problem, using such
techniques as those presented in D-Wave's system documentation's
:std:doc:`sysdocs_gettingstarted:doc_handbook`.

Example: Formulating a Max-Cut Problem as a BQM
-----------------------------------------------

The four-node `maximum cut <https://en.wikipedia.org/wiki/Maximum_cut>`_
problem shown in this figure,

.. figure:: ../_images/four_node_star_graph.png
    :align: center
    :scale: 40 %
    :name: four_node_star_graph
    :alt: Four-node star graph

    Star graph with four nodes.

can be represented, as shown in the
`dwave-examples <https://github.com/dwave-examples/maximum-cut>`_ Maximum Cut
example, by a QUBO:

.. math::

   Q = \begin{bmatrix} -3 & 2 & 2 & 2\\
                        0 & -1 & 0 & 0\\
                        0 & 0 & -1 & 0\\
                        0 & 0 & 0 & -1
       \end{bmatrix}

For learning and testing with small BQMs, constructing BQMs in Python is
convenient:

>>> qubo = {(0, 0): -3, (1, 1): -1, (0, 1): 2, (2, 2): -1,
...         (0, 2): 2, (3, 3): -1, (0, 3): 2}
>>> bqm = dimod.BQM.from_qubo(qubo)

For performance, especially with very large BQMs, you might read the data from a
file using methods, such as :func:`~dimod.binary.BinaryQuadraticModel.from_file`
or from NumPy arrays.

Example: Interaction Between Integer Variables
----------------------------------------------

This example constructs a QM with an interaction between two integer variables.

>>> qm = dimod.QuadraticModel()
>>> qm.add_variables_from('INTEGER', ['i', 'j'])
>>> qm.add_quadratic('i', 'j', 1.5)
