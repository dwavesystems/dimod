.. _intro_qm:

===============================
Quadratic Models: Unconstrained
===============================

:term:`Sampler`\ s such as D-Wave quantum computers accept unconstrained models
(binary quadratic models, for D-Wave systems: binary because variables are
represented by qubits that return two states and quadratic because you can
configure coupling strengths between pairs of qubits). Hybrid quantum-classical
samplers can accept non-binary models; for example, quadratic models with
discrete variables.

When using such samplers to handle problems with constraints, you typically
formulate the constraints as penalties: see
:std:doc:`sysdocs_gettingstarted:doc_getting_started`.
(:ref:`Constrained models <intro_cqm>`, such as the
:class:`dimod.ConstrainedQuadraticModel`, can support constraints natively.)

Supported Models
================

* **Quadratic Models**

  The :term:`quadratic model` (QM) class, :class:`dimod.QuadraticModel`, encodes
  polynomials of binary, integer, and discrete variables, with all terms of degree
  two or less.

  For the QM class, its attributes and methods, see :ref:`qm`.

* **Binary Quadratic Models**

  The :term:`binary quadratic model` (BQM) class, :class:`dimod.BinaryQuadraticModel`,
  encodes :term:`Ising` and quadratic unconstrained binary optimization
  (\ :term:`QUBO`\ ) models used by samplers such as the D-Wave system.

  For an introduction to BQMs, see
  :std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

  For the BQM class, its attributes and methods, see :ref:`bqm`.

* **Discrete Quadratic Models**

  The :term:`discrete quadratic model` (BQM) class,
  :class:`dimod.DiscreteQuadraticModel`, encodes polynomials of discrete variables,
  with all terms of degree two or less.

  For an introduction to DQMs, see
  :std:doc:`Discrete Quadratic Models <oceandocs:concepts/dqm>`.

See examples of using QPU solvers and `Leap <https://cloud.dwavesys.com/leap>`_
hybrid solvers on these models in the
`dwave-examples GitHub repository <https://github.com/dwave-examples>`_.

Model Construction
==================

dimod provides a variety of model generators. These are especially useful for testing
code and learning.

Example: dimod BQM Generator
----------------------------

>>> bqm = dimod.generators.random.ran_r(1, 7, cls=dimod.AdjVectorBQM)

Typically you construct a model when reformulating your problem, using such
techniques as those presented in D-Wave's system documentation's
:std:doc:`oceandocs:doc_handbook`.

Example: Formulating a Max-Cut Problem as a BQM
-----------------------------------------------

The small four-node `maximum cut <https://en.wikipedia.org/wiki/Maximum_cut>`_
problem shown in this figure,

.. figure:: ../_images/four_node_star_graph.png
    :align: center
    :scale: 40 %
    :name: four_node_star_graph
    :alt: Four-node star graph

    Star graph with four nodes.

Can be represented, as shown in the
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
file using methods,
such as :meth:`~dimod.bqm.adjvectorbqm.AdjVectorBQM.from_file` or from NumPy arrays.

Example: Interaction Between Integer Variables
----------------------------------------------

This example constructs a QM with an interaction between to integer variables.

>>> qm = QuadraticModel()
>>> qm.add_variables_from('INTEGER', ['i', 'j'])
>>> qm.add_quadratic('i', 'j', 1.5)
