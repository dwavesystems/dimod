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
:class:`ConstrainedQuadraticModel`, can support constraints natively.)

Quadratic Models
================

dimod provides a :term:`quadratic model` (QM) class

Binary Quadratic Models
=======================

dimod provides a :term:`binary quadratic model` (BQM) class that encodes
:term:`Ising` and quadratic unconstrained binary optimization (\ :term:`QUBO`\ )
models used by samplers such as the D-Wave system.

For an introduction to BQMs, see :std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

For a description of the BQM class and its methods, see :ref:`bqm`.

BQM Generation
--------------

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

As mentioned above, for learning and testing with small BQMs, dimod's Python dict
representation of BQMs is convenient:

>>> qubo = {(0, 0): -3, (1, 1): -1, (0, 1): 2, (2, 2): -1,
...         (0, 2): 2, (3, 3): -1, (0, 3): 2}
>>> dict_bqm = dimod.BQM.from_qubo(qubo)

Especially for very large BQMs, you might read the data from a file using methods,
such as :meth:`~dimod.bqm.adjvectorbqm.AdjVectorBQM.from_file` or others,
described in the documentation of each class.

Additionally, dimod provides a variety of :ref:`BQM generators <bqm>`.

>>> map_bqm = dimod.generators.random.ran_r(1, 7, cls=dimod.AdjVectorBQM)

BQM Attributes
--------------

dimod's BQM objects provide access to a number of attributes and views. See the
documentation for a particular type of BQM class under :ref:`bqm`.

>>> dict_bqm.shape
(4, 3)

>>> list(map_bqm.variables)
[0, 1, 2, 3, 4, 5, 6]

BQM Methods
-----------

BQMs support a large number of methods, many common, some particular to a class,
described under the documentation for :ref:`each class <bqm>`, to enable you to
build and manipulate BQMs.

>>> map_bqm.num_interactions
21
>>> map_bqm.remove_interaction(5, 6)
>>> map_bqm.num_interactions
20

Discrete Quadratic Models
=========================

For an introduction to DQMs, see :std:doc:`Discrete Quadratic Models <oceandocs:concepts/dqm>`.

See examples of using `Leap <https://cloud.dwavesys.com/leap>`_ hybrid DQM
solvers in the `dwave-examples GitHub repository <https://github.com/dwave-examples>`_.
