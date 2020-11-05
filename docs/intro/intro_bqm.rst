.. _intro_bqm:

=======================
Binary Quadratic Models
=======================

dimod provides a :term:`binary quadratic model` (BQM) class that encodes
:term:`Ising` and quadratic unconstrained binary optimization (\ :term:`QUBO`\ )
models used by samplers such as the D-Wave system.

For an introduction to BQMs, see :std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

dimod can represent BQMs with classes suited to different needs for
performance and ease of use. Beginners might start with the simple Python
:ref:`AdjDictBQM <adjdictbqm_dimod>` class (:class:`dimod.BinaryQuadraticModel`),
and later switch to a higher-performing class with a C++ implementation.

For descriptions of all supported BQM representations, see :ref:`bqm`.

BQM Generation
==============

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

>>> qubo = dict({(0, 0): -3, (1, 1): -1, (0, 1): 2, (2, 2): -1,
...              (0, 2): 2, (3, 3): -1, (0, 3): 2})
>>> dict_bqm = dimod.BQM.from_qubo(qubo)

When working with large, unchanging BQMs, you might use
dimod's :ref:`AdjArrayBQM <adjarraybqm_dimod>` class for performance.

>>> import numpy as np
>>> q_array = np.array([[-3.0, 2, 2, 2],
...                     [0, -1.0, 0.0, 0.0],
...                     [0, 0, -1.0, 0.0],
...                     [0, 0, 0, -1.0]])
>>> array_bqm = dimod.AdjArrayBQM(q_array, 'BINARY')

Especially for very large BQMs, you might read the data from a file using methods,
such as :meth:`~dimod.bqm.adjarraybqm.AdjArrayBQM.from_file` or others,
described in the documentation of each class.

Additionally, dimod provides a variety of :ref:`BQM generators <bqm>`.

>>> map_bqm = dimod.generators.random.ran_r(1, 7, cls=dimod.AdjVectorBQM)

BQM Attributes
==============

dimod's BQM objects provide access to a number of attributes and views. See the
documentation for a particular type of BQM class under :ref:`bqm`.

>>> dict_bqm.num_interactions
3
>>> dict_bqm.spin    # doctest:+ELLIPSIS
SpinView({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, ... -1.5, 'SPIN')

>>> map_bqm.variables
KeysView({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0})

BQM Methods
===========

BQMs support a large number of methods, many common, some particular to a class,
described under the documentation for :ref:`each class <bqm>`, to enable you to
build and manipulate BQMs.

>>> len(map_bqm.quadratic)
21
>>> map_bqm.remove_interaction(5, 6)
>>> len(map_bqm.quadratic)
20
