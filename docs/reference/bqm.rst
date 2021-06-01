.. _bqm:

=======================
Binary Quadratic Models
=======================

For an introduction to BQMs, see
:std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

Class
=====

.. currentmodule:: dimod.binary

.. autoclass:: BinaryQuadraticModel

.. autoclass:: BQM

Attributes
----------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.adj
   BinaryQuadraticModel.binary
   BinaryQuadraticModel.dtype
   BinaryQuadraticModel.linear
   BinaryQuadraticModel.num_interactions
   BinaryQuadraticModel.num_variables
   BinaryQuadraticModel.offset
   BinaryQuadraticModel.quadratic
   BinaryQuadraticModel.shape
   BinaryQuadraticModel.spin
   BinaryQuadraticModel.variables
   BinaryQuadraticModel.vartype

Class Methods
-------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.empty
   BinaryQuadraticModel.from_coo
   BinaryQuadraticModel.from_file
   BinaryQuadraticModel.from_ising
   BinaryQuadraticModel.from_numpy_vectors
   BinaryQuadraticModel.from_qubo

Methods
-------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.add_linear
   BinaryQuadraticModel.add_linear_equality_constraint
   BinaryQuadraticModel.add_linear_from
   BinaryQuadraticModel.add_linear_from_array
   BinaryQuadraticModel.add_quadratic
   BinaryQuadraticModel.add_quadratic_from
   BinaryQuadraticModel.add_quadratic_from_dense
   BinaryQuadraticModel.add_variable
   BinaryQuadraticModel.change_vartype
   BinaryQuadraticModel.contract_variables
   BinaryQuadraticModel.copy
   BinaryQuadraticModel.degree
   BinaryQuadraticModel.energies
   BinaryQuadraticModel.energy
   BinaryQuadraticModel.fix_variable
   BinaryQuadraticModel.fix_variables
   BinaryQuadraticModel.flip_variable
   BinaryQuadraticModel.is_linear
   BinaryQuadraticModel.iter_neighborhood
   BinaryQuadraticModel.iter_quadratic
   BinaryQuadraticModel.get_linear
   BinaryQuadraticModel.get_quadratic
   BinaryQuadraticModel.normalize
   BinaryQuadraticModel.reduce_linear
   BinaryQuadraticModel.reduce_neighborhood
   BinaryQuadraticModel.reduce_quadratic
   BinaryQuadraticModel.relabel_variables
   BinaryQuadraticModel.relabel_variables_as_integers
   BinaryQuadraticModel.remove_interaction
   BinaryQuadraticModel.remove_interactions_from
   BinaryQuadraticModel.remove_variable
   BinaryQuadraticModel.resize
   BinaryQuadraticModel.scale
   BinaryQuadraticModel.set_linear
   BinaryQuadraticModel.set_quadratic
   BinaryQuadraticModel.to_coo
   BinaryQuadraticModel.to_file
   BinaryQuadraticModel.to_ising
   BinaryQuadraticModel.to_numpy_vectors
   BinaryQuadraticModel.to_qubo
   BinaryQuadraticModel.update




Functions
=========

Generic constructor:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   as_bqm


Generating BQMs:

.. currentmodule:: dimod.generators

.. autosummary::
   :toctree: generated/

   chimera_anticluster
   combinations
   frustrated_loop
   gnm_random_bqm
   gnp_random_bqm
   randint
   ran_r
   uniform

Fixing variables:

.. currentmodule:: dimod.roof_duality

.. autosummary::
   :toctree: generated/

   fix_variables

Traversing as a graph:

.. currentmodule:: dimod.traversal

.. autosummary::
   :toctree: generated/

   connected_components
   bfs_variables


Converting to and from other data structures:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   to_networkx_graph
   from_networkx_graph

See also: :ref:`serialization functions<serialization_dimod>`

Usage
=====

Ocean represents BQMs with four objects that differ in the
data structures used to encode the BQM.

* :ref:`AdjDictBQM <adjdictbqm_dimod>`: Uses Python dictionaries
* :ref:`AdjVectorBQM <adjvectorbqm_dimod>`: Uses C++ vectors

The documentation for each class outlines some of the advantages and
disadvantages of the different implementations.

.. todo: we'll probably want a comparative adj/disadv table here but for now
.. I am not sure how to best represent it succinctly

All these BQM implementations use an adjacency structure in which each variable
tracks its own linear bias and its neighborhood. The figure below
shows the graph and adjacency representations for an example BQM,

.. math::

   E(x) = .5 x_0 - 3 x_1 - x_0 x_1 + x_0 x_2 + 2 x_0 x_3 + x_2 x_3

.. figure:: ../_images/adj-reference.png
    :align: center
    :name: Adjacency Structure
    :alt: Adjacency Structure

    Adjacency structure of a 4-variable binary quadratic model.

The performance of various operations depends on your selected BQM implementation;
the following table compares the complexity for a BQM of `v` variables.

.. csv-table:: Complexity of various operations
   :header:, AdjDictBQM, AdjVectorBQM

   add_variable, O(1) [#first]_,  O(1) [#third]_
   add_interaction, O(1) [#second]_, O(v)
   get_linear, O(1) [#first]_, O(1)
   get_quadratic, O(1) [#second]_, O(log v)
   num_variables, O(1), O(1)
   num_interactions, O(v), O(v)

.. todo: add the remove variable and remove_interaction

.. [#first] Average case, amortized worst case is O(v)
.. [#third] Amortized
.. [#second] Average case, amortized worst case is O(v^2)

It is worth noting that although the AdjDictBQM is superior in terms of
complexity, in practice it is much slower for large BQMs.

.. todo: maybe we should put some time-comparisons, fixing system etc?

.. todo: link to https://github.com/dwavesystems/dimod/pull/746 intro_bqm
