.. _bqm:

=======================
Binary Quadratic Models
=======================

For an introduction to BQMs, see
:std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

BQM Classes
===========

.. currentmodule:: dimod

.. we need this toctree for the autosummary links below work
.. toctree::
   :hidden:

   bqm_classes/adjdictbqm
   bqm_classes/adjvectorbqm
   bqm_classes/binary_quadratic_model.rst

.. autosummary::

   BinaryQuadraticModel
   AdjDictBQM
   AdjVectorBQM


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
