.. _bqm:

=======================
Binary Quadratic Models
=======================

:term:`Binary quadratic model`\ s (BQMs) are described under
:std:doc:`Binary Quadratic Models <oceandocs:concepts/bqm>`.

BQM Classes
===========

.. currentmodule:: dimod

.. we need this toctree for the autosummary links below work
.. toctree::
   :hidden:

   bqm_classes/adjarraybqm
   bqm_classes/adjdictbqm
   bqm_classes/adjmapbqm
   bqm_classes/adjvectorbqm
   bqm_classes/binary_quadratic_model.rst

.. autosummary::

   BinaryQuadraticModel
   AdjArrayBQM
   AdjDictBQM
   AdjMapBQM
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

In Ocean, there are four objects that represent BQMs, differentiated by the
data structure used to encode their structure and biases.

* :ref:`AdjArrayBQM <adjarraybqm_dimod>`: Uses c++ vectors as arrays
* :ref:`AdjDictBQM <adjdictbqm_dimod>`: Uses python dictionaries
* :ref:`AdjMapBQM <adjmapbqm_dimod>`: Uses c++ maps
* :ref:`AdjVectorBQM <adjvectorbqm_dimod>`: Uses c++ vectors

The documentation for each class outlines some of the advantages and
disadvantages of the different representations.

.. todo: we'll probably want a comparative adj/disadv table here but for now
.. I am not sure how to best represent it succinctly

All of the BQM types use an adjacency structure, in which each variable
tracks its own linear bias and its neighborhood. For instance, given
a BQM,

.. math::

   E(x) = .5 x_0 - 3 x_1 - x_0 x_1 + x_0 x_2 + 2 x_0 x_3 + x_2 x_3

its graph and adjacency representations are

.. figure:: ../_images/adj-reference.png
    :align: center
    :name: Adjacency Structure
    :alt: Adjacency Structure

    The adjacency structure of a 4-variable binary quadratic model.

The performance of various operations will depend on which binary quadratic
model implementation you are using. Let `v` be the number of variables in the
BQM

.. csv-table:: Complexity of various operations
   :header: , AdjArrayBQM, AdjDictBQM, AdjMapBQM, AdjVectorBQM

   add_variable, n/a, O(1) [#first]_, O(1) [#third]_, O(1) [#third]_
   add_interaction, n/a, O(1) [#second]_, O(log v), O(v)
   get_linear, O(1), O(1) [#first]_, O(1), O(1)
   get_quadratic, O(log v), O(1) [#second]_, O(log v), O(log v)
   num_variables, O(1), O(1), O(1), O(1)
   num_interactions, O(v), O(v), O(v), O(v)

.. todo: add the remove variable and remove_interaction

.. [#first] Average case, amortized worst case is O(v)
.. [#third] Amortized
.. [#second] Average case, amortized worst case is O(v^2)

It is worth noting that although the AdjDictBQM is superior in terms of
complexity, in practice it is much slower for large BQMs.

.. todo: maybe we should put some time-comparisons, fixing system etc?

.. todo: some examples showing how to construct them and maybe some comparisons
