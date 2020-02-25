.. _bqm:

=======================
Binary Quadratic Models
=======================

.. todo: probably we'll want a shorter summary


.. automodule:: dimod.binary_quadratic_model

Contents
========

BQM Classes
-----------

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
---------

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

Traversing:

.. currentmodule:: dimod.traversal

.. autosummary::
   :toctree: generated/

   connected_components
   bfs_variables


Converting to and from other data types:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   to_networkx_graph
   from_networkx_graph

See also: :ref:`serialization functions<serialization_dimod>`

Usage
=====

This section is concerned with using binary quadratic models, the following
links describe how to formulate problems

* :std:doc:`Getting Started with the D-Wave System <sysdocs_gettingstarted:doc_getting_started>`
   Introduces key concepts such as objective functions, Ising model, QUBOs, and graphs, explains
   how these models are used to represent problems, and provides some simple examples.
* :std:doc:`D-Wave Problem-Solving Handbook <sysdocs_gettingstarted:doc_handbook>`
   Provides a variety of techniques for, and examples of, reformulating problems as BQMs.
* :std:doc:`Ocean's Getting Started <oceandocs:getting_started>`
   Describes and demonstrates the use of BQM in the context of Ocean software.

In Ocean, there are four objects that represent BQMs

* AdjDictBQM: Uses python dictionaries
* AdjArrayBQM: Uses c++ vectors as arrays
* AdjMapBQM: Uses c++ maps
* AdjVectorBQM: Uses c++ vectors

All of these BQM types use an adjacency structure, in which each variable
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

It is worth noting, that although the AdjDictBQM is superior in terms of
complexity, in practice it is much slower for large BQMs.

.. todo: maybe we should put some time-comparisons, fixing system etc?

.. todo: some examples showing how to construct them and maybe some comparisons