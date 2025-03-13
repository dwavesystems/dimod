.. _dimod_utilities:

=========
Utilities
=========

Converting Between Models
=========================

Converting between BQM models:

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   ising_to_qubo
   qubo_to_ising

Converting CQMs to BQMs:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   cqm_to_bqm

Converting higher-order models to BQMs:

.. autosummary::
   :toctree: generated/

   make_quadratic
   make_quadratic_cqm
   reduce_binary_polynomial

Decorators
==========

.. currentmodule:: dimod.decorators

.. automodule:: dimod.decorators

.. autosummary::
   :toctree: generated/

   bqm_index_labels
   bqm_structured
   forwarding_method
   graph_argument
   nonblocking_sample_method
   vartype_argument


Energy Calculations
===================

BQM energy:

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   ising_energy
   qubo_energy

Higher-order model energy:

.. automodule:: dimod.higherorder.utils

.. autosummary::
   :toctree: generated/

   poly_energies
   poly_energy


Fixing Variables (Moved)
========================

.. currentmodule:: dimod.roof_duality

.. autosummary::
   :toctree: generated/

   fix_variables

Graph Functions
===============

Converting between BQMs and NetworkX graph:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   to_networkx_graph
   from_networkx_graph

Traversing BQMs as a graph:

.. currentmodule:: dimod.traversal

.. autosummary::
   :toctree: generated/

   connected_components
   bfs_variables



.. _dimod_serialization:

Serialization
=============

COOrdinate
----------
.. currentmodule:: dimod.serialization.coo
.. automodule:: dimod.serialization.coo

.. autosummary::
   :toctree: generated/

   dump
   dumps
   load
   loads

FileView
--------

.. currentmodule:: dimod.serialization.fileview
.. automodule:: dimod.serialization.fileview

.. autosummary::
   :toctree: generated/

   FileView
   load

JSON
----

.. currentmodule:: dimod.serialization.json
.. automodule:: dimod.serialization.json

.. autofunction:: DimodEncoder
.. autofunction:: DimodDecoder
.. autofunction:: dimod_object_hook

LP
--

.. currentmodule:: dimod.lp
.. automodule:: dimod

.. autosummary::
   :toctree: generated/

   lp.dump
   lp.dumps
   lp.load
   lp.loads

Structure of Composed Sampler
=============================

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   child_structure_dfs

Summing Models
==============

Fast summation on models:

.. currentmodule:: dimod.binary

.. autosummary::
   :toctree: generated/

   quicksum


Testing
=======

.. automodule:: dimod.testing

.. currentmodule:: dimod.testing.asserts

API Asserts
-----------

.. autosummary::
   :toctree: generated/

   assert_composite_api
   assert_sampler_api
   assert_structured_api

Correctness Asserts
-------------------

.. autosummary::
   :toctree: generated/

   assert_bqm_almost_equal
   assert_response_energies
   assert_sampleset_energies

Test Case Loader
----------------

.. currentmodule:: dimod.testing.sampler

.. autosummary::
   :toctree: generated/

   load_sampler_bqm_tests
