.. _utilities_dimod:

=========
Utilities
=========

.. contents::
    :depth: 3

Decorators
==========

.. currentmodule:: dimod.decorators

.. automodule:: dimod.decorators

.. autosummary::
   :toctree: generated/

   bqm_index_labels
   bqm_index_labelled_input
   bqm_structured
   graph_argument
   vartype_argument


Energy Calculations
===================

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   ising_energy
   qubo_energy


Graph-like
==========

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   child_structure_dfs


.. _serialization_dimod:

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

.. autosummary::
   :toctree: generated/

   FileView

JSON
----

.. currentmodule:: dimod.serialization.json
.. automodule:: dimod.serialization.json

.. autosummary::
   :toctree: generated/

   DimodEncoder
   DimodDecoder
   dimod_object_hook


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


Vartype Conversion
==================

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   ising_to_qubo
   qubo_to_ising
