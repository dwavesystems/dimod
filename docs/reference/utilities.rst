.. _utilities_dimod:

=========
Utilities
=========

.. contents::
    :depth: 3

Energy Calculations
===================

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   ising_energy
   qubo_energy

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


Graph-like
==========

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   child_structure_dfs


Serialization
=============

JSON
----
.. currentmodule:: dimod.serialization.json
.. automodule:: dimod.serialization.json

.. autoclass:: DimodEncoder

.. autoclass:: DimodDecoder

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated/

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




Vartype Conversion
==================

.. currentmodule:: dimod.utilities

.. autosummary::
   :toctree: generated/

   ising_to_qubo
   qubo_to_ising
