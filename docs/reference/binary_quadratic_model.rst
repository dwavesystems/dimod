.. _bqm:

=======================================
Ising, QUBO and Binary Quadratic Models
=======================================

.. automodule:: dimod.binary_quadratic_model

Class
=====

.. currentmodule:: dimod
.. autoclass:: BinaryQuadraticModel


Vartype Properties
==================

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.binary
   BinaryQuadraticModel.spin

Methods
=======

Construction Shortcuts
----------------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.empty

Adding and Removing Variables and Interactions
----------------------------------------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.add_variable
   BinaryQuadraticModel.add_variables_from
   BinaryQuadraticModel.add_interaction
   BinaryQuadraticModel.add_interactions_from
   BinaryQuadraticModel.add_offset
   BinaryQuadraticModel.remove_variable
   BinaryQuadraticModel.remove_variables_from
   BinaryQuadraticModel.remove_interaction
   BinaryQuadraticModel.remove_interactions_from
   BinaryQuadraticModel.remove_offset
   BinaryQuadraticModel.update

Transformations
---------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.contract_variables
   BinaryQuadraticModel.fix_variable
   BinaryQuadraticModel.fix_variables
   BinaryQuadraticModel.flip_variable
   BinaryQuadraticModel.relabel_variables
   BinaryQuadraticModel.scale

Change Vartype
--------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.change_vartype

Copy
----

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.copy

Energy
------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.energy
   BinaryQuadraticModel.energies

Converting to other types
-------------------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.from_coo
   BinaryQuadraticModel.from_ising
   BinaryQuadraticModel.from_numpy_matrix
   BinaryQuadraticModel.from_numpy_vectors
   BinaryQuadraticModel.from_qubo
   BinaryQuadraticModel.from_pandas_dataframe
   BinaryQuadraticModel.from_serializable
   BinaryQuadraticModel.to_coo
   BinaryQuadraticModel.to_ising
   BinaryQuadraticModel.to_networkx_graph
   BinaryQuadraticModel.to_numpy_matrix
   BinaryQuadraticModel.to_numpy_vectors
   BinaryQuadraticModel.to_qubo
   BinaryQuadraticModel.to_pandas_dataframe
   BinaryQuadraticModel.to_serializable

Alias
=====
.. autoclass:: BQM
