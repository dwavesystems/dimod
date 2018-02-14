.. _bqm:

=======================================
Ising, QUBO and Binary Quadratic Models
=======================================

Overview
========
.. automodule:: dimod.binary_quadratic_model

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

Adding and removing variables and interactions
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

Conversion Functions
====================

.. automodule:: dimod.binary_quadratic_model_convert

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   from_ising
   from_numpy_matrix
   from_qubo
   from_pandas_dataframe
   to_ising
   to_networkx_graph
   to_numpy_matrix
   to_qubo
   to_pandas_dataframe