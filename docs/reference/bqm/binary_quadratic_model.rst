.. _bqm:

=====================
Ising, QUBO, and BQMs
=====================

.. automodule:: dimod.binary_quadratic_model

These models and their use in solving problems on the D-Wave system is described
in the following documentation:

* :std:doc:`Getting Started with the D-Wave System <sysdocs_gettingstarted:doc_getting_started>`
   Introduces key concepts such as objective functions, Ising model, QUBOs, and graphs, explains
   how these models are used to represent problems, and provides some simple examples.
* :std:doc:`D-Wave Problem-Solving Handbook <sysdocs_gettingstarted:doc_handbook>`
   Provides a variety of techniques for, and examples of, reformulating problems as BQMs.
* :std:doc:`Solving Problems on a D-Wave System <oceandocs:overview/solving_problems>`
   Describes and demonstrates the use of BQM in the context of Ocean software.

Class
=====

.. currentmodule:: dimod
.. autoclass:: BinaryQuadraticModel


Vartype Properties
==================

QUBO (binary-valued variables) and Ising (spin-valued variables) instances of a BQM.

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
   BinaryQuadraticModel.normalize
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

Converting To and From Other Formats
------------------------------------

.. autosummary::
   :toctree: generated/

   BinaryQuadraticModel.from_coo
   BinaryQuadraticModel.from_ising
   BinaryQuadraticModel.from_networkx_graph
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

.. _COOrdinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

Alias
=====
.. autoclass:: BQM
