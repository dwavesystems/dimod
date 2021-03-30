.. _dqm:

=========================
Discrete Quadratic Models
=========================

:term:`Discrete quadratic model`\ s (DQMs) are described under
:std:doc:`Discrete Quadratic Models <oceandocs:concepts/dqm>`.

DQM Class
=========

.. currentmodule:: dimod

.. autoclass:: DiscreteQuadraticModel

Attributes
----------

.. autosummary::
   :toctree: generated/

   ~DiscreteQuadraticModel.adj
   ~DiscreteQuadraticModel.variables

Methods
-------

.. autosummary::
   :toctree: generated/

   ~DiscreteQuadraticModel.add_linear_equality_constraint
   ~DiscreteQuadraticModel.add_variable
   ~DiscreteQuadraticModel.copy
   ~DiscreteQuadraticModel.from_file
   ~DiscreteQuadraticModel.from_numpy_vectors
   ~DiscreteQuadraticModel.get_linear
   ~DiscreteQuadraticModel.get_linear_case
   ~DiscreteQuadraticModel.get_quadratic
   ~DiscreteQuadraticModel.get_quadratic_case
   ~DiscreteQuadraticModel.num_cases
   ~DiscreteQuadraticModel.num_case_interactions
   ~DiscreteQuadraticModel.num_variable_interactions
   ~DiscreteQuadraticModel.num_variables
   ~DiscreteQuadraticModel.relabel_variables
   ~DiscreteQuadraticModel.relabel_variables_as_integers
   ~DiscreteQuadraticModel.set_linear
   ~DiscreteQuadraticModel.set_linear_case
   ~DiscreteQuadraticModel.set_quadratic
   ~DiscreteQuadraticModel.set_quadratic_case
   ~DiscreteQuadraticModel.to_file
   ~DiscreteQuadraticModel.to_numpy_vectors