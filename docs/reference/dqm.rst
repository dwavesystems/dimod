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
   ~DiscreteQuadraticModel.get_cases
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

CaseLabelDQM Class
==================

.. currentmodule:: dimod

.. autoclass:: CaseLabelDQM

Methods
-------

.. autosummary::
   :toctree: generated/

   ~CaseLabelDQM.add_variable
   ~CaseLabelDQM.get_cases
   ~CaseLabelDQM.get_linear
   ~CaseLabelDQM.get_linear_case
   ~CaseLabelDQM.get_quadratic
   ~CaseLabelDQM.get_quadratic_case
   ~CaseLabelDQM.map_sample
   ~CaseLabelDQM.set_linear
   ~CaseLabelDQM.set_linear_case
   ~CaseLabelDQM.set_quadratic
   ~CaseLabelDQM.set_quadratic_case
