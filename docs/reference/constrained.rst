.. _cqm:

=============================
Quadratic Models: Constrained
=============================

CQM Class
=========

.. currentmodule:: dimod

.. autoclass:: ConstrainedQuadraticModel

.. autoclass:: CQM

Attributes
----------

.. autosummary::
   :toctree: generated/

   ConstrainedQuadraticModel.constraints
   ConstrainedQuadraticModel.objective
   ConstrainedQuadraticModel.variables

Methods
-------

.. autosummary::
   :toctree: generated/

   ConstrainedQuadraticModel.add_constraint
   ConstrainedQuadraticModel.add_constraint_from_comparison
   ConstrainedQuadraticModel.add_constraint_from_iterable
   ConstrainedQuadraticModel.add_constraint_from_model
   ConstrainedQuadraticModel.add_discrete
   ConstrainedQuadraticModel.add_discrete_from_comparison
   ConstrainedQuadraticModel.add_discrete_from_iterable
   ConstrainedQuadraticModel.add_discrete_from_model
   ConstrainedQuadraticModel.add_variable
   ConstrainedQuadraticModel.check_feasible
   ConstrainedQuadraticModel.fix_variable
   ConstrainedQuadraticModel.fix_variables
   ConstrainedQuadraticModel.flip_variable
   ConstrainedQuadraticModel.from_bqm
   ConstrainedQuadraticModel.from_discrete_quadratic_model
   ConstrainedQuadraticModel.from_dqm
   ConstrainedQuadraticModel.from_qm
   ConstrainedQuadraticModel.from_quadratic_model
   ConstrainedQuadraticModel.from_file
   ConstrainedQuadraticModel.from_lp_file
   ConstrainedQuadraticModel.iter_constraint_data
   ConstrainedQuadraticModel.iter_violations
   ConstrainedQuadraticModel.is_almost_equal
   ConstrainedQuadraticModel.is_equal
   ConstrainedQuadraticModel.num_biases
   ConstrainedQuadraticModel.num_quadratic_variables
   ConstrainedQuadraticModel.relabel_constraints
   ConstrainedQuadraticModel.relabel_variables
   ConstrainedQuadraticModel.remove_constraint
   ConstrainedQuadraticModel.set_lower_bound
   ConstrainedQuadraticModel.set_objective
   ConstrainedQuadraticModel.set_upper_bound
   ConstrainedQuadraticModel.substitute_self_loops
   ConstrainedQuadraticModel.to_file
   ConstrainedQuadraticModel.vartype
   ConstrainedQuadraticModel.violations


Sense Class
===========

.. currentmodule:: dimod.sym

.. autoclass:: Sense

Functions
=========

Converting constrained quadratic models to other model types:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   cqm_to_bqm
