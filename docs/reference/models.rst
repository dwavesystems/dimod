.. _dimod_models:

============================
Models: BQM, CQM, QM, Others
============================

This page describes the `dimod` package's quadratic models: classes,
attributes, and methods. For an introduction and the data structure, see :ref:`intro_models`.

.. contents::
    :local:
    :depth: 3

.. _bqm:

Binary Quadratic Models
=======================

.. automodule:: dimod.binary.binary_quadratic_model

For examples, see
`Ocean's Getting Started examples <https://docs.ocean.dwavesys.com/en/stable/getting_started.html#examples>`_
and the BQM examples in `D-Wave's collection of examples <https://github.com/dwave-examples>`_.

BQM Class
---------

.. currentmodule:: dimod.binary

.. autoclass:: BinaryQuadraticModel

.. autoclass:: BQM

BQM Attributes
--------------

.. autosummary::
   :toctree: generated/

   ~BinaryQuadraticModel.adj
   ~BinaryQuadraticModel.binary
   ~BinaryQuadraticModel.dtype
   ~BinaryQuadraticModel.linear
   ~BinaryQuadraticModel.num_interactions
   ~BinaryQuadraticModel.num_variables
   ~BinaryQuadraticModel.offset
   ~BinaryQuadraticModel.quadratic
   ~BinaryQuadraticModel.shape
   ~BinaryQuadraticModel.spin
   ~BinaryQuadraticModel.variables
   ~BinaryQuadraticModel.vartype

BQM Methods
-----------

.. autosummary::
   :toctree: generated/

   ~BinaryQuadraticModel.add_linear
   ~BinaryQuadraticModel.add_linear_equality_constraint
   ~BinaryQuadraticModel.add_linear_from
   ~BinaryQuadraticModel.add_linear_from_array
   ~BinaryQuadraticModel.add_linear_inequality_constraint
   ~BinaryQuadraticModel.add_quadratic
   ~BinaryQuadraticModel.add_quadratic_from
   ~BinaryQuadraticModel.add_quadratic_from_dense
   ~BinaryQuadraticModel.add_variable
   ~BinaryQuadraticModel.change_vartype
   ~BinaryQuadraticModel.clear
   ~BinaryQuadraticModel.contract_variables
   ~BinaryQuadraticModel.copy
   ~BinaryQuadraticModel.degree
   ~BinaryQuadraticModel.degrees
   ~BinaryQuadraticModel.empty
   ~BinaryQuadraticModel.energies
   ~BinaryQuadraticModel.energy
   ~BinaryQuadraticModel.fix_variable
   ~BinaryQuadraticModel.fix_variables
   ~BinaryQuadraticModel.flip_variable
   ~BinaryQuadraticModel.from_coo
   ~BinaryQuadraticModel.from_file
   ~BinaryQuadraticModel.from_ising
   ~BinaryQuadraticModel.from_numpy_vectors
   ~BinaryQuadraticModel.from_qubo
   ~BinaryQuadraticModel.from_serializable
   ~BinaryQuadraticModel.is_almost_equal
   ~BinaryQuadraticModel.is_equal
   ~BinaryQuadraticModel.is_linear
   ~BinaryQuadraticModel.iter_linear
   ~BinaryQuadraticModel.iter_neighborhood
   ~BinaryQuadraticModel.get_linear
   ~BinaryQuadraticModel.get_quadratic
   ~BinaryQuadraticModel.maximum_energy_delta
   ~BinaryQuadraticModel.nbytes
   ~BinaryQuadraticModel.normalize
   ~BinaryQuadraticModel.reduce_linear
   ~BinaryQuadraticModel.reduce_neighborhood
   ~BinaryQuadraticModel.reduce_quadratic
   ~BinaryQuadraticModel.relabel_variables
   ~BinaryQuadraticModel.relabel_variables_as_integers
   ~BinaryQuadraticModel.remove_interaction
   ~BinaryQuadraticModel.remove_interactions_from
   ~BinaryQuadraticModel.remove_variable
   ~BinaryQuadraticModel.resize
   ~BinaryQuadraticModel.scale
   ~BinaryQuadraticModel.set_linear
   ~BinaryQuadraticModel.set_quadratic
   ~BinaryQuadraticModel.to_coo
   ~BinaryQuadraticModel.to_file
   ~BinaryQuadraticModel.to_ising
   ~BinaryQuadraticModel.to_numpy_vectors
   ~BinaryQuadraticModel.to_polystring
   ~BinaryQuadraticModel.to_qubo
   ~BinaryQuadraticModel.to_serializable
   ~BinaryQuadraticModel.update

Generic BQM Constructor
-----------------------

.. currentmodule:: dimod.binary

.. autosummary::
   :toctree: generated/

   as_bqm

.. _cqm:

Constrained Quadratic Model
===========================

.. automodule:: dimod.constrained.constrained

For examples, see
`Ocean's Getting Started examples <https://docs.ocean.dwavesys.com/en/stable/getting_started.html#examples>`_
and the CQM examples in `D-Wave's collection of examples <https://github.com/dwave-examples>`_.

CQM Class
---------

.. currentmodule:: dimod

.. autoclass:: ConstrainedQuadraticModel

.. autoclass:: CQM

CQM Attributes
--------------

.. autosummary::
   :toctree: generated/

   ~ConstrainedQuadraticModel.constraints
   ~ConstrainedQuadraticModel.objective
   ~ConstrainedQuadraticModel.variables

CQM Methods
-----------

.. autosummary::
   :toctree: generated/

   ~ConstrainedQuadraticModel.add_constraint
   ~ConstrainedQuadraticModel.add_constraint_from_comparison
   ~ConstrainedQuadraticModel.add_constraint_from_iterable
   ~ConstrainedQuadraticModel.add_constraint_from_model
   ~ConstrainedQuadraticModel.add_discrete
   ~ConstrainedQuadraticModel.add_discrete_from_comparison
   ~ConstrainedQuadraticModel.add_discrete_from_iterable
   ~ConstrainedQuadraticModel.add_discrete_from_model
   ~ConstrainedQuadraticModel.add_variable
   ~ConstrainedQuadraticModel.add_variables
   ~ConstrainedQuadraticModel.check_feasible
   ~ConstrainedQuadraticModel.fix_variable
   ~ConstrainedQuadraticModel.fix_variables
   ~ConstrainedQuadraticModel.flip_variable
   ~ConstrainedQuadraticModel.from_bqm
   ~ConstrainedQuadraticModel.from_discrete_quadratic_model
   ~ConstrainedQuadraticModel.from_dqm
   ~ConstrainedQuadraticModel.from_qm
   ~ConstrainedQuadraticModel.from_quadratic_model
   ~ConstrainedQuadraticModel.from_file
   ~ConstrainedQuadraticModel.from_lp_file
   ~ConstrainedQuadraticModel.iter_constraint_data
   ~ConstrainedQuadraticModel.iter_violations
   ~ConstrainedQuadraticModel.is_almost_equal
   ~ConstrainedQuadraticModel.is_equal
   ~ConstrainedQuadraticModel.is_linear
   ~ConstrainedQuadraticModel.lower_bound
   ~ConstrainedQuadraticModel.num_biases
   ~ConstrainedQuadraticModel.num_quadratic_variables
   ~ConstrainedQuadraticModel.relabel_constraints
   ~ConstrainedQuadraticModel.relabel_variables
   ~ConstrainedQuadraticModel.remove_constraint
   ~ConstrainedQuadraticModel.set_lower_bound
   ~ConstrainedQuadraticModel.set_objective
   ~ConstrainedQuadraticModel.set_upper_bound
   ~ConstrainedQuadraticModel.spin_to_binary
   ~ConstrainedQuadraticModel.substitute_self_loops
   ~ConstrainedQuadraticModel.to_file
   ~ConstrainedQuadraticModel.upper_bound
   ~ConstrainedQuadraticModel.vartype
   ~ConstrainedQuadraticModel.violations

Sense Class
-----------

.. currentmodule:: dimod.sym

.. autoclass:: Sense

.. _qm:

Quadratic Models
================

.. automodule:: dimod.quadratic.quadratic_model

For examples, see
`Ocean's Getting Started examples <https://docs.ocean.dwavesys.com/en/stable/getting_started.html#examples>`_
and the examples in `D-Wave's collection of examples <https://github.com/dwave-examples>`_.


QM Class
--------

.. currentmodule:: dimod

.. autoclass:: QuadraticModel

.. autoclass:: QM

QM Attributes
-------------

.. autosummary::
   :toctree: generated/

   ~QuadraticModel.adj
   ~QuadraticModel.dtype
   ~QuadraticModel.linear
   ~QuadraticModel.num_interactions
   ~QuadraticModel.num_variables
   ~QuadraticModel.offset
   ~QuadraticModel.quadratic
   ~QuadraticModel.shape
   ~QuadraticModel.variables

QM Methods
----------

.. autosummary::
   :toctree: generated/

   ~QuadraticModel.add_linear
   ~QuadraticModel.add_linear_from
   ~QuadraticModel.add_quadratic
   ~QuadraticModel.add_quadratic_from
   ~QuadraticModel.add_variable
   ~QuadraticModel.add_variables_from
   ~QuadraticModel.add_variables_from_model
   ~QuadraticModel.change_vartype
   ~QuadraticModel.clear
   ~QuadraticModel.copy
   ~QuadraticModel.degree
   ~QuadraticModel.energies
   ~QuadraticModel.energy
   ~QuadraticModel.fix_variable
   ~QuadraticModel.fix_variables
   ~QuadraticModel.flip_variable
   ~QuadraticModel.from_bqm
   ~QuadraticModel.from_file
   ~QuadraticModel.get_linear
   ~QuadraticModel.get_quadratic
   ~QuadraticModel.is_almost_equal
   ~QuadraticModel.is_equal
   ~QuadraticModel.is_linear
   ~QuadraticModel.iter_linear
   ~QuadraticModel.iter_neighborhood
   ~QuadraticModel.iter_quadratic
   ~QuadraticModel.lower_bound
   ~QuadraticModel.nbytes
   ~QuadraticModel.set_lower_bound
   ~QuadraticModel.set_upper_bound
   ~QuadraticModel.reduce_linear
   ~QuadraticModel.reduce_neighborhood
   ~QuadraticModel.reduce_quadratic
   ~QuadraticModel.relabel_variables
   ~QuadraticModel.relabel_variables_as_integers
   ~QuadraticModel.remove_interaction
   ~QuadraticModel.remove_variable
   ~QuadraticModel.scale
   ~QuadraticModel.set_linear
   ~QuadraticModel.set_quadratic
   ~QuadraticModel.spin_to_binary
   ~QuadraticModel.to_file
   ~QuadraticModel.to_polystring
   ~QuadraticModel.update
   ~QuadraticModel.upper_bound
   ~QuadraticModel.vartype

Additional Models
=================

.. _dqm:

Discrete Quadratic Models
-------------------------

For an introduction to DQMs, see
:std:doc:`Concepts: Discrete Quadratic Models <oceandocs:concepts/dqm>`.

DQM Class
~~~~~~~~~

.. currentmodule:: dimod

.. autoclass:: DiscreteQuadraticModel

DQM Attributes
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ~DiscreteQuadraticModel.adj
   ~DiscreteQuadraticModel.variables

DQM Methods
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ~DiscreteQuadraticModel.add_linear_equality_constraint
   ~DiscreteQuadraticModel.add_variable
   ~DiscreteQuadraticModel.copy
   ~DiscreteQuadraticModel.energies
   ~DiscreteQuadraticModel.energy
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
~~~~~~~~~~~~~~~~~~

.. currentmodule:: dimod

.. autoclass:: CaseLabelDQM

CaseLabelDQM Methods
~~~~~~~~~~~~~~~~~~~~

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

.. _higher_order:

Higher-Order Models
-------------------

Sometimes it is nice to work with problems that are not restricted to quadratic
interactions.

Binary Polynomial Class
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dimod.higherorder.polynomial

.. autoclass:: BinaryPolynomial

Binary Polynomial Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   BinaryPolynomial.copy
   BinaryPolynomial.energies
   BinaryPolynomial.energy
   BinaryPolynomial.from_hising
   BinaryPolynomial.from_hubo
   BinaryPolynomial.normalize
   BinaryPolynomial.relabel_variables
   BinaryPolynomial.scale
   BinaryPolynomial.to_binary
   BinaryPolynomial.to_hising
   BinaryPolynomial.to_hubo
   BinaryPolynomial.to_spin
