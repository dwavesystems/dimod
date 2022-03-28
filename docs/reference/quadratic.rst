.. _unconstrained_qm:

===============================
Quadratic Models: Unconstrained
===============================

This page describes the `dimod` package's unconstrained quadratic models: classes,
attributes, methods, and some functions related to these classes. For an introduction
and the data structure, see :ref:`intro_qm`.

For reference documentation of **constrained** quadratic models, see
:ref:`cqm`.

.. contents::
    :local:
    :depth: 3

.. _qm:

Quadratic Models
================

For an introduction to QMs, see
:std:doc:`Concepts: Quadratic Models <oceandocs:concepts/qm>`.

QM Class
--------

.. currentmodule:: dimod

.. autoclass:: QuadraticModel

.. autoclass:: QM

Attributes
~~~~~~~~~~

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

Methods
~~~~~~~

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

.. _bqm:

Binary Quadratic Models
=======================

For an introduction to BQMs, see
:std:doc:`Concepts: Binary Quadratic Models <oceandocs:concepts/bqm>`.

BQM Class
---------

.. currentmodule:: dimod.binary

.. autoclass:: BinaryQuadraticModel

.. autoclass:: BQM

Attributes
~~~~~~~~~~

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

Methods
~~~~~~~

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


BQM Functions
-------------

Generic constructor:

.. currentmodule:: dimod.binary

.. autosummary::
   :toctree: generated/

   as_bqm

Adding models symbolically:

.. currentmodule:: dimod.binary

.. autosummary::
   :toctree: generated/

   quicksum

Fixing variables:

.. currentmodule:: dimod.roof_duality

.. autosummary::
   :toctree: generated/

   fix_variables

Traversing as a graph:

.. currentmodule:: dimod.traversal

.. autosummary::
   :toctree: generated/

   connected_components
   bfs_variables


Converting to and from other data structures:

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   to_networkx_graph
   from_networkx_graph

See also: :ref:`serialization functions<serialization_dimod>`

.. _dqm:

Discrete Quadratic Models
=========================

For an introduction to DQMs, see
:std:doc:`Concepts: Discrete Quadratic Models <oceandocs:concepts/dqm>`.

DQM Class
---------

.. currentmodule:: dimod

.. autoclass:: DiscreteQuadraticModel

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ~DiscreteQuadraticModel.adj
   ~DiscreteQuadraticModel.variables

Methods
~~~~~~~

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
------------------

.. currentmodule:: dimod

.. autoclass:: CaseLabelDQM

Methods
~~~~~~~

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
