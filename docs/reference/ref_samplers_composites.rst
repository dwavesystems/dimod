.. _ref_samplers:

=======================
Samplers and Composites
=======================

The `dimod` package includes several example samplers and composed samplers.

.. contents::
    :depth: 3

Samplers
========

.. automodule:: dimod.reference.samplers
.. currentmodule:: dimod.reference.samplers

Exact Solver
------------

.. automodule:: dimod.reference.samplers.exact_solver

Class
~~~~~

.. autoclass:: ExactSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ExactSolver.sample
   ExactSolver.sample_ising
   ExactSolver.sample_qubo

Random Sampler
--------------

.. automodule:: dimod.reference.samplers.random_sampler

Class
~~~~~

.. autoclass:: RandomSampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   RandomSampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   RandomSampler.sample
   RandomSampler.sample_ising
   RandomSampler.sample_qubo

Simulated Annealing Sampler
---------------------------

.. automodule:: dimod.reference.samplers.simulated_annealing

Class
~~~~~

.. autoclass:: SimulatedAnnealingSampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SimulatedAnnealingSampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   SimulatedAnnealingSampler.sample
   SimulatedAnnealingSampler.sample_ising
   SimulatedAnnealingSampler.sample_qubo


Composites
==========

.. currentmodule:: dimod.reference.composites

.. automodule:: dimod.reference.composites


Fixed Variable Composite
------------------------

.. automodule:: dimod.reference.composites.fixedvariable

Class
~~~~~

.. autoclass:: FixedVariableComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedVariableComposite.child
   FixedVariableComposite.children
   FixedVariableComposite.parameters
   FixedVariableComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedVariableComposite.sample
   FixedVariableComposite.sample_ising
   FixedVariableComposite.sample_qubo


Roof Duality Composite
----------------------

.. automodule:: dimod.reference.composites.roofduality

Class
~~~~~

.. autoclass:: RoofDualityComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   RoofDualityComposite.child
   RoofDualityComposite.children
   RoofDualityComposite.parameters
   RoofDualityComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   RoofDualityComposite.sample
   RoofDualityComposite.sample_ising
   RoofDualityComposite.sample_qubo


Scale Composite
---------------

.. automodule:: dimod.reference.composites.scalecomposite

Class
~~~~~

.. autoclass:: ScaleComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ScaleComposite.child
   ScaleComposite.children
   ScaleComposite.parameters
   ScaleComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ScaleComposite.sample
   ScaleComposite.sample_ising
   ScaleComposite.sample_qubo



Spin Reversal Transform Composite
---------------------------------

.. automodule:: dimod.reference.composites.spin_transform

Class
~~~~~

.. autoclass:: SpinReversalTransformComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SpinReversalTransformComposite.child
   SpinReversalTransformComposite.children
   SpinReversalTransformComposite.parameters
   SpinReversalTransformComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   SpinReversalTransformComposite.sample
   SpinReversalTransformComposite.sample_ising
   SpinReversalTransformComposite.sample_qubo

Structured Composite
--------------------

.. automodule:: dimod.reference.composites.structure

Class
~~~~~

.. autoclass:: StructureComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   StructureComposite.child
   StructureComposite.children
   StructureComposite.parameters
   StructureComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   StructureComposite.sample
   StructureComposite.sample_ising
   StructureComposite.sample_qubo


Truncate Composite
------------------

.. automodule:: dimod.reference.composites.truncatecomposite

Class
~~~~~

.. autoclass:: TruncateComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   TruncateComposite.child
   TruncateComposite.children
   TruncateComposite.parameters
   TruncateComposite.properties


Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   TruncateComposite.sample
   TruncateComposite.sample_ising
   TruncateComposite.sample_qubo
