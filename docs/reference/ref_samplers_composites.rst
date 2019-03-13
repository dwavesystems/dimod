.. _ref_samplers:

=================================
Reference Samplers and Composites
=================================

The `dimod` package includes several example samplers and composed samplers.

Samplers
========

.. attention:: The reference samplers included in the `dimod` package are intended as an aid for
    coding and testing; they are not optimized for performance or intended for benchmarking.

.. automodule:: dimod.reference.samplers

Exact Sampler
-------------

.. automodule:: dimod.reference.samplers.exact_solver

Class
~~~~~

.. currentmodule:: dimod.reference.samplers
.. autoclass:: ExactSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ExactSolver.sample

Simulated Annealing
-------------------

.. automodule:: dimod.reference.samplers.simulated_annealing

Class
~~~~~

.. currentmodule:: dimod.reference.samplers
.. autoclass:: SimulatedAnnealingSampler

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   SimulatedAnnealingSampler.sample

Random Sampler
--------------

.. automodule:: dimod.reference.samplers.random_sampler

Class
~~~~~

.. currentmodule:: dimod.reference.samplers
.. autoclass:: RandomSampler

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   RandomSampler.sample

Composites
==========

.. currentmodule:: dimod.reference.composites

.. automodule:: dimod.reference.composites


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
