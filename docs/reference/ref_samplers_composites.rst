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

.. automodule:: dimod.reference.composites

Spin Reversal Transform Composite
---------------------------------

.. automodule:: dimod.reference.composites.spin_transform

Class
~~~~~

.. currentmodule:: dimod.reference.composites
.. autoclass:: SpinReversalTransformComposite

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   SpinReversalTransformComposite.sample

Structured Composite
--------------------

.. automodule:: dimod.reference.composites.structure

Class
~~~~~

.. currentmodule:: dimod.reference.composites
.. autoclass:: StructureComposite

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   StructureComposite.sample

Higher Order Composite
----------------------

.. automodule:: dimod.reference.composites.higherordercomposite

Class
~~~~~

.. currentmodule:: dimod.reference.composites
.. autoclass:: HigherOrderComposite

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   HigherOrderComposite.sample_ising

Fixed Variable Composite
------------------------

.. automodule:: dimod.reference.composites.fixedvariable

Class
~~~~~

.. currentmodule:: dimod.reference.composites
.. autoclass:: FixedVariableComposite

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   FixedVariableComposite.sample

Scale Composite
---------------

.. automodule:: dimod.reference.composites.scalecomposite

Class
~~~~~

.. currentmodule:: dimod.reference.composites
.. autoclass:: ScaleComposite

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ScaleComposite.sample
   ScaleComposite.sample_ising
