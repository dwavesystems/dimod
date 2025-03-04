.. _dimod_samplers_composites:

=======================
Samplers and Composites
=======================

.. _dimod_samplers:

Samplers
========

The `dimod` package includes several example samplers.

Other Ocean packages provide production samplers; for example, the
:std:doc:`dwave-system <oceandocs:docs_system/sdk_index>` package provides
:std:doc:`samplers for D-Wave systems <oceandocs:docs_system/reference/samplers>`
and :std:doc:`dwave-neal <oceandocs:docs_neal/sdk_index>` provides
a simulated-annealing sampler.

.. automodule:: dimod.reference.samplers
.. currentmodule:: dimod.reference.samplers

Exact Solver
------------

.. autoclass:: ExactSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ExactSolver.sample
   ExactSolver.sample_ising
   ExactSolver.sample_qubo

Exact CQM Solver
----------------

.. autoclass:: ExactCQMSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ExactCQMSolver.sample_cqm

Exact DQM Solver
----------------

.. autoclass:: ExactDQMSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ExactDQMSolver.sample_dqm

Identity Sampler
----------------

.. autoclass:: IdentitySampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   IdentitySampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   IdentitySampler.sample
   IdentitySampler.sample_ising
   IdentitySampler.sample_qubo

Null Sampler
------------

.. autoclass:: NullSampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   NullSampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   NullSampler.sample
   NullSampler.sample_ising
   NullSampler.sample_qubo

Random Sampler
--------------

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

.. tip:: :obj:`neal.sampler.SimulatedAnnealingSampler` is a more performant
  implementation of simulated annealing you can use for solving problems.

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


.. _dimod_higher_order_samplers:

Higher-Order Samplers
=====================

The `dimod` package includes the following example higher-order samplers.

.. currentmodule:: dimod.reference.samplers

Exact Polynomial Solver
-----------------------

A simple exact solver for testing and debugging code using your local CPU.

Note:
    This sampler is designed for use in testing. Because it calculates the
    energy for every possible sample, it is very slow.

Class
~~~~~

.. autoclass:: ExactPolySolver

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   ExactPolySolver.sample_hising
   ExactPolySolver.sample_hubo
   ExactPolySolver.sample_poly

.. _dimod_composites:

Composites
==========

The `dimod` package includes several example composed samplers:

.. currentmodule:: dimod.reference.composites

The :std:doc:`dwave-system <oceandocs:docs_system/sdk_index>` package provides
additional :std:doc:`composites for D-Wave systems <oceandocs:docs_system/reference/composites>`
such as those used for :term:`minor-embedding`.

Structure Composite
-------------------

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

Tracking Composite
------------------

.. automodule:: dimod.reference.composites.tracking

Class
~~~~~

.. autoclass:: TrackingComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   TrackingComposite.input
   TrackingComposite.inputs
   TrackingComposite.output
   TrackingComposite.outputs
   TrackingComposite.parameters
   TrackingComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   TrackingComposite.clear
   TrackingComposite.sample
   TrackingComposite.sample_ising
   TrackingComposite.sample_qubo

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

.. _dimod_higher_order_composites:

Higher-Order Composites
=======================

The `dimod` package includes several example higher-order composed samplers.

.. currentmodule:: dimod.reference.composites

HigherOrderComposite
--------------------

.. autoclass:: HigherOrderComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   HigherOrderComposite.child
   HigherOrderComposite.children
   HigherOrderComposite.parameters
   HigherOrderComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   HigherOrderComposite.sample_poly
   HigherOrderComposite.sample_hising
   HigherOrderComposite.sample_hubo

PolyFixedVariableComposite
--------------------------

.. autoclass:: PolyFixedVariableComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyFixedVariableComposite.child
   PolyFixedVariableComposite.children
   PolyFixedVariableComposite.parameters
   PolyFixedVariableComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyFixedVariableComposite.sample_poly
   PolyFixedVariableComposite.sample_hising
   PolyFixedVariableComposite.sample_hubo

PolyScaleComposite
------------------

.. autoclass:: PolyScaleComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyScaleComposite.child
   PolyScaleComposite.children
   PolyScaleComposite.parameters
   PolyScaleComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyScaleComposite.sample_poly
   PolyScaleComposite.sample_hising
   PolyScaleComposite.sample_hubo

PolyTruncateComposite
---------------------

.. autoclass:: PolyTruncateComposite

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyTruncateComposite.child
   PolyTruncateComposite.children
   PolyTruncateComposite.parameters
   PolyTruncateComposite.properties

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/

   PolyTruncateComposite.sample_poly
   PolyTruncateComposite.sample_hising
   PolyTruncateComposite.sample_hubo