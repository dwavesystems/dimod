.. _quadratic_samplers:

========
Samplers
========

The `dimod` package includes several example samplers.

.. contents::
    :local:
    :depth: 1

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
   :toctree: ../generated/

   ExactSolver.sample
   ExactSolver.sample_ising
   ExactSolver.sample_qubo

Exact CQM Solver
----------------

.. autoclass:: ExactCQMSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   ExactCQMSolver.sample_cqm

Exact DQM Solver
----------------

.. autoclass:: ExactDQMSolver

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   ExactDQMSolver.sample_dqm

Identity Sampler
----------------

.. autoclass:: IdentitySampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   IdentitySampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   IdentitySampler.sample
   IdentitySampler.sample_ising
   IdentitySampler.sample_qubo

Null Sampler
------------

.. autoclass:: NullSampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   NullSampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   NullSampler.sample
   NullSampler.sample_ising
   NullSampler.sample_qubo

Random Sampler
--------------

.. autoclass:: RandomSampler

Properties
~~~~~~~~~~

.. autosummary::
   :toctree: ../generated/

   RandomSampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

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
   :toctree: ../generated/

   SimulatedAnnealingSampler.parameters

Methods
~~~~~~~

.. autosummary::
   :toctree: ../generated/

   SimulatedAnnealingSampler.sample
   SimulatedAnnealingSampler.sample_ising
   SimulatedAnnealingSampler.sample_qubo
