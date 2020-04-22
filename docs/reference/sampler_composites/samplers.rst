.. _quadratic_samplers:

========
Samplers
========

The `dimod` package includes several example samplers.

.. contents::
    :depth: 3

.. automodule:: dimod.reference.samplers
.. currentmodule:: dimod.reference.samplers

Exact Solver
------------

A simple exact solver for testing and debugging code using your local CPU.

Note:
    This sampler is designed for use in testing. Because it calculates the
    energy for every possible sample, it is very slow.


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

Identity Sampler
----------------

.. automodule:: dimod.reference.samplers.identity_sampler

Class
~~~~~

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

.. automodule:: dimod.reference.samplers.null_sampler

Class
~~~~~

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
