.. _sampleset:

=======
Samples
=======

dimod :term:`sampler`\ s sample from a problem's :term:`objective function`, such
as a BQM, and return an iterable of samples contained in a :class:`.SampleSet` class.
In addition to containing the returned solutions and some additional
information, and providing methods to work with the solutions, :class:`.SampleSet`
is also used, for example, by :std:doc:`dwave-hybrid <hybrid:index>`,
which iterates sets of samples through samplers to solve arbitrary QUBOs. dimod
provides functionality for creating and manipulating samples.

sample_like Objects
===================

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   as_samples

SampleSet
=========

.. autoclass:: SampleSet

Properties
----------

.. autosummary::
   :toctree: generated/

   SampleSet.first
   SampleSet.info
   SampleSet.record
   SampleSet.variables
   SampleSet.vartype


Methods
-------

.. autosummary::
   :toctree: generated/

   SampleSet.aggregate
   SampleSet.append_variables
   SampleSet.change_vartype
   SampleSet.copy
   SampleSet.data
   SampleSet.done
   SampleSet.from_future
   SampleSet.from_samples
   SampleSet.from_samples_bqm
   SampleSet.from_serializable
   SampleSet.lowest
   SampleSet.resolve
   SampleSet.relabel_variables
   SampleSet.samples
   SampleSet.slice
   SampleSet.to_pandas_dataframe
   SampleSet.to_serializable
   SampleSet.truncate

Utility Functions
=================

.. autosummary::
   :toctree: generated/

   concatenate
