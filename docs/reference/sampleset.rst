.. _sampleset:

=======
Samples
=======

sample_like Objects
===================

.. currentmodule:: dimod

.. autosummary::
   :toctree: generated/

   as_samples

Utility Functions
=================

.. autosummary::
   :toctree: generated/

   concatenate

SampleSet
=========

Class
-----

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
   SampleSet.relabel_variables
   SampleSet.samples
   SampleSet.to_pandas_dataframe
   SampleSet.to_serializable
   SampleSet.truncate
