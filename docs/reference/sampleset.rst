.. _sampleset:

=======
Samples
=======

Returned solutions and samples are described under
:std:doc:`Binary Quadratic Models <oceandocs:concepts/solutions>`.

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
   SampleSet.filter
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

   append_data_vectors
   append_variables
   concatenate
   drop_variables
   keep_variables

Printing
========

.. currentmodule:: dimod.serialization.format

.. autofunction:: Formatter
.. autofunction:: set_printoptions
