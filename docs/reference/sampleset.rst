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

SampleSet
=========

Class
-----

.. autoclass:: SampleSet

Properties
----------

.. autosummary::
   :toctree: generated/

   SampleSet.record
   SampleSet.variables
   SampleSet.info
   SampleSet.vartype
   SampleSet.first


Methods
-------

Viewing a Response
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.samples
   SampleSet.data

Constructing a Response
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.from_samples
   SampleSet.from_samples_bqm
   SampleSet.from_future

Transformations
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.change_vartype
   SampleSet.relabel_variables

Copy
~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.copy

Aggregate
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.aggregate

Done
~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.done

Utility Functions
=================

.. autosummary::
   :toctree: generated/

   concatenate
