.. _sampleset:

=========
SampleSet
=========

.. automodule:: dimod.sampleset

Class
=====

.. currentmodule:: dimod
.. autoclass:: SampleSet

Properties
==========

.. autosummary::
   :toctree: generated/

   SampleSet.record
   SampleSet.variables
   SampleSet.info
   SampleSet.vartype
   SampleSet.first


Methods
=======

Viewing a Response
------------------

.. autosummary::
   :toctree: generated/

   SampleSet.samples
   SampleSet.data

Constructing a Response
-----------------------

.. autosummary::
   :toctree: generated/

   SampleSet.from_samples
   SampleSet.from_future
   SampleSet.from_response

Transformations
---------------

.. autosummary::
   :toctree: generated/

   SampleSet.change_vartype
   SampleSet.relabel_variables

Copy
----

.. autosummary::
   :toctree: generated/

   SampleSet.copy

Done
----

.. autosummary::
   :toctree: generated/

   SampleSet.done
