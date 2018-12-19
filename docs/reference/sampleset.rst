.. _sampleset:

=======
Samples
=======

This module is under development as a possible replacement for :obj:`.Response`.

sample_like Objects
===================

.. automodule:: dimod.sampleset

.. autosummary::
   :toctree: generated/

   as_samples

SampleSet
=========

Class
-----

.. currentmodule:: dimod
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

Done
~~~~

.. autosummary::
   :toctree: generated/

   SampleSet.done
