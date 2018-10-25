.. _response:

========
Response
========

See also :ref:`sampleset`. 

.. automodule:: dimod.response

Class
=====

.. currentmodule:: dimod
.. autoclass:: Response

Properties
==========

.. autosummary::
   :toctree: generated/

   Response.record
   Response.variable_labels
   Response.label_to_idx
   Response.info
   Response.vartype


Methods
=======

Viewing a Response
------------------

.. autosummary::
   :toctree: generated/

   Response.samples
   Response.data

Constructing a Response
-----------------------

.. autosummary::
   :toctree: generated/

   Response.from_future
   Response.from_samples

Transformations
---------------

.. autosummary::
   :toctree: generated/

   Response.change_vartype
   Response.relabel_variables

Copy
----

.. autosummary::
   :toctree: generated/

   Response.copy

Done
----

.. autosummary::
   :toctree: generated/

   Response.done
