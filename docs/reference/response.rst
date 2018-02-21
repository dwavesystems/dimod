.. _response:

========
Response
========

Overview
========

.. automodule:: dimod.response

.. currentmodule:: dimod
.. autoclass:: Response


Properties
==========

.. autosummary::
   :toctree: generated/

   Response.samples_matrix
   Response.data_vectors

Methods
=======

Viewing a Response
------------------

.. autosummary::
   :toctree: generated/

   Response.samples
   Response.data

Constructing or updating a Response
-----------------------------------

.. autosummary::
   :toctree: generated/

   Response.from_matrix
   Response.from_dicts
   Response.from_pandas
   Response.from_futures
   Response.update

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
