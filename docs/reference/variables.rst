.. _variables:

=========
Variables
=========

.. currentmodule:: dimod.variables

.. automodule:: dimod.variables

Variables Class
===============

.. autoclass:: Variables

Properties
----------

.. autosummary::
   :toctree: generated/

    ~Variables.is_range

Methods
-------

.. autosummary::
   :toctree: generated/

    ~Variables.index
    ~Variables.count
    ~Variables.to_serializable

Mutation Methods
----------------

.. Caution::

   The :class:`.Variables` class comes with a number of semi-private methods
   that allow other classes to manipulate its contents. These are intended to
   be used by parent classes, not by the user. Modifying a :class:`.Variables`
   object that is an attribute of a class results in undefined behaviour.

.. autosummary::
   :toctree: generated/

    ~Variables._append
    ~Variables._extend
    ~Variables._pop
    ~Variables._relabel
    ~Variables._relabel_as_integers
