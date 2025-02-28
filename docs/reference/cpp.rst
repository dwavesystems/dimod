.. _dimod_cppdocs:

=======
C++ API
=======

This page describes the `dimod` package's C++ API.

.. contents::
    :local:
    :depth: 3

Models
======

For the Python API and descriptions of the various models, see :ref:`dimod_models`.

Binary Quadratic Model (BQM)
----------------------------

.. doxygenclass:: dimod::BinaryQuadraticModel
    :members:
    :project: dimod

Constrained Quadratic Model (CQM)
---------------------------------

.. doxygenclass:: dimod::ConstrainedQuadraticModel
    :members:
    :project: dimod

Constraints
~~~~~~~~~~~

.. doxygenclass:: dimod::Constraint
    :members:
    :project: dimod

Quadratic Model (QM)
--------------------

.. doxygenclass:: dimod::QuadraticModel
    :members:
    :project: dimod

Variable Type (Vartype)
=======================

Vartype
-------

.. doxygenenum:: dimod::Vartype
   :project: dimod

vartype_info
------------

.. doxygenclass:: dimod::vartype_info
    :members:
    :undoc-members:
    :project: dimod

vartype_limits
--------------

.. Todo: vartype_limits. Getting it to look nice is possible but fiddly

.. doxygenclass:: dimod::vartype_limits
    :members:
    :undoc-members:
    :project: dimod


dimod Abstract Base Class (`dimod::abc`)
========================================

.. doxygenclass:: dimod::abc::QuadraticModelBase
    :members:
    :protected-members:
    :project: dimod

.. Todo: dimod lp

dimod Utilities (`dimod::utils`)
================================

.. doxygenfunction:: zip_sort
   :project: dimod
